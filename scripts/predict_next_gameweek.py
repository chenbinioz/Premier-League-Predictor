#!/usr/bin/env python
"""
predict_next_gameweek.py
========================
Pre-match inference pipeline for the 2026/27 EPL season.

Workflow per fixture
--------------------
1. Fetch upcoming fixtures for the target gameweek (API-Football).
2. Skip any fixture that already has a locked prediction in the DB.
3. Load live team states (Elo + rolling xG/goals) from live_team_states.
4. Build the 75-feature NDC vector and 18-feature XGBoost vector.
5. Run Neural Dixon-Coles → λ, μ, ρ → scoreline grid → 1X2 probs.
6. Run XGBoost calibrated classifier → blended 1X2 probs.
7. Lock prediction into predictions_26_27.

Usage
-----
  python scripts/predict_next_gameweek.py --gameweek 1
  python scripts/predict_next_gameweek.py --gameweek 1 --dry-run
  python scripts/predict_next_gameweek.py --gameweek 1 --no-fetch --db data/live/epl_2627.db

Flags
-----
  --gameweek  INT    Target gameweek (required).
  --db        PATH   Path to SQLite DB (default: data/live/epl_2627.db).
  --dry-run          Print predictions without writing to DB.
  --no-fetch         Skip the API call; use fixtures already in the DB.
  --blend     FLOAT  Weight on NDC vs XGBoost (0=XGBoost only, 1=NDC only, default 0.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Must be set before torch import to prevent macOS OpenMP segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import torch
from dotenv import load_dotenv

load_dotenv()

from models.neural_dixon_coles import (
    NeuralDixonColes,
    parse_grid_outputs,
    predict_scoreline_grid,
)
from src.data.state_manager import TeamStateManager
from src.data.live_fetchers import fetch_upcoming_fixtures

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH      = REPO_ROOT / "data" / "live" / "epl_2627.db"
NDC_MODEL_PATH       = REPO_ROOT / "models" / "bin" / "neural_dixon_coles.pt"
NDC_SCALER_PATH      = REPO_ROOT / "models" / "bin" / "ndc_scaler.joblib"
NDC_FEATURES_PATH    = REPO_ROOT / "models" / "bin" / "ndc_feature_names.json"
XGB_MODEL_PATH       = REPO_ROOT / "models" / "calibrated_xgb_outcome.pkl"

# ── Seasonal mean imputation constants ────────────────────────────────────────
# Pre-computed from last 2 seasons of epl_model_features.csv.
# Used for features not tracked in live_team_states (shots, corners, fouls).
_SEASON_MEANS: dict[str, float] = {
    "Home_ShotsOnTarget_roll3":  4.3239,
    "Away_ShotsOnTarget_roll3":  4.4453,
    "Home_Corners_roll3":        5.0189,
    "Away_Corners_roll3":        5.1171,
    "Home_Fouls_roll3":         10.9542,
    "Away_Fouls_roll3":         10.8878,
    "Home_ShotsOnTarget_roll5":  4.3453,
    "Away_ShotsOnTarget_roll5":  4.4397,
    "Home_Corners_roll5":        5.0452,
    "Away_Corners_roll5":        5.1064,
    "Home_Fouls_roll5":         10.9576,
    "Away_Fouls_roll5":         10.8818,
    "Home_ShotsOnTarget_roll10": 4.3995,
    "Away_ShotsOnTarget_roll10": 4.4229,
    "Home_Corners_roll10":       5.0686,
    "Away_Corners_roll10":       5.1075,
    "Home_Fouls_roll10":        10.9333,
    "Away_Fouls_roll10":        10.9259,
    "Home_xG_Created_Venue_roll5":  1.7912,
    "Home_xG_Conceded_Venue_roll5": 1.5308,
    "Away_xG_Created_Venue_roll5":  1.5364,
    "Away_xG_Conceded_Venue_roll5": 1.7850,
}

# ── XGBoost feature list (must match training order exactly) ──────────────────
XGB_FEATURE_COLS = [
    "Elo_Diff", "Home_Elo", "Away_Elo",
    "xG_Attack_Diff_roll3", "xG_Defense_Diff_roll3",
    "xG_Attack_Diff_roll5", "xG_Defense_Diff_roll5",
    "xG_Attack_Diff_roll10", "xG_Defense_Diff_roll10",
    "Corner_Diff_roll5", "Foul_Diff_roll5",
    "Venue_xG_Attack_Diff", "Expected_Match_xG",
    "Rest_Diff", "Congestion_Diff",
    "Bookie_Prob_H", "Bookie_Prob_D", "Bookie_Prob_A",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_ndc() -> tuple[NeuralDixonColes, Any, list[str]] | tuple[None, None, None]:
    """Load the Neural Dixon-Coles model, scaler, and feature name list."""
    for p in [NDC_MODEL_PATH, NDC_SCALER_PATH, NDC_FEATURES_PATH]:
        if not p.exists():
            logger.warning("NDC artifact missing: %s — NDC inference disabled.", p)
            return None, None, None

    checkpoint   = torch.load(NDC_MODEL_PATH, map_location="cpu", weights_only=True)
    n_features   = checkpoint["n_features"]
    ndc_model    = NeuralDixonColes(n_features=n_features)
    ndc_model.load_state_dict(checkpoint["state_dict"])
    ndc_model.eval()

    scaler        = joblib.load(NDC_SCALER_PATH)
    feature_names = json.loads(NDC_FEATURES_PATH.read_text())

    logger.info("Loaded NDC model (%d features).", n_features)
    return ndc_model, scaler, feature_names


def _load_xgb() -> Any | None:
    """Load the calibrated XGBoost classifier."""
    if not XGB_MODEL_PATH.exists():
        logger.warning("XGBoost model missing at %s — XGB inference disabled.", XGB_MODEL_PATH)
        return None
    model = joblib.load(XGB_MODEL_PATH)
    logger.info("Loaded XGBoost calibrated classifier.")
    return model


# ── Feature vector builders ───────────────────────────────────────────────────

def build_ndc_feature_vector(
    home_state: dict[str, Any],
    away_state: dict[str, Any],
    feature_names: list[str],
    bookie_h: float = 0.40,
    bookie_d: float = 0.27,
    bookie_a: float = 0.33,
) -> np.ndarray:
    """
    Construct the 75-element feature vector expected by the NDC model.

    Features tracked in live_team_states are read directly.
    Features not tracked (shots on target, corners, fouls, rest, venue splits)
    are imputed from seasonal means or sensible defaults.

    Args:
        home_state:    Dict from TeamStateManager.get_team_state().
        away_state:    Dict from TeamStateManager.get_team_state().
        feature_names: Ordered list from ndc_feature_names.json.
        bookie_h/d/a:  Prior 1X2 odds fractions (sum to 1.0).

    Returns:
        np.ndarray of shape (1, len(feature_names)), dtype float32.
    """
    h = home_state
    a = away_state

    # Helper: use avg from state if window is non-empty, else seasonal mean
    def _hv(key: str, fallback: float = 0.0) -> float:
        return float(h.get(key, fallback) or fallback)

    def _av(key: str, fallback: float = 0.0) -> float:
        return float(a.get(key, fallback) or fallback)

    h_elo     = _hv("current_elo", 1500.0)
    a_elo     = _av("current_elo", 1500.0)
    h_xg_for  = _hv("avg_xg_for",  1.5)
    h_xg_con  = _hv("avg_xg_against", 1.2)
    a_xg_for  = _av("avg_xg_for",  1.3)
    a_xg_con  = _av("avg_xg_against", 1.3)
    h_g_for   = _hv("avg_goals_for",  1.4)
    h_g_con   = _hv("avg_goals_against", 1.2)
    a_g_for   = _av("avg_goals_for",  1.2)
    a_g_con   = _av("avg_goals_against", 1.3)

    # Build lookup for all 75 features
    feature_map: dict[str, float] = {
        # Elo
        "Home_Elo":   h_elo,
        "Away_Elo":   a_elo,

        # Last-match single stats (use rolling averages as proxy)
        "Home_GoalsScored":       h_g_for,
        "Home_GoalsConceded":     h_g_con,
        "Home_xG_Created":        h_xg_for,
        "Home_xG_Conceded":       h_xg_con,
        "Home_ShotsOnTarget":     _SEASON_MEANS["Home_ShotsOnTarget_roll5"],
        "Home_Opp_ShotsOnTarget": _SEASON_MEANS["Away_ShotsOnTarget_roll5"],
        "Home_YellowCards":       1.6,
        "Home_RedCards":          0.05,
        "Home_Corners":           _SEASON_MEANS["Home_Corners_roll5"],
        "Home_Fouls":             _SEASON_MEANS["Home_Fouls_roll5"],
        "Home_Rest_Days":         4.0,
        "Home_Congestion_Flag":   0.0,

        "Away_GoalsScored":       a_g_for,
        "Away_GoalsConceded":     a_g_con,
        "Away_xG_Created":        a_xg_for,
        "Away_xG_Conceded":       a_xg_con,
        "Away_ShotsOnTarget":     _SEASON_MEANS["Away_ShotsOnTarget_roll5"],
        "Away_Opp_ShotsOnTarget": _SEASON_MEANS["Home_ShotsOnTarget_roll5"],
        "Away_YellowCards":       1.7,
        "Away_RedCards":          0.05,
        "Away_Corners":           _SEASON_MEANS["Away_Corners_roll5"],
        "Away_Fouls":             _SEASON_MEANS["Away_Fouls_roll5"],
        "Away_Rest_Days":         4.0,
        "Away_Congestion_Flag":   0.0,

        # Rolling stats — xG and goals from live state
        "Home_GoalsScored_roll3":   h_g_for,
        "Home_GoalsConceded_roll3": h_g_con,
        "Home_xG_Created_roll3":    h_xg_for,
        "Home_xG_Conceded_roll3":   h_xg_con,
        "Away_GoalsScored_roll3":   a_g_for,
        "Away_GoalsConceded_roll3": a_g_con,
        "Away_xG_Created_roll3":    a_xg_for,
        "Away_xG_Conceded_roll3":   a_xg_con,

        "Home_GoalsScored_roll5":   h_g_for,
        "Home_GoalsConceded_roll5": h_g_con,
        "Home_xG_Created_roll5":    h_xg_for,
        "Home_xG_Conceded_roll5":   h_xg_con,
        "Away_GoalsScored_roll5":   a_g_for,
        "Away_GoalsConceded_roll5": a_g_con,
        "Away_xG_Created_roll5":    a_xg_for,
        "Away_xG_Conceded_roll5":   a_xg_con,

        "Home_GoalsScored_roll10":   h_g_for,
        "Home_GoalsConceded_roll10": h_g_con,
        "Home_xG_Created_roll10":    h_xg_for,
        "Home_xG_Conceded_roll10":   h_xg_con,
        "Away_GoalsScored_roll10":   a_g_for,
        "Away_GoalsConceded_roll10": a_g_con,
        "Away_xG_Created_roll10":    a_xg_for,
        "Away_xG_Conceded_roll10":   a_xg_con,

        # Rolling shots / corners / fouls — seasonal means
        "Home_ShotsOnTarget_roll3":  _SEASON_MEANS["Home_ShotsOnTarget_roll3"],
        "Away_ShotsOnTarget_roll3":  _SEASON_MEANS["Away_ShotsOnTarget_roll3"],
        "Home_Corners_roll3":        _SEASON_MEANS["Home_Corners_roll3"],
        "Away_Corners_roll3":        _SEASON_MEANS["Away_Corners_roll3"],
        "Home_Fouls_roll3":          _SEASON_MEANS["Home_Fouls_roll3"],
        "Away_Fouls_roll3":          _SEASON_MEANS["Away_Fouls_roll3"],
        "Home_ShotsOnTarget_roll5":  _SEASON_MEANS["Home_ShotsOnTarget_roll5"],
        "Away_ShotsOnTarget_roll5":  _SEASON_MEANS["Away_ShotsOnTarget_roll5"],
        "Home_Corners_roll5":        _SEASON_MEANS["Home_Corners_roll5"],
        "Away_Corners_roll5":        _SEASON_MEANS["Away_Corners_roll5"],
        "Home_Fouls_roll5":          _SEASON_MEANS["Home_Fouls_roll5"],
        "Away_Fouls_roll5":          _SEASON_MEANS["Away_Fouls_roll5"],
        "Home_ShotsOnTarget_roll10": _SEASON_MEANS["Home_ShotsOnTarget_roll10"],
        "Away_ShotsOnTarget_roll10": _SEASON_MEANS["Away_ShotsOnTarget_roll10"],
        "Home_Corners_roll10":       _SEASON_MEANS["Home_Corners_roll10"],
        "Away_Corners_roll10":       _SEASON_MEANS["Away_Corners_roll10"],
        "Home_Fouls_roll10":         _SEASON_MEANS["Home_Fouls_roll10"],
        "Away_Fouls_roll10":         _SEASON_MEANS["Away_Fouls_roll10"],

        # Venue-split xG — fallback to overall rolling
        "Home_xG_Created_Venue_roll5":  h_xg_for,
        "Home_xG_Conceded_Venue_roll5": h_xg_con,
        "Away_xG_Created_Venue_roll5":  a_xg_for,
        "Away_xG_Conceded_Venue_roll5": a_xg_con,

        # Bookie odds proxy (from NDC prior or caller-supplied)
        "B365H": 1.0 / bookie_h if bookie_h > 0 else 3.0,
        "B365D": 1.0 / bookie_d if bookie_d > 0 else 3.7,
        "B365A": 1.0 / bookie_a if bookie_a > 0 else 3.0,
    }

    # Assemble in the exact order recorded during training
    vec = np.array([feature_map.get(f, 0.0) for f in feature_names], dtype=np.float32)
    return vec.reshape(1, -1)


def build_xgb_feature_vector(
    home_state: dict[str, Any],
    away_state: dict[str, Any],
    bookie_h: float,
    bookie_d: float,
    bookie_a: float,
) -> np.ndarray:
    """
    Build the 18-element XGBoost feature vector from live team states.

    Returns np.ndarray of shape (1, 18).
    """
    h_elo    = float(home_state.get("current_elo", 1500.0))
    a_elo    = float(away_state.get("current_elo", 1500.0))
    h_xg_for = float(home_state.get("avg_xg_for",  1.5) or 1.5)
    h_xg_con = float(home_state.get("avg_xg_against", 1.2) or 1.2)
    a_xg_for = float(away_state.get("avg_xg_for",  1.3) or 1.3)
    a_xg_con = float(away_state.get("avg_xg_against", 1.3) or 1.3)

    h_cor = _SEASON_MEANS["Home_Corners_roll5"]
    a_cor = _SEASON_MEANS["Away_Corners_roll5"]
    h_foul = _SEASON_MEANS["Home_Fouls_roll5"]
    a_foul = _SEASON_MEANS["Away_Fouls_roll5"]

    # Venue xG splits — use overall rolling as proxy
    h_venue_xg_for = float(home_state.get("avg_xg_for",  1.7) or 1.7)
    a_venue_xg_con = float(away_state.get("avg_xg_against", 1.5) or 1.5)

    # Derived differentials (mirrors app.py derivation logic)
    elo_diff             = h_elo - a_elo
    xg_attack_diff_roll3 = h_xg_for - a_xg_con
    xg_defense_diff_roll3 = a_xg_for - h_xg_con
    xg_attack_diff_roll5 = h_xg_for - a_xg_con
    xg_defense_diff_roll5 = a_xg_for - h_xg_con
    xg_attack_diff_roll10 = h_xg_for - a_xg_con
    xg_defense_diff_roll10 = a_xg_for - h_xg_con
    corner_diff_roll5    = h_cor - a_cor
    foul_diff_roll5      = h_foul - a_foul
    venue_xg_attack_diff = h_venue_xg_for - a_venue_xg_con
    expected_match_xg    = h_xg_for + a_xg_for

    vec = np.array([
        elo_diff,
        h_elo, a_elo,
        xg_attack_diff_roll3,  xg_defense_diff_roll3,
        xg_attack_diff_roll5,  xg_defense_diff_roll5,
        xg_attack_diff_roll10, xg_defense_diff_roll10,
        corner_diff_roll5,
        foul_diff_roll5,
        venue_xg_attack_diff,
        expected_match_xg,
        0.0,   # Rest_Diff         — not tracked, use 0
        0.0,   # Congestion_Diff   — not tracked, use 0
        bookie_h, bookie_d, bookie_a,
    ], dtype=np.float32)
    return vec.reshape(1, -1)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_ndc_inference(
    ndc_model: NeuralDixonColes,
    scaler: Any,
    feature_vec: np.ndarray,
) -> dict[str, Any]:
    """Run NDC inference and return λ, μ, ρ + parsed grid outputs."""
    scaled = scaler.transform(feature_vec).astype(np.float32)
    x_t    = torch.from_numpy(scaled)

    with torch.no_grad():
        ndc_model.eval()
        lam_t, mu_t, rho_t = ndc_model(x_t)

    lam = float(lam_t.item())
    mu  = float(mu_t.item())
    rho = float(rho_t.item())

    grid   = predict_scoreline_grid(lam, mu, rho, max_goals=6)
    parsed = parse_grid_outputs(grid, top_k=5)

    return {"lambda": lam, "mu": mu, "rho": rho, "grid": grid, "parsed": parsed}


def run_xgb_inference(
    xgb_model: Any,
    feature_vec: np.ndarray,
) -> tuple[float, float, float]:
    """Run XGBoost calibrated classifier. Returns (home_win%, draw%, away_win%)."""
    import pandas as pd
    X = pd.DataFrame(feature_vec, columns=XGB_FEATURE_COLS)
    probs = xgb_model.predict_proba(X)[0]
    # Calibrated classifiers return [away, draw, home] order (label 0,1,2)
    return float(probs[2]), float(probs[1]), float(probs[0])


def _blend_probs(
    ndc_h: float, ndc_d: float, ndc_a: float,
    xgb_h: float, xgb_d: float, xgb_a: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Weighted average of NDC and XGBoost 1X2 probabilities."""
    h = alpha * ndc_h + (1 - alpha) * xgb_h
    d = alpha * ndc_d + (1 - alpha) * xgb_d
    a = alpha * ndc_a + (1 - alpha) * xgb_a
    total = h + d + a
    return h / total, d / total, a / total


# ── Prediction derivation ─────────────────────────────────────────────────────

def _ndc_implied_odds(
    ndc_h: float, ndc_d: float, ndc_a: float
) -> tuple[float, float, float]:
    """
    Convert NDC 1X2 probabilities (0-1) to implied bookie probability fractions.
    Adds a small margin so they sum to slightly more than 1 (realistic).
    """
    margin = 1.05
    raw_h = ndc_h * margin
    raw_d = ndc_d * margin
    raw_a = ndc_a * margin
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def _top1_score(parsed: dict) -> str:
    top = parsed.get("top_scorelines", [{}])
    return top[0].get("scoreline", "?-?") if top else "?-?"


def _topk_scores(parsed: dict, top_n: int = 3) -> list[str]:
    """Return the top-K scoreline strings from the NDC grid."""
    top = parsed.get("top_scorelines", [])
    return [item.get("scoreline", "?-?") for item in top[:top_n] if item.get("scoreline")]


# ── CLI orchestrator ──────────────────────────────────────────────────────────

def predict_gameweek(
    gameweek: int,
    db_path:  Path,
    dry_run:  bool,
    no_fetch: bool,
    blend:    float,
    force:    bool = False,
) -> None:
    """Main prediction loop for a given gameweek."""

    # ── Load models ────────────────────────────────────────────────────────────
    ndc_model, ndc_scaler, ndc_feature_names = _load_ndc()
    xgb_model = _load_xgb()

    if not ndc_model and not xgb_model:
        logger.error("No models loaded — aborting.")
        sys.exit(1)

    ndc_ok = ndc_model is not None
    xgb_ok = xgb_model is not None

    # ── Fetch or load fixtures ─────────────────────────────────────────────────
    with TeamStateManager(db_path) as sm:
        if no_fetch:
            pending = [
                f for f in sm.get_pending_fixtures()
                if f["gameweek"] == gameweek
            ]
            logger.info("--no-fetch: using %d pending fixtures from DB.", len(pending))
            fixtures = [
                {
                    "api_match_id": None,
                    "db_match_id":  f["match_id"],   # carry the DB PK through
                    "gameweek":     f["gameweek"],
                    "match_date":   f["match_date"],
                    "home_team":    f["home_team"],
                    "away_team":    f["away_team"],
                }
                for f in pending
            ]
        else:
            fixtures = fetch_upcoming_fixtures(gameweek)
            logger.info("Fetched %d fixtures from API-Football for GW%d.", len(fixtures), gameweek)

            # Upsert fixtures into DB
            existing_fix = {
                (f["home_team"], f["away_team"]): f["match_id"]
                for f in sm.get_pending_fixtures()
            }
            for fix in fixtures:
                key = (fix["home_team"], fix["away_team"])
                if key not in existing_fix:
                    mid = sm.add_fixture(
                        gameweek=fix["gameweek"],
                        match_date=fix["match_date"],
                        home_team=fix["home_team"],
                        away_team=fix["away_team"],
                    )
                    fix["db_match_id"] = mid
                else:
                    fix["db_match_id"] = existing_fix[key]

        # ── Check for already-locked predictions ───────────────────────────────
        import sqlite3
        conn_check = sqlite3.connect(db_path)
        locked_ids = set()
        if not force:
            locked_ids = {
                r[0] for r in conn_check.execute(
                    "SELECT match_id FROM predictions_26_27"
                ).fetchall()
            }
        conn_check.close()

        # ── Per-fixture inference loop ──────────────────────────────────────────
        results_table = []

        for fix in fixtures:
            home, away = fix["home_team"], fix["away_team"]
            db_mid = fix.get("db_match_id") or fix.get("match_id")

            if db_mid and db_mid in locked_ids and not force:
                logger.info("⏭  GW%d: %s vs %s — prediction already locked, skipping.",
                            gameweek, home, away)
                continue

            # Load team states
            try:
                home_state = sm.get_team_state(home)
                away_state = sm.get_team_state(away)
            except KeyError as e:
                logger.warning("Team state not found: %s — skipping fixture.", e)
                continue

            # ── NDC inference ──────────────────────────────────────────────────
            ndc_result = None
            bookie_h, bookie_d, bookie_a = 0.40, 0.27, 0.33  # neutral prior

            if ndc_ok:
                ndc_vec    = build_ndc_feature_vector(
                    home_state, away_state, ndc_feature_names,
                    bookie_h=bookie_h, bookie_d=bookie_d, bookie_a=bookie_a,
                )
                ndc_result = run_ndc_inference(ndc_model, ndc_scaler, ndc_vec)
                ndc_p      = ndc_result["parsed"]
                ndc_h      = ndc_p["home_win"] / 100.0
                ndc_d      = ndc_p["draw"]     / 100.0
                ndc_a      = ndc_p["away_win"] / 100.0

                # Update bookie proxy from NDC output (implied margin)
                bookie_h, bookie_d, bookie_a = _ndc_implied_odds(ndc_h, ndc_d, ndc_a)

            # ── XGBoost inference ──────────────────────────────────────────────
            xgb_h, xgb_d, xgb_a = 0.40, 0.27, 0.33  # fallback
            if xgb_ok:
                xgb_vec        = build_xgb_feature_vector(
                    home_state, away_state, bookie_h, bookie_d, bookie_a
                )
                xgb_h, xgb_d, xgb_a = run_xgb_inference(xgb_model, xgb_vec)

            # ── Blend ──────────────────────────────────────────────────────────
            if ndc_ok and xgb_ok:
                final_h, final_d, final_a = _blend_probs(
                    ndc_h, ndc_d, ndc_a,
                    xgb_h, xgb_d, xgb_a,
                    alpha=blend,
                )
            elif ndc_ok:
                final_h, final_d, final_a = ndc_h, ndc_d, ndc_a
            else:
                final_h, final_d, final_a = xgb_h, xgb_d, xgb_a

            top_score = (
                _top1_score(ndc_result["parsed"])
                if ndc_result else f"{int(round(final_h*2))}-{int(round(final_a*2))}"
            )

            if ndc_result:
                score_candidates = _topk_scores(ndc_result["parsed"], top_n=3)
                if score_candidates:
                    top_score = " • ".join(score_candidates)
                else:
                    top_score = _top1_score(ndc_result["parsed"])
            else:
                top_score = (
                    f"{int(round(final_h*2))}-{int(round(final_a*2))} • "
                    f"{int(round(final_h*1.5))}-{int(round(final_a*1.2))} • "
                    f"{int(round(final_h*1.2))}-{int(round(final_a*1.5))}"
                )

            # ── Record ─────────────────────────────────────────────────────────
            results_table.append({
                "GW":        gameweek,
                "Home":      home,
                "Away":      away,
                "H%":        f"{final_h:.1%}",
                "D%":        f"{final_d:.1%}",
                "A%":        f"{final_a:.1%}",
                "NDC%":      f"{ndc_h:.1%} / {ndc_d:.1%} / {ndc_a:.1%}" if ndc_result else "—",
                "XGB%":      f"{xgb_h:.1%} / {xgb_d:.1%} / {xgb_a:.1%}",
                "Score":     top_score,
                "λ":         f"{ndc_result['lambda']:.2f}" if ndc_result else "—",
                "μ":         f"{ndc_result['mu']:.2f}"     if ndc_result else "—",
                "db_mid":    db_mid,
                # Raw floats for DB write
                "_h": final_h, "_d": final_d, "_a": final_a,
                "_ndc_h": ndc_h if ndc_result else None,
                "_ndc_d": ndc_d if ndc_result else None,
                "_ndc_a": ndc_a if ndc_result else None,
                "_xgb_h": xgb_h,
                "_xgb_d": xgb_d,
                "_xgb_a": xgb_a,
                "_ndc_lam":  ndc_result["lambda"]  if ndc_result else None,
                "_ndc_mu":   ndc_result["mu"]       if ndc_result else None,
                "_ndc_rho":  ndc_result["rho"]      if ndc_result else None,
            })

        # ── Print summary table ────────────────────────────────────────────────
        print(f"\n{'━' * 86}")
        print(f"  GW{gameweek} PREDICTIONS {'(DRY RUN — not written to DB)' if dry_run else ''}")
        print(f"{'━' * 86}")
        header = f"  {'Home':<22} {'Away':<22} {'Final':>18} {'NDC':>18} {'XGB':>18}  {'Score':>8}  {'λ':>5} {'μ':>5}"
        print(header)
        print(f"  {'-' * 82}")
        for r in results_table:
            print(
                f"  {r['Home']:<22} {r['Away']:<22} {r['H%']:>18} {r['NDC%']:>18} {r['XGB%']:>18}"
                f"  {r['Score']:>8}  {r['λ']:>5} {r['μ']:>5}"
            )
        print(f"{'━' * 86}")
        print(f"  Blend: NDC×{blend:.0%} + XGB×{1-blend:.0%} | "
              f"NDC={'✓' if ndc_ok else '✗'} | XGB={'✓' if xgb_ok else '✗'}")
        print(f"{'━' * 86}\n")

        # ── Write to DB ────────────────────────────────────────────────────────
        if not dry_run:
            written = 0
            for r in results_table:
                if r["db_mid"] is None:
                    logger.warning("No DB match_id for %s vs %s — skipping DB write.", r["Home"], r["Away"])
                    continue
                sm.lock_prediction(
                    match_id=r["db_mid"],
                    home_prob=r["_h"],
                    draw_prob=r["_d"],
                    away_prob=r["_a"],
                    predicted_score=r["Score"],
                    ndc_home_prob=r["_ndc_h"],
                    ndc_draw_prob=r["_ndc_d"],
                    ndc_away_prob=r["_ndc_a"],
                    xgb_home_prob=r["_xgb_h"],
                    xgb_draw_prob=r["_xgb_d"],
                    xgb_away_prob=r["_xgb_a"],
                    ndc_lambda=r["_ndc_lam"],
                    ndc_mu=r["_ndc_mu"],
                    ndc_rho=r["_ndc_rho"],
                )
                written += 1
            logger.info("✅  Locked %d predictions into %s.", written, db_path)
        else:
            logger.info("Dry-run complete — no DB writes performed.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def get_auto_gameweek(db_path: Path) -> int:
    import sqlite3
    if not db_path.exists():
        logger.error("Database %s does not exist. Cannot determine automatic gameweek.", db_path)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT MIN(gameweek) FROM fixtures_26_27 WHERE status = 'pending'")
    row = cur.fetchone()
    conn.close()
    if row and row[0] is not None:
        return int(row[0])
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock pre-match predictions for a Premier League gameweek.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gameweek", "-g", type=int, required=False,
        help="Target gameweek number (1–38). Required if --auto is not set.",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Automatically predict the next gameweek with pending fixtures.",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help="Path to the SQLite live database.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print predictions without writing them to the database.",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip API-Football call; use pending fixtures already in the DB.",
    )
    parser.add_argument(
        "--blend", type=float, default=0.5,
        help="Weight on NDC vs XGBoost (0=XGBoost only, 1=NDC only).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate predictions even for matches already locked in the live database.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    target_gw = args.gameweek
    if args.auto:
        target_gw = get_auto_gameweek(args.db)
        logger.info("Auto mode: Selected Gameweek %d for prediction.", target_gw)
    elif target_gw is None:
        logger.error("Either --gameweek (-g) or --auto must be provided.")
        sys.exit(1)
        
    predict_gameweek(
        gameweek=target_gw,
        db_path=args.db,
        dry_run=args.dry_run,
        no_fetch=args.no_fetch,
        blend=args.blend,
        force=args.force,
    )
