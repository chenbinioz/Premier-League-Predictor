"""
Premier League Predictive Engine — Streamlit App
=================================================
Combines:
  • XGBoost Calibrated Classifier  → 1X2 outcome probabilities
  • Neural Dixon-Coles (PyTorch)   → λ, μ, ρ → scoreline grid & 1X2 implied
  • Static Dixon-Coles MLE Model  → Baseline comparison
"""

# ── MUST be set before any `import torch` to prevent the OpenMP / macOS
#    segfault that occurs when PyTorch 2.x initialises its thread pool
#    inside Streamlit's forked worker process. ──────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────

import json
import re
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from scipy.stats import poisson
import sqlite3
from sklearn.metrics import log_loss
from dotenv import load_dotenv
load_dotenv()

# Path setup so we can import the models package
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from models.neural_dixon_coles import (
    NeuralDixonColes,
    parse_grid_outputs,
    predict_scoreline_grid,
)
from scripts.predict_next_gameweek import (
    build_ndc_feature_vector,
    build_xgb_feature_vector,
    run_ndc_inference,
    run_xgb_inference,
)
from src.data.live_fetchers import fetch_from_football_data_csv, normalize_team_name

# Page Configuration
st.set_page_config(
    page_title="Premier League Predictive Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS — Light Premium Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background — light theme */
    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 50%, #eef2f7 100%); }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1d4ed8, #6d28d9, #be185d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #4b5563;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Section titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #cbd5e1, transparent);
        margin-left: 0.75rem;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.25rem 1rem;
        transition: border-color 0.2s ease;
        box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    }
    [data-testid="metric-container"]:hover { border-color: rgba(99, 102, 241, 0.5); }
    [data-testid="metric-container"] label { color: #4b5563 !important; font-size: 0.8rem !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #1e293b !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }

    /* Generate button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
    .stButton > button:active { transform: translateY(0); }

    /* Selectboxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 10px !important;
        color: #1e293b !important;
    }

    /* Comparison table */
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .compare-table th {
        background: rgba(99, 102, 241, 0.08);
        color: #4338ca;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.6rem 0.9rem;
        text-align: left;
        border-bottom: 1px solid rgba(99, 102, 241, 0.15);
    }
    .compare-table td {
        color: #1e293b;
        font-size: 0.9rem;
        padding: 0.55rem 0.9rem;
        border-bottom: 1px solid rgba(203, 213, 225, 0.8);
    }
    .compare-table tr:hover td { background: rgba(99, 102, 241, 0.04); }
    .compare-table .highlight { color: #2563eb; font-weight: 600; }

    /* Badges */
    .badge-ndc {
        display: inline-block;
        background: rgba(109, 40, 217, 0.1);
        border: 1px solid rgba(109, 40, 217, 0.3);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: #6d28d9;
        letter-spacing: 0.05em;
        vertical-align: middle;
        margin-left: 0.4rem;
    }
    .badge-xgb {
        display: inline-block;
        background: rgba(190, 24, 93, 0.08);
        border: 1px solid rgba(190, 24, 93, 0.25);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: #be185d;
        letter-spacing: 0.05em;
        vertical-align: middle;
        margin-left: 0.4rem;
    }
    .badge-dc {
        display: inline-block;
        background: rgba(5, 150, 105, 0.08);
        border: 1px solid rgba(5, 150, 105, 0.25);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: #059669;
        letter-spacing: 0.05em;
        vertical-align: middle;
        margin-left: 0.4rem;
    }

    .stAlert { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1. Load Assets & Models
# ---------------------------------------------------------------------------
@st.cache_resource
def load_xgb_assets():
    model    = joblib.load("models/calibrated_xgb_outcome.pkl")
    features = pd.read_csv("data/processed/epl_model_features.csv")

    # Derive differential and engineered features
    features["Elo_Diff"] = features["Home_Elo"] - features["Away_Elo"]
    for w in [3, 5, 10]:
        features[f"xG_Attack_Diff_roll{w}"]  = features[f"Home_xG_Created_roll{w}"]  - features[f"Away_xG_Conceded_roll{w}"]
        features[f"xG_Defense_Diff_roll{w}"] = features[f"Away_xG_Created_roll{w}"]  - features[f"Home_xG_Conceded_roll{w}"]
        features[f"Corner_Diff_roll{w}"]     = features[f"Home_Corners_roll{w}"]      - features[f"Away_Corners_roll{w}"]
        features[f"Foul_Diff_roll{w}"]       = features[f"Home_Fouls_roll{w}"]        - features[f"Away_Fouls_roll{w}"]

    features["Venue_xG_Attack_Diff"] = (
        features["Home_xG_Created_Venue_roll5"] - features["Away_xG_Conceded_Venue_roll5"]
    )
    features["Expected_Match_xG"] = features["Home_xG_Created_roll5"] + features["Away_xG_Created_roll5"]
    features["Rest_Diff"]         = features["Home_Rest_Days"]         - features["Away_Rest_Days"]
    features["Congestion_Diff"]   = features["Home_Congestion_Flag"]   - features["Away_Congestion_Flag"]

    raw_margin              = (1 / features["B365H"]) + (1 / features["B365D"]) + (1 / features["B365A"])
    features["Bookie_Prob_H"] = (1 / features["B365H"]) / raw_margin
    features["Bookie_Prob_D"] = (1 / features["B365D"]) / raw_margin
    features["Bookie_Prob_A"] = (1 / features["B365A"]) / raw_margin

    return model, features

@st.cache_resource
def load_ndc_assets():
    model_path    = "models/bin/neural_dixon_coles.pt"
    scaler_path   = "models/bin/ndc_scaler.joblib"
    features_path = "models/bin/ndc_feature_names.json"

    missing = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]
    if missing:
        return None, None, None

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    n_features = checkpoint["n_features"]
    
    ndc_model = NeuralDixonColes(n_features=n_features)
    ndc_model.load_state_dict(checkpoint["state_dict"])
    ndc_model.eval()

    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_names = json.load(f)

    return ndc_model, scaler, feature_names

@st.cache_data(ttl=10)
def load_live_tracker_data(db_path):
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df_fixtures = pd.read_sql_query("SELECT * FROM fixtures_26_27", conn)
        df_predictions = pd.read_sql_query("SELECT * FROM predictions_26_27", conn)
        conn.close()
        return df_fixtures, df_predictions
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=30)
def load_live_team_states(db_path):
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
        df_states = pd.read_sql_query("SELECT * FROM live_team_states", conn)
        conn.close()
        if df_states.empty:
            return {}
        return df_states.set_index("team_name").to_dict(orient="index")
    except Exception:
        return {}


def build_live_fixture_model_snapshot(home_team: str, away_team: str, live_states: dict[str, dict]) -> dict[str, dict | list[str] | None]:
    home_state = live_states.get(home_team)
    away_state = live_states.get(away_team)
    if not home_state or not away_state:
        return {"xgb": None, "ndc": None}

    bookie_h, bookie_d, bookie_a = 0.40, 0.27, 0.33
    ndc_vec = build_ndc_feature_vector(home_state, away_state, ndc_feature_names, bookie_h, bookie_d, bookie_a)
    ndc_result = run_ndc_inference(ndc_model, ndc_scaler, ndc_vec)
    ndc_probs = {
        "Home": ndc_result["parsed"]["home_win"] / 100.0,
        "Draw": ndc_result["parsed"]["draw"] / 100.0,
        "Away": ndc_result["parsed"]["away_win"] / 100.0,
    }
    ndc_top3 = [item["scoreline"] for item in ndc_result["parsed"]["top_scorelines"][:3]]

    bookie_h, bookie_d, bookie_a = _ndc_implied_odds(ndc_probs["Home"], ndc_probs["Draw"], ndc_probs["Away"])
    xgb_vec = build_xgb_feature_vector(home_state, away_state, bookie_h, bookie_d, bookie_a)
    xgb_home, xgb_draw, xgb_away = run_xgb_inference(xgb_model, xgb_vec)
    xgb_probs = {"Home": xgb_home, "Draw": xgb_draw, "Away": xgb_away}

    return {
        "xgb": {"probs": xgb_probs},
        "ndc": {
            "lambda": ndc_result["lambda"],
            "mu": ndc_result["mu"],
            "rho": ndc_result["rho"],
            "probs": ndc_probs,
            "top3": ndc_top3,
        },
    }


def get_live_tracker_status(db_path: str) -> dict:
    db_exists = os.path.exists(db_path)
    understat_available = False
    try:
        import understatapi  # noqa: F401
        understat_available = True
    except Exception:
        understat_available = False

    if understat_available:
        return {
            "source_name": "Understat (live season feed)",
            "summary": "The tracker is reading live 2026/27 fixtures and results from Understat, with a football-data.co.uk CSV fallback when needed.",
            "api_configured": True,
            "db_exists": db_exists,
        }

    if db_exists:
        return {
            "source_name": "football-data.co.uk CSV fallback",
            "summary": "Understat is unavailable right now, so the tracker is using the football-data.co.uk season CSV as the fallback source.",
            "api_configured": False,
            "db_exists": True,
        }

    return {
        "source_name": "No live data source configured",
        "summary": "No live season feed is available. The tracker is waiting for an active Understat or fallback data source.",
        "api_configured": False,
        "db_exists": False,
    }


def _actual_outcome_label(home_goals, away_goals) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return ""
    home_goals = int(home_goals)
    away_goals = int(away_goals)
    if home_goals > away_goals:
        return "Home"
    if home_goals < away_goals:
        return "Away"
    return "Draw"


def _predicted_outcome_label(home_prob: float, draw_prob: float, away_prob: float) -> str:
    probs = {
        "Home": float(home_prob),
        "Draw": float(draw_prob),
        "Away": float(away_prob),
    }
    return max(probs, key=probs.get)


def _scoreline_string(home_goals, away_goals) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return ""
    return f"{int(home_goals)} - {int(away_goals)}"


def _static_dc_summary(home_team: str, away_team: str, top_n: int = 3) -> dict | None:
    if not dc_params:
        return None
    home_xg, away_xg, grid = _static_dc_grid(home_team, away_team, dc_params, max_goals=6)
    parsed = parse_grid_outputs(grid, top_k=top_n)
    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "parsed": parsed,
        "scorelines": [item["scoreline"] for item in parsed["top_scorelines"]],
    }


def _gameweek_status_label(gw_df: pd.DataFrame, today: pd.Timestamp | None = None) -> str:
    if gw_df.empty:
        return "upcoming"

    today = today or pd.Timestamp.now().normalize()
    statuses = set(gw_df["status"].astype(str).str.lower().tolist())

    if statuses == {"completed"}:
        return "completed"

    dates = pd.to_datetime(gw_df["match_date"], errors="coerce")
    first_date = dates.min()
    if pd.notna(first_date) and today < first_date.normalize():
        return "upcoming"

    if "completed" in statuses and "pending" in statuses:
        return "in progress"

    if pd.notna(first_date) and today >= first_date.normalize():
        return "in progress"

    return "upcoming"


def _parse_scoreline_candidates(raw_value, top_n: int = 3) -> list[str]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    candidates = [part.strip() for part in re.split(r"\s*[•|/]\s*", text) if part.strip()]
    if len(candidates) > 1:
        return candidates[:top_n]
    return [text]


def _probability_triplet(row: pd.Series, prefix: str, fallback_prefix: str | None = None) -> tuple[float, float, float] | None:
    """Read a 1X2 probability triplet from a row, with an optional fallback prefix."""
    prefixes = [prefix]
    if fallback_prefix:
        prefixes.append(fallback_prefix)

    for current_prefix in prefixes:
        values = [
            row.get(f"{current_prefix}_home_prob"),
            row.get(f"{current_prefix}_draw_prob"),
            row.get(f"{current_prefix}_away_prob"),
        ]
        if all(pd.notna(value) for value in values):
            return float(values[0]), float(values[1]), float(values[2])

    return None


def _odds_to_prob_triplet(home_odds: object, draw_odds: object, away_odds: object) -> tuple[float, float, float] | None:
    try:
        h = float(home_odds)
        d = float(draw_odds)
        a = float(away_odds)
    except (TypeError, ValueError):
        return None

    if any(v <= 0 for v in (h, d, a)):
        return None

    implied = (1.0 / h) + (1.0 / d) + (1.0 / a)
    return (1.0 / h) / implied, (1.0 / d) / implied, (1.0 / a) / implied


@st.cache_data(ttl=300)
def _load_bookie_odds_map() -> dict[tuple[str, str, str], tuple[float, float, float]]:
    mapping: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for row in fetch_from_football_data_csv():
        home = normalize_team_name(row.get("home_team"))
        away = normalize_team_name(row.get("away_team"))
        match_date = str(row.get("date") or "")
        if not home or not away or not match_date:
            continue
        triplet = _odds_to_prob_triplet(row.get("B365H"), row.get("B365D"), row.get("B365A"))
        if triplet is not None:
            mapping[(home, away, match_date)] = triplet
    return mapping


def _bookie_prob_triplet_from_row(row: pd.Series) -> tuple[float, float, float] | None:
    odds_h = row.get("B365H")
    odds_d = row.get("B365D")
    odds_a = row.get("B365A")
    if all(pd.notna(v) for v in [odds_h, odds_d, odds_a]):
        return _odds_to_prob_triplet(odds_h, odds_d, odds_a)

    bookie_map = _load_bookie_odds_map()
    key = (
        normalize_team_name(row.get("home_team")),
        normalize_team_name(row.get("away_team")),
        str(row.get("match_date") or ""),
    )
    return bookie_map.get(key)


def _ndc_implied_odds(ndc_h: float, ndc_d: float, ndc_a: float) -> tuple[float, float, float]:
    margin = 1.05
    raw_h = ndc_h * margin
    raw_d = ndc_d * margin
    raw_a = ndc_a * margin
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def _format_hits(hits: int, total: int) -> str:
    return f"{hits}/{total}" if total else "0/0"


def build_gameweek_overview(df_fixtures: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
    if df_fixtures.empty:
        return pd.DataFrame(columns=[
            "Gameweek", "Status", "Fixtures", "Played",
            "Final Prob Hits", "Final Prob Total",
            "NDC Prob Hits", "NDC Prob Total",
            "XGB Prob Hits", "XGB Prob Total",
            "Bookie Prob Hits", "Bookie Prob Total",
            "NDC Score Hits", "NDC Score Total",
            "Static Prob Hits", "Static Prob Total", "Static Score Hits", "Static Score Total",
        ])

    df_merged = pd.merge(df_fixtures, df_predictions, on="match_id", how="left")
    today = pd.Timestamp.now().normalize()
    rows = []

    for gw in range(1, 39):
        gw_df = df_merged[df_merged["gameweek"] == gw].copy()
        if gw_df.empty:
            rows.append({
                "Gameweek": gw,
                "Status": "upcoming",
                "Fixtures": 0,
                "Played": 0,
                "Final Prob Hits": 0,
                "Final Prob Total": 0,
                "NDC Prob Hits": 0,
                "NDC Prob Total": 0,
                "XGB Prob Hits": 0,
                "XGB Prob Total": 0,
                "Bookie Prob Hits": 0,
                "Bookie Prob Total": 0,
                "NDC Score Hits": 0,
                "NDC Score Total": 0,
                "Static Prob Hits": 0,
                "Static Prob Total": 0,
                "Static Score Hits": 0,
                "Static Score Total": 0,
            })
            continue

        total_fixtures = len(gw_df)
        played_df = gw_df[gw_df["status"].astype(str).str.lower() == "completed"].copy()
        gw_status = _gameweek_status_label(gw_df, today=today)

        final_prob_hits = 0
        ndc_prob_hits = 0
        xgb_prob_hits = 0
        bookie_prob_hits = 0
        ndc_score_hits = 0
        static_prob_hits = 0
        static_score_hits = 0
        ndc_total = 0
        xgb_total = 0
        bookie_total = 0

        for _, row in played_df.iterrows():
            actual_outcome = _actual_outcome_label(row["home_goals"], row["away_goals"])
            actual_score = _scoreline_string(row["home_goals"], row["away_goals"])

            final_probs = _probability_triplet(row, "predicted")
            ndc_probs = _probability_triplet(row, "ndc", fallback_prefix="predicted")
            xgb_probs = _probability_triplet(row, "xgb", fallback_prefix="predicted")
            bookie_probs = _bookie_prob_triplet_from_row(row)

            if final_probs:
                final_outcome = _predicted_outcome_label(*final_probs)
                final_prob_hits += int(final_outcome == actual_outcome)

            if ndc_probs:
                ndc_total += 1
                ndc_outcome = _predicted_outcome_label(*ndc_probs)
                ndc_prob_hits += int(ndc_outcome == actual_outcome)

            if xgb_probs:
                xgb_total += 1
                xgb_outcome = _predicted_outcome_label(*xgb_probs)
                xgb_prob_hits += int(xgb_outcome == actual_outcome)

            if bookie_probs:
                bookie_total += 1
                bookie_outcome = _predicted_outcome_label(*bookie_probs)
                bookie_prob_hits += int(bookie_outcome == actual_outcome)

            ndc_candidates = _parse_scoreline_candidates(row.get("predicted_score"), top_n=3)
            ndc_score_hits += int(actual_score in ndc_candidates)

            static_summary = _static_dc_summary(str(row["home_team"]), str(row["away_team"]), top_n=3)
            if static_summary:
                static_parsed = static_summary["parsed"]
                static_outcome = _predicted_outcome_label(
                    static_parsed["home_win"] / 100.0,
                    static_parsed["draw"] / 100.0,
                    static_parsed["away_win"] / 100.0,
                )
                static_prob_hits += int(static_outcome == actual_outcome)
                static_score_hits += int(actual_score in static_summary["scorelines"])

        rows.append({
            "Gameweek": gw,
            "Status": gw_status,
            "Fixtures": total_fixtures,
            "Played": len(played_df),
            "Final Prob Hits": final_prob_hits,
            "Final Prob Total": total_fixtures,
            "NDC Prob Hits": ndc_prob_hits,
            "NDC Prob Total": ndc_total,
            "XGB Prob Hits": xgb_prob_hits,
            "XGB Prob Total": xgb_total,
            "Bookie Prob Hits": bookie_prob_hits,
            "Bookie Prob Total": bookie_total,
            "NDC Score Hits": ndc_score_hits,
            "NDC Score Total": total_fixtures,
            "Static Prob Hits": static_prob_hits,
            "Static Prob Total": total_fixtures,
            "Static Score Hits": static_score_hits,
            "Static Score Total": total_fixtures,
        })

    return pd.DataFrame(rows)


def render_fixture_card(row: pd.Series) -> None:
    home_team = row["home_team"]
    away_team = row["away_team"]
    date_str = row["match_date"]
    status = row["status"]
    live_snapshot = build_live_fixture_model_snapshot(home_team, away_team, live_team_states)

    st.markdown(
        f"""
        <div style="padding: 1rem 1.1rem; border: 1px solid rgba(99,102,241,0.18); border-left: 5px solid #3b82f6; border-radius: 16px; background: rgba(255,255,255,0.85); box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 1rem;">
            <div style="display:flex; justify-content:space-between; gap: .75rem; font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">
                <span>Gameweek {row['gameweek']}</span>
                <span>{date_str}</span>
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 0.6rem; line-height: 1.4;">
                🏠 {home_team} <span style="color:#9ca3af; font-weight: 400;">vs</span> ✈️ {away_team}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    xgb_snapshot = live_snapshot["xgb"]
    ndc_snapshot = live_snapshot["ndc"]
    static_summary = _static_dc_summary(home_team, away_team, top_n=3)

    prob_cols = st.columns(2)
    with prob_cols[0]:
        st.markdown("**XGBoost probabilities**")
        if xgb_snapshot:
            xgb_probs = xgb_snapshot["probs"]
            xgb_outcome = max(xgb_probs, key=xgb_probs.get)
            st.caption(f"Outcome: {xgb_outcome} ({xgb_probs[xgb_outcome]*100:.1f}%)")
            st.caption(
                f"H {xgb_probs['Home']*100:.0f}% | D {xgb_probs['Draw']*100:.0f}% | A {xgb_probs['Away']*100:.0f}%"
            )
        else:
            st.caption("XGBoost snapshot unavailable")

    with prob_cols[1]:
        st.markdown("**Neural Dixon-Coles probabilities**")
        if ndc_snapshot:
            ndc_probs = ndc_snapshot["probs"]
            ndc_outcome = max(ndc_probs, key=ndc_probs.get)
            ndc_top3 = ndc_snapshot["top3"]
            st.caption(f"Outcome: {ndc_outcome} ({ndc_probs[ndc_outcome]*100:.1f}%)")
            st.caption(f"NDC top 3: {' • '.join(ndc_top3)}")
            st.caption(
                f"H {ndc_probs['Home']*100:.0f}% | D {ndc_probs['Draw']*100:.0f}% | A {ndc_probs['Away']*100:.0f}%"
            )
        else:
            st.caption("NDC snapshot unavailable")

    if static_summary:
        st.caption(f"Static DC top 3: {' • '.join(static_summary['scorelines'][:3])}")
    else:
        st.caption("Static DC top 3: unavailable")

    if status == "completed":
        h_g = int(row["home_goals"])
        a_g = int(row["away_goals"])
        actual_outcome = home_team if h_g > a_g else away_team if h_g < a_g else "Draw"
        actual_score = f"{h_g} - {a_g}"

        if xgb_snapshot:
            xgb_probs = xgb_snapshot["probs"]
            xgb_outcome = max(xgb_probs, key=xgb_probs.get)
            xgb_prob_correct = xgb_outcome == actual_outcome
            st.markdown(
                f"<div style='margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.45rem;'>"
                f"<span style='padding: 0.35rem 0.65rem; border-radius: 999px; background: {'#d1fae5' if xgb_prob_correct else '#fee2e2'}; color: {'#065f46' if xgb_prob_correct else '#991b1b'}; font-weight: 700;'>XGBoost Prob {'✅' if xgb_prob_correct else '❌'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if ndc_snapshot:
            ndc_probs = ndc_snapshot["probs"]
            ndc_outcome = max(ndc_probs, key=ndc_probs.get)
            ndc_prob_correct = ndc_outcome == actual_outcome
            ndc_score_correct = actual_score in ndc_snapshot["top3"]
            st.markdown(
                f"<div style='margin-top: 0.35rem; display: flex; flex-wrap: wrap; gap: 0.45rem;'>"
                f"<span style='padding: 0.35rem 0.65rem; border-radius: 999px; background: {'#d1fae5' if ndc_prob_correct else '#fee2e2'}; color: {'#065f46' if ndc_prob_correct else '#991b1b'}; font-weight: 700;'>NDC Prob {'✅' if ndc_prob_correct else '❌'}</span>"
                f"<span style='padding: 0.35rem 0.65rem; border-radius: 999px; background: {'#d1fae5' if ndc_score_correct else '#fee2e2'}; color: {'#065f46' if ndc_score_correct else '#991b1b'}; font-weight: 700;'>NDC Score {'✅' if ndc_score_correct else '❌'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if static_summary:
            static_parsed = static_summary["parsed"]
            static_probs = {
                "Home": static_parsed["home_win"] / 100.0,
                "Draw": static_parsed["draw"] / 100.0,
                "Away": static_parsed["away_win"] / 100.0,
            }
            static_outcome = max(static_probs, key=static_probs.get)
            static_prob_correct = static_outcome == actual_outcome
            static_score_correct = actual_score in static_summary["scorelines"]
            st.markdown(
                f"<div style='margin-top: 0.35rem; display: flex; flex-wrap: wrap; gap: 0.45rem;'>"
                f"<span style='padding: 0.35rem 0.65rem; border-radius: 999px; background: {'#d1fae5' if static_prob_correct else '#fee2e2'}; color: {'#065f46' if static_prob_correct else '#991b1b'}; font-weight: 700;'>Static Prob {'✅' if static_prob_correct else '❌'}</span>"
                f"<span style='padding: 0.35rem 0.65rem; border-radius: 999px; background: {'#d1fae5' if static_score_correct else '#fee2e2'}; color: {'#065f46' if static_score_correct else '#991b1b'}; font-weight: 700;'>Static Score {'✅' if static_score_correct else '❌'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Actual result:** {h_g} - {a_g}")
    else:
        st.caption("⏳ Match pending")

    st.markdown("<hr style='margin: 0.8rem 0 0; border: 0; border-top: 1px solid rgba(203,213,225,0.9);'>", unsafe_allow_html=True)

# XGBoost feature columns
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

# Static Dixon-Coles Grid Helper
def _static_dc_grid(home_team, away_team, dc_params, max_goals=6):
    alpha_h = dc_params["attack"].get(home_team, 1.0)
    beta_h  = dc_params["defense"].get(home_team, 1.0)
    alpha_a = dc_params["attack"].get(away_team, 1.0)
    beta_a  = dc_params["defense"].get(away_team, 1.0)
    gamma   = dc_params["home_adv"]
    home_xg = alpha_h * beta_a * gamma
    away_xg = alpha_a * beta_h
    home_probs = poisson.pmf(np.arange(max_goals + 1), home_xg)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_xg)
    grid = np.outer(home_probs, away_probs)
    
    rho = dc_params.get("rho", 0.0)
    if rho != 0.0:
        eps = 1e-10
        grid[0, 0] *= max(1.0 - home_xg * away_xg * rho, eps)
        grid[1, 0] *= (1.0 + away_xg * rho)
        grid[0, 1] *= (1.0 + home_xg * rho)
        grid[1, 1] *= max(1.0 - rho, eps)
        
    total = grid.sum()
    if total > 0:
        grid /= total
    return home_xg, away_xg, grid

def _static_dc_top_scorelines(home_team, away_team, dc_params, max_goals=6, top_n=5):
    home_xg, away_xg, grid = _static_dc_grid(home_team, away_team, dc_params, max_goals=max_goals)
    scorelines = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scorelines.append({
                "Scoreline": f"{h} - {a}",
                "Probability": grid[h][a]
            })
    df_scores = (
        pd.DataFrame(scorelines)
        .sort_values("Probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return home_xg, away_xg, df_scores

# ---------------------------------------------------------------------------
# Reconstruct Live Elo Standings Helper
# ---------------------------------------------------------------------------
@st.cache_data
def compute_live_elos(df, k_factor=20, home_advantage=60):
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    elo_dict = {}
    last_season = {}
    
    seasons_list = sorted(df_sorted["Season"].unique().tolist())
    previous_season_teams = set()
    
    for season in seasons_list:
        df_season = df_sorted[df_sorted["Season"] == season]
        current_season_teams = set(df_season["HomeTeam"]).union(set(df_season["AwayTeam"]))
        
        if not elo_dict:
            for team in current_season_teams:
                elo_dict[team] = 1500.0
        else:
            for team in current_season_teams:
                if team in previous_season_teams:
                    elo_dict[team] = 0.80 * elo_dict[team] + 0.20 * 1500.0
                else:
                    elo_dict[team] = 1420.0
                    
        previous_season_teams = current_season_teams
        
        for _, row in df_season.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            last_season[home] = season
            last_season[away] = season
            
            home_elo = elo_dict[home]
            away_elo = elo_dict[away]
            
            prob_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0))
            prob_away = 1.0 - prob_home
            
            if row["FTHG"] > row["FTAG"]:
                res_home, res_away = 1.0, 0.0
            elif row["FTHG"] < row["FTAG"]:
                res_home, res_away = 0.0, 1.0
            else:
                res_home, res_away = 0.5, 0.5
                
            elo_dict[home] = home_elo + k_factor * (res_home - prob_home)
            elo_dict[away] = away_elo + k_factor * (res_away - prob_away)
            
    return elo_dict, last_season

# Load data assets
xgb_model, df_features = load_xgb_assets()
ndc_model, ndc_scaler, ndc_feature_names = load_ndc_assets()
ndc_available = ndc_model is not None

dc_params = None
if os.path.exists("models/dc_mle_params.pkl"):
    dc_params = joblib.load("models/dc_mle_params.pkl")

# UI — Hero Header
st.markdown(
    """
    <div class="hero-header">
        <h1>⚽ Premier League Predictive Engine</h1>
        <p>Neural Dixon-Coles · XGBoost Ensemble · Weekend Match Forecasting</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ndc_available:
    st.warning(
        "⚠️ Neural Dixon-Coles artifacts not found in `models/bin/`. "
        "Run `python scripts/train_neural_dixon_coles.py` to generate them.",
        icon="⚠️",
    )

# Load Live database
LIVE_DB_PATH = os.environ.get("LIVE_DB_PATH", "data/live/epl_2627.db")
if not os.path.isabs(LIVE_DB_PATH):
    LIVE_DB_PATH = os.path.join(REPO_ROOT, LIVE_DB_PATH)

df_fixtures, df_predictions = load_live_tracker_data(LIVE_DB_PATH)
live_team_states = load_live_team_states(LIVE_DB_PATH)

# Determine default active gameweek based on dates/pending matches
default_gw = 1
if not df_fixtures.empty:
    pending_fixtures = df_fixtures[df_fixtures["status"] == "pending"]
    if not pending_fixtures.empty:
        default_gw = int(pending_fixtures["gameweek"].min())
    else:
        default_gw = int(df_fixtures["gameweek"].max())
default_gw = max(1, min(default_gw, 38))

# Sidebar selectbox for gameweek navigator
st.sidebar.markdown("### 📊 Live 26/27 Tracker Settings")
selected_gw = st.sidebar.selectbox(
    "Select Gameweek",
    options=list(range(1, 39)),
    index=int(default_gw - 1),
    key="tracker_gameweek"
)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_elo, tab_live = st.tabs(["🔮 Match Forecast", "📈 Live Elo Ratings", "📊 Live 26/27 Tracker"])

# ── TAB 1: Match Forecast ──────────────────────────────────────────────────
with tab_forecast:
    teams = sorted(df_features["HomeTeam"].unique())

    if "has_forecast" not in st.session_state:
        st.session_state["has_forecast"] = False

    sel_col1, sel_col2, btn_col = st.columns([2, 2, 1])
    with sel_col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=0, key="home_team")
    with sel_col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=1, key="away_team")
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Forecast ⚡", key="generate_btn", use_container_width=True)

    if generate:
        st.session_state["has_forecast"] = True
        st.session_state["active_home"] = home_team
        st.session_state["active_away"] = away_team

    if st.session_state.get("has_forecast", False):
        home_team = st.session_state.get("active_home", home_team)
        away_team = st.session_state.get("active_away", away_team)

        if home_team == away_team:
            st.error("Home and Away teams must be different.")
        else:
            use_live_state = (
                live_team_states
                and home_team in live_team_states
                and away_team in live_team_states
            )
            match_vector = df_features[
                (df_features["HomeTeam"] == home_team) &
                (df_features["AwayTeam"] == away_team)
            ].iloc[-1:]

            if not use_live_state and match_vector.empty:
                st.error(f"No historical data found for **{home_team}** vs **{away_team}**.")
            else:
                if use_live_state:
                    st.caption("⚡ **Live Mode**: Forecasting using real-time 2026/27 team states (Elo rating & 10-match rolling form)")
                    snapshot = build_live_fixture_model_snapshot(home_team, away_team, live_team_states)
                    xgb_snap = snapshot.get("xgb")
                    ndc_snap = snapshot.get("ndc")

                    if xgb_snap:
                        xgb_home_w = xgb_snap["probs"]["Home"]
                        xgb_draw   = xgb_snap["probs"]["Draw"]
                        xgb_away_w = xgb_snap["probs"]["Away"]
                    else:
                        xgb_home_w, xgb_draw, xgb_away_w = 0.33, 0.33, 0.34

                    if ndc_snap and ndc_available:
                        ndc_lam  = ndc_snap["lambda"]
                        ndc_mu   = ndc_snap["mu"]
                        ndc_rho  = ndc_snap["rho"]
                        ndc_grid   = predict_scoreline_grid(ndc_lam, ndc_mu, ndc_rho, max_goals=6)
                        ndc_parsed = parse_grid_outputs(ndc_grid, top_k=3)
                    else:
                        ndc_lam, ndc_mu, ndc_rho = None, None, None
                        ndc_grid, ndc_parsed = None, None
                else:
                    # XGBoost fallback
                    X_live_xgb = match_vector[XGB_FEATURE_COLS]
                    xgb_probs  = xgb_model.predict_proba(X_live_xgb)[0]
                    xgb_home_w = float(xgb_probs[2])
                    xgb_draw   = float(xgb_probs[1])
                    xgb_away_w = float(xgb_probs[0])

                    # NDC fallback
                    ndc_lam, ndc_mu, ndc_rho = None, None, None
                    ndc_grid = None
                    ndc_parsed = None

                    if ndc_available:
                        ndc_row = match_vector[ndc_feature_names].values.astype(np.float32)
                        ndc_row_scaled = ndc_scaler.transform(ndc_row)
                        X_live_ndc = torch.from_numpy(ndc_row_scaled)

                        with torch.no_grad():
                            ndc_model.eval()
                            lam_t, mu_t, rho_t = ndc_model(X_live_ndc)

                        ndc_lam  = float(lam_t.item())
                        ndc_mu   = float(mu_t.item())
                        ndc_rho  = float(rho_t.item())
                        ndc_grid   = predict_scoreline_grid(ndc_lam, ndc_mu, ndc_rho, max_goals=6)
                        ndc_parsed = parse_grid_outputs(ndc_grid, top_k=3)

                # Match Overview Metrics
                st.markdown(
                    f'<div class="section-title">📊 Match Overview — {home_team} vs {away_team}</div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric(f"🏠 {home_team} Win", f"{xgb_home_w:.1%}")
                m2.metric("🤝 Draw",             f"{xgb_draw:.1%}")
                m3.metric(f"✈️ {away_team} Win", f"{xgb_away_w:.1%}")

                if ndc_lam is not None:
                    m4.metric("λ Home xG",  f"{ndc_lam:.2f}")
                    m5.metric("μ Away xG",  f"{ndc_mu:.2f}")
                    m6.metric("ρ Draw Bias", f"{ndc_rho:+.3f}")
                else:
                    m4.metric("λ Home xG",  "—")
                    m5.metric("μ Away xG",  "—")
                    m6.metric("ρ Draw Bias", "—")

                # 1X2 Comparison Table
                st.markdown(
                    '<div class="section-title">⚖️ 1X2 Model Comparison</div>',
                    unsafe_allow_html=True,
                )

                if ndc_parsed:
                    ndc_home_w = ndc_parsed["home_win"]
                    ndc_draw   = ndc_parsed["draw"]
                    ndc_away_w = ndc_parsed["away_win"]

                    table_html = f"""
                    <div class="glass-card">
                    <table class="compare-table">
                      <thead>
                        <tr>
                          <th>Outcome</th>
                          <th>XGBoost <span class="badge-xgb">CLF</span></th>
                          <th>Neural Dixon-Coles <span class="badge-ndc">NDC</span></th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>🏠 {home_team} Win</td>
                          <td class="highlight">{xgb_home_w:.1%}</td>
                          <td class="highlight">{ndc_home_w:.1f}%</td>
                        </tr>
                        <tr>
                          <td>🤝 Draw</td>
                          <td>{xgb_draw:.1%}</td>
                          <td>{ndc_draw:.1f}%</td>
                        </tr>
                        <tr>
                          <td>✈️ {away_team} Win</td>
                          <td>{xgb_away_w:.1%}</td>
                          <td>{ndc_away_w:.1f}%</td>
                        </tr>
                      </tbody>
                    </table>
                    </div>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    cols = st.columns(3)
                    cols[0].metric(f"{home_team} Win", f"{xgb_home_w:.1%}")
                    cols[1].metric("Draw",             f"{xgb_draw:.1%}")
                    cols[2].metric(f"{away_team} Win", f"{xgb_away_w:.1%}")

                # Scorelines side-by-side
                st.markdown(
                    '<div class="section-title">🎯 Top 3 Scoreline Candidates</div>',
                    unsafe_allow_html=True,
                )

                score_ndc_col, score_dc_col = st.columns(2, gap="large")

                with score_ndc_col:
                    st.markdown('Neural Dixon-Coles <span class="badge-ndc">NDC</span>', unsafe_allow_html=True)
                    if ndc_parsed:
                        df_ndc = pd.DataFrame({
                            "Scoreline": [s["scoreline"] for s in ndc_parsed["top_scorelines"]],
                            "Prob": [s["probability"] for s in ndc_parsed["top_scorelines"]],
                        })
                        df_ndc.index = df_ndc.index + 1
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.table(df_ndc)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.caption(f"λ={ndc_lam:.2f} · μ={ndc_mu:.2f} · ρ={ndc_rho:+.3f}")
                    else:
                        st.info("Train NDC model to see these predictions.")

                with score_dc_col:
                    st.markdown('Static Dixon-Coles <span class="badge-dc">MLE</span>', unsafe_allow_html=True)
                    if dc_params:
                        dc_xg_h, dc_xg_a, df_dc = _static_dc_top_scorelines(home_team, away_team, dc_params, top_n=3)
                        df_dc["Prob"] = df_dc["Probability"].map(lambda p: f"{p:.1%}")
                        df_dc = df_dc[["Scoreline", "Prob"]].copy()
                        df_dc.index = df_dc.index + 1
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.table(df_dc)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.caption(f"xG: {home_team} {dc_xg_h:.2f} · {away_team} {dc_xg_a:.2f}")
                    else:
                        st.info("`models/dc_mle_params.pkl` not found.")

                # Heatmaps
                st.markdown(
                    '<div class="section-title">🌡️ 2D Scoreline Probability Heatmaps</div>',
                    unsafe_allow_html=True,
                )

                hm_ndc_col, hm_dc_col = st.columns(2, gap="large")

                def create_heatmap_fig(grid_data, title):
                    max_g = grid_data.shape[0]
                    labels = list(range(max_g))
                    fig = px.imshow(
                        grid_data * 100,
                        x=labels,
                        y=labels,
                        color_continuous_scale="Blues",
                        labels={"x": f"Away Goals ({away_team})", "y": f"Home Goals ({home_team})", "color": "Prob (%)"},
                        text_auto=".1f",
                        title=title,
                        aspect="equal",
                    )
                    fig.update_traces(
                        texttemplate="%{z:.1f}%",
                        textfont={"size": 11, "color": "black"},
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Inter", "color": "#1e293b"},
                        title_font={"size": 14, "color": "#1e293b"},
                        coloraxis_colorbar={
                            "title": {"text": "Prob (%)", "font": {"color": "#4b5563"}},
                            "tickfont": {"color": "#4b5563"},
                            "bgcolor": "rgba(255,255,255,0.9)",
                            "bordercolor": "rgba(99,102,241,0.2)",
                        },
                        margin={"l": 60, "r": 20, "t": 60, "b": 60},
                        xaxis={"title_font": {"color": "#4b5563"}, "tickfont": {"color": "#4b5563"}},
                        yaxis={"title_font": {"color": "#4b5563"}, "tickfont": {"color": "#4b5563"}},
                    )
                    return fig

                with hm_ndc_col:
                    if ndc_grid is not None:
                        fig_ndc = create_heatmap_fig(ndc_grid, f"Neural Dixon-Coles (NDC) — {home_team} vs {away_team}")
                        st.plotly_chart(fig_ndc, use_container_width=True)

                with hm_dc_col:
                    if dc_params:
                        _, _, static_grid = _static_dc_grid(home_team, away_team, dc_params, max_goals=6)
                        fig_dc = create_heatmap_fig(static_grid, f"Static Dixon-Coles (MLE) — {home_team} vs {away_team}")
                        st.plotly_chart(fig_dc, use_container_width=True)

                # Elo History Trajectory
                st.markdown(
                    '<div class="section-title">📈 10-Season Elo Rating Trajectory</div>',
                    unsafe_allow_html=True,
                )

                @st.cache_data
                def extract_elo_history(df):
                    h_df = df[["Date", "HomeTeam", "Home_Elo"]].rename(columns={"HomeTeam": "Team", "Home_Elo": "Elo"})
                    a_df = df[["Date", "AwayTeam", "Away_Elo"]].rename(columns={"AwayTeam": "Team", "Away_Elo": "Elo"})
                    return pd.concat([h_df, a_df]).sort_values(by=["Team", "Date"]).reset_index(drop=True)

                elo_df = extract_elo_history(df_features)
                all_teams_list = sorted(elo_df["Team"].unique())
                default_teams = [t for t in [home_team, away_team, "Man City", "Arsenal", "Liverpool"] if t in all_teams_list]

                selected_teams = st.multiselect(
                    "Compare 10-Season Elo Trajectories:",
                    all_teams_list,
                    default=default_teams,
                    key="elo_team_selector",
                )

                if selected_teams:
                    sub_elo = elo_df[elo_df["Team"].isin(selected_teams)]
                    fig_elo = px.line(
                        sub_elo,
                        x="Date",
                        y="Elo",
                        color="Team",
                        title="Premier League Elo Ratings (2016/17 - 2025/26)",
                        labels={"Date": "Date", "Elo": "Elo Rating", "Team": "Club"},
                    )
                    fig_elo.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Inter", "color": "#1e293b"},
                        title_font={"size": 14, "color": "#1e293b"},
                        hovermode="x unified",
                        xaxis={"showgrid": True, "gridcolor": "rgba(203, 213, 225, 0.4)"},
                        yaxis={"showgrid": True, "gridcolor": "rgba(203, 213, 225, 0.4)"},
                        margin={"l": 40, "r": 20, "t": 50, "b": 40},
                    )
                    st.plotly_chart(fig_elo, use_container_width=True)

# ── TAB 2: Live Elo Ratings ────────────────────────────────────────────────
with tab_elo:
    st.markdown('<div class="section-title">🏆 Current Premier League Elo Standings</div>', unsafe_allow_html=True)
    
    elo_ratings, last_active_season = compute_live_elos(df_features)
    latest_season_str = sorted(df_features["Season"].unique().tolist())[-1]
    active_teams_set = set(df_features[df_features["Season"] == latest_season_str]["HomeTeam"])
    
    # Build dataframe for standings table
    elo_data = []
    for team, elo in elo_ratings.items():
        is_active = team in active_teams_set
        elo_data.append({
            "Team": team,
            "Elo Rating": round(elo, 1),
            "Status": "Active" if is_active else "Relegated/Historical",
            "Last Active Season": last_active_season.get(team, "Unknown")
        })
        
    df_elo_table = pd.DataFrame(elo_data).sort_values(by="Elo Rating", ascending=False).reset_index(drop=True)
    df_elo_table.index = df_elo_table.index + 1
    df_elo_table.index.name = "Rank"
    
    # Toggle to filter active teams only
    filter_active = st.checkbox("Show active 2025/26 season teams only", value=True, key="filter_active_teams")
    if filter_active:
        display_df = df_elo_table[df_elo_table["Status"] == "Active"].copy()
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.index.name = "Rank"
    else:
        display_df = df_elo_table
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(display_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bar chart for active teams
    active_elos_df = df_elo_table[df_elo_table["Status"] == "Active"].sort_values(by="Elo Rating", ascending=True)
    fig_bar = px.bar(
        active_elos_df,
        x="Elo Rating",
        y="Team",
        orientation="h",
        title="Live Premier League Elo Ratings (Active Teams)",
        labels={"Elo Rating": "Elo Rating", "Team": "Club"},
        color="Elo Rating",
        color_continuous_scale="Blues",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#1e293b"},
        title_font={"size": 14, "color": "#1e293b"},
        margin={"l": 100, "r": 20, "t": 50, "b": 40},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── TAB 3: Live 26/27 Tracker ──────────────────────────────────────────────
with tab_live:
    live_status = get_live_tracker_status(LIVE_DB_PATH)
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 1rem 1.25rem; border-left: 5px solid {'#f59e0b' if not live_status['api_configured'] else '#10b981'};">
            <div style="font-size: 0.8rem; font-weight: 700; color: #4b5563; text-transform: uppercase; letter-spacing: 0.08em;">Live Data Source</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #111827; margin-top: 0.2rem;">{live_status['source_name']}</div>
            <div style="font-size: 0.9rem; color: #4b5563; margin-top: 0.35rem;">{live_status['summary']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">📊 Live 2026/27 Season Tracker</div>', unsafe_allow_html=True)
    
    # 1. Top-Level KPI Banner
    if df_fixtures.empty or df_predictions.empty:
        st.info("No predictions or fixtures found in the live database yet. Run the prediction script to populate it.")
    else:
        df_merged = pd.merge(df_fixtures, df_predictions, on="match_id", how="inner")
        total_predicted = len(df_merged)

        # Filter for completed matches that have predictions
        df_completed_preds = df_merged[df_merged["status"] == "completed"]
        total_completed = len(df_completed_preds)

        # Calculate correct predictions
        def is_correct(row):
            h_g = row["home_goals"]
            a_g = row["away_goals"]
            if h_g > a_g:
                actual = "Home"
            elif h_g < a_g:
                actual = "Away"
            else:
                actual = "Draw"
                
            probs = {
                "Home": row["predicted_home_prob"],
                "Draw": row["predicted_draw_prob"],
                "Away": row["predicted_away_prob"]
            }
            predicted = max(probs, key=probs.get)
            return actual == predicted
            
        # Head-to-head model metric calculator across completed 2026/27 matches
        def compute_model_metrics(df_completed):
            if df_completed.empty:
                return {
                    "blend": {"hits": 0, "acc": 0.0, "loss": 0.0, "total": 0},
                    "xgb":   {"hits": 0, "acc": 0.0, "loss": 0.0, "total": 0},
                    "ndc":   {"hits": 0, "acc": 0.0, "loss": 0.0, "total": 0},
                    "bookie": {"hits": 0, "acc": 0.0, "loss": 0.0, "total": 0},
                }

            def _sf(val, fallback):
                if val is None or pd.isna(val):
                    return float(fallback)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return float(fallback)

            hits_blend, hits_xgb, hits_ndc, hits_bookie = 0, 0, 0, 0
            y_true_blend, y_true_xgb, y_true_ndc, y_true_bookie = [], [], [], []
            probs_blend, probs_xgb, probs_ndc, probs_bookie = [], [], [], []

            for _, row in df_completed.iterrows():
                h_g = row["home_goals"]
                a_g = row["away_goals"]
                actual = "Home" if h_g > a_g else "Away" if h_g < a_g else "Draw"
                actual_idx = 2 if h_g > a_g else 0 if h_g < a_g else 1

                # Blended Ensemble (always present for locked rows)
                b_h = _sf(row.get("predicted_home_prob"), 0.33)
                b_d = _sf(row.get("predicted_draw_prob"), 0.33)
                b_a = _sf(row.get("predicted_away_prob"), 0.34)
                p_blend = {"Home": b_h, "Draw": b_d, "Away": b_a}
                if max(p_blend, key=p_blend.get) == actual:
                    hits_blend += 1
                y_true_blend.append(actual_idx)
                probs_blend.append([b_a, b_d, b_h])

                # XGBoost Classifier
                x_h = row.get("xgb_home_prob")
                x_d = row.get("xgb_draw_prob")
                x_a = row.get("xgb_away_prob")
                if all(pd.notna(v) for v in [x_h, x_d, x_a]):
                    x_h = _sf(x_h, 0.33)
                    x_d = _sf(x_d, 0.33)
                    x_a = _sf(x_a, 0.34)
                    p_xgb = {"Home": x_h, "Draw": x_d, "Away": x_a}
                    if max(p_xgb, key=p_xgb.get) == actual:
                        hits_xgb += 1
                    y_true_xgb.append(actual_idx)
                    probs_xgb.append([x_a, x_d, x_h])

                # Neural Dixon-Coles
                n_h = row.get("ndc_home_prob")
                n_d = row.get("ndc_draw_prob")
                n_a = row.get("ndc_away_prob")
                if all(pd.notna(v) for v in [n_h, n_d, n_a]):
                    n_h = _sf(n_h, 0.33)
                    n_d = _sf(n_d, 0.33)
                    n_a = _sf(n_a, 0.34)
                    p_ndc = {"Home": n_h, "Draw": n_d, "Away": n_a}
                    if max(p_ndc, key=p_ndc.get) == actual:
                        hits_ndc += 1
                    y_true_ndc.append(actual_idx)
                    probs_ndc.append([n_a, n_d, n_h])

                # Bookmaker benchmark
                bookie_triplet = _bookie_prob_triplet_from_row(row)
                if bookie_triplet is not None:
                    b_h, b_d, b_a = bookie_triplet
                    p_bookie = {"Home": b_h, "Draw": b_d, "Away": b_a}
                    if max(p_bookie, key=p_bookie.get) == actual:
                        hits_bookie += 1
                    y_true_bookie.append(actual_idx)
                    probs_bookie.append([b_a, b_d, b_h])

            def _log_loss_calc(y_t, p_list):
                if not p_list:
                    return 0.0
                arr = np.nan_to_num(np.array(p_list, dtype=np.float64), nan=1.0 / 3.0)
                arr = np.clip(arr, 1e-9, None)
                arr = arr / arr.sum(axis=1, keepdims=True)
                return float(log_loss(np.array(y_t, dtype=int), arr, labels=[0, 1, 2]))

            total_blend = len(y_true_blend)
            total_xgb = len(y_true_xgb)
            total_ndc = len(y_true_ndc)
            total_bookie = len(y_true_bookie)

            return {
                "blend": {"hits": hits_blend, "acc": (hits_blend / total_blend) * 100 if total_blend else 0.0, "loss": _log_loss_calc(y_true_blend, probs_blend), "total": total_blend},
                "xgb":   {"hits": hits_xgb,   "acc": (hits_xgb / total_xgb) * 100 if total_xgb else 0.0,   "loss": _log_loss_calc(y_true_xgb, probs_xgb), "total": total_xgb},
                "ndc":   {"hits": hits_ndc,   "acc": (hits_ndc / total_ndc) * 100 if total_ndc else 0.0,   "loss": _log_loss_calc(y_true_ndc, probs_ndc), "total": total_ndc},
                "bookie": {"hits": hits_bookie, "acc": (hits_bookie / total_bookie) * 100 if total_bookie else 0.0, "loss": _log_loss_calc(y_true_bookie, probs_bookie), "total": total_bookie},
            }

        model_stats = compute_model_metrics(df_completed_preds)
        gw_overview = build_gameweek_overview(df_fixtures, df_predictions)

        if "tracker_gw_selected" not in st.session_state:
            st.session_state["tracker_gw_selected"] = int(selected_gw)
        selected_gw = int(st.session_state["tracker_gw_selected"])

        # Display Model Race Banner
        st.markdown('### ⚔️ 2026/27 Live Model Race & Leaderboard', unsafe_allow_html=True)
        st.caption("Pitting the Blended Ensemble, XGBoost, Neural Dixon-Coles, and the Bookie benchmark head-to-head on completed 2026/27 season fixtures.")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        blend_info = model_stats["blend"]
        xgb_info   = model_stats["xgb"]
        ndc_info   = model_stats["ndc"]
        bookie_info = model_stats["bookie"]

        kpi1.metric(
            "🔮 Blended Ensemble (50/50)",
            f"{blend_info['acc']:.1f}% Acc",
            delta=f"{blend_info['hits']}/{total_completed} Correct | Loss: {blend_info['loss']:.3f}"
        )
        kpi2.metric(
            "⚡ XGBoost Classifier",
            f"{xgb_info['acc']:.1f}% Acc",
            delta=f"{xgb_info['hits']}/{total_completed} Correct | Loss: {xgb_info['loss']:.3f}"
        )
        kpi3.metric(
            "🧠 Neural Dixon-Coles",
            f"{ndc_info['acc']:.1f}% Acc",
            delta=f"{ndc_info['hits']}/{total_completed} Correct | Loss: {ndc_info['loss']:.3f}"
        )
        kpi4.metric(
            "📉 Bookie Odds Benchmark",
            f"{bookie_info['acc']:.1f}% Acc",
            delta=f"{bookie_info['hits']}/{bookie_info['total']} Correct | Loss: {bookie_info['loss']:.3f}"
        )

        st.markdown("### 🧭 Gameweek Overview", unsafe_allow_html=True)
        st.caption("Status is based on fixture dates/results: completed, in progress, or upcoming. Accuracy counts use the completed fixtures for each gameweek.")

        overview_df = gw_overview[[
            "Gameweek", "Status", "Played", "Fixtures",
            "Final Prob Hits", "Final Prob Total",
            "NDC Prob Hits", "NDC Prob Total", "NDC Score Hits", "NDC Score Total",
            "XGB Prob Hits", "XGB Prob Total",
            "Bookie Prob Hits", "Bookie Prob Total",
            "Static Prob Hits", "Static Prob Total", "Static Score Hits", "Static Score Total",
        ]].copy()
        overview_df["Final Prob"] = overview_df.apply(lambda r: _format_hits(int(r["Final Prob Hits"]), int(r["Final Prob Total"])), axis=1)
        overview_df["NDC Prob"] = overview_df.apply(lambda r: _format_hits(int(r["NDC Prob Hits"]), int(r["NDC Prob Total"])), axis=1)
        overview_df["NDC Score"] = overview_df.apply(lambda r: _format_hits(int(r["NDC Score Hits"]), int(r["NDC Score Total"])), axis=1)
        overview_df["XGB Prob"] = overview_df.apply(lambda r: _format_hits(int(r["XGB Prob Hits"]), int(r["XGB Prob Total"])), axis=1)
        overview_df["Bookie Prob"] = overview_df.apply(lambda r: _format_hits(int(r["Bookie Prob Hits"]), int(r["Bookie Prob Total"])), axis=1)
        overview_df["Static Prob"] = overview_df.apply(lambda r: _format_hits(int(r["Static Prob Hits"]), int(r["Static Prob Total"])), axis=1)
        overview_df["Static Score"] = overview_df.apply(lambda r: _format_hits(int(r["Static Score Hits"]), int(r["Static Score Total"])), axis=1)
        overview_df = overview_df[["Gameweek", "Status", "Played", "Fixtures", "Final Prob", "NDC Prob", "XGB Prob", "Bookie Prob", "Static Prob"]].copy()
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

        st.markdown("#### Click a gameweek", unsafe_allow_html=True)
        gw_rows = [list(range(1, 14)), list(range(14, 27)), list(range(27, 39))]
       
        for row_gws in gw_rows:
            cols = st.columns(len(row_gws))
            for col, gw in zip(cols, row_gws):
                gw_row = gw_overview[gw_overview["Gameweek"] == gw]
                if gw_row.empty:
                    continue
                gw_row = gw_row.iloc[0]
                final_prob = _format_hits(int(gw_row["Final Prob Hits"]), int(gw_row["Final Prob Total"]))
                ndc_prob = _format_hits(int(gw_row["NDC Prob Hits"]), int(gw_row["NDC Prob Total"]))
                xgb_prob = _format_hits(int(gw_row["XGB Prob Hits"]), int(gw_row["XGB Prob Total"]))
                bookie_prob = _format_hits(int(gw_row["Bookie Prob Hits"]), int(gw_row["Bookie Prob Total"]))
                static_prob = _format_hits(int(gw_row["Static Prob Hits"]), int(gw_row["Static Prob Total"]))
                label = f"GW{gw}\n{gw_row['Status']}\nFinal {final_prob} | NDC {ndc_prob} | XGB {xgb_prob} | Bookie {bookie_prob}"
                if col.button(label, key=f"gw_button_{gw}", use_container_width=True):
                    st.session_state["tracker_gw_selected"] = gw
                    st.rerun()

        selected_row = gw_overview[gw_overview["Gameweek"] == selected_gw]
        if not selected_row.empty:
            selected_row = selected_row.iloc[0]
            st.markdown(f"### 🗓️ Gameweek {selected_gw} Summary", unsafe_allow_html=True)
            s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
            s1.metric("Status", selected_row["Status"])
            s2.metric("Played", f"{selected_row['Played']}/{selected_row['Fixtures']}")
            s3.metric("Final Prob", _format_hits(int(selected_row["Final Prob Hits"]), int(selected_row["Final Prob Total"])))
            s4.metric("NDC Prob", _format_hits(int(selected_row["NDC Prob Hits"]), int(selected_row["NDC Prob Total"])))
            s5.metric("XGB Prob", _format_hits(int(selected_row["XGB Prob Hits"]), int(selected_row["XGB Prob Total"])))
            s6.metric("Bookie Prob", _format_hits(int(selected_row["Bookie Prob Hits"]), int(selected_row["Bookie Prob Total"])))
            s7.metric("Static Prob", _format_hits(int(selected_row["Static Prob Hits"]), int(selected_row["Static Prob Total"])))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🗓️ Gameweek {selected_gw} Fixtures & Predictions")
        
        # 2. Get fixtures for the selected gameweek
        df_gw = pd.merge(
            df_fixtures[df_fixtures["gameweek"] == selected_gw],
            df_predictions,
            on="match_id",
            how="left"
        ).sort_values("match_date")
        
        if df_gw.empty:
            st.info(f"No fixtures scheduled or seeded for Gameweek {selected_gw}.")
        else:
            # Match grid
            m_col1, m_col2 = st.columns(2)
            for idx, (_, row) in enumerate(df_gw.iterrows()):
                target_col = m_col1 if idx % 2 == 0 else m_col2
                with target_col:
                    render_fixture_card(row)