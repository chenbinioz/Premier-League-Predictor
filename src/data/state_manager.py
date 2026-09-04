"""
state_manager.py
================
TeamStateManager — sliding-window Elo + rolling stats updater for the
2026/27 live database.

Usage:
    from src.data.state_manager import TeamStateManager

    sm = TeamStateManager("data/live/epl_2627.db")

    # Lock a pre-match prediction
    sm.lock_prediction(
        match_id=1,
        home_prob=0.52, draw_prob=0.25, away_prob=0.23,
        predicted_score="2 - 1",
        ndc_lambda=1.8, ndc_mu=1.1, ndc_rho=-0.04,
    )

    # After the match completes
    sm.update_after_match(
        match_id=1,
        home_team="Arsenal", away_team="Chelsea",
        home_goals=2, away_goals=1,
        home_xg=1.95, away_xg=1.12,
    )

    sm.close()
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Elo constants — must match build_features.py ──────────────────────────────
ELO_K             = 20    # K-factor
ELO_HOME_ADV      = 60    # Home-advantage bonus in Elo points
WINDOW_SIZE       = 10    # Length of the sliding rolling arrays


class TeamStateManager:
    """
    Manages per-team state in `live_team_states` after each completed fixture.

    Responsibilities:
      • Read the current state for any team.
      • Slide the 10-match rolling arrays (xG, goals) forward.
      • Recalculate rolling averages.
      • Apply standard Elo update (with home-field advantage).
      • Write predictions to `predictions_26_27`.
      • Mark fixtures complete in `fixtures_26_27`.
    """

    # ── Init ──────────────────────────────────────────────────────────────────
    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                "Run `python src/db/init_live_db.py` first."
            )
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")   # safer concurrent reads
        self._ensure_prediction_columns()

    def _ensure_prediction_columns(self) -> None:
        """Add newer per-model probability columns to legacy live databases."""
        cur = self._conn.execute("PRAGMA table_info(predictions_26_27)")
        existing = {row[1] for row in cur.fetchall()}
        required = {
            "ndc_home_prob": "REAL",
            "ndc_draw_prob": "REAL",
            "ndc_away_prob": "REAL",
            "xgb_home_prob": "REAL",
            "xgb_draw_prob": "REAL",
            "xgb_away_prob": "REAL",
        }
        for column_name, column_type in required.items():
            if column_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE predictions_26_27 ADD COLUMN {column_name} {column_type}"
                )
        self._conn.commit()

    # ── Public read ───────────────────────────────────────────────────────────
    def get_team_state(self, team: str) -> dict[str, Any]:
        """
        Return the full `live_team_states` row for *team* as a dict.
        JSON columns are automatically decoded to Python lists.
        Raises KeyError if the team is not found.
        """
        cur = self._conn.execute(
            "SELECT * FROM live_team_states WHERE team_name = ?", (team,)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"Team '{team}' not found in live_team_states.")

        state = dict(row)
        # Decode JSON arrays
        for col in (
            "last_10_xg_for", "last_10_xg_against",
            "last_10_goals_for", "last_10_goals_against",
        ):
            state[col] = json.loads(state[col])
        return state

    def get_all_team_states(self) -> list[dict[str, Any]]:
        """Return all team states, sorted by Elo descending."""
        cur = self._conn.execute(
            "SELECT * FROM live_team_states ORDER BY current_elo DESC"
        )
        rows = cur.fetchall()
        result = []
        for row in rows:
            state = dict(row)
            for col in (
                "last_10_xg_for", "last_10_xg_against",
                "last_10_goals_for", "last_10_goals_against",
            ):
                state[col] = json.loads(state[col])
            result.append(state)
        return result

    # ── Core match update ─────────────────────────────────────────────────────
    def update_after_match(
        self,
        match_id:    int,
        home_team:   str,
        away_team:   str,
        home_goals:  int,
        away_goals:  int,
        home_xg:     float,
        away_xg:     float,
    ) -> None:
        """
        Process a completed result:
          1. Fetch current states for both teams.
          2. Compute Elo update for home and away.
          3. Slide rolling windows and recalculate averages.
          4. Persist both team states and mark fixture as completed.

        Args:
            match_id:   Primary key in fixtures_26_27.
            home_team:  Canonical team name (e.g. "Arsenal").
            away_team:  Canonical team name (e.g. "Chelsea").
            home_goals: Full-time home goals scored.
            away_goals: Full-time away goals scored.
            home_xg:    Home expected goals (from Opta / StatsBomb or NDC λ).
            away_xg:    Away expected goals.
        """
        home_state = self.get_team_state(home_team)
        away_state = self.get_team_state(away_team)

        # 1. Elo update
        new_home_elo, new_away_elo = self._compute_elo_update(
            home_elo=home_state["current_elo"],
            away_elo=away_state["current_elo"],
            home_goals=home_goals,
            away_goals=away_goals,
        )

        # 2. Slide windows
        self._update_sliding_window(
            team=home_team,
            new_elo=new_home_elo,
            goals_for=home_goals,
            goals_against=away_goals,
            xg_for=home_xg,
            xg_against=away_xg,
            current_state=home_state,
        )
        self._update_sliding_window(
            team=away_team,
            new_elo=new_away_elo,
            goals_for=away_goals,
            goals_against=home_goals,
            xg_for=away_xg,
            xg_against=home_xg,
            current_state=away_state,
        )

        # 3. Mark fixture completed
        self.mark_fixture_complete(match_id, home_goals, away_goals)

    # ── Fixture management ────────────────────────────────────────────────────
    def mark_fixture_complete(
        self, match_id: int, home_goals: int, away_goals: int
    ) -> None:
        """Set status='completed' and record goals in fixtures_26_27."""
        self._conn.execute(
            """
            UPDATE fixtures_26_27
            SET status = 'completed', home_goals = ?, away_goals = ?
            WHERE match_id = ?
            """,
            (home_goals, away_goals, match_id),
        )
        self._conn.commit()

    def add_fixture(
        self,
        gameweek:   int,
        match_date: str,   # 'YYYY-MM-DD'
        home_team:  str,
        away_team:  str,
    ) -> int:
        """Insert a pending fixture and return its new match_id."""
        cur = self._conn.execute(
            """
            INSERT INTO fixtures_26_27 (gameweek, match_date, home_team, away_team)
            VALUES (?, ?, ?, ?)
            """,
            (gameweek, match_date, home_team, away_team),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_pending_fixtures(self) -> list[dict[str, Any]]:
        """Return all unplayed matches as a list of dicts."""
        cur = self._conn.execute(
            "SELECT * FROM fixtures_26_27 WHERE status = 'pending' ORDER BY match_date"
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Prediction ledger ─────────────────────────────────────────────────────
    def lock_prediction(
        self,
        match_id:        int,
        home_prob:       float,
        draw_prob:       float,
        away_prob:       float,
        predicted_score: str,
        ndc_home_prob:   Optional[float] = None,
        ndc_draw_prob:   Optional[float] = None,
        ndc_away_prob:   Optional[float] = None,
        xgb_home_prob:   Optional[float] = None,
        xgb_draw_prob:   Optional[float] = None,
        xgb_away_prob:   Optional[float] = None,
        ndc_lambda:      Optional[float] = None,
        ndc_mu:          Optional[float] = None,
        ndc_rho:         Optional[float] = None,
    ) -> None:
        """
        Write a pre-match forecast to predictions_26_27.
        Uses INSERT OR REPLACE so re-running the pipeline is idempotent.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO predictions_26_27
                (match_id, predicted_home_prob, predicted_draw_prob,
                 predicted_away_prob, predicted_score,
                 ndc_home_prob, ndc_draw_prob, ndc_away_prob,
                 xgb_home_prob, xgb_draw_prob, xgb_away_prob,
                 ndc_lambda, ndc_mu, ndc_rho, locked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id, home_prob, draw_prob, away_prob,
                predicted_score,
                ndc_home_prob, ndc_draw_prob, ndc_away_prob,
                xgb_home_prob, xgb_draw_prob, xgb_away_prob,
                ndc_lambda, ndc_mu, ndc_rho,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    # ── Private helpers ───────────────────────────────────────────────────────
    def _compute_elo_update(
        self,
        home_elo:    float,
        away_elo:    float,
        home_goals:  int,
        away_goals:  int,
    ) -> tuple[float, float]:
        """
        Standard Elo exchange with home-field advantage.

        Formula (mirrors calculate_10_season_elo in build_features.py):
            P(home) = 1 / (1 + 10^((away_elo - (home_elo + HFA)) / 400))
            delta   = K * (actual - P(home))
            new_home_elo = home_elo + delta
            new_away_elo = away_elo - delta

        Returns:
            (new_home_elo, new_away_elo)
        """
        prob_home = 1.0 / (
            1.0 + 10.0 ** ((away_elo - (home_elo + ELO_HOME_ADV)) / 400.0)
        )
        prob_away = 1.0 - prob_home

        if home_goals > away_goals:
            result_home, result_away = 1.0, 0.0
        elif home_goals < away_goals:
            result_home, result_away = 0.0, 1.0
        else:
            result_home, result_away = 0.5, 0.5

        new_home = home_elo + ELO_K * (result_home - prob_home)
        new_away = away_elo + ELO_K * (result_away - prob_away)
        return new_home, new_away

    def _update_sliding_window(
        self,
        team:          str,
        new_elo:       float,
        goals_for:     int,
        goals_against: int,
        xg_for:        float,
        xg_against:    float,
        current_state: dict[str, Any],
    ) -> None:
        """
        Append the new match's stats to each rolling array (max WINDOW_SIZE),
        recalculate averages, and persist the updated state.
        """
        def _slide(arr: list, new_val: float) -> list:
            updated = arr + [new_val]
            if len(updated) > WINDOW_SIZE:
                updated = updated[-WINDOW_SIZE:]
            return updated

        new_xg_for      = _slide(current_state["last_10_xg_for"],      xg_for)
        new_xg_against  = _slide(current_state["last_10_xg_against"],   xg_against)
        new_goals_for   = _slide(current_state["last_10_goals_for"],    goals_for)
        new_goals_against = _slide(current_state["last_10_goals_against"], goals_against)

        avg_xg_for      = sum(new_xg_for)      / len(new_xg_for)
        avg_xg_against  = sum(new_xg_against)  / len(new_xg_against)
        avg_goals_for   = sum(new_goals_for)   / len(new_goals_for)
        avg_goals_against = sum(new_goals_against) / len(new_goals_against)

        self._conn.execute(
            """
            UPDATE live_team_states SET
                current_elo           = ?,
                last_10_xg_for        = ?,
                last_10_xg_against    = ?,
                last_10_goals_for     = ?,
                last_10_goals_against = ?,
                avg_xg_for            = ?,
                avg_xg_against        = ?,
                avg_goals_for         = ?,
                avg_goals_against     = ?,
                last_updated          = ?
            WHERE team_name = ?
            """,
            (
                round(new_elo, 4),
                json.dumps(new_xg_for),
                json.dumps(new_xg_against),
                json.dumps(new_goals_for),
                json.dumps(new_goals_against),
                round(avg_xg_for, 4),
                round(avg_xg_against, 4),
                round(avg_goals_for, 4),
                round(avg_goals_against, 4),
                datetime.now(timezone.utc).isoformat(),
                team,
            ),
        )
        self._conn.commit()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "TeamStateManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
