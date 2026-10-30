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
        """Add newer per-model probability columns and shots/corners/fouls tracking columns to live databases."""
        # 1. predictions_26_27
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

        # 2. live_team_states (shots on target, corners, fouls)
        cur = self._conn.execute("PRAGMA table_info(live_team_states)")
        existing_states = {row[1] for row in cur.fetchall()}
        required_states = {
            "last_10_sot_for": "TEXT NOT NULL DEFAULT '[]'",
            "last_10_sot_against": "TEXT NOT NULL DEFAULT '[]'",
            "last_10_corners_for": "TEXT NOT NULL DEFAULT '[]'",
            "last_10_corners_against": "TEXT NOT NULL DEFAULT '[]'",
            "last_10_fouls_for": "TEXT NOT NULL DEFAULT '[]'",
            "last_10_fouls_against": "TEXT NOT NULL DEFAULT '[]'",
            "avg_sot_for": "REAL NOT NULL DEFAULT 0.0",
            "avg_sot_against": "REAL NOT NULL DEFAULT 0.0",
            "avg_corners_for": "REAL NOT NULL DEFAULT 0.0",
            "avg_corners_against": "REAL NOT NULL DEFAULT 0.0",
            "avg_fouls_for": "REAL NOT NULL DEFAULT 0.0",
            "avg_fouls_against": "REAL NOT NULL DEFAULT 0.0",
        }
        for column_name, column_def in required_states.items():
            if column_name not in existing_states:
                self._conn.execute(
                    f"ALTER TABLE live_team_states ADD COLUMN {column_name} {column_def}"
                )

        # 3. fixtures_26_27
        cur = self._conn.execute("PRAGMA table_info(fixtures_26_27)")
        existing_fixtures = {row[1] for row in cur.fetchall()}
        required_fixtures = {
            "home_shots_on_target": "INTEGER",
            "away_shots_on_target": "INTEGER",
            "home_corners": "INTEGER",
            "away_corners": "INTEGER",
            "home_fouls": "INTEGER",
            "away_fouls": "INTEGER",
        }
        for column_name, column_def in required_fixtures.items():
            if column_name not in existing_fixtures:
                self._conn.execute(
                    f"ALTER TABLE fixtures_26_27 ADD COLUMN {column_name} {column_def}"
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
            "last_10_sot_for", "last_10_sot_against",
            "last_10_corners_for", "last_10_corners_against",
            "last_10_fouls_for", "last_10_fouls_against",
        ):
            val = state.get(col)
            if isinstance(val, str):
                try:
                    state[col] = json.loads(val)
                except Exception:
                    state[col] = []
            elif not isinstance(val, list):
                state[col] = []
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
                "last_10_sot_for", "last_10_sot_against",
                "last_10_corners_for", "last_10_corners_against",
                "last_10_fouls_for", "last_10_fouls_against",
            ):
                val = state.get(col)
                if isinstance(val, str):
                    try:
                        state[col] = json.loads(val)
                    except Exception:
                        state[col] = []
                elif not isinstance(val, list):
                    state[col] = []
            result.append(state)
        return result

    # ── Core match update ─────────────────────────────────────────────────────
    def update_after_match(
        self,
        match_id:      int,
        home_team:     str,
        away_team:     str,
        home_goals:    int,
        away_goals:    int,
        home_xg:       float,
        away_xg:       float,
        home_sot:      int | None = None,
        away_sot:      int | None = None,
        home_corners:  int | None = None,
        away_corners:  int | None = None,
        home_fouls:    int | None = None,
        away_fouls:    int | None = None,
    ) -> None:
        """
        Process a completed result:
          1. Fetch current states for both teams.
          2. Compute Elo update for home and away.
          3. Slide rolling windows (xG, goals, shots on target, corners, fouls).
          4. Persist both team states and mark fixture as completed.
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
            sot_for=home_sot,
            sot_against=away_sot,
            corners_for=home_corners,
            corners_against=away_corners,
            fouls_for=home_fouls,
            fouls_against=away_fouls,
            current_state=home_state,
        )
        self._update_sliding_window(
            team=away_team,
            new_elo=new_away_elo,
            goals_for=away_goals,
            goals_against=home_goals,
            xg_for=away_xg,
            xg_against=home_xg,
            sot_for=away_sot,
            sot_against=home_sot,
            corners_for=away_corners,
            corners_against=home_corners,
            fouls_for=away_fouls,
            fouls_against=home_fouls,
            current_state=away_state,
        )

        # 3. Mark fixture completed
        self.mark_fixture_complete(
            match_id=match_id,
            home_goals=home_goals,
            away_goals=away_goals,
            home_sot=home_sot,
            away_sot=away_sot,
            home_corners=home_corners,
            away_corners=away_corners,
            home_fouls=home_fouls,
            away_fouls=away_fouls,
        )

    # ── Fixture management ────────────────────────────────────────────────────
    def mark_fixture_complete(
        self,
        match_id:     int,
        home_goals:   int,
        away_goals:   int,
        home_sot:     int | None = None,
        away_sot:     int | None = None,
        home_corners: int | None = None,
        away_corners: int | None = None,
        home_fouls:   int | None = None,
        away_fouls:   int | None = None,
    ) -> None:
        """Set status='completed' and record match stats in fixtures_26_27."""
        self._conn.execute(
            """
            UPDATE fixtures_26_27
            SET status = 'completed',
                home_goals = ?,
                away_goals = ?,
                home_shots_on_target = COALESCE(?, home_shots_on_target),
                away_shots_on_target = COALESCE(?, away_shots_on_target),
                home_corners         = COALESCE(?, home_corners),
                away_corners         = COALESCE(?, away_corners),
                home_fouls           = COALESCE(?, home_fouls),
                away_fouls           = COALESCE(?, away_fouls)
            WHERE match_id = ?
            """,
            (
                home_goals, away_goals,
                home_sot, away_sot,
                home_corners, away_corners,
                home_fouls, away_fouls,
                match_id,
            ),
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
        team:            str,
        new_elo:         float,
        goals_for:       int,
        goals_against:   int,
        xg_for:          float,
        xg_against:      float,
        sot_for:         int | float | None,
        sot_against:     int | float | None,
        corners_for:     int | float | None,
        corners_against: int | float | None,
        fouls_for:       int | float | None,
        fouls_against:   int | float | None,
        current_state:   dict[str, Any],
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

        def _get_list(key: str, fallback_val: float) -> list:
            raw = current_state.get(key)
            if isinstance(raw, list):
                return list(raw)
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [fallback_val] * 5

        new_xg_for        = _slide(current_state.get("last_10_xg_for") or [1.4] * 5, xg_for)
        new_xg_against    = _slide(current_state.get("last_10_xg_against") or [1.3] * 5, xg_against)
        new_goals_for     = _slide(current_state.get("last_10_goals_for") or [1.4] * 5, goals_for)
        new_goals_against = _slide(current_state.get("last_10_goals_against") or [1.3] * 5, goals_against)

        # Shots on target (fall back to current avg or seasonal 4.35)
        cur_sot_for = _get_list("last_10_sot_for", 4.35)
        cur_sot_con = _get_list("last_10_sot_against", 4.45)
        new_sot_for = _slide(cur_sot_for, float(sot_for if sot_for is not None else (sum(cur_sot_for) / len(cur_sot_for) if cur_sot_for else 4.35)))
        new_sot_con = _slide(cur_sot_con, float(sot_against if sot_against is not None else (sum(cur_sot_con) / len(cur_sot_con) if cur_sot_con else 4.45)))

        # Corners (fall back to current avg or seasonal 5.05)
        cur_cor_for = _get_list("last_10_corners_for", 5.05)
        cur_cor_con = _get_list("last_10_corners_against", 5.11)
        new_cor_for = _slide(cur_cor_for, float(corners_for if corners_for is not None else (sum(cur_cor_for) / len(cur_cor_for) if cur_cor_for else 5.05)))
        new_cor_con = _slide(cur_cor_con, float(corners_against if corners_against is not None else (sum(cur_cor_con) / len(cur_cor_con) if cur_cor_con else 5.11)))

        # Fouls (fall back to current avg or seasonal 10.95)
        cur_foul_for = _get_list("last_10_fouls_for", 10.95)
        cur_foul_con = _get_list("last_10_fouls_against", 10.88)
        new_foul_for = _slide(cur_foul_for, float(fouls_for if fouls_for is not None else (sum(cur_foul_for) / len(cur_foul_for) if cur_foul_for else 10.95)))
        new_foul_con = _slide(cur_foul_con, float(fouls_against if fouls_against is not None else (sum(cur_foul_con) / len(cur_foul_con) if cur_foul_con else 10.88)))

        avg_xg_for          = sum(new_xg_for)        / len(new_xg_for)
        avg_xg_against      = sum(new_xg_against)    / len(new_xg_against)
        avg_goals_for       = sum(new_goals_for)     / len(new_goals_for)
        avg_goals_against   = sum(new_goals_against) / len(new_goals_against)

        avg_sot_for         = sum(new_sot_for)       / len(new_sot_for)
        avg_sot_against     = sum(new_sot_con)       / len(new_sot_con)
        avg_corners_for     = sum(new_cor_for)       / len(new_cor_for)
        avg_corners_against = sum(new_cor_con)       / len(new_cor_con)
        avg_fouls_for       = sum(new_foul_for)      / len(new_foul_for)
        avg_fouls_against   = sum(new_foul_con)      / len(new_foul_con)

        self._conn.execute(
            """
            UPDATE live_team_states SET
                current_elo           = ?,
                last_10_xg_for        = ?,
                last_10_xg_against    = ?,
                last_10_goals_for     = ?,
                last_10_goals_against = ?,
                last_10_sot_for       = ?,
                last_10_sot_against   = ?,
                last_10_corners_for   = ?,
                last_10_corners_against = ?,
                last_10_fouls_for     = ?,
                last_10_fouls_against = ?,
                avg_xg_for            = ?,
                avg_xg_against        = ?,
                avg_goals_for         = ?,
                avg_goals_against     = ?,
                avg_sot_for           = ?,
                avg_sot_against       = ?,
                avg_corners_for       = ?,
                avg_corners_against   = ?,
                avg_fouls_for         = ?,
                avg_fouls_against     = ?,
                last_updated          = ?
            WHERE team_name = ?
            """,
            (
                round(new_elo, 4),
                json.dumps(new_xg_for),
                json.dumps(new_xg_against),
                json.dumps(new_goals_for),
                json.dumps(new_goals_against),
                json.dumps(new_sot_for),
                json.dumps(new_sot_con),
                json.dumps(new_cor_for),
                json.dumps(new_cor_con),
                json.dumps(new_foul_for),
                json.dumps(new_foul_con),
                round(avg_xg_for, 4),
                round(avg_xg_against, 4),
                round(avg_goals_for, 4),
                round(avg_goals_against, 4),
                round(avg_sot_for, 4),
                round(avg_sot_against, 4),
                round(avg_corners_for, 4),
                round(avg_corners_against, 4),
                round(avg_fouls_for, 4),
                round(avg_fouls_against, 4),
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
