"""
init_live_db.py
===============
Initialises the SQLite live database for the 2026/27 Premier League season.

Steps:
  1. Create (or recreate) data/live/epl_2627.db.
  2. Execute schema DDL from schema.sql.
  3. Seed live_team_states by carrying over final Elo ratings from the
     2025/26 processed feature set, applying inter-season decay.

26/27 Promoted teams (enter at Elo 1420):
  Coventry City, Hull City, Ipswich

25/26 Relegated (NOT seeded into 26/27 table):
  West Ham, Burnley, Wolves
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
DB_DIR      = REPO_ROOT / "data" / "live"
DB_PATH     = DB_DIR / "epl_2627.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "epl_model_features.csv"

# ── Season configuration ──────────────────────────────────────────────────────
RELEGATED_25_26  = {"West Ham", "Burnley", "Wolves"}
PROMOTED_26_27   = {
    "Coventry":  1420.0,   # Coventry City
    "Hull":      1420.0,   # Hull City
    "Ipswich":   1420.0,   # Ipswich Town  (already canonical in pipeline)
}

# Standard Elo season-boundary decay (matching build_features.py)
ELO_DECAY_RETAIN   = lambda elo: 0.80 * elo + 0.20 * 1500.0
ELO_START_PROMOTED = 1420.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_final_elo_dict() -> dict[str, float]:
    """
    Read the processed feature CSV and return a {team: elo} dict
    representing each team's Elo rating at the END of the 2025/26 season.
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Processed features not found at {FEATURES_PATH}. "
            "Run scripts/fetch_10_seasons.py then src/features/build_features.py first."
        )

    df = pd.read_csv(FEATURES_PATH)
    seasons = sorted(df["Season"].unique().tolist())
    last_season = seasons[-1]
    df_last = df[df["Season"] == last_season].sort_values("Date")

    # Take the last recorded Elo for each team in 25/26
    elo_dict: dict[str, float] = {}
    for _, row in df_last.iterrows():
        elo_dict[row["HomeTeam"]] = float(row["Home_Elo"])
        elo_dict[row["AwayTeam"]] = float(row["Away_Elo"])

    return elo_dict


def _team_last_10_stats(df: pd.DataFrame, team: str) -> dict[str, list]:
    """Return the most recent 10 stats for a team from the prior season."""
    rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy().sort_values("Date")
    xg_for: list[float] = []
    xg_against: list[float] = []
    goals_for: list[float] = []
    goals_against: list[float] = []
    sot_for: list[float] = []
    sot_against: list[float] = []
    corners_for: list[float] = []
    corners_against: list[float] = []
    fouls_for: list[float] = []
    fouls_against: list[float] = []

    for _, row in rows.iterrows():
        if row["HomeTeam"] == team:
            xg_for.append(float(row["Home_xG_Created"]))
            xg_against.append(float(row["Home_xG_Conceded"]))
            goals_for.append(int(row["FTHG"]))
            goals_against.append(int(row["FTAG"]))
            sot_for.append(float(row.get("Home_ShotsOnTarget", 4.35)))
            sot_against.append(float(row.get("Home_Opp_ShotsOnTarget", 4.45)))
            corners_for.append(float(row.get("Home_Corners", 5.05)))
            corners_against.append(float(row.get("Away_Corners", 5.11)))
            fouls_for.append(float(row.get("Home_Fouls", 10.95)))
            fouls_against.append(float(row.get("Away_Fouls", 10.88)))
        else:
            xg_for.append(float(row["Away_xG_Created"]))
            xg_against.append(float(row["Away_xG_Conceded"]))
            goals_for.append(int(row["FTAG"]))
            goals_against.append(int(row["FTHG"]))
            sot_for.append(float(row.get("Away_ShotsOnTarget", 4.45)))
            sot_against.append(float(row.get("Away_Opp_ShotsOnTarget", 4.35)))
            corners_for.append(float(row.get("Away_Corners", 5.11)))
            corners_against.append(float(row.get("Home_Corners", 5.05)))
            fouls_for.append(float(row.get("Away_Fouls", 10.88)))
            fouls_against.append(float(row.get("Home_Fouls", 10.95)))

    return {
        "xg_for": xg_for[-10:],
        "xg_against": xg_against[-10:],
        "goals_for": goals_for[-10:],
        "goals_against": goals_against[-10:],
        "sot_for": sot_for[-10:],
        "sot_against": sot_against[-10:],
        "corners_for": corners_for[-10:],
        "corners_against": corners_against[-10:],
        "fouls_for": fouls_for[-10:],
        "fouls_against": fouls_against[-10:],
    }


def _load_final_season_team_stats() -> dict[str, dict[str, Any]]:
    """Create realistic start-of-season team states from the prior season's final match stats."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Processed features not found at {FEATURES_PATH}. "
            "Run scripts/fetch_10_seasons.py then src/features/build_features.py first."
        )

    df = pd.read_csv(FEATURES_PATH)
    if df.empty:
        return {}

    # Target 2025/26 as the prior season boundary baseline
    seasons = sorted([s for s in df["Season"].unique().tolist() if s != "2026/27"])
    last_season = seasons[-1] if seasons else sorted(df["Season"].unique().tolist())[-1]
    df_last = df[df["Season"] == last_season].sort_values("Date").reset_index(drop=True)

    teams = sorted(set(df_last["HomeTeam"].unique()) | set(df_last["AwayTeam"].unique()))
    stats: dict[str, dict[str, Any]] = {}

    for team in teams:
        res = _team_last_10_stats(df_last, team)
        if not res["xg_for"]:
            continue

        stats[team] = {
            "last_10_xg_for": res["xg_for"],
            "last_10_xg_against": res["xg_against"],
            "last_10_goals_for": res["goals_for"],
            "last_10_goals_against": res["goals_against"],
            "last_10_sot_for": res["sot_for"],
            "last_10_sot_against": res["sot_against"],
            "last_10_corners_for": res["corners_for"],
            "last_10_corners_against": res["corners_against"],
            "last_10_fouls_for": res["fouls_for"],
            "last_10_fouls_against": res["fouls_against"],
            "avg_xg_for": float(sum(res["xg_for"]) / len(res["xg_for"])),
            "avg_xg_against": float(sum(res["xg_against"]) / len(res["xg_against"])),
            "avg_goals_for": float(sum(res["goals_for"]) / len(res["goals_for"])),
            "avg_goals_against": float(sum(res["goals_against"]) / len(res["goals_against"])),
            "avg_sot_for": float(sum(res["sot_for"]) / len(res["sot_for"])) if res["sot_for"] else 4.35,
            "avg_sot_against": float(sum(res["sot_against"]) / len(res["sot_against"])) if res["sot_against"] else 4.45,
            "avg_corners_for": float(sum(res["corners_for"]) / len(res["corners_for"])) if res["corners_for"] else 5.05,
            "avg_corners_against": float(sum(res["corners_against"]) / len(res["corners_against"])) if res["corners_against"] else 5.11,
            "avg_fouls_for": float(sum(res["fouls_for"]) / len(res["fouls_for"])) if res["fouls_for"] else 10.95,
            "avg_fouls_against": float(sum(res["fouls_against"]) / len(res["fouls_against"])) if res["fouls_against"] else 10.88,
        }

    return stats


def _build_26_27_team_elos(elo_dict_25_26: dict[str, float]) -> dict[str, float]:
    """
    Apply inter-season Elo transition rules to produce 26/27 starting Elos.

    - Retained teams  →  0.80 * Elo_final + 0.20 * 1500
    - Relegated       →  excluded
    - Promoted        →  1420
    """
    result: dict[str, float] = {}

    # Retained teams (25/26 → 26/27, not relegated)
    for team, elo in elo_dict_25_26.items():
        if team in RELEGATED_25_26:
            continue
        result[team] = ELO_DECAY_RETAIN(elo)

    # Promoted teams
    for team, start_elo in PROMOTED_26_27.items():
        result[team] = start_elo

    return result


def _init_db(conn: sqlite3.Connection, schema_sql: str) -> None:
    """Create tables from DDL."""
    conn.executescript(schema_sql)
    conn.commit()


def _seed_team_states(conn: sqlite3.Connection, team_elos: dict[str, float]) -> None:
    """Insert one row per team into live_team_states with realistic prior-season form."""
    prior_stats = _load_final_season_team_stats()
    rows = []
    for team, elo in sorted(team_elos.items()):
        stats = prior_stats.get(team, {})

        xg_for = stats.get("last_10_xg_for", []) or [1.4] * 5
        xg_against = stats.get("last_10_xg_against", []) or [1.3] * 5
        goals_for = stats.get("last_10_goals_for", []) or [1.4] * 5
        goals_against = stats.get("last_10_goals_against", []) or [1.3] * 5
        sot_for = stats.get("last_10_sot_for", []) or [4.35] * 5
        sot_against = stats.get("last_10_sot_against", []) or [4.45] * 5
        corners_for = stats.get("last_10_corners_for", []) or [5.05] * 5
        corners_against = stats.get("last_10_corners_against", []) or [5.11] * 5
        fouls_for = stats.get("last_10_fouls_for", []) or [10.95] * 5
        fouls_against = stats.get("last_10_fouls_against", []) or [10.88] * 5

        rows.append((
            team,
            round(elo, 2),
            json.dumps(xg_for[-10:]),
            json.dumps(xg_against[-10:]),
            json.dumps(goals_for[-10:]),
            json.dumps(goals_against[-10:]),
            json.dumps(sot_for[-10:]),
            json.dumps(sot_against[-10:]),
            json.dumps(corners_for[-10:]),
            json.dumps(corners_against[-10:]),
            json.dumps(fouls_for[-10:]),
            json.dumps(fouls_against[-10:]),
            float(stats.get("avg_xg_for", 1.4)),
            float(stats.get("avg_xg_against", 1.3)),
            float(stats.get("avg_goals_for", 1.4)),
            float(stats.get("avg_goals_against", 1.3)),
            float(stats.get("avg_sot_for", 4.35)),
            float(stats.get("avg_sot_against", 4.45)),
            float(stats.get("avg_corners_for", 5.05)),
            float(stats.get("avg_corners_against", 5.11)),
            float(stats.get("avg_fouls_for", 10.95)),
            float(stats.get("avg_fouls_against", 10.88)),
        ))

    conn.executemany(
        """
        INSERT OR REPLACE INTO live_team_states
            (team_name, current_elo,
             last_10_xg_for, last_10_xg_against,
             last_10_goals_for, last_10_goals_against,
             last_10_sot_for, last_10_sot_against,
             last_10_corners_for, last_10_corners_against,
             last_10_fouls_for, last_10_fouls_against,
             avg_xg_for, avg_xg_against, avg_goals_for, avg_goals_against,
             avg_sot_for, avg_sot_against, avg_corners_for, avg_corners_against,
             avg_fouls_for, avg_fouls_against)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def _print_summary(conn: sqlite3.Connection) -> None:
    """Print a nicely formatted summary table of the seeded team states."""
    cur = conn.execute(
        "SELECT team_name, current_elo FROM live_team_states ORDER BY current_elo DESC"
    )
    rows = cur.fetchall()
    print(f"\n{'Rank':<5} {'Team':<25} {'Starting Elo':>12}")
    print("-" * 45)
    for rank, (team, elo) in enumerate(rows, start=1):
        tag = " ← PROMOTED" if team in PROMOTED_26_27 else ""
        print(f"{rank:<5} {team:<25} {elo:>12.1f}{tag}")
    print(f"\nTotal teams seeded: {len(rows)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print("=== EPL 26/27 Live DB Initialiser ===\n")
    print(f"  DB path:   {DB_PATH}")
    print(f"  Schema:    {SCHEMA_PATH}")
    print(f"  Features:  {FEATURES_PATH}\n")

    # 1. Load 25/26 final Elo ratings
    print("1. Loading 25/26 final Elo ratings from feature CSV...")
    elo_25_26 = _load_final_elo_dict()
    print(f"   Found {len(elo_25_26)} teams in 25/26 dataset.")

    # 2. Apply decay + promotion rules
    print("2. Applying inter-season Elo transitions...")
    elo_26_27 = _build_26_27_team_elos(elo_25_26)
    print(f"   → {len(elo_26_27)} teams in 26/27 system "
          f"({len(PROMOTED_26_27)} promoted, "
          f"{len(RELEGATED_25_26)} relegated / excluded).")

    # 3. Connect and create schema
    print(f"3. Creating SQLite database at {DB_PATH} ...")
    schema_sql = SCHEMA_PATH.read_text()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    _init_db(conn, schema_sql)
    print("   Tables created: fixtures_26_27, live_team_states, predictions_26_27.")

    # 4. Seed team states
    print("4. Seeding live_team_states...")
    _seed_team_states(conn, elo_26_27)

    # 5. Summary
    _print_summary(conn)

    conn.close()
    print(f"\n✅  Initialisation complete. DB ready at: {DB_PATH}")


if __name__ == "__main__":
    main()
