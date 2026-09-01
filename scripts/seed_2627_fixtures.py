"""
seed_2627_fixtures.py
====================
Initialises the 2026/27 live tracker database and populates it with the actual
Premier League schedule from Understat when available.

Fallback mode: if the live season feed is unavailable, the script reverts to the
historical local placeholder schedule so the app still has a working database.
"""

import os
import sys
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.init_live_db import main as init_db
from src.data.live_fetchers import _read_understat_matches, normalize_team_name

DB_PATH = REPO_ROOT / "data" / "live" / "epl_2627.db"

# Placeholder fallback used only when Understat is unavailable.
GW1_FIXTURES = [
    ("2026-08-14", "Arsenal", "Coventry"),
    ("2026-08-15", "Aston Villa", "Newcastle"),
    ("2026-08-15", "Brighton", "Fulham"),
    ("2026-08-15", "Chelsea", "Crystal Palace"),
    ("2026-08-15", "Everton", "Leeds"),
    ("2026-08-15", "Ipswich", "Liverpool"),
    ("2026-08-15", "Man City", "Hull"),
    ("2026-08-15", "Nott'm Forest", "Brentford"),
    ("2026-08-16", "Man United", "Bournemouth"),
    ("2026-08-17", "Tottenham", "Sunderland"),
]

GW2_FIXTURES = [
    ("2026-08-22", "Bournemouth", "Chelsea"),
    ("2026-08-22", "Brentford", "Aston Villa"),
    ("2026-08-22", "Coventry", "Everton"),
    ("2026-08-22", "Crystal Palace", "Man City"),
    ("2026-08-22", "Fulham", "Nott'm Forest"),
    ("2026-08-22", "Hull", "Tottenham"),
    ("2026-08-22", "Leeds", "Arsenal"),
    ("2026-08-22", "Newcastle", "Ipswich"),
    ("2026-08-23", "Liverpool", "Brighton"),
    ("2026-08-24", "Sunderland", "Man United"),
]


def _seed_placeholder_fixtures():
    fixtures_to_insert = []
    for date_str, home, away in GW1_FIXTURES:
        fixtures_to_insert.append((1, date_str, home, away, "pending", None, None))
    for date_str, home, away in GW2_FIXTURES:
        fixtures_to_insert.append((2, date_str, home, away, "pending", None, None))
    return fixtures_to_insert


def _seed_understat_fixtures():
    rows = _read_understat_matches()
    if not rows:
        return None

    ordered = sorted(rows, key=lambda item: (item.get("date") or "", item.get("match_id") or 0))
    fixtures = []
    for idx, row in enumerate(ordered, start=1):
        gw = ((idx - 1) // 10) + 1
        home = normalize_team_name(row.get("home_team"))
        away = normalize_team_name(row.get("away_team"))
        if not home or not away:
            continue
        status = "completed" if row.get("is_result") else "pending"
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        fixtures.append((
            gw,
            row.get("date"),
            home,
            away,
            status,
            int(hg) if hg is not None else None,
            int(ag) if ag is not None else None,
        ))
    return fixtures


def seed_fixtures():
    print("1. Initializing clean 2026/27 database...")
    init_db()

    print("\n2. Loading 2026/27 schedule from Understat...")
    fixtures_to_insert = _seed_understat_fixtures() or _seed_placeholder_fixtures()

    if fixtures_to_insert == _seed_placeholder_fixtures():
        print("   Understat data unavailable; using local placeholder fixtures fallback.")
    else:
        print(f"   Loaded {len(fixtures_to_insert)} actual fixtures from Understat.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO fixtures_26_27 (gameweek, match_date, home_team, away_team, status, home_goals, away_goals)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        fixtures_to_insert,
    )
    conn.commit()
    conn.close()
    print(f"   Successfully inserted {len(fixtures_to_insert)} fixtures into fixtures_26_27.")


if __name__ == "__main__":
    seed_fixtures()
