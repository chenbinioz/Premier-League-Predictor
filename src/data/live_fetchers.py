"""
live_fetchers.py
================
Understat-backed live data adapter for the 2026/27 Premier League tracker.

This module intentionally preserves the project’s historical call patterns while
switching the live season source away from API-Sports / RapidAPI to Understat,
which exposes the 2026/27 season without a blocking paid plan.

Public API contract:
  - normalize_team_name(raw_name: str) -> str
  - fetch_upcoming_fixtures(gameweek: int) -> list[dict]
  - fetch_weekend_results(gameweek: int) -> list[dict]
  - fetch_live_xg(*args, **kwargs) -> tuple[float | None, float | None] | dict

Backward compatibility:
  - Existing code calls fetch_live_xg(home_team, away_team, match_date)
    and expects a 2-tuple: (home_xg, away_xg)
  - Newer code may call fetch_live_xg(match_id) and expect a dict:
    {'home_xg': float, 'away_xg': float}
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

UNDERSTAT_SEASON = os.environ.get("UNDERSTAT_SEASON", "2026")
UNDERSTAT_LEAGUE = os.environ.get("UNDERSTAT_LEAGUE", "EPL")
CSV_FALLBACK_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"


CANONICAL_TEAM_MAP: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull": "Hull",
    "Hull City": "Hull",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Man City": "Man City",
    "Manchester United": "Man United",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "Sheffield United": "Sheffield United",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Burnley FC": "Burnley",
    "Brentford FC": "Brentford",
    "Bournemouth AFC": "Bournemouth",
    "Watford": "Watford",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Middlesbrough": "Middlesbrough",
    "Birmingham City": "Birmingham City",
    "Blackburn Rovers": "Blackburn Rovers",
    "Blackpool": "Blackpool",
    "Portsmouth": "Portsmouth",
    "Reading": "Reading",
}


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            return raw[:10]
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def normalize_team_name(raw_name: str) -> str:
    """Map a raw Understat or scraped source team name to the project canonical name."""
    if raw_name is None:
        return ""
    clean = str(raw_name).strip()
    if not clean:
        return ""
    canonical = CANONICAL_TEAM_MAP.get(clean)
    if canonical is not None:
        return canonical
    return clean


def _assign_gameweeks(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign a sequential gameweek number using the actual season order.

    Understat does not expose a gameweek field in the raw match payload. To keep the
    tracker compatible with the rest of the project, we assign gameweeks in the same
    order as the season schedule: each block of 10 matches is treated as one fixture
    round, matching the Premier League structure.
    """
    ordered = sorted(rows, key=lambda item: (item.get("date") or "", item.get("match_id") or 0))
    out: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        row_copy = dict(row)
        row_copy["gameweek"] = ((index - 1) // 10) + 1
        row_copy["match_date"] = row_copy.get("date") or row_copy.get("match_date") or ""
        row_copy["date"] = row_copy["match_date"]
        out.append(row_copy)
    return out


def _read_understat_matches() -> list[dict[str, Any]]:
    """Load all EPL matches for the configured season from Understat using understatapi."""
    try:
        from understatapi import UnderstatClient
    except ImportError as exc:  # pragma: no cover - guard for environments without the package
        logger.warning("understatapi is not installed; falling back to football-data CSV only: %s", exc)
        return []

    try:
        with UnderstatClient() as client:
            raw_rows = client.league(UNDERSTAT_LEAGUE).get_match_data(season=UNDERSTAT_SEASON)
    except Exception as exc:  # pragma: no cover - runtime external dependency issue
        logger.warning("Understat fetch failed for season %s/%s: %s", UNDERSTAT_SEASON, UNDERSTAT_LEAGUE, exc)
        return []

    cleaned: list[dict[str, Any]] = []
    for row in raw_rows or []:
        h_info = row.get("h") or {}
        a_info = row.get("a") or {}
        goals = row.get("goals") or {}
        xg = row.get("xG") or {}

        home_team = normalize_team_name((h_info or {}).get("title"))
        away_team = normalize_team_name((a_info or {}).get("title"))
        if not home_team or not away_team:
            continue

        home_goals = _safe_int(goals.get("h"))
        away_goals = _safe_int(goals.get("a"))
        home_xg = _safe_float(xg.get("h"))
        away_xg = _safe_float(xg.get("a"))

        cleaned.append({
            "match_id": int(str(row.get("id") or 0).strip() or 0),
            "date": _normalise_date(row.get("datetime")),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "status": "completed" if bool(row.get("isResult")) else "pending",
            "is_result": bool(row.get("isResult")),
        })

    return sorted(cleaned, key=lambda item: (item.get("date") or "", item.get("match_id") or 0))


def fetch_from_football_data_csv() -> list[dict[str, Any]]:
    """Offline backup for completed results when Understat is unavailable.

    Reads the football-data.co.uk CSV for the 2026/27 season and returns a list of
    canonicalized rows containing at least Date/HomeTeam/AwayTeam/FTHG/FTAG.
    """
    try:
        df = pd.read_csv(CSV_FALLBACK_URL)
    except Exception as exc:  # pragma: no cover - network fallback may be unavailable
        logger.warning("football-data csv backup failed: %s", exc)
        return []

    if df.empty:
        return []

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("football-data csv missing expected columns: %s", sorted(missing))
        return []

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            date_value = _normalise_date(row.get("Date"))
            home_team = normalize_team_name(row.get("HomeTeam"))
            away_team = normalize_team_name(row.get("AwayTeam"))
            if not date_value or not home_team or not away_team:
                continue
            rows.append({
                "date": date_value,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": _safe_int(row.get("FTHG")) or 0,
                "away_goals": _safe_int(row.get("FTAG")) or 0,
                "status": "completed",
                "match_id": 0,
                "home_xg": None,
                "away_xg": None,
            })
        except Exception:
            continue
    return rows


def fetch_upcoming_fixtures(gameweek: int) -> list[dict[str, Any]]:
    """Return the pending fixtures for the requested gameweek from Understat.

    The function preserves the project contract for older code by returning both a
    canonical "match_date" field and a compatibility "date" alias.
    """
    if gameweek < 1:
        return []

    rows = _read_understat_matches()
    if not rows:
        return []

    rows = _assign_gameweeks(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("is_result") and row.get("gameweek") == gameweek:
            out.append({
                "match_id": int(row["match_id"]),
                "api_match_id": int(row["match_id"]),
                "gameweek": int(row["gameweek"]),
                "match_date": row["match_date"],
                "date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "status": "pending",
            })
    return out


def fetch_weekend_results(gameweek: int) -> list[dict[str, Any]]:
    """Return completed results for the requested gameweek from Understat.

    Returns a list of dicts with values suitable for live-state updates. Includes both
    "match_date" and "date" for compatibility with the existing pipeline.
    """
    if gameweek < 1:
        return []

    rows = _read_understat_matches()
    if rows:
        rows = _assign_gameweeks(rows)
        results: list[dict[str, Any]] = []
        for row in rows:
            if row.get("is_result") and row.get("gameweek") == gameweek:
                results.append({
                    "match_id": int(row["match_id"]),
                    "api_match_id": int(row["match_id"]),
                    "gameweek": int(row["gameweek"]),
                    "match_date": row["match_date"],
                    "date": row["match_date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_goals": int(row["home_goals"] or 0),
                    "away_goals": int(row["away_goals"] or 0),
                    "home_xg": float(row["home_xg"]) if row.get("home_xg") is not None else 0.0,
                    "away_xg": float(row["away_xg"]) if row.get("away_xg") is not None else 0.0,
                    "status": "completed",
                })
        if results:
            return results

    csv_rows = fetch_from_football_data_csv()
    if not csv_rows:
        return []

    csv_dates = sorted({row["date"] for row in csv_rows if row.get("date")})
    date_to_gw = {date_value: index + 1 for index, date_value in enumerate(csv_dates)}

    results = []
    for row in csv_rows:
        if row.get("status") == "completed" and date_to_gw.get(row.get("date"), 0) == gameweek:
            results.append({
                "match_id": int(row.get("match_id") or 0),
                "api_match_id": int(row.get("match_id") or 0),
                "gameweek": int(gameweek),
                "match_date": row["date"],
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": int(row.get("home_goals") or 0),
                "away_goals": int(row.get("away_goals") or 0),
                "home_xg": float(row.get("home_xg") or 0.0),
                "away_xg": float(row.get("away_xg") or 0.0),
                "status": "completed",
            })
    return results


def fetch_live_xg(*args: Any, **kwargs: Any) -> tuple[float | None, float | None] | dict[str, float | None]:
    """Fetch xG for a match.

    Backward-compatible signatures:
      - fetch_live_xg(home_team, away_team, match_date)
        -> (home_xg, away_xg)
      - fetch_live_xg(match_id)
        -> {'home_xg': float, 'away_xg': float}
    """
    if len(args) == 1 and isinstance(args[0], int):
        match_id = int(args[0])
        rows = _read_understat_matches()
        for row in rows:
            if int(row.get("match_id") or 0) == match_id:
                return {
                    "home_xg": float(row["home_xg"]) if row.get("home_xg") is not None else None,
                    "away_xg": float(row["away_xg"]) if row.get("away_xg") is not None else None,
                }
        return {"home_xg": None, "away_xg": None}

    if len(args) == 3:
        home_team, away_team, match_date = args
        rows = _read_understat_matches()
        normalized_date = _normalise_date(match_date)
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)
        for row in rows:
            if row.get("date") != normalized_date:
                continue
            if row.get("home_team") == home_norm and row.get("away_team") == away_norm:
                return (
                    float(row["home_xg"]) if row.get("home_xg") is not None else None,
                    float(row["away_xg"]) if row.get("away_xg") is not None else None,
                )
        return (None, None)

    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
        home_team, away_team = args
        rows = _read_understat_matches()
        for row in rows:
            if row.get("home_team") == normalize_team_name(home_team) and row.get("away_team") == normalize_team_name(away_team):
                return (
                    float(row["home_xg"]) if row.get("home_xg") is not None else None,
                    float(row["away_xg"]) if row.get("away_xg") is not None else None,
                )
        return (None, None)

    raise TypeError(
        "fetch_live_xg requires either (match_id) or (home_team, away_team, match_date) "
        "or (home_team, away_team)."
    )


__all__ = [
    "normalize_team_name",
    "fetch_upcoming_fixtures",
    "fetch_weekend_results",
    "fetch_live_xg",
    "fetch_from_football_data_csv",
]
