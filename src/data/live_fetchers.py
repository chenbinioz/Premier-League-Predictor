"""Canonical 2026/27 data adapters.

Official Premier League FPL fixtures own schedule/round identity. Football-data
supplies observed scores and match statistics plus real upcoming prices.
Understat is optional xG enrichment; no date or groups-of-ten round guessing is
used. Every network read has a bounded timeout on the primary pipeline path.
"""

from __future__ import annotations
from contextlib import closing

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
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
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


def _safe_float(value):
    import math
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    value = _safe_float(value)
    return int(value) if value is not None and value >= 0 and value.is_integer() else None


def _normalise_date(value):
    if value is None:
        return ""
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ""


def normalize_team_name(raw_name):
    if raw_name is None:
        return ""
    clean = str(raw_name).strip()
    return CANONICAL_TEAM_MAP.get(clean, clean)


def _read_understat_matches():
    try:
        from understatapi import UnderstatClient
        with UnderstatClient() as client:
            raw = client.league(UNDERSTAT_LEAGUE).get_match_data(season=UNDERSTAT_SEASON)
        if not isinstance(raw, list):
            raise ValueError("Understat response has no match list")
    except Exception as exc:
        logger.warning("Understat unavailable: %s", exc)
        return []
    out = []
    for r in raw:
        goals, xg = r.get("goals") or {}, r.get("xG") or {}
        hg, ag = _safe_int(goals.get("h")), _safe_int(goals.get("a"))
        complete = r.get("isResult") is True and hg is not None and ag is not None
        dt = _normalise_date(r.get("datetime"))
        if not dt:
            continue
        out.append({"api_match_id": str(r.get("id", "")), "match_date": dt, "date": dt,
                    "home_team": normalize_team_name((r.get("h") or {}).get("title")),
                    "away_team": normalize_team_name((r.get("a") or {}).get("title")),
                    "home_goals": hg, "away_goals": ag, "home_xg": _safe_float(xg.get("h")),
                    "away_xg": _safe_float(xg.get("a")), "status": "completed" if complete else "pending",
                    "is_result": complete, "result_source": "understat", "xg_source": "understat"})
    return out


def _kickoff(match_date, match_time):
    from datetime import timezone
    from zoneinfo import ZoneInfo
    try:
        dt = datetime.strptime(f"{match_date} {match_time}", "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def fetch_from_football_data_csv(csv_path=None):
    """Observed results only; missing goals never become completed 0-0s."""
    from datetime import timezone
    try:
        if csv_path is not None:
            frame = pd.read_csv(csv_path)
        else:
            from io import StringIO
            response = requests.get(CSV_FALLBACK_URL, timeout=(5, 25))
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(frame):
            raise ValueError("Results CSV is missing required columns")
    except Exception as exc:
        logger.warning("football-data unavailable: %s", exc)
        return []
    observed = datetime.now(timezone.utc).isoformat()
    out = []
    for r in frame.to_dict("records"):
        dt = _normalise_date(r.get("Date"))
        hg, ag = _safe_int(r.get("FTHG")), _safe_int(r.get("FTAG"))
        if not dt or hg is None or ag is None:
            continue
        item = {"match_date": dt, "date": dt, "home_team": normalize_team_name(r.get("HomeTeam")),
                "away_team": normalize_team_name(r.get("AwayTeam")), "home_goals": hg, "away_goals": ag,
                "status": "completed", "result_source": "football-data", "observed_at": observed,
                "kickoff_at": _kickoff(dt, r.get("Time")), "xg_source": "football-data"}
        for key, column in {"home_sot": "HST", "away_sot": "AST", "home_shots": "HS", "away_shots": "AS",
                            "home_corners": "HC", "away_corners": "AC", "home_fouls": "HF", "away_fouls": "AF"}.items():
            item[key] = _safe_int(r.get(column))
        for key, col in {"home_xg": "HxG", "away_xg": "AxG", "B365H": "B365H", "B365D": "B365D", "B365A": "B365A"}.items():
            item[key] = _safe_float(r.get(col))
        if item["home_team"] and item["away_team"] and item["home_team"] != item["away_team"]:
            out.append(item)
    return out


def fetch_results(csv_path=None, include_understat=True):
    """Merge sources by season fixture identity, never by inferred gameweek.

    CSV is authoritative for scores, dates and match stats; Understat enriches xG.
    Conflicting scores fail explicitly instead of mixing incompatible observations.
    """
    from datetime import timezone
    understat = _read_understat_matches() if include_understat else []
    csv = fetch_from_football_data_csv(csv_path)
    merged = {(r["home_team"], r["away_team"]): dict(r) for r in understat if r["status"] == "completed"}
    for row in csv:
        key = row["home_team"], row["away_team"]
        old = merged.get(key)
        if old and (old["home_goals"], old["away_goals"]) != (row["home_goals"], row["away_goals"]):
            raise ValueError(f"Result sources disagree for {key}")
        merged[key] = {**(old or {}), **row}
        if old and old.get("home_xg") is not None and old.get("away_xg") is not None:
            merged[key].update(home_xg=old["home_xg"], away_xg=old["away_xg"], xg_source="understat")
    observed = datetime.now(timezone.utc).isoformat()
    for row in merged.values():
        row.setdefault("observed_at", observed)
        row["xg_source"] = row.get("xg_source") if row.get("home_xg") is not None and row.get("away_xg") is not None else None
    if not csv and not merged:
        raise RuntimeError("No usable results source; ingestion did not succeed")
    return sorted(merged.values(), key=lambda r: (r["match_date"], r["home_team"]))


def _known_fixtures(db_path=None):
    import sqlite3
    from pathlib import Path
    path = Path(db_path or os.environ.get("LIVE_DB_PATH", "data/live/epl_2627.db"))
    if not path.exists():
        return []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn, conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM fixtures_26_27")]


def fetch_weekend_results(gameweek, fixtures=None, csv_path=None):
    known = fixtures if fixtures is not None else _known_fixtures()
    lookup = {(f["home_team"], f["away_team"]): f for f in known if f["gameweek"] == gameweek}
    return [{**r, "gameweek": gameweek, "match_id": lookup[(r["home_team"], r["away_team"])]["match_id"]}
            for r in fetch_results(csv_path) if (r["home_team"], r["away_team"]) in lookup]


def fetch_upcoming_fixtures(gameweek):
    # The persisted schedule owns round identity. No groups-of-ten/date guessing.
    return [f for f in _known_fixtures() if f["gameweek"] == gameweek and f["status"] == "pending"]


def fetch_live_xg(*args, **kwargs):
    rows = _read_understat_matches()
    for row in rows:
        if len(args) == 1 and str(row.get("api_match_id")) == str(args[0]):
            return {"home_xg": row.get("home_xg"), "away_xg": row.get("away_xg")}
        if len(args) >= 2 and (row["home_team"], row["away_team"]) == tuple(map(normalize_team_name, args[:2])):
            if len(args) == 2 or row["match_date"] == _normalise_date(args[2]):
                return row.get("home_xg"), row.get("away_xg")
    return {"home_xg": None, "away_xg": None} if len(args) == 1 else (None, None)


def fetch_official_schedule(csv_path=None):
    """Official FPL event IDs supply rounds, including rescheduled fixtures.

    A snapshot CSV is supported for reproducible offline repair; its provenance
    must be supplied by this adapter, not by grouping dates or counting matches.
    """
    from datetime import timezone
    from zoneinfo import ZoneInfo
    if csv_path:
        rows = pd.read_csv(csv_path).to_dict('records')
        return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in rows]
    response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/', timeout=(5, 20))
    response.raise_for_status()
    teams = {t['id']: normalize_team_name(t['name']) for t in response.json()['teams']}
    response = requests.get('https://fantasy.premierleague.com/api/fixtures/', timeout=(5, 20))
    response.raise_for_status()
    observed = datetime.now(timezone.utc).isoformat()
    out = []
    for row in response.json():
        kickoff = datetime.fromisoformat(row['kickoff_time'].replace('Z','+00:00')) if row.get('kickoff_time') else None
        out.append({'home_team':teams[row['team_h']], 'away_team':teams[row['team_a']],
                    'gameweek':row.get('event'), 'match_date':kickoff.astimezone(ZoneInfo('Europe/London')).date().isoformat() if kickoff else None,
                    'kickoff_at':kickoff.isoformat() if kickoff else None, 'official_match_id':str(row['id']),
                    'schedule_source':'premier-league-fpl', 'schedule_observed_at':observed,
                    'schedule_status':'scheduled' if kickoff else 'postponed',
                    'official_home_goals':row.get('team_h_score') if row.get('finished') else None,
                    'official_away_goals':row.get('team_a_score') if row.get('finished') else None})
    if len(out) != 380 or len({(r['home_team'],r['away_team']) for r in out}) != 380:
        raise ValueError('Official schedule must contain exactly 380 unique EPL fixtures')
    if any(r['match_date'] and not '2026-07-01' <= r['match_date'] < '2027-07-01' for r in out):
        raise ValueError('Official schedule API is serving a different season')
    return out


def fetch_upcoming_odds(csv_path=None):
    """Real pre-match quotes; unavailable prices never become NDC proxies."""
    from datetime import timezone
    from src.features.prematch import normalized_market
    try:
        if csv_path:
            frame = pd.read_csv(csv_path)
        else:
            from io import StringIO
            response = requests.get('https://www.football-data.co.uk/fixtures.csv', timeout=(5,15))
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
        if not {'Div','Date','HomeTeam','AwayTeam','B365H','B365D','B365A'}.issubset(frame):
            return []
    except Exception as exc:
        logger.warning('Upcoming bookmaker quotes unavailable: %s',exc)
        return []
    observed = datetime.now(timezone.utc).isoformat()
    out = []
    for raw in frame[frame.Div=='E0'].to_dict('records'):
        odds = [_safe_float(raw.get(k)) for k in ('B365H','B365D','B365A')]
        if normalized_market(odds) is None:
            continue
        out.append({'home_team':normalize_team_name(raw['HomeTeam']), 'away_team':normalize_team_name(raw['AwayTeam']),
                    'match_date':_normalise_date(raw['Date']), 'odds_observed_at':observed,
                    **dict(zip(('B365H','B365D','B365A'),odds))})
    return out
