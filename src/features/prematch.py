"""One versioned pre-match feature contract for historical replay and serving.

Observations enter state only after the feature vector for their match is built.
The shot-quality proxy is deliberately used consistently; provider xG is stored
separately and must not silently replace this model input.
"""
from __future__ import annotations

import copy
import math
from datetime import date

CONTRACT = "prematch-v2-shot-proxy"
METRICS = {"GoalsScored": 1.35, "GoalsConceded": 1.35,
           "xG_Created": 1.5, "xG_Conceded": 1.5,
           "ShotsOnTarget": 4.4, "Corners": 5.1, "Fouls": 11.0}
FORM_FEATURES = [f"{side}_{metric}_roll{window}"
                 for side in ("Home", "Away") for window in (3, 5, 10) for metric in METRICS]
NDC_FEATURES = ["Home_Elo", "Away_Elo"] + FORM_FEATURES + [
    f"{side}_{name}" for side in ("Home", "Away") for name in (
        "Rest_Days", "Congestion_Flag", "xG_Created_Venue_roll5",
        "xG_Conceded_Venue_roll5", "Form_Missing_Fraction")]
XGB_BASE_FEATURES = ["Elo_Diff", "Home_Elo", "Away_Elo"] + [
    f"xG_{kind}_Diff_roll{window}" for window in (3, 5, 10) for kind in ("Attack", "Defense")
] + ["Corner_Diff_roll5", "Foul_Diff_roll5", "Venue_xG_Attack_Diff", "Expected_Match_xG",
     "Rest_Diff", "Congestion_Diff", "Home_Form_Missing_Fraction", "Away_Form_Missing_Fraction"]
MARKET_FEATURES = ["Bookie_Prob_H", "Bookie_Prob_D", "Bookie_Prob_A"]


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def shot_proxy(shots_on_target, shots):
    sot, total = finite(shots_on_target), finite(shots)
    if sot is None or total is None or not 0 <= sot <= total:
        return None
    return .32 * sot + .03 * (total - sot)


def normalized_market(odds):
    values = [finite(v) for v in odds]
    if any(v is None or v <= 1 for v in values):
        return None
    implied = [1 / v for v in values]
    return [v / sum(implied) for v in implied]


def mean_history(history, metric, season, window, venue=None):
    observations = [r for r in history if venue is None or r["venue"] == venue][-window:]
    known = [(finite(r.get(metric)), .5 ** max(0, int(season[:4]) - int(r["season"][:4]))) for r in observations]
    known = [(v, w) for v, w in known if v is not None]
    return sum(v * w for v, w in known) / sum(w for _, w in known) if known else METRICS[metric]


class FeatureEngine:
    def __init__(self, snapshot=None):
        state = copy.deepcopy(snapshot or {})
        self.teams = state.get("teams", {})
        self.season = state.get("season")
        self.active = set(state.get("active", []))

    def snapshot(self):
        return {"contract": CONTRACT, "season": self.season, "active": sorted(self.active), "teams": copy.deepcopy(self.teams)}

    def start_season(self, season, teams):
        teams = set(teams)
        if season == self.season:
            return
        for team in teams:
            old = self.teams.get(team, {"elo": 1500.0, "history": []})
            if self.season is not None and team not in self.active:
                # A returning/promoted club's years-old top-flight matches are
                # not current form. Use explicit priors until new data arrives.
                old["history"] = []
            old["elo"] = (1500.0 if self.season is None else
                          .8 * old["elo"] + .2 * 1500 if team in self.active else 1420.0)
            self.teams[team] = old
        self.active, self.season = teams, season

    def features(self, home, away, match_date, odds=None):
        if home == away or home not in self.teams or away not in self.teams:
            raise ValueError("Fixture teams must be distinct members of the initialized season")
        target = date.fromisoformat(match_date[:10])
        out = {}
        for side, team, venue in (("Home", home, "Home"), ("Away", away, "Away")):
            state = self.teams[team]
            history = state["history"]
            if any(r["date"] >= match_date for r in history):
                raise ValueError(f"State for {team} contains observations on/after prediction date")
            out[f"{side}_Elo"] = state["elo"]
            for window in (3, 5, 10):
                for metric in METRICS:
                    out[f"{side}_{metric}_roll{window}"] = mean_history(history, metric, self.season, window)
            for metric in ("xG_Created", "xG_Conceded"):
                out[f"{side}_{metric}_Venue_roll5"] = mean_history(history, metric, self.season, 5, venue)
            rest = (target - date.fromisoformat(history[-1]["date"])).days if history else 14
            out[f"{side}_Rest_Days"] = float(min(rest, 21))
            out[f"{side}_Congestion_Flag"] = float(rest < 4)
            recent = history[-10:]
            out[f"{side}_Form_Missing_Fraction"] = (
                sum(finite(r.get(m)) is None for r in recent for m in METRICS) / (len(recent) * len(METRICS))
                if recent else 1.0)
        out["Elo_Diff"] = out["Home_Elo"] - out["Away_Elo"]
        for w in (3, 5, 10):
            out[f"xG_Attack_Diff_roll{w}"] = out[f"Home_xG_Created_roll{w}"] - out[f"Away_xG_Conceded_roll{w}"]
            out[f"xG_Defense_Diff_roll{w}"] = out[f"Away_xG_Created_roll{w}"] - out[f"Home_xG_Conceded_roll{w}"]
        for name, metric in (("Corner", "Corners"), ("Foul", "Fouls")):
            out[f"{name}_Diff_roll5"] = out[f"Home_{metric}_roll5"] - out[f"Away_{metric}_roll5"]
        out["Venue_xG_Attack_Diff"] = out["Home_xG_Created_Venue_roll5"] - out["Away_xG_Conceded_Venue_roll5"]
        out["Expected_Match_xG"] = out["Home_xG_Created_roll5"] + out["Away_xG_Created_roll5"]
        out["Rest_Diff"] = out["Home_Rest_Days"] - out["Away_Rest_Days"]
        out["Congestion_Diff"] = out["Home_Congestion_Flag"] - out["Away_Congestion_Flag"]
        market = normalized_market(odds) if odds is not None else None
        if market:
            out.update(dict(zip(MARKET_FEATURES, market)))
        return out

    def observe(self, row):
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        if any(finite(v) is None or float(v) < 0 or not float(v).is_integer() for v in (hg, ag)):
            raise ValueError("Completed matches require non-negative integer goals")
        h, a = self.teams[home], self.teams[away]
        if any(s["history"] and s["history"][-1]["date"] >= row["match_date"] for s in (h, a)):
            raise ValueError("Team observations must be strictly chronological")
        expected = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - 60) / 400))
        actual = 1 if hg > ag else .5 if hg == ag else 0
        delta = 20 * (actual - expected)
        h["elo"] += delta
        a["elo"] -= delta
        hp = shot_proxy(row.get("home_sot"), row.get("home_shots"))
        ap = shot_proxy(row.get("away_sot"), row.get("away_shots"))
        for state, side, venue, gf, ga, xgf, xga in (
            (h, "home", "Home", hg, ag, hp, ap), (a, "away", "Away", ag, hg, ap, hp)):
            observation = {"season": self.season, "date": row["match_date"], "venue": venue,
                           "GoalsScored": int(gf), "GoalsConceded": int(ga),
                           "xG_Created": xgf, "xG_Conceded": xga,
                           "ShotsOnTarget": finite(row.get(side + "_sot")),
                           "Corners": finite(row.get(side + "_corners")), "Fouls": finite(row.get(side + "_fouls"))}
            state["history"].append(observation)
            # Keep enough history for both the overall and venue windows.
            history = state["history"]
            keep = set(range(max(0, len(history) - 10), len(history)))
            for v in ("Home", "Away"):
                keep.update([i for i, r in enumerate(history) if r["venue"] == v][-5:])
            state["history"] = [r for i, r in enumerate(history) if i in keep]


def historical_rows(frame):
    """Canonical observations from football-data rows, sorted deterministically."""
    from src.data.live_fetchers import normalize_team_name, _normalise_date
    rows = []
    for raw in frame.to_dict("records"):
        rows.append({"season": str(raw["Season"]), "match_date": _normalise_date(raw["Date"]),
                     "home_team": normalize_team_name(raw["HomeTeam"]), "away_team": normalize_team_name(raw["AwayTeam"]),
                     "home_goals": raw["FTHG"], "away_goals": raw["FTAG"],
                     **{field: finite(raw.get(col)) for field, col in {
                         "home_sot": "HST", "away_sot": "AST", "home_shots": "HS", "away_shots": "AS",
                         "home_corners": "HC", "away_corners": "AC", "home_fouls": "HF", "away_fouls": "AF",
                         "B365H": "B365H", "B365D": "B365D", "B365A": "B365A"}.items()}})
    return sorted(rows, key=lambda r: (r["match_date"], r["home_team"], r["away_team"]))


def replay_history(frame, before_season=None):
    engine = FeatureEngine()
    rows = historical_rows(frame)
    for season in sorted({r["season"] for r in rows}):
        if before_season and season >= before_season:
            continue
        matches = [r for r in rows if r["season"] == season]
        engine.start_season(season, {r[k] for r in matches for k in ("home_team", "away_team")})
        for row in matches:
            engine.observe(row)
    return engine
