"""Record and evaluate paired Elo and opponent-adjusted forecasts."""
import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest_challengers import form_columns
from scripts.backtest_models import OUTCOMES, replay
from scripts.train_models import labels, score
from src.data.state_manager import RESULT_COLUMNS, SEASON, digest
from src.features.prematch import CONTRACT

LEDGER = ROOT / "data/live/shadow_forecasts.json"
HISTORY = [ROOT / "data/backtest/epl_1516.csv", ROOT / "data/raw/epl_10_seasons_raw.csv"]
FEATURES = {"elo": ["Elo_Diff"], "elo_form_10": ["Elo_Diff", *form_columns(10)]}
TRIAL = "elo-form-10-v1"


def load_ledger(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_ledger(path, ledger):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)


def read_snapshot(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        fixtures = [dict(row) for row in conn.execute(
            "SELECT * FROM fixtures_26_27 ORDER BY match_date, home_team, away_team"
        )]
        latest = conn.execute(
            "SELECT status, observed_at FROM ingestion_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    return fixtures, latest


def select_targets(fixtures, ledger, cutoff):
    targets = []
    for row in fixtures:
        key = f"{SEASON}:{row['match_id']}"
        if key in ledger or row["status"] != "pending" or row["schedule_status"] == "postponed":
            continue
        if not row.get("kickoff_at"):
            continue
        kickoff = datetime.fromisoformat(row["kickoff_at"])
        if cutoff < kickoff <= cutoff + timedelta(hours=30):
            targets.append((key, row, kickoff))
    return targets


def generate(fixtures, targets, cutoff, state_observed_at):
    history = pd.concat([pd.read_csv(path) for path in HISTORY], ignore_index=True)
    history = history[history.Season < SEASON]
    # Reuse the historical cutoff for training and replay the exact same
    # chronological observations into the engine for live feature generation.
    training, engine = replay(history, 24, 3, return_engine=True)
    engine.start_season(SEASON, {row[side] for row in fixtures for side in ("home_team", "away_team")})
    observed = [row for row in fixtures if row["status"] == "completed"]
    for row in observed:
        timestamp = row.get("result_observed_at")
        if not timestamp or datetime.fromisoformat(timestamp) > cutoff:
            raise ValueError(f"Result lacks an observation available at issuance for fixture {row['match_id']}")
        observation = {field: row[column] for field, column in RESULT_COLUMNS.items()}
        engine.observe({**observation, "home_team": row["home_team"], "away_team": row["away_team"]})
    y = labels(training)
    models = {
        name: make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(training[columns], y)
        for name, columns in FEATURES.items()
    }
    training_hash = digest(training[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                                     *FEATURES["elo_form_10"]]].to_dict("records"))
    forecasts = {}
    for key, row, kickoff in targets:
        home, away = row["home_team"], row["away_team"]
        features = {
            **engine.features(home, away, row["match_date"]),
            **engine.challenger_features(home, away),
        }
        inputs = {column: features[column] for column in FEATURES["elo_form_10"]}
        forecasts[key] = {
            "trial": TRIAL, "feature_contract": CONTRACT, "training_hash": training_hash,
            "training_matches": len(training), "home_team": home, "away_team": away,
            "match_id": row["match_id"], "kickoff_at": kickoff.isoformat(),
            "information_cutoff": cutoff.isoformat(), "state_observed_at": state_observed_at,
            "features": inputs,
            "probabilities": {
                name: dict(zip(OUTCOMES, model.predict_proba(pd.DataFrame([inputs])[FEATURES[name]])[0]))
                for name, model in models.items()
            },
        }
    return forecasts


def finalize(forecasts, issued_at):
    ready = {}
    for key, forecast in forecasts.items():
        kickoff = datetime.fromisoformat(forecast["kickoff_at"])
        if issued_at < kickoff:
            ready[key] = {**forecast, "issued_at": issued_at.isoformat(),
                          "lead_hours": (kickoff - issued_at).total_seconds() / 3600}
    return ready


def evaluate(ledger, fixtures):
    results = {row["match_id"]: row for row in fixtures}
    outcomes, pairs = [], []
    pending = excluded = 0
    for forecast in ledger.values():
        row = results.get(forecast["match_id"])
        if (row is None or not row.get("kickoff_at") or
            (row["home_team"], row["away_team"]) != (forecast["home_team"], forecast["away_team"]) or
            datetime.fromisoformat(row["kickoff_at"]) != datetime.fromisoformat(forecast["kickoff_at"])):
            excluded += 1
            continue
        if row["status"] != "completed":
            pending += 1
            continue
        outcomes.append(int(np.sign(row["home_goals"] - row["away_goals"])) + 1)
        pairs.append(forecast["probabilities"])
    scores = {
        name: score(np.array(outcomes), [[pair[name][outcome] for outcome in OUTCOMES] for pair in pairs])
        for name in FEATURES
    } if pairs else {}
    return {"locked": len(ledger), "scored": len(pairs), "pending": pending,
            "excluded_changed_fixture": excluded, "scores": scores,
            "log_loss_delta_vs_elo": scores["elo_form_10"]["log_loss"] - scores["elo"]["log_loss"]
            if scores else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/live/epl_2627.db")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ledger = load_ledger(args.ledger)
    fixtures, latest = read_snapshot(args.db)
    if args.evaluate:
        print(json.dumps(evaluate(ledger, fixtures), indent=2))
        return
    cutoff = datetime.now(timezone.utc)
    if (not latest or latest["status"] != "ok" or
        not cutoff - timedelta(hours=36) <= datetime.fromisoformat(latest["observed_at"]) <= cutoff):
        raise ValueError("A successful reconciliation within 36 hours is required")
    targets = select_targets(fixtures, ledger, cutoff)
    forecasts = generate(fixtures, targets, cutoff, latest["observed_at"]) if targets else {}
    forecasts = finalize(forecasts, datetime.now(timezone.utc))
    if forecasts and not args.dry_run:
        write_ledger(args.ledger, {**ledger, **forecasts})
    print(json.dumps({"new_pairs": len(forecasts), "dry_run": args.dry_run,
                      **evaluate({**ledger, **forecasts}, fixtures)}, indent=2))


if __name__ == "__main__":
    main()
