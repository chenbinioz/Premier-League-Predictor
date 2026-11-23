"""Compare EPL probability forecasts without writing training or production assets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.neural_dixon_coles import predict_scoreline_grid
from scripts.backtest_challengers import (
    DYNAMIC_FEATURES, FORM_WINDOWS, ChallengerEngine, bounded_rho, fit_dixon_coles, form_columns,
)
from scripts.train_models import labels, ndc_probs, score
from scripts.train_static_poisson import fit_static_poisson
from src.features.prematch import CONTRACT, XGB_BASE_FEATURES, historical_rows

LEGACY_SEASONS = ("2024/25", "2025/26")
DEFAULT_SEASONS = ("2018/19", "2019/20", "2020/21", "2021/22",
                   "2022/23", "2023/24", *LEGACY_SEASONS)
OUTCOMES = ("away", "draw", "home")


def replay(raw, horizon_hours, result_lag_hours, *, return_engine=False):
    """Estimate when a historical result and its statistics could be used."""
    # Canonical rows normalize teams and dates using the live pipeline rules.
    rows = historical_rows(raw)
    from src.data.live_fetchers import normalize_team_name, _normalise_date

    times = {}
    for item in raw.to_dict("records"):
        key = (item["Season"], _normalise_date(item["Date"]),
               normalize_team_name(item["HomeTeam"]), normalize_team_name(item["AwayTeam"]))
        if key in times:
            raise ValueError(f"Duplicate historical fixture: {key}")
        times[key] = item.get("Time")
    engine = ChallengerEngine()
    output = []
    prior_season_available = None
    for season in sorted({row["season"] for row in rows}):
        matches = [row for row in rows if row["season"] == season]
        engine.start_season(season, {row[side] for row in matches for side in ("home_team", "away_team")})
        timeline = []
        for row in matches:
            home, away, day = row["home_team"], row["away_team"], row["match_date"]
            midnight = pd.Timestamp(day).tz_localize("Europe/London")
            time = times[season, day, home, away]
            naive_kickoff = pd.to_datetime(f"{day} {time}", format="%Y-%m-%d %H:%M", errors="coerce")
            estimated = pd.isna(naive_kickoff)
            kickoff = midnight if estimated else naive_kickoff.tz_localize("Europe/London")
            # Missing kickoff times use the earliest possible forecast cutoff and
            # the latest possible kickoff for the observation availability proxy.
            latest_kickoff = midnight + pd.DateOffset(days=1) if estimated else kickoff
            available = latest_kickoff + pd.Timedelta(hours=result_lag_hours)
            timeline.append((kickoff.tz_convert("UTC"), available.tz_convert("UTC"), estimated, row))
        timeline.sort(key=lambda match: (match[0], match[3]["home_team"]))
        if prior_season_available is not None:
            first_cutoff = timeline[0][0] - pd.Timedelta(hours=horizon_hours)
            if prior_season_available > first_cutoff:
                raise ValueError("The forecast horizon and result lag overlap a season boundary")
        prior_season_available = max(match[1] for match in timeline)
        next_result = 0
        for kickoff, available, estimated, row in timeline:
            cutoff = kickoff - pd.Timedelta(hours=horizon_hours)
            # Preserve chronological state when an earlier observation is delayed.
            while next_result < len(timeline):
                _, previous_available, _, previous = timeline[next_result]
                if previous_available > cutoff or previous["match_date"] >= row["match_date"]:
                    break
                engine.observe(previous)
                next_result += 1
            home, away = row["home_team"], row["away_team"]
            output.append({
                "Date": row["match_date"], "Season": season, "HomeTeam": home, "AwayTeam": away,
                "FTHG": row["home_goals"], "FTAG": row["away_goals"],
                "kickoff_utc": kickoff.isoformat(), "cutoff_utc": cutoff.isoformat(),
                "kickoff_estimated": estimated, "result_available_utc": available.isoformat(),
                **{key: row[key] for key in ("B365H", "B365D", "B365A")},
                **engine.features(home, away, row["match_date"]),
                **engine.challenger_features(home, away),
            })
        for _, _, _, row in timeline[next_result:]:
            engine.observe(row)
    frame = pd.DataFrame(output)
    return (frame, engine) if return_engine else frame


def baseline_predictions(train, test):
    y = labels(train)
    yield "league_prior", np.tile(np.bincount(y, minlength=3) / len(train), (len(test), 1))
    elo = ["Elo_Diff"]
    specifications = {"elo_logistic": elo, "form_logistic": XGB_BASE_FEATURES}
    rolling = set(DYNAMIC_FEATURES).issubset(train)
    if rolling:
        for window in FORM_WINDOWS:
            name = "opponent_adjusted_form" if window == 5 else f"opponent_adjusted_form_{window}"
            specifications[name] = [*elo, *form_columns(window)]
        specifications.update({
            "dynamic_strength": DYNAMIC_FEATURES,
            "elo_dynamic_strength": [*elo, *DYNAMIC_FEATURES],
            "elo_form_dynamic_strength": [*elo, *form_columns(5), *DYNAMIC_FEATURES],
        })
    for name, features in specifications.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1))
        model.fit(train[features], y)
        yield name, model.predict_proba(test[features])
    yield "static_poisson", score_model_probabilities(fit_static_poisson(train), test)
    if rolling:
        yield "weighted_dixon_coles", score_model_probabilities(fit_dixon_coles(train), test)


def score_model_probabilities(model, test):
    probabilities = []
    attack, defense = model["attack"], model["defense"]
    base = model.get("base_rate", 1.0)
    for home, away in zip(test.HomeTeam, test.AwayTeam):
        home_rate = base * model["home_adv"] * attack.get(home, 1.0) * defense.get(away, 1.0)
        away_rate = base * attack.get(away, 1.0) * defense.get(home, 1.0)
        rho = bounded_rho(home_rate, away_rate, model.get("strength", 0.0))
        grid = predict_scoreline_grid(home_rate, away_rate, rho, max_goals=12)
        probabilities.append([np.triu(grid, 1).sum(), np.trace(grid), np.tril(grid, -1).sum()])
    return np.asarray(probabilities)


def paired_loss(rows, reference, draws=2000):
    """Resample calendar weeks within each season on paired fixtures."""
    result = {}
    rng = np.random.default_rng(42)
    rows = rows.assign(week=pd.to_datetime(rows.Date).dt.to_period("W"))
    wide = rows.pivot(index=["Season", "week", "Date", "HomeTeam", "AwayTeam"],
                      columns="model", values="match_log_loss")
    for model in wide.columns.drop(reference):
        differences = (wide[model] - wide[reference]).dropna()
        if differences.empty:
            continue
        totals = np.zeros(draws)
        counts = np.zeros(draws)
        weeks = differences.groupby(level=["Season", "week"]).agg(["sum", "count"])
        for _, group in weeks.groupby(level="Season"):
            sample = rng.integers(len(group), size=(draws, len(group)))
            totals += group["sum"].to_numpy()[sample].sum(axis=1)
            counts += group["count"].to_numpy()[sample].sum(axis=1)
        result[model] = {"n": len(differences), "delta": float(differences.mean()),
                         "interval_95": np.quantile(totals / counts, [0.025, 0.975]).tolist()}
    return result


def evaluate(frame, seasons, legacy=False):
    from src.features.prematch import normalized_market

    outputs = []
    folds = {}
    for season in seasons:
        test = frame[frame.Season == season]
        train = frame[frame.Season <= "2022/23"] if legacy else frame[frame.Season < season]
        if not legacy and not test.empty:
            train = train[pd.to_datetime(train.result_available_utc, utc=True) <
                          pd.to_datetime(test.cutoff_utc, utc=True).min()]
        if train.empty or test.empty:
            raise ValueError(f"No training or evaluation matches for {season}")
        predictions = dict(baseline_predictions(train, test))
        if legacy:
            from scripts.predict_next_gameweek import _load_ndc, _load_xgb, model_manifest

            manifest = model_manifest()
            training = manifest["training"]
            if season not in (training["validation_season"], training["test_season"]):
                raise ValueError(f"{season} is outside the deployed bundle's evaluation seasons")
            neural, scaler, _ = _load_ndc()
            predictions["ndc"] = ndc_probs(neural, scaler, test)
            predictions["xgb_market_free"] = _load_xgb().predict_proba(test[XGB_BASE_FEATURES])
            predictions["blend"] = (training["ndc_weight"] * predictions["ndc"]
                                    + training["xgb_weight"] * predictions["xgb_market_free"])
        market = np.array([normalized_market(odds) or [np.nan] * 3
                           for odds in test[["B365A", "B365D", "B365H"]].to_numpy()])
        covered = np.isfinite(market).all(axis=1)
        predictions["bookmaker_untimed"] = market
        y = labels(test)
        metrics = {}
        matched_metrics = {}
        for name, probabilities in predictions.items():
            valid = np.isfinite(probabilities).all(axis=1)
            if not valid.any():
                continue
            probabilities[valid] /= probabilities[valid].sum(axis=1, keepdims=True)
            metrics[name] = score(y[valid], probabilities[valid])
            matched = covered & valid
            if matched.any():
                matched_metrics[name] = score(y[matched], probabilities[matched])
            columns = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not legacy:
                columns += ["kickoff_utc", "cutoff_utc", "kickoff_estimated", "result_available_utc"]
            details = test.loc[valid, columns].copy()
            details["model"] = name
            for index, outcome in enumerate(OUTCOMES):
                details[f"p_{outcome}"] = probabilities[valid, index]
            details["match_log_loss"] = -np.log(np.clip(
                probabilities[valid, y[valid]], np.finfo(float).eps, 1))
            outputs.append(details)
        folds[season] = {"train_matches": len(train), "test_matches": len(test),
                         "train_end": train.Date.max(), "market_matches": int(covered.sum()),
                         "estimated_kickoffs": None if legacy else int(test.kickoff_estimated.sum()),
                         "scores": metrics, "matched_market_scores": matched_metrics}
    rows = pd.concat(outputs, ignore_index=True)
    return rows, {"folds": folds, "paired_vs_elo": paired_loss(rows, "elo_logistic")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-start", choices=("2015/16", "2016/17"), default="2015/16",
                        help="Use 2016/17 to measure the added season on identical evaluation folds")
    parser.add_argument("--horizon-hours", type=float, default=24)
    parser.add_argument("--result-lag-hours", type=float, default=3,
                        help="Assumed kickoff-to-availability delay for historical scores and stats")
    args = parser.parse_args()
    if not np.isfinite([args.horizon_hours, args.result_lag_hours]).all() or min(
        args.horizon_hours, args.result_lag_hours
    ) <= 0:
        parser.error("horizon and result lag must be positive and finite")
    raw_path = ROOT / "data/raw/epl_10_seasons_raw.csv"
    processed_path = ROOT / "data/processed/epl_model_features.csv"
    history_path = ROOT / "data/backtest/epl_1516.csv"
    raw = pd.concat([pd.read_csv(history_path), pd.read_csv(raw_path)], ignore_index=True)
    raw = raw[raw.Season.between(args.history_start, DEFAULT_SEASONS[-1])]
    rolling, rolling_summary = evaluate(replay(raw, args.horizon_hours, args.result_lag_hours), DEFAULT_SEASONS)
    legacy, legacy_summary = evaluate(pd.read_csv(processed_path), LEGACY_SEASONS, legacy=True)
    from scripts.predict_next_gameweek import model_manifest

    summary = {
        "legacy_model_version": model_manifest()["model_version"],
        "feature_contract": CONTRACT,
        "history_start": raw.Season.min(),
        "2015_16_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(processed_path.read_bytes()).hexdigest(),
        "rolling_horizon_hours": args.horizon_hours,
        "assumed_kickoff_to_observation_hours": args.result_lag_hours,
        "rolling": rolling_summary,
        "legacy": legacy_summary,
        "notes": [
            "Rolling replay preserves chronological state, so a delayed observation can hold back later observations.",
            "Only results from earlier local dates enter state, even at horizons shorter than 24 hours.",
            "Missing local kickoff times use midnight for target cutoffs and next midnight for availability.",
            "Historical odds are untimed and appear only as a retrospective reference, never as model inputs.",
            "Legacy evaluates the deployed bundle with its original dated inputs and base training cutoff.",
            "Paired intervals resample calendar weeks within each season and are descriptive.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(args.output_dir / "rolling_predictions.csv", index=False)
    legacy.to_csv(args.output_dir / "legacy_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote backtest results to {args.output_dir}")


if __name__ == "__main__":
    main()
