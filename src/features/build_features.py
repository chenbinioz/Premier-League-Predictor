"""Build leakage-free historical inputs through the live pre-match engine."""
import os
import sys
from pathlib import Path
import pandas as pd
REPO_ROOT = str(Path(__file__).resolve().parents[2])
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data/raw/epl_10_seasons_raw.csv")
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, "data/processed/epl_model_features.csv")


def build_features(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Generate every input before observing the target result, via serving's engine."""
    from src.features.prematch import FeatureEngine, historical_rows, CONTRACT
    frame = pd.read_csv(raw_path)
    rows = historical_rows(frame)
    engine, output = FeatureEngine(), []
    for season in sorted({r["season"] for r in rows}):
        matches = [r for r in rows if r["season"] == season]
        engine.start_season(season, {r[k] for r in matches for k in ("home_team", "away_team")})
        for row in matches:
            features = engine.features(row["home_team"], row["away_team"], row["match_date"],
                                       [row.get(k) for k in ("B365H", "B365D", "B365A")])
            output.append({"Date": row["match_date"], "Season": season, "HomeTeam": row["home_team"],
                           "AwayTeam": row["away_team"], "FTHG": row["home_goals"], "FTAG": row["away_goals"],
                           "Feature_Contract": CONTRACT, **{k: row.get(k) for k in ("B365H", "B365D", "B365A")}, **features})
            engine.observe(row)
    result = pd.DataFrame(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Built {len(result)} chronological pre-match rows ({CONTRACT})")
    return result


if __name__ == "__main__":
    sys.path.insert(0, REPO_ROOT)
    build_features()
