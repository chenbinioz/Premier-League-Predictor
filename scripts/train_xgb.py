"""
Train Calibrated XGBoost Match Outcome Classifier
=================================================
Trains an XGBoost multi-class classifier on 10 EPL seasons of processed match
features, calibrates predicted probabilities using isotonic regression (CV=3),
and handles draw class imbalance via sample weighting (L12 / Recommendation #6).

Usage:
    python scripts/train_xgb.py [--draw-weight 1.35] [--output models/calibrated_xgb_outcome.pkl]

Features (18):
    - Elo ratings & Elo difference
    - Multi-scale rolling xG attack & defense differences (windows 3, 5, 10)
    - Pressure metrics (Corner & Foul differences, roll5)
    - Venue-specific xG difference & expected total match xG
    - Schedule context (Rest days difference, Congestion difference)
    - Bookmaker implied probabilities (B365 margin-normalized)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xgb")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "epl_model_features.csv"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "calibrated_xgb_outcome.pkl"

FEATURE_COLS = [
    "Elo_Diff", "Home_Elo", "Away_Elo",
    "xG_Attack_Diff_roll3", "xG_Defense_Diff_roll3",
    "xG_Attack_Diff_roll5", "xG_Defense_Diff_roll5",
    "xG_Attack_Diff_roll10", "xG_Defense_Diff_roll10",
    "Corner_Diff_roll5", "Foul_Diff_roll5",
    "Venue_xG_Attack_Diff", "Expected_Match_xG",
    "Rest_Diff", "Congestion_Diff",
    "Bookie_Prob_H", "Bookie_Prob_D", "Bookie_Prob_A",
]


def prepare_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Computes differential features and target labels from processed dataset."""
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Outcome target: 2 = Home Win, 1 = Draw, 0 = Away Win
    conditions = [
        df["FTHG"] > df["FTAG"],
        df["FTHG"] == df["FTAG"],
        df["FTHG"] < df["FTAG"],
    ]
    df["Target"] = np.select(conditions, [2, 1, 0], default=1)

    # Elo diff
    df["Elo_Diff"] = df["Home_Elo"] - df["Away_Elo"]

    # Multi-scale rolling differentials
    for w in [3, 5, 10]:
        df[f"xG_Attack_Diff_roll{w}"] = df[f"Home_xG_Created_roll{w}"] - df[f"Away_xG_Conceded_roll{w}"]
        df[f"xG_Defense_Diff_roll{w}"] = df[f"Away_xG_Created_roll{w}"] - df[f"Home_xG_Conceded_roll{w}"]
        df[f"Corner_Diff_roll{w}"] = df[f"Home_Corners_roll{w}"] - df[f"Away_Corners_roll{w}"]
        df[f"Foul_Diff_roll{w}"] = df[f"Home_Fouls_roll{w}"] - df[f"Away_Fouls_roll{w}"]

    # Venue & match dynamics
    df["Venue_xG_Attack_Diff"] = df["Home_xG_Created_Venue_roll5"] - df["Away_xG_Conceded_Venue_roll5"]
    df["Expected_Match_xG"] = df["Home_xG_Created_roll5"] + df["Away_xG_Created_roll5"]

    # Schedule context
    df["Rest_Diff"] = df["Home_Rest_Days"] - df["Away_Rest_Days"]
    df["Congestion_Diff"] = df["Home_Congestion_Flag"] - df["Away_Congestion_Flag"]

    # Market baseline (margin-normalized implied probabilities)
    raw_margin = (1.0 / df["B365H"]) + (1.0 / df["B365D"]) + (1.0 / df["B365A"])
    df["Bookie_Prob_H"] = (1.0 / df["B365H"]) / raw_margin
    df["Bookie_Prob_D"] = (1.0 / df["B365D"]) / raw_margin
    df["Bookie_Prob_A"] = (1.0 / df["B365A"]) / raw_margin

    X = df[FEATURE_COLS].copy()
    y = df["Target"].copy()

    # Fill any remaining NaNs cleanly
    X = X.ffill().bfill().fillna(0.0)

    return X, y


def train_calibrated_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    draw_weight: float = 1.35,
    cv: int = 3,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Trains base XGBoost with draw sample weighting and calibrates with Isotonic Regression."""
    # Apply sample weighting for draw class (Target == 1) to address class imbalance
    sample_weights = np.where(y_train == 1, draw_weight, 1.0)

    base_xgb = XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.015,
        reg_alpha=1.5,
        reg_lambda=2.5,
        subsample=0.75,
        colsample_bytree=0.75,
        objective="multi:softprob",
        num_class=3,
        random_state=random_state,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method="isotonic",
        cv=cv,
    )

    logger.info(
        "Fitting XGBoost + CalibratedClassifierCV (draw_weight=%.2f, cv=%d)...",
        draw_weight,
        cv,
    )
    calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
    logger.info("Fitting complete.")

    return calibrated_model


def evaluate_model(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluates probability calibration and classification performance."""
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    # Benchmark against bookmaker odds
    bookie_probs = X_test[["Bookie_Prob_A", "Bookie_Prob_D", "Bookie_Prob_H"]].values
    bookie_loss = log_loss(y_test, bookie_probs)
    bookie_acc = accuracy_score(y_test, np.argmax(bookie_probs, axis=1))

    logger.info("\n" + "=" * 50)
    logger.info("=== TEST SET PERFORMANCE (80/20 CHRONOLOGICAL SPLIT) ===")
    logger.info("=" * 50)
    logger.info("Model Accuracy:    %.1f%%  | Model Log-Loss:    %.4f", acc * 100, loss)
    logger.info("Bookie Accuracy:   %.1f%%  | Bookie Log-Loss:   %.4f", bookie_acc * 100, bookie_loss)
    logger.info("-" * 50)
    logger.info("Classification Report:\n%s", classification_report(
        y_test, preds, target_names=["Away Win (0)", "Draw (1)", "Home Win (2)"], digits=3, zero_division=0
    ))

    draw_preds = int(np.sum(preds == 1))
    actual_draws = int(np.sum(y_test == 1))
    logger.info("Draw Predictions: %d / %d (Actual Draws: %d)", draw_preds, len(y_test), actual_draws)
    logger.info("Mean Predicted Probabilities: Away=%.1f%%, Draw=%.1f%%, Home=%.1f%%",
                np.mean(probs[:, 0]) * 100, np.mean(probs[:, 1]) * 100, np.mean(probs[:, 2]) * 100)

    return {
        "loss": loss,
        "acc": acc,
        "bookie_loss": bookie_loss,
        "bookie_acc": bookie_acc,
        "draw_preds": draw_preds,
        "actual_draws": actual_draws,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Calibrated XGBoost with Draw Weighting")
    parser.add_argument(
        "--draw-weight",
        type=float,
        default=1.35,
        help="Sample weight multiplier for draws (class 1). Default: 1.35",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for probability calibration. Default: 3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path for saved model pickle. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Evaluate without saving model pickle to disk",
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        logger.error("Data file not found at %s. Please run build_features first.", DATA_PATH)
        sys.exit(1)

    logger.info("Loading features from %s...", DATA_PATH)
    df_raw = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df_raw)

    split_idx = int(len(X) * 0.80)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    logger.info("Dataset shape: %d total matches (%d features).", len(X), len(FEATURE_COLS))
    logger.info("Training set: %d matches | Test set: %d matches.", len(X_train), len(X_test))

    model = train_calibrated_xgb(X_train, y_train, draw_weight=args.draw_weight, cv=args.cv)
    evaluate_model(model, X_test, y_test)

    if not args.no_save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logger.info("Model saved successfully to %s", output_path)


if __name__ == "__main__":
    main()
