"""
Optimize Ensemble Blend Weights on Validation Season (2024/25)
==============================================================
Optimizes the ensemble weighting between Neural Dixon-Coles (NDC) and
Calibrated XGBoost using Nelder-Mead optimization and Logistic Regression
on the 2024/25 validation season (380 matches), and verifies out-of-sample
generalization on the 2025/26 test season (380 matches).

Usage:
    python scripts/optimize_blend.py [--save] [--output models/bin/blend_weights.json]

Outputs:
    - Detailed comparative metrics (Log-Loss, Accuracy, Draw Recall/Precision)
    - Optional JSON artifact with learned optimal weights for live inference
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must be set before torch/scipy import to prevent macOS OpenMP segfault
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize_blend")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.neural_dixon_coles import NeuralDixonColes, parse_grid_outputs, predict_scoreline_grid
from scripts.train_xgb import FEATURE_COLS as XGB_FEATURE_COLS
from scripts.train_xgb import prepare_features

DATA_PATH = REPO_ROOT / "data" / "processed" / "epl_model_features.csv"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "bin" / "blend_weights.json"


def load_data_and_splits():
    """Loads feature dataset and splits into Train, Val (2024/25), and Test (2025/26)."""
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    conditions = [
        df["FTHG"] > df["FTAG"],
        df["FTHG"] == df["FTAG"],
        df["FTHG"] < df["FTAG"],
    ]
    df["Target"] = np.select(conditions, [2, 1, 0], default=1)

    val_mask = df["Season"] == "2024/25"
    test_mask = df["Season"] == "2025/26"

    return df, val_mask, test_mask


def vectorized_dixon_coles(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray, max_goals: int = 6) -> np.ndarray:
    """Vectorized calculation of match probabilities from lambda, mu, rho."""
    k = np.arange(max_goals + 1)
    fact = np.array([1, 1, 2, 6, 24, 120, 720][:max_goals + 1], dtype=np.float64)
    lam = lam[:, None, None]
    mu = mu[:, None, None]
    rho = rho[:, None, None]

    k_h = k[None, :, None]
    k_a = k[None, None, :]
    fact_h = fact[None, :, None]
    fact_a = fact[None, None, :]

    pmf_h = np.exp(-lam) * (lam ** k_h) / fact_h
    pmf_a = np.exp(-mu) * (mu ** k_a) / fact_a

    grid = pmf_h * pmf_a  # shape (N, 7, 7)

    eps = 1e-10
    grid[:, 0, 0] *= np.maximum(1.0 - (lam * mu * rho)[:, 0, 0], eps)
    grid[:, 1, 0] *= (1.0 + (mu * rho)[:, 0, 0])
    grid[:, 0, 1] *= (1.0 + (lam * rho)[:, 0, 0])
    grid[:, 1, 1] *= np.maximum(1.0 - rho[:, 0, 0], eps)

    # Normalize grid
    grid_sum = grid.sum(axis=(1, 2), keepdims=True)
    grid /= np.where(grid_sum > 0, grid_sum, 1.0)

    tril_mask = np.tril(np.ones((max_goals + 1, max_goals + 1), dtype=bool), -1)
    triu_mask = np.triu(np.ones((max_goals + 1, max_goals + 1), dtype=bool), 1)
    diag_mask = np.eye(max_goals + 1, dtype=bool)

    p_home = (grid * tril_mask).sum(axis=(1, 2))
    p_draw = (grid * diag_mask).sum(axis=(1, 2))
    p_away = (grid * triu_mask).sum(axis=(1, 2))

    p_all = np.column_stack([p_away, p_draw, p_home])
    p_all /= p_all.sum(axis=1, keepdims=True)
    return p_all


def prepare_ndc_inference_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms historical dataframe to mirror the pre-match live inference feature vector
    constructed by `build_ndc_feature_vector()` in scripts/predict_next_gameweek.py.

    At pre-match inference time, same-match single stats (GoalsScored, GoalsConceded,
    xG, ShotsOnTarget, Corners, Fouls, Cards) are unknown and proxied by rolling 5-match averages
    and league constants.
    """
    df_inf = df.copy()
    for side in ["Home", "Away"]:
        df_inf[f"{side}_GoalsScored"] = df_inf[f"{side}_GoalsScored_roll5"]
        df_inf[f"{side}_GoalsConceded"] = df_inf[f"{side}_GoalsConceded_roll5"]
        df_inf[f"{side}_xG_Created"] = df_inf[f"{side}_xG_Created_roll5"]
        df_inf[f"{side}_xG_Conceded"] = df_inf[f"{side}_xG_Conceded_roll5"]
        df_inf[f"{side}_ShotsOnTarget"] = df_inf[f"{side}_ShotsOnTarget_roll5"]
        df_inf[f"{side}_Opp_ShotsOnTarget"] = (
            df_inf["Away_ShotsOnTarget_roll5"] if side == "Home" else df_inf["Home_ShotsOnTarget_roll5"]
        )
        df_inf[f"{side}_YellowCards"] = 1.6 if side == "Home" else 1.7
        df_inf[f"{side}_RedCards"] = 0.05
        df_inf[f"{side}_Corners"] = df_inf[f"{side}_Corners_roll5"]
        df_inf[f"{side}_Fouls"] = df_inf[f"{side}_Fouls_roll5"]
    return df_inf


def generate_ndc_probabilities(df_sub: pd.DataFrame) -> np.ndarray:
    """Generates (N, 3) array of [Away, Draw, Home] probabilities using Neural Dixon-Coles."""
    ndc_pt = torch.load(REPO_ROOT / "models" / "bin" / "neural_dixon_coles.pt", map_location="cpu", weights_only=False)
    ndc_scaler = joblib.load(REPO_ROOT / "models" / "bin" / "ndc_scaler.joblib")
    with open(REPO_ROOT / "models" / "bin" / "ndc_feature_names.json") as f:
        ndc_features = json.load(f)

    model = NeuralDixonColes(n_features=len(ndc_features))
    model.load_state_dict(ndc_pt["state_dict"])
    model.eval()

    df_inf = prepare_ndc_inference_df(df_sub)
    X = df_inf[ndc_features].copy().ffill().bfill().fillna(0.0).values
    X_scaled = ndc_scaler.transform(X)

    with torch.no_grad():
        t_X = torch.tensor(X_scaled, dtype=torch.float32)
        lam, mu, rho = model(t_X)
        lam = lam.squeeze().numpy()
        mu = mu.squeeze().numpy()
        rho = rho.squeeze().numpy()

    return vectorized_dixon_coles(lam, mu, rho, max_goals=6)


def generate_xgb_probabilities(X_all: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Generates (N, 3) array of [Away, Draw, Home] probabilities using Calibrated XGBoost."""
    xgb_model = joblib.load(REPO_ROOT / "models" / "calibrated_xgb_outcome.pkl")
    X_sub = X_all.loc[mask, XGB_FEATURE_COLS]
    # Calibrated classifier returns [Away(0), Draw(1), Home(2)]
    return xgb_model.predict_proba(X_sub).astype(np.float64)


def optimize_scalar_blend(
    p_ndc: np.ndarray,
    p_xgb: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """
    Finds optimal scalar weight alpha in [0, 1] minimizing Log-Loss:
        P_blend = alpha * P_ndc + (1 - alpha) * P_xgb
    """
    def loss_func(params):
        # Param is unbounded z; alpha = sigmoid(z)
        z = params[0]
        alpha = expit(z)
        p_blend = alpha * p_ndc + (1.0 - alpha) * p_xgb
        # Clip for numerical stability
        p_blend = np.clip(p_blend, 1e-12, 1.0 - 1e-12)
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        return log_loss(y_true, p_blend)

    res = minimize(loss_func, [0.0], method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-5})
    opt_alpha = float(expit(res.x[0]))
    min_loss = float(res.fun)
    return opt_alpha, min_loss


def optimize_classwise_blend(
    p_ndc: np.ndarray,
    p_xgb: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Finds class-specific weights [alpha_away, alpha_draw, alpha_home] in [0, 1]^3
    minimizing Log-Loss via Nelder-Mead:
        P_unnorm[c] = alpha_c * P_ndc[c] + (1 - alpha_c) * P_xgb[c]
        P_blend[c]  = P_unnorm[c] / sum(P_unnorm)
    """
    def loss_func(params):
        alphas = expit(params)  # shape (3,)
        p_unnorm = alphas * p_ndc + (1.0 - alphas) * p_xgb
        p_blend = p_unnorm / p_unnorm.sum(axis=1, keepdims=True)
        p_blend = np.clip(p_blend, 1e-12, 1.0 - 1e-12)
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        return log_loss(y_true, p_blend)

    init_params = [0.0, 0.0, 0.0]  # sigmoid(0) = 0.5 for each class
    res = minimize(loss_func, init_params, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-5})
    opt_alphas = expit(res.x)
    min_loss = float(res.fun)
    return opt_alphas, min_loss


def apply_blend(
    p_ndc: np.ndarray,
    p_xgb: np.ndarray,
    alphas: float | np.ndarray,
) -> np.ndarray:
    """Applies scalar or per-class blend weights and normalizes."""
    if isinstance(alphas, (float, int)):
        p_blend = alphas * p_ndc + (1.0 - alphas) * p_xgb
    else:
        p_unnorm = alphas * p_ndc + (1.0 - alphas) * p_xgb
        p_blend = p_unnorm / p_unnorm.sum(axis=1, keepdims=True)
    return p_blend / p_blend.sum(axis=1, keepdims=True)


def train_logistic_stacker(
    p_ndc_val: np.ndarray,
    p_xgb_val: np.ndarray,
    y_val: np.ndarray,
) -> LogisticRegression:
    """Trains a multi-class Logistic Regression stacking meta-classifier."""
    # Stack NDC and XGB predicted probabilities as 6 meta-features
    X_meta = np.hstack([p_ndc_val, p_xgb_val])
    stacker = LogisticRegression(
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    stacker.fit(X_meta, y_val)
    return stacker


def main():
    parser = argparse.ArgumentParser(description="Optimize Ensemble Blend Weights")
    parser.add_argument("--save", action="store_true", help="Save optimal blend configuration to models/bin/blend_weights.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output path for weights JSON")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" ENSEMBLE BLEND OPTIMIZATION — 2024/25 VALIDATION SEASON")
    print("=" * 70)

    # 1. Load data
    df, val_mask, test_mask = load_data_and_splits()
    y_val = df.loc[val_mask, "Target"].values
    y_test = df.loc[test_mask, "Target"].values

    logger.info("Validation set: %d matches (2024/25)", len(y_val))
    logger.info("Test set:       %d matches (2025/26)", len(y_test))

    # 2. Generate predictions
    logger.info("Generating predictions from NDC...")
    p_ndc_val = generate_ndc_probabilities(df[val_mask])
    p_ndc_test = generate_ndc_probabilities(df[test_mask])

    logger.info("Generating predictions from Calibrated XGBoost...")
    X_all, _ = prepare_features(df)
    p_xgb_val = generate_xgb_probabilities(X_all, val_mask)
    p_xgb_test = generate_xgb_probabilities(X_all, test_mask)

    # 3. Baseline Individual Models
    ndc_val_loss, ndc_val_acc = log_loss(y_val, p_ndc_val), accuracy_score(y_val, np.argmax(p_ndc_val, axis=1))
    ndc_test_loss, ndc_test_acc = log_loss(y_test, p_ndc_test), accuracy_score(y_test, np.argmax(p_ndc_test, axis=1))

    xgb_val_loss, xgb_val_acc = log_loss(y_val, p_xgb_val), accuracy_score(y_val, np.argmax(p_xgb_val, axis=1))
    xgb_test_loss, xgb_test_acc = log_loss(y_test, p_xgb_test), accuracy_score(y_test, np.argmax(p_xgb_test, axis=1))

    # 4. Baseline 50/50 Blend
    p_50_val = apply_blend(p_ndc_val, p_xgb_val, 0.50)
    p_50_test = apply_blend(p_ndc_test, p_xgb_test, 0.50)
    b50_val_loss, b50_val_acc = log_loss(y_val, p_50_val), accuracy_score(y_val, np.argmax(p_50_val, axis=1))
    b50_test_loss, b50_test_acc = log_loss(y_test, p_50_test), accuracy_score(y_test, np.argmax(p_50_test, axis=1))

    # 5. Method A: Optimal Scalar Blend (Nelder-Mead)
    opt_alpha, opt_scalar_val_loss = optimize_scalar_blend(p_ndc_val, p_xgb_val, y_val)
    p_scalar_val = apply_blend(p_ndc_val, p_xgb_val, opt_alpha)
    p_scalar_test = apply_blend(p_ndc_test, p_xgb_test, opt_alpha)
    scalar_test_loss, scalar_test_acc = log_loss(y_test, p_scalar_test), accuracy_score(y_test, np.argmax(p_scalar_test, axis=1))
    scalar_val_acc = accuracy_score(y_val, np.argmax(p_scalar_val, axis=1))

    # 6. Method B: Optimal Class-Wise Blend (Nelder-Mead)
    opt_alphas, opt_cw_val_loss = optimize_classwise_blend(p_ndc_val, p_xgb_val, y_val)
    p_cw_val = apply_blend(p_ndc_val, p_xgb_val, opt_alphas)
    p_cw_test = apply_blend(p_ndc_test, p_xgb_test, opt_alphas)
    cw_test_loss, cw_test_acc = log_loss(y_test, p_cw_test), accuracy_score(y_test, np.argmax(p_cw_test, axis=1))
    cw_val_acc = accuracy_score(y_val, np.argmax(p_cw_val, axis=1))

    # 7. Method C: Logistic Regression Stacker
    stacker = train_logistic_stacker(p_ndc_val, p_xgb_val, y_val)
    X_meta_val = np.hstack([p_ndc_val, p_xgb_val])
    X_meta_test = np.hstack([p_ndc_test, p_xgb_test])
    p_stack_val = stacker.predict_proba(X_meta_val)
    p_stack_test = stacker.predict_proba(X_meta_test)
    stack_val_loss, stack_val_acc = log_loss(y_val, p_stack_val), accuracy_score(y_val, np.argmax(p_stack_val, axis=1))
    stack_test_loss, stack_test_acc = log_loss(y_test, p_stack_test), accuracy_score(y_test, np.argmax(p_stack_test, axis=1))

    # 8. Display Results
    print("\n" + "-" * 75)
    print(f"{'Model / Strategy':<32} {'Val Log-Loss':<14} {'Val Acc':<10} {'Test Log-Loss':<14} {'Test Acc':<10}")
    print("-" * 75)
    print(f"{'Neural Dixon-Coles (NDC)':<32} {ndc_val_loss:<14.4f} {ndc_val_acc:<10.1%} {ndc_test_loss:<14.4f} {ndc_test_acc:<10.1%}")
    print(f"{'Calibrated XGBoost':<32} {xgb_val_loss:<14.4f} {xgb_val_acc:<10.1%} {xgb_test_loss:<14.4f} {xgb_test_acc:<10.1%}")
    print(f"{'Baseline 50/50 Blend':<32} {b50_val_loss:<14.4f} {b50_val_acc:<10.1%} {b50_test_loss:<14.4f} {b50_test_acc:<10.1%}")
    print(f"{f'Optimal Scalar Blend (α={opt_alpha:.3f})':<32} {opt_scalar_val_loss:<14.4f} {scalar_val_acc:<10.1%} {scalar_test_loss:<14.4f} {scalar_test_acc:<10.1%}")
    print(f"{'Class-Wise Nelder-Mead Blend':<32} {opt_cw_val_loss:<14.4f} {cw_val_acc:<10.1%} {cw_test_loss:<14.4f} {cw_test_acc:<10.1%}")
    print(f"{'Logistic Regression Stacking':<32} {stack_val_loss:<14.4f} {stack_val_acc:<10.1%} {stack_test_loss:<14.4f} {stack_test_acc:<10.1%}")
    print("-" * 75)

    print(f"\n[Learned Class-Wise Weights (NDC %)]: Away={opt_alphas[0]:.1%}, Draw={opt_alphas[1]:.1%}, Home={opt_alphas[2]:.1%}")
    print(f"[Learned Class-Wise Weights (XGB %)]: Away={1-opt_alphas[0]:.1%}, Draw={1-opt_alphas[1]:.1%}, Home={1-opt_alphas[2]:.1%}")

    # Determine recommended configuration
    # Check if class-wise or scalar is best
    best_config = {
        "validation_season": "2024/25",
        "test_season": "2025/26",
        "optimal_scalar_alpha": round(opt_alpha, 4),
        "scalar_blend": {
            "ndc_weight": round(opt_alpha, 4),
            "xgb_weight": round(1.0 - opt_alpha, 4),
            "val_loss": round(opt_scalar_val_loss, 4),
            "val_acc": round(scalar_val_acc, 4),
            "test_loss": round(scalar_test_loss, 4),
            "test_acc": round(scalar_test_acc, 4),
        },
        "classwise_blend": {
            "away_ndc_weight": round(float(opt_alphas[0]), 4),
            "draw_ndc_weight": round(float(opt_alphas[1]), 4),
            "home_ndc_weight": round(float(opt_alphas[2]), 4),
            "away_xgb_weight": round(float(1.0 - opt_alphas[0]), 4),
            "draw_xgb_weight": round(float(1.0 - opt_alphas[1]), 4),
            "home_xgb_weight": round(float(1.0 - opt_alphas[2]), 4),
            "val_loss": round(opt_cw_val_loss, 4),
            "val_acc": round(cw_val_acc, 4),
            "test_loss": round(cw_test_loss, 4),
            "test_acc": round(cw_test_acc, 4),
        },
        "logistic_stacking": {
            "val_loss": round(stack_val_loss, 4),
            "val_acc": round(stack_val_acc, 4),
            "test_loss": round(stack_test_loss, 4),
            "test_acc": round(stack_test_acc, 4),
            "classes": [int(c) for c in stacker.classes_],
            "coefficients": np.round(stacker.coef_, 4).tolist(),
            "intercepts": np.round(stacker.intercept_, 4).tolist(),
        },
        "baseline_50_50": {
            "val_loss": round(b50_val_loss, 4),
            "val_acc": round(b50_val_acc, 4),
            "test_loss": round(b50_test_loss, 4),
            "test_acc": round(b50_test_acc, 4),
        },
    }

    if args.save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\n✅  Saved optimal blend weights to {out_path}")

        stacker_path = out_path.parent / "blend_stacker.joblib"
        joblib.dump(stacker, stacker_path)
        print(f"✅  Saved Logistic Regression stacker model to {stacker_path}")

    return best_config


if __name__ == "__main__":
    main()
