"""
Train Neural Dixon-Coles Model across 10 Seasons
=================================================
Reads `data/processed/epl_model_features.csv`, performs strict 10-season
temporal splits (Train: 2016/17–2023/24, Val: 2024/25, Test: 2025/26),
fits StandardScaler on the training set ONLY, trains NeuralDixonColes
PyTorch model with early stopping on Validation NLL, and saves artifacts to `models/bin/`.

Usage:
    python scripts/train_neural_dixon_coles.py

Outputs:
    models/bin/neural_dixon_coles.pt      – PyTorch state_dict & metrics
    models/bin/ndc_scaler.joblib          – Fitted StandardScaler
    models/bin/ndc_feature_names.json     – List of input feature column names
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Make the repo root importable so we can import models.neural_dixon_coles
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.neural_dixon_coles import DixonColesLoss, NeuralDixonColes  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH     = os.path.join(REPO_ROOT, "data", "processed", "epl_model_features.csv")
ARTIFACT_DIR  = os.path.join(REPO_ROOT, "models", "bin")

LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
PATIENCE      = 10
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EXCLUDE_COLS = {
    "Date", "Season", "HomeTeam", "AwayTeam", "Referee",
    "FTHG", "FTAG",
}

TRAIN_SEASONS = {
    "2016/17", "2017/18", "2018/19", "2019/20",
    "2020/21", "2021/22", "2022/23", "2324", "2023/24",
}
VAL_SEASONS = {"2024/25", "2425"}
TEST_SEASONS = {"2025/26", "2526"}


def load_and_split_10_seasons(path: str):
    """
    Loads dataset and splits strictly by season:
      - Train: 2016/17 through 2023/24 (~3,040 matches)
      - Val  : 2024/25 (~380 matches)
      - Test : 2025/26 (~380 matches)
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    train_mask = df["Season"].astype(str).isin(TRAIN_SEASONS)
    val_mask   = df["Season"].astype(str).isin(VAL_SEASONS)
    test_mask  = df["Season"].astype(str).isin(TEST_SEASONS)

    # Fallback to ratio split if season column values differ
    if not train_mask.any():
        print("  [Warning] Season labels did not match expected set, using 80/10/10 ratio split.")
        n = len(df)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        train_mask = pd.Series(np.arange(n) < n_train, index=df.index)
        val_mask   = pd.Series((np.arange(n) >= n_train) & (np.arange(n) < n_train + n_val), index=df.index)
        test_mask  = pd.Series(np.arange(n) >= n_train + n_val, index=df.index)

    def extract_split(mask):
        sub_df = df[mask]
        X = sub_df[feature_cols].values.astype(np.float32)
        yh = sub_df["FTHG"].values.astype(np.float32)
        ya = sub_df["FTAG"].values.astype(np.float32)
        return X, yh, ya

    X_train, yh_train, ya_train = extract_split(train_mask)
    X_val,   yh_val,   ya_val   = extract_split(val_mask)
    X_test,  yh_test,  ya_test  = extract_split(test_mask)

    return (
        X_train, yh_train, ya_train,
        X_val,   yh_val,   ya_val,
        X_test,  yh_test,  ya_test,
        feature_cols,
    )


def make_dataloader(X, yh, ya, batch_size, shuffle=False):
    tensors = (
        torch.from_numpy(X),
        torch.from_numpy(yh),
        torch.from_numpy(ya),
    )
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for X_b, yh_b, ya_b in loader:
        X_b, yh_b, ya_b = X_b.to(device), yh_b.to(device), ya_b.to(device)

        optimizer.zero_grad()
        lam, mu, rho = model(X_b)
        loss = criterion(lam, mu, rho, yh_b, ya_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for X_b, yh_b, ya_b in loader:
        X_b, yh_b, ya_b = X_b.to(device), yh_b.to(device), ya_b.to(device)

        lam, mu, rho = model(X_b)
        loss = criterion(lam, mu, rho, yh_b, ya_b)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"[NDC Training] Using device: {device}")

    # 1. Load data and split by 10 seasons
    print(f"[NDC Training] Loading data from {DATA_PATH} ...")
    (
        X_train, yh_train, ya_train,
        X_val,   yh_val,   ya_val,
        X_test,  yh_test,  ya_test,
        feature_cols,
    ) = load_and_split_10_seasons(DATA_PATH)

    print(f"             → Features ({len(feature_cols)}): {feature_cols[:5]} ...")
    print(f"             → Train set: {len(X_train)} matches (Seasons 2016/17–2023/24)")
    print(f"             → Val set  : {len(X_val)} matches (Season 2024/25)")
    print(f"             → Test set : {len(X_test)} matches (Season 2025/26)")

    # 2. Impute and Scale (Fit ONLY on Train set)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_val   = imputer.transform(X_val).astype(np.float32)
    X_test  = imputer.transform(X_test).astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # 3. Build DataLoaders
    train_loader = make_dataloader(X_train, yh_train, ya_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_dataloader(X_val,   yh_val,   ya_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_dataloader(X_test,  yh_test,  ya_test,  BATCH_SIZE, shuffle=False)

    # 4. Initialise Model, Loss, Optimizer
    n_features = X_train.shape[1]
    model      = NeuralDixonColes(n_features=n_features).to(device)
    criterion  = DixonColesLoss(eps=1e-7)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Training loop with early stopping
    print("\n[NDC Training] Starting training ...")
    print(f"{'Epoch':>6}  {'Train NLL':>10}  {'Val NLL':>10}  {'Best':>5}  {'LR':>8}")
    print("-" * 50)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " ✓" if is_best else ""
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
            f"{'yes' if is_best else 'no':>5}  {current_lr:.2e}{marker}"
        )

        if patience_count >= PATIENCE:
            print(f"\n[NDC Training] Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

    # Restore best checkpoint & evaluate on Test set
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[NDC Training] Final Evaluation:")
    print(f"  → Best Val NLL  (2024/25): {best_val_loss:.4f}")
    print(f"  → Holdout Test NLL (2025/26): {test_loss:.4f}")

    # 6. Save Artifacts
    model_path    = os.path.join(ARTIFACT_DIR, "neural_dixon_coles.pt")
    scaler_path   = os.path.join(ARTIFACT_DIR, "ndc_scaler.joblib")
    features_path = os.path.join(ARTIFACT_DIR, "ndc_feature_names.json")

    torch.save(
        {
            "state_dict":   model.state_dict(),
            "n_features":   n_features,
            "best_val_nll": best_val_loss,
            "test_nll":     test_loss,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\n[NDC Training] ✅ Artifacts saved successfully to {ARTIFACT_DIR}:")
    print(f"  → Model  : {model_path}")
    print(f"  → Scaler : {scaler_path}")
    print(f"  → Features: {features_path}")

    # Sanity check prediction
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(X_test[:1]).to(device)
        lam, mu, rho = model(sample)
        print(f"\n[NDC Training] Sanity check — Sample Test Prediction:")
        print(f"  λ (home xG)  = {lam.item():.4f}")
        print(f"  μ (away xG)  = {mu.item():.4f}")
        print(f"  ρ (draw bias)= {rho.item():.4f}")

    print("\n[NDC Training] 🏁 Done.")


if __name__ == "__main__":
    main()
