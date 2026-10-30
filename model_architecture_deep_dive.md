# Premier League Predictor — Model Architecture Deep Dive

## Overview

The system runs **three models in parallel** plus a **Bookmaker benchmark**, blending their outputs for the final prediction:

| Model | Type | Output |
|---|---|---|
| **Neural Dixon-Coles (NDC)** | PyTorch neural network | λ (home xG), μ (away xG), ρ (low-score bias) → scoreline grid → 1X2 probs |
| **XGBoost Calibrated Classifier** | Gradient boosted trees + Platt scaling | Direct 1X2 win/draw/lose probabilities |
| **Static Dixon-Coles (MLE)** | Classical max-likelihood estimation | Baseline 1X2 probs + scoreline grid |
| **Blended Ensemble** | Weighted average (50% NDC + 50% XGB) | Final reported probability |

---

## 1. Data Pipeline

### Training Data
- **Source**: `football-data.co.uk` CSVs covering **10 EPL seasons** (2016/17 – 2025/26)
- **Raw file**: `data/raw/epl_10_seasons_raw.csv`
- **Key raw columns used**: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `Home_xG`, `Away_xG`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `B365H`, `B365D`, `B365A`

> [!NOTE]
> If `Home_xG` / `Away_xG` are missing in the raw data, they are **estimated from shots**: `xG ≈ 0.32 × shots_on_target + 0.03 × (shots - shots_on_target)`. This is a rough approximation.

### Live 2026/27 Data
- **Source**: Understat API (fetched daily via GitHub Actions) → `data/live/epl_2627.db` (SQLite)
- **Tables**: `fixtures_26_27`, `predictions_26_27`, `live_team_states`
- **Fallback**: `football-data.co.uk` CSV for current season (for bookmaker odds and match results when Understat is unavailable)

---

## 2. Feature Engineering ([`build_features.py`](file:///Users/chenbozhang/premier-league-predictor/src/features/build_features.py))

Run once offline to produce `data/processed/epl_model_features.csv`. Generates ~75 features per match.

### 2a. Elo Ratings
- **Algorithm**: Standard Elo with `K=20`, `home_advantage=60`
- **Inter-season decay**: At every season boundary, existing team Elos regress toward mean: `Elo_new = 0.80 × Elo_old + 0.20 × 1500`. Newly promoted teams start at `1420`.
- **Output features**: `Home_Elo`, `Away_Elo`

### 2b. Multi-Scale Decayed Rolling Form
Rolling windows over **3, 5, and 10** preceding matches, applied to:

| Statistic | Used For |
|---|---|
| Goals Scored / Conceded | General form |
| xG Created / xG Conceded | Attack/defence quality |
| Shots on Target | Chance creation |
| Corners | Territorial pressure |
| Fouls | Physical play / disruption |

**Off-season decay**: Prior-season matches receive weight `α=0.5` per season boundary crossed (so previous season counts half as much as the current season). This prevents stale pre-season form contaminating early-season predictions.

### 2c. Venue-Specific xG Form
- Home teams' rolling `xG_Created_Venue_roll5` and `xG_Conceded_Venue_roll5` (computed only over home matches)
- Away teams' same metrics computed only over away matches

### 2d. Schedule Context
- `Rest_Days`: Days since last match per team
- `Congestion_Flag`: Binary flag if rest < 4 days

### 2e. Bookmaker Prior (B365 odds)
- `B365H`, `B365D`, `B365A` are included as raw features (and normalized to implied probabilities `Bookie_Prob_H/D/A` for XGBoost)

---

## 3. Neural Dixon-Coles (NDC) Model

### Architecture ([`neural_dixon_coles.py`](file:///Users/chenbozhang/premier-league-predictor/models/neural_dixon_coles.py))

```
Input (n_features ≈ 75)
    ↓
Linear(n → 64) → BatchNorm1d → SiLU → Dropout(0.2)
    ↓
Linear(64 → 32) → BatchNorm1d → SiLU
    ↓ (three output heads)
head_lambda: Linear(32→1) → Softplus  →  λ > 0  (Home expected goals)
head_mu:     Linear(32→1) → Softplus  →  μ > 0  (Away expected goals)
head_rho:    Linear(32→1) → Tanh×0.25 →  ρ ∈ (-0.25, 0.25)  (Low-score dependency)
```

### Training ([`train_neural_dixon_coles.py`](file:///Users/chenbozhang/premier-league-predictor/scripts/train_neural_dixon_coles.py))

| Setting | Value |
|---|---|
| **Train Seasons** | 2016/17 – 2023/24 (~3,040 matches) |
| **Validation Season** | 2024/25 (~380 matches) |
| **Test Season** | 2025/26 (~380 matches) |
| **Loss** | Dixon-Coles NLL (Poisson PMF + τ low-score correction) |
| **Optimizer** | AdamW, `lr=1e-3`, `weight_decay=1e-4` |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Batch size** | 64 |
| **Max epochs** | 50 with early stopping (patience=10 on val NLL) |
| **Input scaling** | `StandardScaler` fit on train split only |

### The Dixon-Coles Loss (τ correction)
The τ correction accounts for the well-known statistical dependency between low-score outcomes (0-0, 1-0, 0-1, 1-1) in football — they are significantly under/over-represented by a naive independent Poisson model:

```
τ(0,0) = 1 - λ·μ·ρ   (adjusts 0-0 probability)
τ(1,0) = 1 + μ·ρ     (adjusts 1-0 probability)
τ(0,1) = 1 + λ·ρ     (adjusts 0-1 probability)
τ(1,1) = 1 - ρ       (adjusts 1-1 probability)
τ(h,a) = 1           for h+a ≥ 2 (no correction needed)
```

### Inference
1. Build 75-feature vector from live team states → scale with saved `StandardScaler`
2. Forward pass: NDC network → (λ, μ, ρ)
3. `predict_scoreline_grid(λ, μ, ρ)` builds a **7×7 probability matrix** over scores 0–6 goals each
4. Aggregate: `home_win = sum of lower triangle`, `draw = diagonal`, `away_win = upper triangle`

---

## 4. XGBoost Calibrated Classifier

### Features Used (18 features)

| Feature | Description |
|---|---|
| `Elo_Diff` | Home Elo minus Away Elo |
| `Home_Elo`, `Away_Elo` | Absolute Elo ratings |
| `xG_Attack_Diff_roll3/5/10` | `Home_xG_for - Away_xG_against` at each window |
| `xG_Defense_Diff_roll3/5/10` | `Away_xG_for - Home_xG_against` at each window |
| `Corner_Diff_roll5` | Home corners minus Away corners (5-match rolling) |
| `Foul_Diff_roll5` | Home fouls minus Away fouls (5-match rolling) |
| `Venue_xG_Attack_Diff` | Home venue xG_for minus Away venue xG_against |
| `Expected_Match_xG` | `Home_xG_for + Away_xG_for` (5-match rolling) |
| `Rest_Diff` | Home rest days minus Away rest days |
| `Congestion_Diff` | Home congestion flag minus Away congestion flag |
| `Bookie_Prob_H/D/A` | Bookmaker implied probabilities (margin-normalized) |

### Model Details
- Multi-class XGBoost classifier (Home Win / Draw / Away Win)
- Wrapped with `CalibratedClassifierCV` (Platt scaling) to produce well-calibrated probabilities
- Trained on `data/processed/epl_model_features.csv`

---

## 5. Static Dixon-Coles (MLE Baseline)

- Fits classical Dixon-Coles attack/defence strength parameters per team via maximum likelihood estimation
- Uses historical data only — does not adapt to live form
- Saved in `models/dc_mle_params.pkl`
- Used as a static baseline comparison in the app (not part of the blended ensemble)

---

## 6. Live State Management ([`state_manager.py`](file:///Users/chenbozhang/premier-league-predictor/src/data/state_manager.py))

After each completed match, `TeamStateManager.update_after_match()`:
1. **Fetches** current state for both teams from `live_team_states`
2. **Updates Elo** using `K=20` and `home_advantage=60` (identical parameters to training)
3. **Slides rolling arrays** (max 10 entries) for xG and goals
4. **Recalculates averages** (`avg_xg_for`, `avg_goals_for`, etc.)
5. **Persists** back to SQLite

> [!IMPORTANT]
> The live state only tracks **xG and goals**. Shots, corners, and fouls are **imputed from pre-computed seasonal means** at inference time — a significant limitation.

---

## 7. Blended Ensemble

At inference time, for each match:
```python
# 50/50 blend
blend_H = 0.5 * ndc_H + 0.5 * xgb_H
blend_D = 0.5 * ndc_D + 0.5 * xgb_D
blend_A = 0.5 * ndc_A + 0.5 * xgb_A
```
The blend weight (`--blend` flag, default 0.5) is fixed at prediction time.

---

## 8. Known Limitations & Improvement Opportunities

### 🔴 High Priority

#### L1. Shots/Corners/Fouls Are Imputed, Not Tracked Live
The live `TeamStateManager` previously only tracked xG and goals. Shots on target, corners, and fouls were replaced with static seasonal averages for all teams.

**Status**: ✅ **Implemented**. `live_team_states` now tracks 10-match rolling arrays and averages for `sot_for`, `sot_against`, `corners_for`, `corners_against`, `fouls_for`, and `fouls_against`. Live updates fetch and enrich match events from Understat and football-data.co.uk feeds, feeding genuine territorial pressure and chance creation metrics into NDC (windows 3, 5, 10) and XGBoost (`Corner_Diff_roll5`, `Foul_Diff_roll5`).

#### L2. Rest Days and Congestion Are Zero/Imputed at Inference
`Rest_Diff` and `Congestion_Diff` are hardcoded to `0.0` in the live XGBoost feature vector because the live pipeline doesn't compute them from the fixture schedule.

**Status**: ✅ **Implemented**. Rest days and congestion flags are now dynamically computed from the `fixtures_26_27` schedule table.

#### L3. Blend Weight Is Fixed at 50/50
The 50/50 blend was previously a fixed default heuristic.

**Status**: ✅ **Implemented via `scripts/optimize_blend.py`**.
- Learned optimal blend weights using Nelder-Mead optimization and Logistic Regression on the 2024/25 validation season (380 matches) and verified out-of-sample on the 2025/26 test season (380 matches):
  - **Optimal Scalar Blend**: $\alpha_{\text{NDC}} = 0.6519$, $\alpha_{\text{XGB}} = 0.3481$ (Val Log-Loss improved from 1.0214 to 1.0184).
  - **Optimal Class-Wise Blend**:
    - **Away**: NDC = `60.7%`, XGB = `39.3%`
    - **Draw**: NDC = `100.0%`, XGB = `0.0%` (Neural Dixon-Coles with bivariate Poisson + tau low-score correction decisively outperforms tree models on draw distributions).
    - **Home**: NDC = `57.6%`, XGB = `42.4%`
    - Val Log-Loss improved to `1.0145` (from 1.0214 baseline), Val Accuracy improved to `51.6%` (from 51.1%).
  - **Logistic Regression Stacking Meta-Classifier**: Val Loss = `1.0114`, Val Acc = `52.9%`, Test Loss = `1.0544`, Test Acc = `45.5%`.
- Artifacts generated: `models/bin/blend_weights.json` and `models/bin/blend_stacker.joblib`.
- Default inference pipeline in `scripts/predict_next_gameweek.py` and `app.py` live leaderboard automatically load and execute the optimized class-wise blend weights.

#### L4. xG Estimation Is Approximate
If Understat xG is unavailable, xG is proxied via `0.32 × shots_on_target + 0.03 × (shots - shots_on_target)`. This is a simplification and loses a lot of shot quality information.

**Fix**: Use Understat's per-shot xG model (which accounts for shot type, position, etc.) consistently across all seasons.

---

### 🟡 Medium Priority

#### L5. Static Dixon-Coles Doesn't Update During Season
The MLE baseline `dc_mle_params.pkl` uses historical attack/defence strengths and never updates for the 2026/27 season. A team like Leeds (newly promoted) will have no attack/defence parameter.

**Fix**: Retrain the Static DC model at the start of each new season incorporating the previous season's data, or move to an online MLE that updates weekly.

#### L6. No Head-to-Head (H2H) Features
Historical H2H records between teams can carry predictive power (e.g. "Arsenal historically overperform against Man City at the Emirates"). Currently not used.

**Fix**: Add H2H rolling win rate, average goals over the last 5 head-to-head encounters as additional features.

#### L7. No Injury / Suspension Data
Key player absences (star striker suspended, first-choice goalkeeper injured) are completely ignored.

**Fix**: Integrate a player availability API (e.g. FantasyPremierLeague API, or scraping official Premier League team news) to create a "squad strength index" feature.

#### L8. Elo K-Factor Is Fixed
`K=20` and `home_advantage=60` were chosen as reasonable defaults. These are not optimised for EPL data.

**Fix**: Run a grid search or gradient descent to find the K and HFA combination that minimises log-loss on the historical 10-season dataset.

---

### 🟢 Lower Priority / Enhancements

#### L9. NDC Architecture Could Be Deeper / Use Attention
The current NDC backbone is a simple 2-layer MLP (`64 → 32`). A deeper network or using attention over the rolling sequence directly (instead of pre-aggregated rolling averages) might capture more complex patterns.

#### L10. No Uncertainty Quantification
The model produces point probabilities but no confidence intervals. A Bayesian approach (e.g. Monte Carlo dropout, deep ensembles) would allow the app to show "confident" vs "uncertain" predictions.

#### L11. Feature Leakage in Static DC
The MLE-fitted static Dixon-Coles parameters are computed on historical full-season data including the test matches — technically a form of future leakage when used as a live baseline.

#### L12. Class Imbalance in XGBoost
Home wins (~45%) outnumber draws (~25%) and away wins (~30%) in EPL data. The XGBoost classifier was previously unweighted, causing it to almost never predict draws under argmax (only 8 out of 764 test games).

**Status**: ✅ **Implemented via `scripts/train_xgb.py` and `notebooks/02_xgb_model.ipynb`**. Draw samples are now weighted at `1.35x` during training and probability calibration, boosting draw predictions from 8 to 70 out of 764 test games and improving log-loss from 1.0071 to 1.0061.

---

## 9. Recommended Next Improvements (Priority Order)

1. ~~**Track shots, corners, fouls in live state** (L1)~~ — ✅ **Implemented** (tracked in `live_team_states` across 3, 5, 10 match rolling windows)
2. ~~**Compute rest days from fixture schedule** (L2)~~ — ✅ **Implemented** (calculated from `fixtures_26_27` schedule)
3. ~~**Optimise blend weights** (L3)~~ — ✅ **Implemented** (learned optimal class-wise blend via Nelder-Mead & logistic stacking on 2024/25 validation season)
4. **Tune Elo K-factor and HFA** (L8) — grid search on historical data
5. **Add H2H features** (L6) — moderate effort, could meaningfully improve draw prediction
6. ~~**Add class weighting for draws in XGBoost** (L12)~~ — ✅ **Implemented** (sample weight 1.35x in `scripts/train_xgb.py`)

