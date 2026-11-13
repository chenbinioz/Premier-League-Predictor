Project audit — 4 November 2026

This review covers the local checkout at `3b333c8`, the SQLite database, all training and inference scripts, both notebooks, saved artifacts, the Streamlit display, the public football-data results feed, and the latest GitHub Actions job. Application code, model artifacts, forecasts, and results have not been repaired or regenerated. `scripts/audit_project.py` is a new, read-only reproduction tool.

**1. Findings at a glance**

| Concern | Finding | Priority |
|---|---|---|
| Live Elo display | Reads the historical CSV, ending 31 August, rather than live state | High |
| Live Elo calculation | Updates correctly for the eight GW9 results already ingested; all ratings exactly reproduce from the preceding snapshot | Working for that interval |
| Missing GW9 results | Two results are available upstream but remain pending locally and remotely | High |
| Collection reliability | Understat parsing fails; CSV dates and gameweek assignment are broken; job succeeds after zero updates | Critical |
| Neural model evaluation | Training inputs contain the target match's goals and other post-match information | Critical |
| Live XGBoost inputs | Trained with genuine bookmaker probabilities; live inference supplies NDC outputs instead | Critical |
| Scoreline display grading | Correct stored top-three scores are all graded as misses on fixture cards | High |
| Historical prediction integrity | GW1–2 locks are after their match dates; 15 scoreline lists disagree with their stored NDC parameters | High |
| Ensemble labeling | All 90 stored forecasts differ from today's optimized blend; GW3–9 are exactly 50/50 | Medium |
| Comparison with 69% claim | Source could not be identified; task, baseline, and validation method are unspecified | Unverified |

**2. Elo: the display is stale, while the updater does change ratings**

`app.py:1329` calls `compute_live_elos(df_features)`. `load_xgb_assets()` loads `data/processed/epl_model_features.csv`; that CSV contains 3,820 matches and ends on **31 August 2026**. Daily live updates change SQLite, not this CSV. Clearing the cache will therefore not repair the rating display.

Examples from the current checkout:

| Team | Elo tab calculation | SQLite current Elo |
|---|---:|---:|
| Arsenal | 1664.62 | 1672.30 |
| Man City | 1646.50 | 1653.97 |

To test the updater independently, I extracted the database from commit `558864e` (26 October), replayed the eight completed GW9 results in date order using K=20 and home advantage=60, and compared every team's Elo with today's database. Maximum error: **0.0000**. For example, Man City fell from **1668.6796 to 1653.9689** after the Brighton defeat.

Liverpool, Arsenal, Newcastle, and Everton have not received their GW9 updates because those two fixtures remain pending. The other 16 teams have state timestamps on 1 November. This is both a display-source problem and an incomplete-ingestion problem, rather than a broken Elo formula.

Additional state risks:

- `init_live_db._load_final_elo_dict()` chooses the latest season in the CSV, which now includes 2026/27, even though initialization describes a 2025/26 baseline. It also reads pre-match Elo from the last fixture, omitting that fixture's update. Its statistics loader explicitly excludes 2026/27, so Elo and form can initialize from different periods. This is a defect in initialization; it does not prove how the current database was originally initialized.
- Seeding completed fixtures directly in `seed_2627_fixtures.py` does not replay their results into team states. The updater subsequently only examines pending fixtures.
- Updates are not atomic or independently idempotent. In a temporary database copy, applying the Liverpool–Arsenal result twice changed Liverpool **1588.6086 → 1579.2893 → 1570.5016**. Injecting a failure before Everton's state update left Newcastle's changed Elo committed while the fixture remained pending. The normal caller's pending filter helps with routine reruns, but cannot recover this partial failure safely.

**3. GW9: the current scheduled job fails silently**

The live database contains **380 fixtures, 88 completed results, and 90 locked forecasts**. GW9 has eight completed results and two pending fixtures:

| Fixture | Local date | Local status | Public CSV result |
|---|---|---|---|
| Liverpool–Arsenal | 1 November | Pending | 0–1 |
| Newcastle–Everton | 2 November | Pending | 0–1 |

The CSV was fetched directly from [football-data's 2026/27 EPL feed](https://www.football-data.co.uk/mmz4281/2627/E0.csv). These scores were verified, not inferred from the passage of the fixture dates. There is also a source date disagreement: Aston Villa–Fulham is 1 November in SQLite and 31 October in the fetched CSV. A canonical fixture record needs to reconcile dates independently of scores.

The remote branch's latest commit is also `3b333c8`; merely pulling GitHub will not add the missing scores. Scheduled runs on 2–4 November report success. The [4 November workflow run](https://github.com/chenbinioz/Premier-League-Predictor/actions/runs/41825240174) contains:

```text
Discovered gameweeks needing updates: [9]
Understat fetch failed for season 2026/EPL: Expecting value: line 1 column 1 (char 0)
GW9 processing complete. Updated 0 fixtures.
Auto mode: Selected Gameweek 9 for prediction.
Fetched 0 fixtures from Understat for GW9.
```

The failure chain is reproducible:

1. Understat returns a page the parser cannot consume. A separate direct request during this audit returned an HTML error page with HTTP 200 and no match-data payload.
2. `fetch_weekend_results()` falls back to football-data CSV.
3. `_normalise_date()` does not parse `DD/MM/YYYY` strings. It returns them unchanged through `raw[:10]`.
4. The fallback sorts these strings and assigns **one gameweek per distinct date**. Even with correct date parsing, Friday, Saturday, Sunday, and Monday are not separate gameweeks.
5. On the fetched feed, asking the fallback for GW9 returns **Coventry–Newcastle on 12 October**, not either pending GW9 fixture.
6. The updater cannot match that result against Liverpool–Arsenal or Newcastle–Everton, writes nothing, and exits successfully.
7. Auto prediction chooses the minimum pending gameweek, so it remains on GW9. Prediction fetching has no database-fixture fallback when Understat is unavailable. No GW10 predictions have been locked.

Understat's own gameweek assignment is also heuristic: it sorts fixtures by date and puts every ten into a round. Postponements and rescheduling can move fixtures between these artificial rounds. Results should reconcile against canonical fixture identity, not a source-specific derived gameweek.

Further collection inconsistencies:

- **All 88 completed fixtures have NULL home/away shots on target, corners, and fouls.** There are **528 missing values** across these six columns. The state manager appends previous averages when these values are unavailable, which can make synthetic form look like observed form. The public CSV has these fields now; completed fixtures are never revisited for enrichment.
- A partially successful Understat response causes an immediate return, so CSV results are not merged in to fill missing matches in that round.
- The updater ignores the result object's xG and fetches the whole Understat season again per match; if unavailable, it substitutes goals. The fixture table does not retain xG or its provenance, preventing faithful state replay later.
- Missing CSV goals are converted to zero and the row is marked completed. Missing/invalid observations should not become real 0–0 results.
- The workflow has no check that overdue fixtures were reconciled and no distinct degraded/failed ingestion status. Missing data can look like a healthy system.

**4. Do stored probabilities and scorelines match the models?**

Some parts do, and some do not:

- **NDC probabilities: consistent.** For all 90 stored forecasts, rebuilding the grid from saved lambda, mu, and rho reproduces stored NDC probabilities exactly at the code's displayed precision. Home/away class ordering and lower/upper grid triangles are correct.
- **GW3–9 scorelines: consistent.** All 70 stored scoreline lists match the reconstructed top three. In GW1–2, **15 of 20 lists disagree** with their stored NDC parameters. Example: match 1 stores `1 - 1 • 0 - 0 • 2 - 1`, while its saved parameters imply `1 - 0 • 0 - 0 • 2 - 0`.
- **Fixture-card score grading: wrong.** `app.py:719` places the entire bullet-separated score string into a one-element list. `app.py:790` then tests exact membership of an individual actual score. Across 88 completed games, parsing the stored lists yields **18 top-three hits**; the fixture-card logic yields **zero**. The gameweek overview uses a parser, so the card and overview can disagree.
- **Stored blend versus current configuration: different generations.** All 90 locked blends differ from the current classwise weights. All 70 GW3–9 locks are exactly the normalized 50/50 mean of their NDC/XGB probabilities. These are legitimate historical forecasts if generated that way, but calling them “Optimized” just because a weights file exists misrepresents their provenance. Preserve old locks and show their actual version.
- **Ad hoc forecast versus locked fixture: different inputs.** The ad hoc page uses current state and default seven-day rest, while the batch predictor computes rest from fixture dates. It also presents XGBoost headline probabilities with NDC scorelines. Missing per-model locks can fall back to a new snapshot calculated after the match. A user should see whether a number is a locked pre-match forecast or a current simulation.

A most likely exact score need not have the same outcome as the most likely 1X2 category: a category sums many scorelines. That difference alone is mathematically valid. However, XGBoost probabilities and NDC scorelines are separate distributions and should be labeled accordingly. XGBoost has no exact-score distribution; the fallback that rounds win probabilities into goals is not a trained scoreline model.

**5. Measured performance and its limits**

I recomputed metrics from locked records against the stored completed results. Bookmaker probabilities are reciprocal Bet365 prices normalized to remove the overround, joined from the fetched CSV. All models use the same 88 matches in this table; ties follow the UI's Home → Draw → Away priority. One bookmaker tie changes its hit count by one under a different rule.

| Model | Correct / 88 | Accuracy | 1X2 log loss, lower is better |
|---|---:|---:|---:|
| Bet365 favourite | 42 | 47.73% | 1.0187 |
| XGBoost | 39 | 44.32% | 1.0542 |
| Neural Dixon–Coles | 36 | 40.91% | 1.0816 |
| Stored blend | 36 | 40.91% | 1.0883 |
| Always predict home | 36 | 40.91% | — |

**GW1 and GW2 must not be advertised as prospective predictions:** all 20 were locked on 4 September, after their August match dates. The timestamp proves they were recorded late, without proving exactly how the original values were computed. The system cannot certify their pre-match provenance. Excluding those rounds:

| Model | Correct / 68 | Accuracy | Log loss |
|---|---:|---:|---:|
| Bet365 favourite | 31 | 45.59% | 1.0314 |
| XGBoost | 27 | 39.71% | 1.0921 |
| Neural Dixon–Coles | 24 | 35.29% | 1.1148 |
| Stored blend | 26 | 38.24% | 1.0988 |

These are small samples; 88 games and a three-hit gap do not establish a statistically reliable long-run difference between XGBoost and the market. GW3 includes one lock on the match date, and the database stores no kickoff times, so strict before-kickoff validation is unavailable for it. The app also permits model/bookmaker coverage to differ; comparisons should always report the common eligible cohort.

**6. The largest modeling defect: target leakage in NDC training**

`train_neural_dixon_coles.py:80` selects every numeric column except a short exclusion list. Although `FTHG` and `FTAG` are excluded, the saved 75-feature list contains exact copies of them:

```text
Home_GoalsScored   == FTHG
Away_GoalsScored   == FTAG
Home_GoalsConceded == FTAG
Away_GoalsConceded == FTHG
```

These equalities hold for all **3,820 rows**. The model also receives the target match's shots, cards, corners, fouls, and xG proxy. They are unavailable before kickoff. The live builder's comment calls them “last-match single stats,” but the feature-building code puts the **current match's** stats there.

At inference, the builder replaces these with rolling averages and card constants. This changes the meaning of the features the model learned. Replacing leaked values at inference does not repair the trained model; it must be retrained using a pre-match-only feature contract.

An illustrative diagnostic with the saved NDC artifact on 2025/26:

| Input construction | Accuracy |
|---|---:|
| Original historical inputs, including target-match statistics — INVALID forecast test | 68.42% |
| Rolling proxies, with genuine historical bookmaker inputs | 39.21% |
| Rolling proxies, with live-style neutral bookmaker inputs | 40.00% |

This is not an accusation about the external 69% claim. It demonstrates why a seemingly strong percentage cannot be accepted without checking what the model knew before the game. The saved neural test NLL of 1.805 is a **scoreline likelihood metric on leaked inputs**, not 1X2 accuracy or directly comparable to the table's log loss.

**7. XGBoost's live market features are not what it trained on**

The XGBoost artifact uses **18 features** and genuine normalized Bet365 probabilities in historical training. Live inference first runs NDC with fixed market priors **40% / 27% / 33%**, then places NDC's output in XGBoost's three bookmaker-feature slots. Adding and renormalizing a 5% margin in `_ndc_implied_odds()` just cancels out; it does not create market information.

The market columns account for approximately **42.6% of the averaged tree gain importance** in the saved classifier. Gain is a diagnostic, not a causal attribution. More directly, keeping every other historical feature fixed and replacing only the market columns gives:

| Saved XGBoost, 2025/26 | Accuracy | Log loss |
|---|---:|---:|
| Real bookmaker inputs | 47.11% | 1.0252 |
| NDC probabilities substituted into market columns | 41.05% | 1.1001 |
| Bookmaker baseline | 48.95% | 1.0185 |

The NDC proxy for this isolated diagnostic uses the existing blend evaluator's rolling-proxy construction with historical market inputs; it is not a complete simulation of the live pipeline. It still demonstrates a large sensitivity to the market substitution. These historical artifact measurements are retrospective diagnostics, not a fresh, independently certified out-of-sample experiment.

The blend is consequently combining dependent predictions, because XGBoost already consumes the NDC prediction. The blend optimizer evaluates XGBoost with real bookmaker inputs, so its validation environment differs from deployment too.

**8. Other reasons to distrust or improve the evaluation**

- Historical “xG” is a hand-built shot proxy: `0.32 * shots_on_target + 0.03 * other_shots`. The processed home column equals this formula to floating-point precision. The raw dataset contains `HxG`/`AxG` for 20 current-season games, but the builder checks different names and ignores them. Live ingestion can use Understat xG or goals, creating three incompatible feature definitions under one name.
- Historical rolling means apply season decay, while live windows use unweighted arrays. Venue-specific historical form becomes overall form live. Missing shots/corners/fouls become self-replenishing average values. Training and serving need the same transformations.
- `calculate_decay_weighted_rolling_mean()` uses the current observation for a team's first appearance instead of an independent prior. This leaks that first match into rolling and venue features.
- XGBoost uses integer `cv=3`, which means stratified folds rather than forward-in-time calibration. Future training-period rows can calibrate past rows. This does not by itself prove leakage from the outer test set. See [scikit-learn's calibration contract](https://scikit-learn.org/1.6/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
- With today's 3,820-row CSV, the 80% XGBoost training boundary is **24 August 2024**, overlapping the blend optimizer's nominal 2024/25 validation season. The notebook's saved output uses a 760-game test, whereas today's script yields 764. There is no artifact manifest establishing the exact training dataset for the pickle. Do not infer its provenance solely from current code.
- `ffill().bfill()` runs before XGBoost's split and crosses team/match boundaries. Future-backed filling is unsafe if missing values occur. NDC's fitted imputer is not saved for inference.
- Draws are weighted 1.35x in both classifier fitting and calibration. That changes calibration's target distribution. Test weighted training with unweighted later calibration against an entirely unweighted model; optimizing draw recall does not necessarily improve natural-frequency probabilities.
- The saved `blend_weights.json` is not a reliable reproducibility record: the current saved artifacts do not reproduce its underlying model results. No model/data hash or per-forecast version is stored. A stacker file is saved, but live inference selects the classwise blend, not that stacker.
- Static “Dixon–Coles” fitting in the notebook is actually independent Poisson attack/defense fitting without a learned rho correction, and fits the full dataframe. Historical evaluation of that same fitted artifact includes matches used to estimate its strengths.
- Persisted scikit-learn objects were created with **1.6.1**; local inference uses **1.9.0**, and the reviewed CI job uses **1.7.2**. Both warn. Pin and reproduce the environment; [cross-version pickle loading is unsupported](https://scikit-learn.org/stable/model_persistence.html).

**9. How this compares with the claimed 69% model**

Your dataset has **3,820 matches**, including ten full historical seasons and 20 current-season matches. XGBoost uses 18 engineered inputs; NDC uses 75, although many are correlated and several are invalid pre-match inputs. You already exceed the quoted 2,000-match sample size. Adding enough columns to reach 50 is not the main missing ingredient.

I could not identify the exact source from the quoted wording. Its 69% cannot be compared without knowing:

- Whether it predicts home/draw/away, a binary event, or only selected fixtures.
- Whether every test feature was available before kickoff.
- Whether testing was chronological, across entire untouched seasons, or a random split.
- How many test games there were, and whether test outcomes influenced feature/hyperparameter selection.
- Whether bookmaker inputs were included, their observation times, and their performance on the identical games.
- What “+15%” means: relative improvement or percentage points. A 69% result that is 15% relatively above baseline implies a **60% baseline**; a 15-point increase implies **54%**.

Your current measured results do not substantiate comparable 69% pre-match performance. The first objective should be a trustworthy, reproducible baseline, then consistent improvement against the market on held-out seasons. Accuracy, probability quality, and financial return are different objectives; a model can improve one and worsen another.

**10. Recommended implementation order**

1. **Repair ingestion and state integrity.** Reconcile completed results by canonical home/away fixture identity within season, persist authoritative dates and external IDs, parse dates explicitly, merge sources per fixture, and enrich completed fixtures when stats arrive. Store xG, observation timestamps, source, and missingness. Use atomic state/result transactions and an applied-result ledger. Replay results chronologically from a known seed to rebuild and verify state. Fail or report degraded freshness when overdue results remain unexplained. Let prediction use database fixtures if schedule fetching fails.
2. **Fix the display and ledger.** Read live Elo from SQLite, parse top-three scores correctly, identify simulated versus locked predictions, and version forecasts. Preserve the original ledger; quarantine or explicitly flag late GW1–2 records and inconsistent scoreline lists. A replacement forecast cannot be retroactively labeled pre-match. Store kickoff timestamps and prevent post-kickoff locks or overwriting original forecasts.
3. **Create one pre-match feature builder.** Use an explicit feature allowlist. Replay historical matches through the same state transitions used live, generating each feature vector before applying that match's result. Test that changing a match's result cannot change its own inputs. Make venue, rest, season decay, xG definitions, imputation, and missingness identical in training and serving.
4. **Retrain separate market-free and market-aware models.** Market-free XGBoost/Poisson models must omit bookmaker features. Market-aware models must receive real, timestamped prices in production. A sensible experiment is a small correction to market log odds, compared against the unchanged market baseline. Keep a simple Elo/logistic and time-decayed Poisson baseline before increasing model complexity.
5. **Evaluate forward in time.** Use multiple expanding seasonal folds, with later disjoint calibration and ensemble-tuning periods, followed by a completely untouched final test season. Group same-day fixtures in splitting where appropriate. Refitting after a published test is fine, but version it as a new artifact. Fit imputers/scalers on training only and save all preprocessing and environment metadata.
6. **Optimize probability quality and report uncertainty.** Compare log loss, multiclass Brier score, calibration curves, accuracy, draw recall, and top-one/top-three exact-score hits separately. Use common match cohorts and matchday/week-level paired resampling for model-versus-market differences. Report coverage and confidence intervals. Do not force draw selections just to balance class counts.
7. **Add information after the baseline is sound.** Candidate experiments: consistent event-derived xG, opponent-adjusted attack/defense form, actual venue form, squad/player availability, expected lineups, transfers/promoted-team priors, and rest/travel across all competitions. Require known-at-prediction timestamps and show out-of-time ablation gains. Tune tree depth, learning rate, regularization, decay, and calibration before trying a deeper neural network. None guarantees a particular accuracy.

**11. Reproduction and review limits**

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/audit_project.py \
  --results-csv /tmp/epl_audit_upstream.csv \
  --output /tmp/epl_project_audit.json
```

`--results-csv` is optional and accepts an independently downloaded current-season football-data E0 CSV. Without it, the script makes no network requests and audits local data/artifacts. The `/tmp` feed is the audit-time snapshot and is not a permanent project dependency. Historical model inference emits the version warnings described above. The script reads the live database with SQLite read-only mode; it writes only the explicitly requested report file.

Additional checks performed outside the script: replay of GW9 Elo from the historical Git snapshot; duplicate-result and injected-failure experiments on a temporary database; inspection of upstream CSV and Understat responses; inspection of current GitHub workflow logs; and execution of the existing UI scoring helpers without starting Streamlit. SQLite integrity check returned `ok`.

This audit identifies current defects and supplies repair priorities. It does not replace the live database, rerun training, change locked predictions, or assert a new production model's performance.
