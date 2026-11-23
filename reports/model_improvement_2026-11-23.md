# Premier League model improvement report

23 November 2026

Start by making **regularized Elo-only logistic regression** a required benchmark and building rolling, point-in-time evaluation. In a retrospective check, that one-feature model beat the deployed neural Dixon–Coles and XGBoost blend on log loss in both 2024/25 and 2025/26. On the latter season, Elo scored 1.0303, the blend 1.0465, and the bookmaker benchmark 1.0185. This makes the first question measurable. Which additional inputs and model structures improve on Elo at the same forecast cutoff across seasons?

In parallel, start collecting timestamped odds and source snapshots, and fix how the daily job handles incomplete feeds. Those changes support both forecast coverage and more credible evaluation. Keep SQLite for now. The local database is about 460 KiB, and the obvious maintenance costs come from redundant replay, publication, and recoverability.

## Measured performance

The saved training report uses 2,660 training matches through 2022/23, 380 in 2023/24 for XGBoost calibration and neural early stopping, 380 in 2024/25 for blend selection, and 380 in 2025/26 for testing. The deployed outcome forecast is roughly 56% neural Dixon–Coles and 44% market-free XGBoost. Exact score candidates come from the neural score distribution. The market-aware XGBoost artifact is an offline challenger. [Training procedure](../scripts/train_models.py) · [Saved evaluation](training_evaluation.json) · [Deployed manifest](../models/bin/manifest.json)

| Forecast | 2024/25 log loss | 2025/26 log loss | 2025/26 Brier | 2025/26 accuracy |
| --- | ---: | ---: | ---: | ---: |
| Bookmaker normalized implied probabilities | 0.9708 | 1.0185 | 0.6115 | 48.95% |
| Elo-only logistic regression | 0.9854 | 1.0303 | 0.6203 | 47.89% |
| Logistic regression on the 17 XGBoost inputs | 0.9973 | 1.0446 | 0.6280 | 47.11% |
| Deployed blend | 0.9986 | 1.0465 | 0.6300 | 47.63% |
| Market-aware XGBoost | 0.9947 | 1.0462 | 0.6300 | 47.63% |
| Market-free XGBoost | 1.0076 | 1.0509 | 0.6342 | 47.63% |
| Neural Dixon–Coles | 1.0045 | 1.0567 | 0.6349 | 48.16% |

Lower log loss and summed multiclass Brier are better. The Elo baseline improved test log loss by **0.01625** versus the blend and narrowed the observed gap to historical bookmaker probabilities from **0.02799** to **0.01174**. It also assigned better probabilities by Brier despite correctly picking one fewer outcome than the neural model. This is why probability quality should drive promotion decisions, with accuracy reported alongside it. Historical bookmaker prices have unverified observation times relative to the intended live cutoff, so this benchmark cannot establish a live head-to-head advantage. [Proper scoring rules](https://doi.org/10.1198/016214506000001437) · [Football forecast scoring study](https://arxiv.org/abs/1908.08980)

For the new read-only calculations, I used the checked-in processed CSV, a training-only standard scaler and multinomial logistic regression with L2 regularization and C=0.1. Both logistic rows use the original training cutoff and have no post-fit calibration. Each comparison contains all 380 matches of the stated season. As a stability check, I also refit both specifications before each evaluation season using all preceding seasons.

| Evaluation season | Prior seasons’ class frequencies | Elo only | All 17 features | Historical bookmaker |
| --- | ---: | ---: | ---: | ---: |
| 2022/23 | 1.0508 | 0.9741 | 0.9741 | 0.9662 |
| 2023/24 | 1.0535 | 0.9286 | 0.9236 | 0.9092 |
| 2024/25 | 1.0818 | 0.9856 | 0.9951 | 0.9708 |
| 2025/26 | 1.0851 | 1.0303 | 1.0452 | 1.0185 |

The richer feature set helped the linear model in 2023/24, was effectively tied in 2022/23, and hurt in the two latest seasons. That makes feature-group ablation a better next experiment than expanding the feature list wholesale. These diagnostics use seasons already examined during this project, and no uncertainty intervals were calculated for this check. Treat Elo as a challenger until prospective evidence is available.

The local SQLite snapshot has 100 completed matches and 110 locked predictions. Only 10 completed matches have an eligible forecast, 90 have quarantined predictions, and none of the 10 has a stored market observation. The historical training score and this small live cohort answer different questions. The processed CSV also contains 20 early 2026/27 rows that the current training split excludes. Before a refresh, reconcile those rows to the canonical results and verify dates, coverage, and source provenance. [Live evaluation eligibility](../src/evaluation.py) · [Forecast locking](../scripts/predict_next_gameweek.py)

## Evaluation before retraining

1. **Set the prediction horizon.** Start with one explicit early forecast, for example approximately 24 hours before each fixture rather than a single gameweek batch. Record scheduled kickoff and its revision, target horizon, actual lock time, and the availability cutoff. If lineup data becomes available, publish a separately labeled late forecast with its own horizon. Keep prior locks and compare each horizon against market quotes available at the same time. For postponements, retain the old lock, define eligibility for the rescheduled match, and report coverage.
2. **Replay what was knowable.** Use UTC timestamps for kickoff, source event time, and first successful observation. Historical replay currently uses match dates and chronological results. For a 24-hour forecast, a result from the previous evening can occur before kickoff yet still be unavailable at the model cutoff. Build as-of joins, apply only observations available by the cutoff, and version corrections. For historical feeds with unknown availability, label the assumed lag and test sensitivity. Keep true timestamped evaluation separate.
3. **Use expanding or sliding temporal folds.** Train before each fold, fit scaling and feature transformations inside training, and use strictly later validation observations for tuning, calibration, and blend selection. Compare fixed expanding versus recent-weighted windows. Keep retrospective development folds distinct from a prospective 2026/27 shadow cohort. On promotion, refit through the most recent eligible data under the selected procedure. Keep the test season out of subsequent hyperparameter selection.
4. **Score forecasts on paired fixtures.** Make 1X2 log loss primary and multiclass Brier secondary. Track per-class reliability, particularly draws, plus accuracy, coverage, and forecast lead time. For score models, evaluate joint score log loss, draw and total-goal calibration, and scoreline hit rate as a descriptive statistic. Keep full distributions or enough parameters to reproduce them. When comparing score models, account for high-score tail mass consistently.
5. **Quantify repeatability.** Save one row of predictions per fixture, horizon, fold and model. Compute paired loss differences and confidence intervals with matchweek or contiguous date blocks. Report season-level effects and slices for promoted clubs, early season, favorites, draws, and stale or missing inputs. Use matched market cohorts and publish the size and missingness of those cohorts alongside model-only results.
6. **Make promotion explicit.** Choose a minimum meaningful gain before experiments. A provisional target is 0.005 lower log loss versus both Elo and the current blend on pooled development folds, improvement in a majority of folds, and no material calibration or coverage regression. Inspect the paired interval and prospective shadow results before changing the default. If a season-sized shadow cohort is impractical, continue logging and retain the challenger until the paired evidence is informative.

## Models to test

| Priority | Experiment | Why it fits this project | How to decide |
| --- | --- | --- | --- |
| 1 | Elo logistic and regularized form logistic | Cheap, deterministic benchmarks expose whether features add information beyond team strength. | Test Elo difference, mean rating, and one home advantage term where identifiable. Add small feature groups in a fixed order and score every fold. |
| 1 | Time-weighted Dixon–Coles | The existing static Poisson fit uses a 216-day decay and rho of zero, and the saved report omits its scores. A fitted low-score correction is a direct challenger for draws and exact scores. | Score the existing baseline first. Then tune decay and regularization, use shrunk team attack and defense, and compare independent versus corrected score distributions. |
| 2 | Dynamic attack and defense | Smooth latent team strengths can react to form with fewer inputs than the neural model. Partial pooling supports teams with little recent EPL history. | Start with a simple penalized or score-driven update, fixed home advantage, and season-transition shrinkage. Test added volatility and venue effects only after the simpler model clears the baseline. |
| 2 | Feature-pruned XGBoost and calibrated blending | The fixed shallow tree model and static blend can be compared against a simpler base with a small tuning budget. | Test feature pruning, recent weighting and low-dimensional temperature scaling using out-of-fold predictions. Retain a blend only if it improves on its strongest constituent across folds. |
| 2 | Market-anchored residual model | Historical normalized odds outperform every checked model. The existing market-aware XGBoost underperforms the raw market benchmark. | Fit a strongly regularized adjustment to market log probabilities or a constrained convex blend. Log the raw market baseline and market-free fallback. Score the adjustment on timestamp-matched fixtures. |
| 3 | Neural score model and extensions | The neural model supplies scorelines. A better goal model could add value even if a simpler model wins at 1X2. | Evaluate goal-rate calibration, low-score cells, and joint likelihood. Try a regularized residual over a Poisson baseline before increasing width. Investigate bivariate counts or alternative dispersion only when residual diagnostics identify a recurring problem. |

Dixon–Coles models the low-score dependence that independent Poisson misses. Dynamic alternatives let attack and defense evolve directly. Both are feasible candidates at this match volume. If the UI is meant to show one coherent probability distribution, either publish the score model’s own 1X2 marginals or reweight the score grid within each outcome class to match the chosen 1X2 probabilities and evaluate the result. Otherwise label the source of the scoreline and outcome forecasts explicitly. [Dixon and Coles](https://doi.org/10.1111/1467-9876.00065) · [Dynamic team-strength model](https://doi.org/10.1111/1467-9884.00308) · [Current serving code](../scripts/predict_next_gameweek.py)

## Feature priorities

| Order | Inputs | Proposed change and test |
| --- | --- | --- |
| 1 | Team strength and promoted teams | Current Elo uses fixed update size 20, home boost 60 and promoted prior 1420. Validate update rate, home edge, offseason shrinkage and a promoted-team prior informed by previous tier performance where comparable. Report early-season error for promoted teams. Try goal margin or xG-based updates as separate challengers. |
| 2 | Chance creation and opponent strength | The current proxy is 0.32 per shot on target and 0.03 per other shot. Preserve it as a fallback. Evaluate timestamped provider xG and non-penalty xG for and against, opponent-adjusted over time, adopting the same definition in training and production. Evaluate both a common provider-covered cohort and all-match fallback performance. |
| 2 | Form windows and home/away splits | The current 3/5/10-match form windows overlap. Replace or supplement them with exponentially weighted attack and defense estimates and shrink thin venue samples toward overall strength. Drop corners and fouls as a group in one ablation. Keep them only if they add repeatable information beyond goals, shots, and strength. |
| 3 | Rest and rotation | Current rest comes from EPL history. Add domestic cups and European fixtures, then derive hours since last match, matches over the last 7/14 days, and minutes played by likely starters where available. Compare effects among congested clubs and across the full fixture set. |
| 3 | Availability and lineups | Start with expected missing minutes or a shrunk player-strength estimate, using timestamped injury, suspension and lineup records. Confirmed lineups belong in the later forecast. Measure incremental value against the same-time market, which may already incorporate the news. |
| 4 | Game state, tactical and contextual features | Explore 11-v-11 or score-state-adjusted xG if reliable event data exists. Consider manager changes, set-piece strength or stylistic interactions when enough observations support them. Skip high-maintenance weather, referee, and head-to-head features until stronger inputs are tested. |

Missingness should carry its own semantics. A provider outage, a promoted club with sparse history, and a match that has yet to finish all warrant different provenance. Retain per-metric coverage and age, capture first observation and any later revision, and use explicit defaults tested inside each fold. [Current feature definitions](../src/features/prematch.py) · [Live source adapters](../src/data/live_fetchers.py)

## Database and pipeline improvements

**Keep the current replay and immutability guarantees.** Schedule and result application currently run in separate transactions. Every accepted schedule fetch updates all fixture rows and rebuilds team state. A substantive result change can trigger a second rebuild, and rebuild starts with historical replay. Measure schedule, rebuild, audit, and publishing time first. Compare schedule values before writing, track fetch freshness in an ingestion record, and rebuild once when the normalized batch changes replay inputs. Cache the season-opening state keyed by the training data and feature contract hashes. Keep full deterministic replay for corrections and audit. At this scale, that is simpler to verify than fine-grained incremental invalidation. [State manager](../src/data/state_manager.py) · [Strict audit](../scripts/audit_project.py)

**Store source history alongside the current view.** Use season-scoped fixture IDs and a source-ID mapping, with aliases for normalized club names. Add append-only fixture revisions, result/stat observations, odds quotes, and pipeline attempts. Deduplicate repeated identical observations by fixture, source, and payload hash while recording the poll attempt separately. Keep accepted current values in materialized tables or views. Index quote lookup by fixture, source and observation time, and use a unique forecast key that includes fixture, horizon, version and issuance policy. Retain forecast input values or an immutable reference to them. Keep migrations versioned and reversible, and preserve existing locks and quarantine flags.

**Record feeds independently of model publication.** The November 23 workflow left seven fixtures overdue and displayed the previous successful audit. The project currently accepts FPL final scores as fallback only when the other results fetch returns no usable results. A partially stale CSV can therefore prevent per-fixture score fallback. Record final scores, provisional scores and stats completeness independently, merge finalized sources per fixture, and quarantine conflicting scores. Publish audited score coverage and enrichment coverage separately. Keep unverified or missing stats out of model state unless a tested fallback contract explicitly supports them. Shadow-test eligibility for forecasts whose inputs remain fresh during a partial outage, including how stale results propagate through league-wide ratings. [Failed workflow job](https://github.com/chenbinioz/Premier-League-Predictor/actions/runs/45192651487/job/137754770390) · [Reconciliation](../scripts/update_live_state.py)

**Create diagnostics for the current attempt.** Write a run-scoped summary from the first step onward with run ID, source response classification, newest result, changed counts, unresolved fixture IDs, and skipped stages. Record an empty or partial feed separately from a transport failure. Use bounded retries for transient failures, align overdue handling across ingestion and prediction, and alert on the age of the last successful result update and upcoming fixtures without a forecast. Publish the audit for that attempt and link to the last successful audit with its timestamp. Add tests for stale partial CSV plus finalized official scores, provisional results, fixture corrections, unchanged source polls, and idempotent reruns. [Workflow](../.github/workflows/epl_live_pipeline.yml)

**Make snapshots easy to restore.** The workflow commits a binary SQLite file and an audit to Git. Keep this during the first experiment if the dashboard depends on it. When changing publication, create a consistent snapshot using the SQLite backup API, strict-audit the candidate, and publish a versioned file with a checksum and atomic latest pointer. Keep a single writer, a previous known-good snapshot, and a restore drill. Store larger raw feeds separately from the published database. PostgreSQL becomes worth considering if concurrent writers or remote SQL access become real requirements. [SQLite backup API](https://sqlite.org/backup.html)

## Implementation sequence

| Phase | Deliverable | Evidence it is ready |
| --- | --- | --- |
| 1 | Read-only benchmark runner with league prior, Elo logistic, current components and static Poisson. Save per-fixture outputs and fold metadata. | Reproduce the stored cohort and the Elo diagnostic, enforce temporal boundaries, and publish paired deltas and coverage. |
| 1 in parallel | Run-specific ingestion diagnostics and timestamped odds collection. | An empty or partial source produces a current failure summary. Every benchmark quote has a source and observation time before its forecast cutoff. |
| 2 | Opponent-adjusted form and a time-weighted Dixon–Coles challenger with promotion priors. | Ablations quantify each feature group, scoreline probabilities are normalized, and replay parity holds. |
| 3 | Dynamic strength or market-anchored challengers and prospective shadow forecasts. | Models are evaluated at the same horizons, produce reproducible snapshots, and meet the agreed promotion criteria. |
| 4 | Incremental publication improvements and optional lineup horizon. | Restore is verified, replay and corrections remain deterministic, and late forecasts are evaluated separately. |

The first implementation session should build the benchmark runner and define the early forecast cutoff. Those two choices let us test better training windows and opponent adjustment immediately while odds and availability history accumulate. This report adds a read-only retrospective experiment. Model artifacts, production data, and workflows were left as they were.

## First implementation results

Run the read-only benchmark from the repository root:

```sh
venv/bin/python -B scripts/backtest_models.py --output-dir /tmp/epl-backtest
```

The runner writes fixture-level probabilities and losses for both protocols, plus fold scores, matched-market scores, data hashes, the deployed model version, and paired calendar-week bootstrap intervals in `summary.json`. The rolling protocol uses expanding training folds with a 24-hour forecast cutoff and a three-hour assumed kickoff-to-observation lag for scores and statistics. Both values are configurable for sensitivity checks. Historical bookmaker odds are included as an untimed reference. The legacy protocol scores the deployed bundle and benchmarks with the original dated features and base training cutoff. [Backtest runner](../scripts/backtest_models.py)

Across 1,520 rolling forecasts from 2022/23 through 2025/26, form logistic scored 0.00480 worse than Elo on average, with a paired 95% interval of −0.00191 to +0.01122. Static Poisson scored 0.02894 worse, with an interval of +0.01697 to +0.04099. Under the legacy protocol, the deployed blend scored 0.01473 worse than Elo across 760 fixtures, with an interval of +0.00323 to +0.02719. The legacy comparison includes the season used to fit the blend weight. These intervals describe the observed fixtures under the chosen replay and bootstrap assumptions. Prospective forecasts with observed timestamps remain the promotion check.

## Expanded history and challenger results

The rolling benchmark now uses 4,180 matches from 2015/16 through 2025/26. Its eight evaluation folds run from 2018/19 through 2025/26, with 1,140 prior matches available for the first fold and 3,800 for the last. The completed-season benchmark ends in 2025/26. The legacy replay stays tied to the deployed bundle's original data and split.

The direct Football-Data download returned an empty file. The supplemental 2015/16 snapshot contains 380 scores and Bet365 1X2 quotes extracted from public Football-Data archives. A second archive agreed on all 380 dates, teams and scores. The supplement has no kickoff times, shots, corners or fouls. Replay preserves missing statistics and applies the existing conservative cutoff rule for missing kickoff times. The 2018/19 fold also has missing kickoff times for all 380 fixtures in the existing data, and uses the same rule. [Results archive](https://github.com/AnishKhetani/premier-league-data/blob/d69530c645c10c7844ee416a6af56a4b392e37be/data/processed/results.parquet) · [Odds archive](https://github.com/AnishKhetani/premier-league-data/blob/d69530c645c10c7844ee416a6af56a4b392e37be/data/processed/results_with_odds.parquet) · [Score cross-check](https://github.com/footballcsv/cache.footballdata/blob/f43a2aa0c65fb061c87e9541e7a567dd58ff933a/2015-16/eng.1.csv) · [Local supplement](../data/backtest/epl_1516.csv)

| Challenger on 3,040 fixtures | Log loss change versus Elo | Paired 95% interval | Seasons ahead of Elo |
| --- | ---: | ---: | ---: |
| Elo plus opponent-adjusted form | −0.00194 | −0.00547 to +0.00178 | 6 of 8 |
| Existing 17-feature form logistic | +0.00867 | +0.00367 to +0.01338 | |
| Static Poisson | +0.01386 | +0.00549 to +0.02180 | |
| Time-weighted Dixon–Coles | +0.01416 | +0.00560 to +0.02226 | 2 of 8 |

The opponent-adjusted challenger adds two inputs to Elo. It measures each side's shot-proxy production and concession relative to what the opponent had been allowing or creating before the match. It dampens the adjustment toward the league prior and summarizes the last five matches with shrinkage. The small 1X2 gain falls short of the provisional 0.005 promotion threshold and the interval includes zero. The time-weighted Dixon–Coles fit uses a 216-day half-life and jointly estimates team strengths, home advantage and a low-score correction bounded by the expected goals. The comparison tests that full specification. It does not isolate the effect of the low-score correction or measure exact-score quality.

Holding the eight evaluation folds fixed, adding the 2015/16 season reduced pooled log loss by 0.00089 for Elo, 0.00226 for Elo plus opponent-adjusted form and 0.00155 for Dixon–Coles. These historical differences are descriptive and combine the effect of the extra labels and earlier team-state initialization. The main benefit of evaluating earlier seasons is a wider test of repeatability.

```sh
venv/bin/python -B scripts/backtest_models.py --output-dir /tmp/epl-backtest-2015
venv/bin/python -B scripts/backtest_models.py --history-start 2016/17 --output-dir /tmp/epl-backtest-2016
```

The next accuracy experiment can keep Elo plus opponent-adjusted form as a challenger and test whether a different adjustment window or a dynamic strength model adds enough value to justify tracking prospectively. The production training inputs, model bundle and live database were left untouched.

## Dynamic strength and form windows

The next experiment kept the same 3,040 fixtures and cutoff assumptions. It compared three opponent-adjusted form windows and a dynamic attack and defense model, with and without Elo. The dynamic model starts from neutral home and away goal rates of 1.5 and 1.2, adjusts each team's attack and defense after an available result using bounded goal residuals, and shrinks ratings toward neutral between seasons. The rates feed the same regularized multinomial model used for the Elo benchmark. These parameter values were fixed for this experiment.

| Model | Pooled log loss change versus Elo | Paired 95% interval | Seasons ahead of Elo | 2025/26 change |
| --- | ---: | ---: | ---: | ---: |
| Elo plus 3-match adjusted form | −0.00301 | −0.00673 to +0.00076 | 5 of 8 | +0.00061 |
| Elo plus 5-match adjusted form | −0.00194 | −0.00577 to +0.00170 | 6 of 8 | −0.00194 |
| Elo plus 10-match adjusted form | −0.00284 | −0.00616 to +0.00059 | 6 of 8 | −0.00650 |
| Dynamic strength | +0.00904 | +0.00253 to +0.01559 | 1 of 8 | +0.00084 |
| Elo plus dynamic strength | −0.00032 | −0.00312 to +0.00257 | 4 of 8 | +0.00007 |
| Elo plus 5-match adjusted form and dynamic strength | −0.00117 | −0.00559 to +0.00321 | 5 of 8 | −0.00019 |

The 3-match window had the best pooled score, and the 10-match window improved six seasons and gave the largest reduction in 2025/26. Both intervals include zero and both pooled gains are below the 0.005 promotion threshold. These window comparisons are exploratory. A fixed 10-match challenger alongside Elo is a reasonable candidate for prospective comparison. Adding dynamic strength weakened the existing 5-match form challenger. This implementation offers little evidence of extra 1X2 value from its goal residuals.

Run the updated comparison with `venv/bin/python -B scripts/backtest_models.py --output-dir /tmp/epl-backtest-dynamic`. The live database and model bundle were left untouched.

## Prospective shadow trial

The daily workflow can now record Elo and Elo plus 10-match opponent-adjusted form as a pair for each pending fixture with a known kickoff in the next 30 hours. The 30-hour window accommodates the daily cadence. Both models use the same reconciled database snapshot and information cutoff, and the ledger captures actual issuance and lead time. The candidate trains through 2025/26 using the expanded historical replay. Each fixture's first pair is retained, and the evaluator excludes fixtures whose kickoff changes after issuance.

The workflow runs the shadow step after the existing production publishing steps and publishes its ledger in a separate commit. The ledger lives at `data/live/shadow_forecasts.json` once the first pair is written. Read it against official reconciled results with:

```sh
venv/bin/python -B scripts/shadow_forecasts.py --evaluate
```

The evaluator reports matched log loss and Brier score, alongside pending and changed fixtures. The trial uses actual forecast times, which will vary by fixture. Track lead-time coverage before comparing the live results to the fixed 24-hour backtest. [Shadow runner](../scripts/shadow_forecasts.py) · [Daily workflow](../.github/workflows/epl_live_pipeline.yml)
