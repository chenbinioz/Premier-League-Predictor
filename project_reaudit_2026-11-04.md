Project repair and second audit — 4 November 2026

**Outcome:** the original critical, high, and medium findings have been addressed locally. The strict second audit passes with **no critical, high, or medium findings**. Its machine-readable evidence is in [reports/latest_audit.json](reports/latest_audit.json). The initial audit is preserved unchanged in [project_audit_2026-11-04.md](project_audit_2026-11-04.md).

**What changed, in priority order**

| Original concern | Implemented repair | Verification |
|---|---|---|
| Critical: failed/inconsistent result collection | Official FPL schedule/round IDs; explicit date parsing; identity-based reconciliation; CSV statistics and official-score cross-check; optional Understat; explicit failure/degraded status | 90 completed results, zero overdue fixtures; official and CSV scores agreed on all 90 |
| Critical: neural target leakage | One shared historical/live feature engine and an explicit 54-feature pre-match allowlist; vectors built before target results enter state | Target-mutation and first-observation tests; no current-match goals/statistics or bookmaker proxies in neural inputs; models retrained |
| Critical: synthetic XGBoost market inputs | 17-feature market-free live classifier; separate 20-feature market-aware classifier trained with real prices | Serving validates the primary feature list; no NDC outputs enter bookmaker feature slots |
| High: missing GW9 results | Liverpool 0–1 Arsenal and Newcastle 0–1 Everton reconciled | GW9 now 10/10 completed |
| High: missing match statistics | Completed fixtures revisited for enrichment; observed shots, shots on target, corners and fouls persisted; unknown observations stay null | Zero missing values across all eight audited statistics columns for 90 completed fixtures |
| High: stale Elo and faulty initialization | Elo tab reads SQLite; full prior-season replay includes final results; one season transition; initialization refuses overwrite | All 20 current states match replay; prior-season seed independently checked against raw history |
| High: partial or duplicate state application | Atomic result/state transactions, result hashes, chronological rebuild on changes and corrections | Duplicate application, correction, and injected partial-failure tests pass |
| High: inconsistent historical forecasts and late locks | Preserve original payloads; quarantine unverified legacy forecasts; record exact inputs, timestamps, model manifest and blend version; prohibit overwrites/late locks | All 90 original forecast payloads exactly match the pre-repair backup; all are excluded from prospective metrics |
| High: scoreline grading/display | Parse score candidates separately; reconstruct inconsistent legacy scoreline displays with a label; distinguish current simulations and locked forecasts | UI tests cover quarantined GW1, current simulations, and historical reconstructions; saved distributions audited |
| Medium: blend labeling/provenance | Per-forecast actual blend version/weight; no label inferred from file existence; scalar mix learned only on validation season | All ten new forecasts reproduce their archived model inference and declared blend |
| Evaluation/preprocessing concerns | Separate fitting, later unweighted calibration, validation, and test periods; no backward fill or draw-weighted calibration; same venue/rest/season transformations in training and serving | Fixed 2,660 / 380 / 380 / 380 chronological stages; feature parity tests |
| xG definition drift | Consistent shot-quality proxy used in both training and serving; provider xG retained separately | Explicit `prematch-v2-shot-proxy` contract and raw observation provenance |
| Unsupported artifact environments | Pinned core/runtime versions, Python 3.13 CI, artifact/code/data/environment hashes, immutable version archives | Bundle identity and hashes validated before loading; historical archived models re-executed on locked vectors |
| Leaking/mislabeled static baseline and obsolete notebook paths | Independent Poisson fitted only on base-training seasons; excluded from locked-performance grading; notebooks delegate to maintained scripts | Correct model-kind/training-cutoff metadata; no full-dataset baseline fit or legacy trainer remains active |

**Additional concerns found during implementation and the second review**

These were fixed before the final audit:

- The inherited future schedule put all GW10 games on Saturday. The [official Premier League fixture API](https://fantasy.premierleague.com/api/fixtures/) and its [team metadata](https://fantasy.premierleague.com/api/bootstrap-static/) supplied all 380 canonical fixtures. GW10 actually has **one Friday fixture, four Saturday fixtures, and five Sunday fixtures**. All fixture identities were retained while dates, kickoff times and rounds were reconciled. This also removes the original full-season schedule provenance concern.
- Returning/promoted clubs could retain top-flight form from many years earlier. Such histories now reset to explicit priors at promotion; their Elo starts at 1420.
- “Live source available” previously meant a Python package was installed. The app now reports actual ingestion success/failure and freshness. Failed/stale collection blocks new production locks.
- A later bookmaker price could otherwise be paired with an earlier forecast. New forecasts snapshot only real prices observed by lock time. Model-only performance is available separately when a fair contemporaneous bookmaker comparison is unavailable.
- Checking a score grid alone does not prove the displayed numbers came from the declared model. The audit now loads each archived model version, reruns its exact saved feature vector, and checks NDC parameters, XGBoost probabilities, scorelines, blend weight/version and final probabilities.
- Refitting models would otherwise remove the ability to reproduce earlier locks. Model artifacts and matching source snapshots are retained under `models/versions/<version>/`; archived source is never automatically executed. The forecast's manifest must match the archived manifest.
- Historical CSV odds no longer inherit an earlier live quote's observation timestamp. Opponent pressure arrays now retain observed values as well as each team's own pressure values. Source/date/season validation and explicit resource cleanup were tightened.

**Current data and ledger**

| Check | Result |
|---|---:|
| Canonical fixtures | 380 |
| Completed results | 90 |
| Pending fixtures | 290 |
| Overdue fixtures | 0 |
| Current team states replayed | 20/20 |
| Applied-result ledger entries | 90 |
| Missing audited result statistics | 0 |
| Original forecasts retained and quarantined | 90 |
| New verified GW10 forecasts | 10 |
| Original forecast payload preservation | Exact match with backup |
| SQLite integrity / foreign keys | Pass |

Re-running reconciliation made **zero result changes**. Re-running prediction made **zero additional forecasts**. GW10 locks are immutable; the version archive allows them to be checked independently of the active bundle after another retrain.

The original database, processed dataset, and model artifacts are retained locally in `data/backups/pre_repair_2026-11-04/`, which is ignored by Git. No GitHub push, deployment, or remote workflow dispatch was performed.

**Retrained historical evaluation**

The fitting period ends in 2022/23. Calibration/early stopping uses 2023/24, blend selection uses 2024/25, and final historical evaluation uses all 380 matches from 2025/26. Hyperparameters were not selected against the final test results. Full results are in [reports/training_evaluation.json](reports/training_evaluation.json).

| Model | 2025/26 accuracy | 1X2 log loss, lower is better |
|---|---:|---:|
| Neural Dixon–Coles, pre-match-only | 48.16% | 1.0567 |
| XGBoost, market-free | 47.63% | 1.0509 |
| Deployed scalar blend | 47.63% | 1.0465 |
| XGBoost, real-market inputs | 47.63% | 1.0462 |
| Bookmaker reference | 48.95% | 1.0185 |

These are historical holdout measurements, not new prospective live accuracy or evidence of an edge over bookmakers. The leakage repairs make the evaluation usable; they do not guarantee beating the market. The repaired forecasts have not yet completed, so prospective performance is currently unavailable rather than 0%.

**Remaining findings**

- **Low — contemporaneous market coverage:** the public upcoming-odds feed had no EPL quotes when the ten early GW10 forecasts were locked. Those records correctly contain no bookmaker snapshot and are excluded from the matched-market race. They will still qualify for the separate model-only performance table once completed. Prices obtained later will not be retroactively assigned to them. The updater collects genuine upcoming prices when available.
- **Low — statistical evidence:** one final historical season and zero completed repaired-version live forecasts cannot establish durable improvement or profitability. Multiple forward seasonal experiments and a longer prospective record remain research work.
- **Low — available football information:** the model currently uses a consistent shot proxy and EPL-only rest history. Event-derived xG, injuries/expected lineups and workload in other competitions remain potential improvements, with known-before-kickoff timestamps required.
- **Informational — legacy evidence:** the 90 original forecasts cannot be rehabilitated into verified pre-match predictions after the fact. They are deliberately retained and quarantined.

**Validation and reproduction**

- **24 regression tests passed**, including training/serving parity, independent first-match priors, source merging/conflicts, date parsing, unknown observations, season transitions, result corrections/idempotence, rollback, opponent statistics, immutable/late locks, official schedule changes, collection failure reporting, postponed-match selection, fair market cohorts, read-only dry runs, and deliberate audit-tampering detection.
- Streamlit `AppTest` passed for initial tracker rendering, current-team forecasting, historical-team reconstruction, and quarantined GW1 rendering. No browser/server startup was required for these checks. An existing app is listening at `http://localhost:8501`; refresh it to load the changes.
- The strict read-only audit passed, including re-execution of all ten new forecasts from their archived model versions. `git diff --check` passed.
- Linux GitHub Actions execution has not been exercised remotely. The updated workflow and pinned environment will take effect when the local changes are committed and pushed.

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/audit_project.py --strict \
  --output reports/latest_audit.json
```

See [README.md](README.md) for live updates, offline replay, training, initialization, and provenance details.
