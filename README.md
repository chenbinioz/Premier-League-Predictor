Premier League predictor

A Streamlit app backed by a replayable SQLite results ledger and a versioned pre-match model bundle. The November 2026 repair replaces inferred gameweeks, target-leaked neural inputs, and synthetic bookmaker features.

**Run locally**

Use Python 3.13 and the pinned runtime versions in `requirements.txt`:

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements.txt
./run_app.sh
```

The app verifies artifact checksums, the feature contract, and model-library versions before loading models. A mismatched or partially replaced bundle fails explicitly.

The launcher always uses `venv/bin/python`, even when Anaconda is active in the terminal. Avoid a bare `streamlit run app.py`, which can select a different installation. In VS Code, the **Run Premier League predictor** task also uses the project interpreter. After changing environments, stop the old server and start the launcher; refreshing a browser alone cannot change a running server's Python environment.

**Update results and predict**

```bash
venv/bin/python scripts/update_live_state.py
venv/bin/python scripts/predict_next_gameweek.py --auto
venv/bin/python scripts/audit_project.py --strict --output reports/latest_audit.json
```

The updater reads the official Premier League FPL schedule, merges observed match statistics from football-data, cross-checks official scores, and collects available real upcoming bookmaker quotes. Understat xG enrichment is optional (`--understat`); an Understat error cannot break the primary results path. Missing goals never become 0–0s. Missing statistics remain unknown and produce explicit degraded status. Failed or stale ingestion blocks new forecast locks.

A deterministic offline replay of the audit-time source snapshots is available:

```bash
venv/bin/python scripts/update_live_state.py \
  --results-csv data/raw/epl_2627_results.csv \
  --schedule-csv data/raw/epl_2627_schedule.csv
```

These snapshots are dated 4 November 2026. They are for reproduction, not a replacement for future live updates. Official gameweeks own round identity; rescheduled dates and kickoff times update without replacing fixture IDs or old forecasts. Postponed fixtures are excluded from automatic pending-match selection until rescheduled.

**Train models**

```bash
venv/bin/python scripts/train_models.py
```

Training rebuilds the historical features and coherent model bundle together:

| Stage | Seasons |
|---|---|
| Base fitting | 2016/17–2022/23 |
| Unweighted calibration / neural early stopping | 2023/24 |
| Ensemble selection | 2024/25 |
| Final historical evaluation | 2025/26 |

Do not tune hyperparameters against the final evaluation. Its results are historical holdout measurements, not prospective live accuracy. The same feature engine generates historical inputs before observing each target match and generates live inputs from replayed observations. It uses a consistent, explicitly defined shot-quality proxy, not interchangeable provider xG or goals. Provider xG is retained separately for research.

The primary XGBoost is market-free. The separate `models/bin/market_xgb.joblib` artifact requires genuine bookmaker features and is not used as a proxy-fed live model. NDC uses only pre-match features. A scalar ensemble is selected on validation log loss. The independent Poisson baseline is fitted only on base-training seasons and is not counted as a locked historical forecast.

Artifact/environment/data/code hashes and temporal boundaries are stored in `models/bin/manifest.json`. Every new forecast stores the manifest, exact feature vector/hash, blend weight/version, state timestamp, and lock timestamp. New model training does not rewrite locked forecasts. Daily updates do not retrain models. Each repaired bundle is retained under `models/versions/<version>/`, so the audit can rerun its models on the exact locked feature vector after a later retrain. Archived source is retained for reference and is never automatically executed.

**Inspect performance and integrity**

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/audit_project.py --strict
```

The audit checks prior-season seeding, all current-season state replays, applied-result hashes, original forecast preservation, model contracts/checksums, distribution consistency, forecast timing, ingestion freshness, and missing observations. Strict mode fails on critical/high/medium findings.

The app separates:

- Original legacy forecasts, retained but quarantined from prospective accuracy.
- Verified model-only performance on completed forecasts.
- A bookmaker race using the same eligible matches and quotes actually observed by forecast time.
- Current simulations, labeled separately from locked forecasts.

Missing comparison data displays as unavailable, not zero accuracy. A later bookmaker quote is never retroactively attached to an earlier forecast.

**Database initialization and recovery**

Initialization requires a canonical schedule and refuses to overwrite an existing database:

```bash
venv/bin/python src/db/init_live_db.py --db /tmp/new-epl.db \
  --schedule data/raw/epl_2627_schedule.csv
```

The original database, processed data, and models are preserved locally under `data/backups/pre_repair_2026-11-04/` (ignored by Git). Original notebooks remain in Git history; current notebooks delegate to maintained scripts. There is no destructive reset or forecast-overwrite option in the pipeline.

Read `project_audit_2026-11-04.md` for the original findings and `project_reaudit_2026-11-04.md` for the repair verification and remaining limitations. The GitHub workflow uses these local changes only after they are committed and pushed.
