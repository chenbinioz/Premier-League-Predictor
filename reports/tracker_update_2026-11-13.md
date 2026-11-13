# Tracker recovery — 13 November 2026

The local database has been updated from current sources: 100 completed matches,
zero overdue fixtures, and 110 locked forecasts. All 100 existing forecasts were
preserved exactly. Ten new forecasts were locked for GW11. All 26 regression
tests and the strict audit passed. No model retraining was performed.

Recovery backup: `data/backups/tracker_update_20261113T212755Z/`. It contains
`original.db`, `candidate.db`, the pre-update working-tree patch, untracked-file
archive, update logs and candidate audit. Backups are ignored by Git.

## View locally

Refresh the running dashboard. If necessary, rerun it with `./run_app.sh`.
Select Gameweek 10 to see the completed results. Model-only performance now has
ten eligible completed forecasts. The matched bookmaker comparison remains
unavailable because contemporaneous bookmaker quotes were not recorded.

## Deployment completed

The coherent repair was merged to main in [PR #1](https://github.com/chenbinioz/Premier-League-Predictor/pull/1).
[Ubuntu validation](https://github.com/chenbinioz/Premier-League-Predictor/actions/runs/43513003358)
passed all 26 tests and the strict audit. The cross-platform audit permits at most
1e-10 absolute floating-point roundoff; structural and observation checks remain
exact, and significant state corruption still fails regression tests.

The [production run](https://github.com/chenbinioz/Premier-League-Predictor/actions/runs/43513466325)
succeeded on 13 November 2026. It fetched 100 results, found zero further result
changes, zero overdue fixtures and no missing match statistics. GW11 forecasts
were already locked, so zero additional forecasts were created. The bot published
results/freshness commit `8de18f2` and final audit commit `0324292`.

The workflow retains the daily 04:00 UTC schedule, publishes audited results
before prediction, and records freshness in its summary and an audit artifact.
The next scheduled run has not occurred yet; its future outcome is unverified.
GitHub scheduling may start later than the configured time.

IDE settings, notebook edits, architecture documentation and local recovery
backups were excluded from deployment and preserved separately. The local
synchronization retains a complete pre-sync Git stash and database backup under
`data/backups/deployment_2026-11-13/`.

## Subsequent local updates

The dashboard reads a local file; GitHub jobs cannot modify that file directly.
Use the sidebar's Run Live Update Pipeline to fetch current results locally.
After the repair is merged and the working tree is clean, Pull Latest from GitHub
can retrieve published updates. If local and remote SQLite files both changed,
preserve both copies and reconcile from sources; do not resolve a binary conflict
by blindly overwriting either database.

For command-line updates, use the project interpreter:

```sh
venv/bin/python scripts/update_live_state.py
venv/bin/python scripts/predict_next_gameweek.py --auto
venv/bin/python scripts/audit_project.py --strict --output reports/latest_audit.json
```

Run each command only after the previous one succeeds. Retain a SQLite backup
before updates. Missing sources or statistics should remain explicit failures;
do not substitute goals, zero scores or old snapshots to make a run pass.
