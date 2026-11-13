# Tracker recovery — 13 November 2026

The local database has been updated from current sources: 100 completed matches,
zero overdue fixtures, and 110 locked forecasts. All 100 existing forecasts were
preserved exactly. Ten new forecasts were locked for GW11. All 25 regression
tests and the strict audit passed. No model retraining was performed.

Recovery backup: `data/backups/tracker_update_20261113T212755Z/`. It contains
`original.db`, `candidate.db`, the pre-update working-tree patch, untracked-file
archive, update logs and candidate audit. Backups are ignored by Git.

## View locally

Refresh the running dashboard. If necessary, rerun it with `./run_app.sh`.
Select Gameweek 10 to see the completed results. Model-only performance now has
ten eligible completed forecasts. The matched bookmaker comparison remains
unavailable because contemporaneous bookmaker quotes were not recorded.

## Finish GitHub deployment

GitHub has not been updated by this recovery. Its existing updater can finish
successfully after a collection failure and contains incorrect inferred rounds.

1. Create a review branch from the current local work. Inspect `git diff` and
   `git status --short` before staging. Preserve the recovery backups locally.
2. Publish the coherent November repair together: runtime requirements, app,
   updater/prediction/audit scripts, source modules, tests, current database,
   processed features, trained model artifacts, manifest, immutable version
   archive, README and workflow. Review ancillary notebook and IDE changes
   separately. Do not publish only the workflow: it depends on untracked local
   scripts and model artifacts. Do not stage backups, secrets or bytecode.
3. Run `venv/bin/python -m unittest discover -s tests -v` and
   `venv/bin/python scripts/audit_project.py --strict` before committing.
4. Push the review branch and review the complete repair before merging to main.
   GitHub's Ubuntu/Python 3.13 dependency installation still needs validation;
   local macOS tests do not establish that installation succeeds there.
5. On main, open Actions → EPL 2026/27 Live Predictive Pipeline → Run workflow,
   choose `both`. Verify update logs, audit timestamp, zero overdue fixtures,
   and successful results/prediction publication steps. The prepared workflow
   publishes audited results before prediction and retains audit diagnostics.
6. Confirm the next scheduled run performs the same checks. A missing new-result
   commit is acceptable when there are no newly completed matches; assess the
   ingestion counts and audit timestamp, not the green badge alone.

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
