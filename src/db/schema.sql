-- ============================================================
-- EPL 26/27 Live Database Schema
-- Backend: SQLite (compatible subset used throughout)
-- ============================================================

-- Drop tables in reverse-dependency order for clean re-initialisation
DROP TABLE IF EXISTS predictions_26_27;
DROP TABLE IF EXISTS live_team_states;
DROP TABLE IF EXISTS fixtures_26_27;

-- ── 1. fixtures_26_27 ────────────────────────────────────────
--    Stores the full 380-match schedule and progressive results.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE fixtures_26_27 (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    gameweek        INTEGER NOT NULL,
    match_date      TEXT    NOT NULL,
    home_team       TEXT    NOT NULL,
    away_team       TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'completed')),
    home_goals      INTEGER,
    away_goals      INTEGER
);

CREATE INDEX idx_fixtures_date   ON fixtures_26_27 (match_date);
CREATE INDEX idx_fixtures_status ON fixtures_26_27 (status);
CREATE INDEX idx_fixtures_gw     ON fixtures_26_27 (gameweek);

-- ── 2. live_team_states ──────────────────────────────────────
--    One row per team. Rolling arrays stored as JSON strings.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE live_team_states (
    team_name             TEXT    PRIMARY KEY,
    current_elo           REAL    NOT NULL DEFAULT 1500.0,
    last_10_xg_for        TEXT    NOT NULL DEFAULT '[]',
    last_10_xg_against    TEXT    NOT NULL DEFAULT '[]',
    last_10_goals_for     TEXT    NOT NULL DEFAULT '[]',
    last_10_goals_against TEXT    NOT NULL DEFAULT '[]',
    avg_xg_for            REAL    NOT NULL DEFAULT 0.0,
    avg_xg_against        REAL    NOT NULL DEFAULT 0.0,
    avg_goals_for         REAL    NOT NULL DEFAULT 0.0,
    avg_goals_against     REAL    NOT NULL DEFAULT 0.0,
    last_updated          TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── 3. predictions_26_27 ────────────────────────────────────
--    Pre-match forecasts, locked before kick-off.
-- ────────────────────────────────────────────────────────────
CREATE TABLE predictions_26_27 (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            INTEGER NOT NULL UNIQUE,
    predicted_home_prob REAL    NOT NULL,
    predicted_draw_prob REAL    NOT NULL,
    predicted_away_prob REAL    NOT NULL,
    predicted_score     TEXT    NOT NULL,
    ndc_home_prob       REAL,
    ndc_draw_prob       REAL,
    ndc_away_prob       REAL,
    xgb_home_prob       REAL,
    xgb_draw_prob       REAL,
    xgb_away_prob       REAL,
    ndc_lambda          REAL,
    ndc_mu              REAL,
    ndc_rho             REAL,
    locked_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (match_id) REFERENCES fixtures_26_27 (match_id)
);

CREATE INDEX idx_preds_match ON predictions_26_27 (match_id);
