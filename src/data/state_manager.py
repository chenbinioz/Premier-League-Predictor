"""Transactional result reconciliation, deterministic state replay, immutable forecasts."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.features.prematch import FeatureEngine, CONTRACT, METRICS, NDC_FEATURES, XGB_BASE_FEATURES, finite, replay_history, normalized_market

ROOT = Path(__file__).resolve().parents[2]
SEASON = '2026/27'
ELO_K, ELO_HOME_ADV, WINDOW_SIZE = 20, 60, 10
RESULT_COLUMNS = {
    'match_date': 'match_date', 'home_goals': 'home_goals', 'away_goals': 'away_goals',
    'home_sot': 'home_shots_on_target', 'away_sot': 'away_shots_on_target',
    'home_shots': 'home_shots', 'away_shots': 'away_shots',
    'home_corners': 'home_corners', 'away_corners': 'away_corners',
    'home_fouls': 'home_fouls', 'away_fouls': 'away_fouls',
    'home_xg': 'home_xg', 'away_xg': 'away_xg', 'xg_source': 'xg_source',
    'result_source': 'result_source', 'observed_at': 'result_observed_at',
    'api_match_id': 'api_match_id', 'kickoff_at': 'kickoff_at',
    'B365H': 'B365H', 'B365D': 'B365D', 'B365A': 'B365A',
}


def digest(value):
    def canonical(item):
        if isinstance(item, dict):
            return {key: canonical(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(val) for val in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def base_signature():
    return digest({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (ROOT/'data/raw/epl_10_seasons_raw.csv', ROOT/'src/features/prematch.py')})


class TeamStateManager:
    def __init__(self, db_path):
        if not Path(db_path).exists():
            raise FileNotFoundError(db_path)
        self._conn = sqlite3.connect(db_path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA foreign_keys=ON')
        # DELETE journal keeps the database self-contained for Git and backups.
        self._conn.execute('PRAGMA journal_mode=DELETE')
        try:
            self._migrate()
        except Exception:
            self._conn.close()
            raise

    def _migrate(self):
        additions = {
            'fixtures_26_27': {
                'home_shots_on_target': 'INTEGER', 'away_shots_on_target': 'INTEGER',
                'home_corners': 'INTEGER', 'away_corners': 'INTEGER', 'home_fouls': 'INTEGER', 'away_fouls': 'INTEGER',
                'home_shots': 'INTEGER', 'away_shots': 'INTEGER', 'home_xg': 'REAL', 'away_xg': 'REAL',
                'xg_source': 'TEXT', 'result_source': 'TEXT', 'result_observed_at': 'TEXT',
                'api_match_id': 'TEXT', 'kickoff_at': 'TEXT', 'schedule_source': "TEXT DEFAULT 'legacy-import'",
                'official_match_id': 'TEXT', 'schedule_observed_at': 'TEXT', 'schedule_status': "TEXT NOT NULL DEFAULT 'scheduled'",
                'B365H': 'REAL', 'B365D': 'REAL', 'B365A': 'REAL', 'odds_observed_at': 'TEXT'},
            'live_team_states': {'engine_state': 'TEXT', **{f'last_10_{m}_{direction}': "TEXT NOT NULL DEFAULT '[]'"
                for m in ('sot', 'corners', 'fouls') for direction in ('for', 'against')},
                **{f'avg_{m}_{direction}': 'REAL NOT NULL DEFAULT 0' for m in ('sot', 'corners', 'fouls') for direction in ('for', 'against')}},
            'predictions_26_27': {
                **{f'{m}_{o}_prob': 'REAL' for m in ('ndc', 'xgb') for o in ('home', 'draw', 'away')},
                'model_version': "TEXT NOT NULL DEFAULT 'legacy-unverified'", 'feature_contract': 'TEXT',
                'blend_version': "TEXT NOT NULL DEFAULT 'legacy-unknown'", 'feature_hash': 'TEXT',
                'state_as_of': 'TEXT', 'prediction_status': "TEXT NOT NULL DEFAULT 'quarantined'",
                'feature_values': 'TEXT', 'blend_alpha': 'REAL', 'model_manifest': 'TEXT',
                'quality_flags': "TEXT NOT NULL DEFAULT 'legacy_model_unverified'",
                'market_home_prob': 'REAL', 'market_draw_prob': 'REAL', 'market_away_prob': 'REAL',
                'market_observed_at': 'TEXT'},
        }
        with self._conn:
            for table, fields in additions.items():
                existing = {r[1] for r in self._conn.execute(f'PRAGMA table_info({table})')}
                for name, definition in fields.items():
                    if name not in existing:
                        self._conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')
            self._conn.execute('CREATE TABLE IF NOT EXISTS pipeline_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self._conn.execute('CREATE TABLE IF NOT EXISTS applied_results (match_id INTEGER PRIMARY KEY REFERENCES fixtures_26_27(match_id), result_hash TEXT NOT NULL, applied_at TEXT NOT NULL)')
            self._conn.execute('CREATE TABLE IF NOT EXISTS ingestion_runs (run_id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL, status TEXT NOT NULL, summary TEXT NOT NULL)')
            duplicates = self._conn.execute('SELECT home_team,away_team FROM fixtures_26_27 GROUP BY home_team,away_team HAVING COUNT(*)>1').fetchall()
            if duplicates:
                raise ValueError('Duplicate season fixtures require explicit reconciliation')
            self._conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS fixture_identity ON fixtures_26_27(home_team,away_team)')
        self._flag_legacy_predictions()
        with self._conn:
            if not self._conn.execute("SELECT 1 FROM pipeline_metadata WHERE key='legacy_forecast_hash'").fetchone():
                rows = self._conn.execute("SELECT match_id,predicted_home_prob,predicted_draw_prob,predicted_away_prob,predicted_score,ndc_lambda,ndc_mu,ndc_rho,locked_at,ndc_home_prob,ndc_draw_prob,ndc_away_prob,xgb_home_prob,xgb_draw_prob,xgb_away_prob FROM predictions_26_27 WHERE model_version='legacy-unverified' ORDER BY match_id").fetchall()
                self._conn.execute('INSERT INTO pipeline_metadata VALUES (?,?)', ('legacy_forecast_hash', digest([list(r) for r in rows])))

    def _flag_legacy_predictions(self):
        import numpy as np
        from models.neural_dixon_coles import predict_scoreline_grid, parse_grid_outputs
        with self._conn:
            rows = self._conn.execute("SELECT p.*,f.match_date,f.kickoff_at FROM predictions_26_27 p JOIN fixtures_26_27 f USING(match_id) WHERE model_version='legacy-unverified'").fetchall()
            for row in rows:
                r = dict(row)
                flags = ['legacy_model_unverified']
                if r['locked_at'][:10] > r['match_date']:
                    flags.append('locked_after_match_date')
                elif r['locked_at'][:10] == r['match_date'] and not r['kickoff_at']:
                    flags.append('kickoff_timing_unverified')
                if all(r.get(k) is not None for k in ('ndc_lambda', 'ndc_mu', 'ndc_rho')):
                    parsed = parse_grid_outputs(predict_scoreline_grid(r['ndc_lambda'], r['ndc_mu'], r['ndc_rho']), top_k=3)
                    scores = ' • '.join(s['scoreline'] for s in parsed['top_scorelines'])
                    if scores != r['predicted_score']:
                        flags.append('scoreline_parameters_disagree')
                blend = 'legacy-unknown'
                values = [r.get(k) for k in ('ndc_home_prob','ndc_draw_prob','ndc_away_prob','xgb_home_prob','xgb_draw_prob','xgb_away_prob')]
                if all(v is not None for v in values):
                    p = (np.array(values[:3]) + np.array(values[3:])) / 2
                    p /= p.sum()
                    if np.max(abs(p - [r['predicted_home_prob'],r['predicted_draw_prob'],r['predicted_away_prob']])) < 1e-6:
                        blend = 'legacy-50-50'
                self._conn.execute('UPDATE predictions_26_27 SET prediction_status=?,quality_flags=?,blend_version=? WHERE match_id=?',
                                   ('quarantined', ','.join(flags), blend, r['match_id']))

    def _base_engine(self):
        cached = self._conn.execute("SELECT value FROM pipeline_metadata WHERE key='base_engine'").fetchone()
        signature = self._conn.execute("SELECT value FROM pipeline_metadata WHERE key='base_engine_signature'").fetchone()
        expected_signature = base_signature()
        if cached and signature and signature[0] == expected_signature:
            snapshot = json.loads(cached[0])
            if snapshot['contract'] != CONTRACT:
                raise ValueError('State contract changed: explicit migration required')
            return FeatureEngine(snapshot)
        engine = replay_history(pd.read_csv(ROOT/'data/raw/epl_10_seasons_raw.csv'), before_season=SEASON)
        teams = {r[0] for r in self._conn.execute('SELECT home_team FROM fixtures_26_27 UNION SELECT away_team FROM fixtures_26_27')}
        engine.start_season(SEASON, teams)
        self._conn.execute('INSERT OR REPLACE INTO pipeline_metadata VALUES (?,?)', ('base_engine', json.dumps(engine.snapshot())))
        self._conn.execute('INSERT OR REPLACE INTO pipeline_metadata VALUES (?,?)', ('base_engine_signature', expected_signature))
        return engine

    def _write_team_state(self, team, state):
        history = state['history'][-10:]
        values = {'team_name': team, 'current_elo': state['elo'], 'engine_state': json.dumps(state), 'last_updated': utcnow()}
        for key, metric in [('xg_for','xG_Created'),('xg_against','xG_Conceded'),('goals_for','GoalsScored'),('goals_against','GoalsConceded'),('sot_for','ShotsOnTarget'),('corners_for','Corners'),('fouls_for','Fouls')]:
            data = [r.get(metric) for r in history]
            known = [v for v in data if v is not None]
            values['last_10_'+key] = json.dumps(data)
            values['avg_'+key] = sum(known)/len(known) if known else METRICS[metric]
        # Preserve observed opponent pressure for inspection/research as well.
        # Prior-season observations without opponent fields remain explicit nulls.
        matches = {r['match_date']:dict(r) for r in self._conn.execute(
            "SELECT * FROM fixtures_26_27 WHERE status='completed' AND (home_team=? OR away_team=?)", (team,team))}
        for metric, column, prior in (('sot','shots_on_target',4.4),('corners','corners',5.1),('fouls','fouls',11.0)):
            data = []
            for observation in history:
                opponent = 'away' if observation['venue']=='Home' else 'home'
                data.append(matches.get(observation['date'],{}).get(f'{opponent}_{column}'))
            known = [v for v in data if v is not None]
            values[f'last_10_{metric}_against'] = json.dumps(data)
            values[f'avg_{metric}_against'] = sum(known)/len(known) if known else prior
        cols = ','.join(values)
        self._conn.execute(f'INSERT OR REPLACE INTO live_team_states ({cols}) VALUES ({",".join("?" for _ in values)})', tuple(values.values()))

    def _rebuild(self):
        engine = self._base_engine()
        self._conn.execute('DELETE FROM applied_results')
        rows = self._conn.execute("SELECT * FROM fixtures_26_27 WHERE status='completed' ORDER BY match_date,home_team,away_team").fetchall()
        for raw in rows:
            r = dict(raw)
            observation = {field: r.get(col) for field,col in RESULT_COLUMNS.items()}
            observation.update(home_team=r['home_team'], away_team=r['away_team'])
            engine.observe(observation)
            fingerprint = {k: observation[k] for k in observation if k not in ('observed_at',)}
            self._conn.execute('INSERT INTO applied_results VALUES (?,?,?)', (r['match_id'], digest(fingerprint), utcnow()))
        for team in sorted(engine.active):
            self._write_team_state(team, engine.teams[team])
        self._conn.execute('INSERT OR REPLACE INTO pipeline_metadata VALUES (?,?)', ('feature_contract', CONTRACT))

    def rebuild_states(self):
        self._conn.execute('BEGIN IMMEDIATE')
        try:
            self._rebuild()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def apply_results(self, results):
        """Atomically reconcile scores/enrichment/corrections and replay chronological state."""
        self._conn.execute('BEGIN IMMEDIATE')
        changed = 0
        try:
            for result in results:
                fixture = self._conn.execute('SELECT * FROM fixtures_26_27 WHERE home_team=? AND away_team=?', (result['home_team'],result['away_team'])).fetchone()
                if fixture is None:
                    raise ValueError(f"Unknown season fixture: {result['home_team']} vs {result['away_team']}")
                for goal in ('home_goals','away_goals'):
                    value = finite(result.get(goal))
                    if value is None or value < 0 or not value.is_integer():
                        raise ValueError('Missing/invalid full-time goals')
                result_date = result.get('match_date') or fixture['match_date']
                if fixture['schedule_source'] == 'premier-league-fpl' and result_date != fixture['match_date']:
                    raise ValueError('Result date conflicts with the official schedule')
                if not '2026-07-01' <= result_date < '2027-07-01':
                    raise ValueError('Result does not belong to the live season')
                if result_date > datetime.now(timezone.utc).date().isoformat():
                    raise ValueError('A future fixture cannot have an observed result')
                for field in ('home_sot','away_sot','home_shots','away_shots','home_corners','away_corners','home_fouls','away_fouls'):
                    value = result.get(field)
                    if value is not None and (finite(value) is None or value < 0 or not float(value).is_integer()):
                        raise ValueError(f'Invalid observed statistic: {field}')
                updates = {column: result[field] for field,column in RESULT_COLUMNS.items() if result.get(field) is not None}
                if fixture['schedule_source'] == 'premier-league-fpl':
                    updates.pop('kickoff_at', None)  # Official kickoff owns forecast timing.
                if any(key in updates for key in ('B365H','B365D','B365A')):
                    # Historical CSV prices are not a newly observed live quote.
                    updates['odds_observed_at'] = None
                # Observation time alone is not a result change.
                substantive = {k:v for k,v in updates.items() if k != 'result_observed_at'}
                if fixture['status']=='completed' and all(fixture[k]==v for k,v in substantive.items()):
                    continue
                updates['status'] = 'completed'
                self._conn.execute(f'UPDATE fixtures_26_27 SET {",".join(k+"=?" for k in updates)} WHERE match_id=?', (*updates.values(),fixture['match_id']))
                changed += 1
            ledger_count = self._conn.execute('SELECT COUNT(*) FROM applied_results').fetchone()[0]
            completed_count = self._conn.execute("SELECT COUNT(*) FROM fixtures_26_27 WHERE status='completed'").fetchone()[0]
            if changed or ledger_count != completed_count:
                self._rebuild()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return changed

    def apply_schedule(self, rows):
        """Update dates/official rounds without changing forecast identity or payloads."""
        keys = {(r['home_team'], r['away_team']) for r in self._conn.execute('SELECT home_team,away_team FROM fixtures_26_27')}
        if len(rows) != len(keys) or {(r['home_team'], r['away_team']) for r in rows} != keys:
            raise ValueError('Official schedule identity differs from the persisted season; explicit review required')
        self._conn.execute('BEGIN IMMEDIATE')
        try:
            for row in rows:
                if row.get('schedule_source') != 'premier-league-fpl':
                    raise ValueError('Unverified schedule source')
                if row.get('gameweek') is not None and not 1 <= int(row['gameweek']) <= 38:
                    raise ValueError('Invalid official round')
                if row.get('match_date') and not '2026-07-01' <= row['match_date'] < '2027-07-01':
                    raise ValueError('Official fixture belongs to a different season')
                values = {k:row[k] for k in ('gameweek','match_date','kickoff_at','official_match_id','schedule_source','schedule_observed_at','schedule_status') if row.get(k) is not None}
                if row.get('kickoff_at') is None:
                    values['kickoff_at'] = None
                self._conn.execute(f'UPDATE fixtures_26_27 SET {",".join(k+"=?" for k in values)} WHERE home_team=? AND away_team=?',(*values.values(),row['home_team'],row['away_team']))
            # Schedule corrections to already completed fixtures affect chronology.
            self._rebuild()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def update_after_match(self, match_id, home_team, away_team, home_goals, away_goals, home_xg=None, away_xg=None, **stats):
        fixture = self._conn.execute('SELECT home_team,away_team FROM fixtures_26_27 WHERE match_id=?',(match_id,)).fetchone()
        if not fixture or tuple(fixture)!=(home_team,away_team):
            raise ValueError('Result identity does not match match_id')
        return self.apply_results([dict(home_team=home_team,away_team=away_team,home_goals=home_goals,away_goals=away_goals,home_xg=home_xg,away_xg=away_xg,**stats)])

    def record_market_quotes(self, rows):
        with self._conn:
            for row in rows:
                fixture = self._conn.execute('SELECT * FROM fixtures_26_27 WHERE home_team=? AND away_team=?',(row['home_team'],row['away_team'])).fetchone()
                if fixture is None or fixture['status']!='pending' or fixture['match_date']!=row['match_date']:
                    continue
                cutoff = fixture['kickoff_at'] or fixture['match_date']+'T00:00:00+00:00'
                observed = datetime.fromisoformat(row['odds_observed_at'])
                if observed.tzinfo is None or observed>=datetime.fromisoformat(cutoff) or observed>datetime.now(timezone.utc):
                    continue
                if normalized_market([row.get(k) for k in ('B365H','B365D','B365A')]) is None:
                    continue
                self._conn.execute('UPDATE fixtures_26_27 SET B365H=?,B365D=?,B365A=?,odds_observed_at=? WHERE match_id=?',
                                   (row['B365H'],row['B365D'],row['B365A'],row['odds_observed_at'],fixture['match_id']))

    def get_team_state(self, team):
        row = self._conn.execute('SELECT * FROM live_team_states WHERE team_name=?',(team,)).fetchone()
        if row is None:
            raise KeyError(team)
        state = dict(row)
        for k in state:
            if k.startswith('last_10_') or k == 'engine_state':
                state[k] = json.loads(state[k]) if state[k] else None
        return state

    def get_all_team_states(self):
        return [self.get_team_state(r[0]) for r in self._conn.execute('SELECT team_name FROM live_team_states ORDER BY current_elo DESC')]

    def get_pending_fixtures(self):
        return [dict(r) for r in self._conn.execute("SELECT * FROM fixtures_26_27 WHERE status='pending' ORDER BY match_date,match_id")]

    def add_fixture(self, gameweek, match_date, home_team, away_team):
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO fixtures_26_27(gameweek,match_date,home_team,away_team) VALUES (?,?,?,?)',(gameweek,match_date,home_team,away_team))
        return self._conn.execute('SELECT match_id FROM fixtures_26_27 WHERE home_team=? AND away_team=?',(home_team,away_team)).fetchone()[0]

    def lock_prediction(self, match_id, home_prob, draw_prob, away_prob, predicted_score, **metadata):
        probabilities = [home_prob,draw_prob,away_prob]
        if any(finite(v) is None or not 0 <= v <= 1 for v in probabilities) or abs(sum(probabilities)-1)>1e-6:
            raise ValueError('Invalid normalized forecast probabilities')
        fixture = self._conn.execute('SELECT * FROM fixtures_26_27 WHERE match_id=?',(match_id,)).fetchone()
        if fixture is None or fixture['status'] != 'pending':
            raise ValueError('Cannot lock forecasts for a completed/missing fixture')
        now = datetime.now(timezone.utc)
        cutoff = datetime.fromisoformat(fixture['kickoff_at']) if fixture['kickoff_at'] else datetime.fromisoformat(fixture['match_date']).replace(tzinfo=timezone.utc)
        if cutoff.tzinfo is None or now >= cutoff:
            raise ValueError('Forecast must be locked before kickoff (before match day if time unknown)')
        if metadata.get('feature_contract') != CONTRACT or not metadata.get('model_version') or not metadata.get('feature_hash'):
            raise ValueError('Verified model/feature provenance is required')
        features = json.loads(metadata.get('feature_values') or '{}')
        required = set(NDC_FEATURES) | set(XGB_BASE_FEATURES)
        if set(features) != required or any(finite(v) is None for v in features.values()) or digest(features) != metadata['feature_hash']:
            raise ValueError('The exact finite pre-match feature vector and its hash are required')
        state_as_of = datetime.fromisoformat(metadata.get('state_as_of') or '')
        if state_as_of.tzinfo is None or state_as_of > now:
            raise ValueError('Forecast state timestamp must precede the lock')
        if metadata.get('blend_alpha') is None or not 0 <= metadata['blend_alpha'] <= 1:
            raise ValueError('The forecast must record the blend weight it used')
        market = [metadata.get(f'market_{side}_prob') for side in ('home','draw','away')]
        if any(v is not None for v in market):
            observed = datetime.fromisoformat(metadata.get('market_observed_at') or '')
            if any(finite(v) is None or not 0 <= v <= 1 for v in market) or abs(sum(market)-1)>1e-6 or observed.tzinfo is None or observed>now:
                raise ValueError('Invalid or future bookmaker snapshot')
        for prefix in ('ndc','xgb'):
            vals = [metadata.get(f'{prefix}_{s}_prob') for s in ('home','draw','away')]
            if any(v is not None for v in vals) and (any(finite(v) is None or not 0<=v<=1 for v in vals) or abs(sum(vals)-1)>1e-6):
                raise ValueError('Invalid per-model probabilities')
        values = dict(match_id=match_id,predicted_home_prob=home_prob,predicted_draw_prob=draw_prob,predicted_away_prob=away_prob,predicted_score=predicted_score,locked_at=now.isoformat(),prediction_status='eligible',quality_flags='',**metadata)
        with self._conn:
            self._conn.execute(f'INSERT INTO predictions_26_27 ({",".join(values)}) VALUES ({",".join("?" for _ in values)}) ON CONFLICT(match_id) DO NOTHING',tuple(values.values()))

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
