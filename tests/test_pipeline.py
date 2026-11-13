"""Regression tests for audited data, forecasting and evaluation invariants."""
from contextlib import closing

import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from models.neural_dixon_coles import predict_scoreline_grid
from src.data import live_fetchers as fetchers
from src.data.state_manager import TeamStateManager,base_signature
from src.evaluation import score_candidates,model_metrics,eligible
from src.features.prematch import FeatureEngine,NDC_FEATURES,XGB_BASE_FEATURES,CONTRACT,normalized_market,shot_proxy
ROOT=Path(__file__).resolve().parents[1]


class ReplayComparisonTests(unittest.TestCase):
    def test_platform_roundoff_is_distinct_from_state_corruption(self):
        from scripts.audit_project import replay_states_equal
        expected = {'elo': 1700.0, 'history': [{'date': '2026-11-01', 'goals': 2, 'proxy': 1.81}]}
        rounded = copy.deepcopy(expected)
        rounded['elo'] += 1e-12
        self.assertTrue(replay_states_equal(expected, rounded))
        for field, value in [('elo', 1700.00000001), ('history', [])]:
            changed = {**expected, field: value}
            self.assertFalse(replay_states_equal(expected, changed))
        changed = copy.deepcopy(expected)
        changed['history'][0]['date'] = '2026-11-02'
        self.assertFalse(replay_states_equal(expected, changed))
        changed['history'][0]['date'] = '2026-11-01'
        changed['history'][0]['goals'] = 3
        self.assertFalse(replay_states_equal(expected, changed))


class FeatureTests(unittest.TestCase):
    def engine(self):
        e=FeatureEngine();e.start_season('2026/27',{'Arsenal','Chelsea'});return e

    def test_no_target_or_current_match_statistics(self):
        e=self.engine();before=e.features('Arsenal','Chelsea','2026-08-21')
        self.assertEqual(before['Home_GoalsScored_roll3'],1.35)
        self.assertNotIn('Home_GoalsScored',NDC_FEATURES)
        self.assertNotIn('B365H',NDC_FEATURES)
        for goals in (0,8):
            branch=self.engine()
            vector=branch.features('Arsenal','Chelsea','2026-08-21')
            branch.observe(dict(home_team='Arsenal',away_team='Chelsea',home_goals=goals,away_goals=1,match_date='2026-08-21'))
            self.assertEqual(before,vector)
        self.assertEqual(len(NDC_FEATURES),len(set(NDC_FEATURES)))

    def test_serving_historical_parity(self):
        from scripts.predict_next_gameweek import _features
        e=self.engine();e.observe(dict(home_team='Arsenal',away_team='Chelsea',home_goals=2,away_goals=1,match_date='2026-08-21',home_sot=5,away_sot=3,home_shots=12,away_shots=9))
        expected=e.features('Chelsea','Arsenal','2026-08-28')
        actual=_features({'engine_state':e.teams['Chelsea']},{'engine_state':json.dumps(e.teams['Arsenal'])},'2026-08-28')
        self.assertEqual(expected,actual)
        with self.assertRaises(ValueError):e.features('Arsenal','Chelsea','2026-08-21')

    def test_missing_observations_remain_missing(self):
        e=self.engine();e.observe(dict(home_team='Arsenal',away_team='Chelsea',home_goals=2,away_goals=0,match_date='2026-08-21'))
        self.assertIsNone(e.teams['Arsenal']['history'][-1]['ShotsOnTarget'])
        self.assertEqual(e.features('Arsenal','Chelsea','2026-08-28')['Home_ShotsOnTarget_roll3'],4.4)
        self.assertIsNone(shot_proxy(None,12));self.assertAlmostEqual(shot_proxy(5,12),1.81)
        self.assertIsNone(normalized_market([np.nan,3,4]))
        self.assertIsNone(normalized_market([1,3,4]))

    def test_venue_and_season_transition(self):
        e=self.engine();e.observe(dict(home_team='Arsenal',away_team='Chelsea',home_goals=4,away_goals=1,match_date='2026-08-21',home_sot=8,away_sot=2,home_shots=15,away_shots=6))
        previous=e.teams['Arsenal']['elo'];e.start_season('2027/28',{'Arsenal','Hull'})
        self.assertAlmostEqual(e.teams['Arsenal']['elo'],.8*previous+300)
        self.assertEqual(e.teams['Hull']['elo'],1420)
        self.assertTrue(np.isfinite(list(e.features('Arsenal','Hull','2027-08-21').values())).all())
        e.start_season('2028/29',{'Chelsea','Hull'})
        self.assertEqual(e.teams['Chelsea']['history'],[])

    def test_distribution_domain_and_normalization(self):
        for lam,mu,rho in [(1.5,1.2,-.1),(8,6,-.25),(8,6,.25)]:
            grid=predict_scoreline_grid(lam,mu,rho,max_goals=12)
            self.assertGreaterEqual(grid.min(),0);self.assertAlmostEqual(float(grid.sum()),1)
        with self.assertRaises(ValueError):predict_scoreline_grid(float('nan'),1,0)


class FetcherTests(unittest.TestCase):
    def test_dates(self):
        for raw in ('02/11/2026','2026-11-02','2026-11-02T15:00:00Z'):
            self.assertEqual(fetchers._normalise_date(raw),'2026-11-02')
        self.assertEqual(fetchers._normalise_date('not-a-date'),'')

    def test_missing_goals_are_not_results(self):
        frame=pd.DataFrame([{'Date':'02/11/2026','HomeTeam':'Arsenal','AwayTeam':'Chelsea','FTHG':np.nan,'FTAG':0}])
        with patch.object(fetchers.pd,'read_csv',return_value=frame):
            self.assertEqual(fetchers.fetch_from_football_data_csv('local.csv'),[])

    def test_partial_sources_merge_and_rounds_are_canonical(self):
        csv=[dict(home_team='Arsenal',away_team='Chelsea',home_goals=1,away_goals=0,match_date='2026-11-02',status='completed')]
        under=[dict(home_team='Hull',away_team='Ipswich',home_goals=2,away_goals=2,match_date='2026-11-01',status='completed')]
        fixtures=[dict(home_team='Arsenal',away_team='Chelsea',gameweek=9,match_id=99)]
        with patch.object(fetchers,'fetch_from_football_data_csv',return_value=csv),patch.object(fetchers,'_read_understat_matches',return_value=under):
            self.assertEqual(len(fetchers.fetch_results()),2)
            results=fetchers.fetch_weekend_results(9,fixtures=fixtures)
            self.assertEqual(len(results),1);self.assertEqual(results[0]['match_id'],99)

    def test_conflicting_sources_fail(self):
        a=dict(home_team='Arsenal',away_team='Chelsea',home_goals=1,away_goals=0,match_date='2026-11-02',status='completed')
        b={**a,'home_goals':2}
        with patch.object(fetchers,'fetch_from_football_data_csv',return_value=[a]),patch.object(fetchers,'_read_understat_matches',return_value=[b]):
            with self.assertRaises(ValueError):fetchers.fetch_results()


class StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory();cls.template=Path(cls.folder.name)/'template.db'
        # Minimal schema independent of the checked-in database's current version.
        with closing(sqlite3.connect(cls.template)) as conn, conn:
            conn.executescript((ROOT/'src/db/schema.sql').read_text())
            conn.executemany('INSERT INTO fixtures_26_27(gameweek,match_date,home_team,away_team) VALUES (?,?,?,?)',[(1,'2026-08-21','Arsenal','Chelsea'),(2,'2099-08-28','Chelsea','Arsenal')])
        with TeamStateManager(cls.template) as sm:
            e=FeatureEngine();e.start_season('2026/27',{'Arsenal','Chelsea'})
            with sm._conn:sm._conn.execute('INSERT INTO pipeline_metadata VALUES (?,?)',('base_engine',json.dumps(e.snapshot())))
            with sm._conn:sm._conn.execute('INSERT INTO pipeline_metadata VALUES (?,?)',('base_engine_signature',base_signature()))
            sm.rebuild_states()

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'test.db'
        with closing(sqlite3.connect(self.template)) as src,closing(sqlite3.connect(self.path)) as dest:src.backup(dest)
        self.sm=TeamStateManager(self.path)
        self.result=dict(home_team='Arsenal',away_team='Chelsea',home_goals=2,away_goals=1,match_date='2026-08-21')

    def tearDown(self):self.sm.close();self.temp.cleanup()

    def test_duplicate_result_and_enrichment(self):
        self.assertEqual(self.sm.apply_results([self.result]),1)
        before=self.sm.get_team_state('Arsenal')
        self.assertEqual(self.sm.apply_results([self.result]),0)
        self.assertEqual(before,self.sm.get_team_state('Arsenal'))
        self.sm.apply_results([{**self.result,'home_sot':5,'home_shots':12}])
        self.assertEqual(before['current_elo'],self.sm.get_team_state('Arsenal')['current_elo'])
        self.assertEqual(self.sm.get_team_state('Arsenal')['last_10_sot_for'],[5.0])

    def test_opponent_pressure_uses_observed_values(self):
        self.sm.apply_results([{**self.result,'home_sot':5,'away_sot':3,'home_corners':6,'away_corners':2,'home_fouls':10,'away_fouls':12}])
        self.assertEqual(self.sm.get_team_state('Arsenal')['last_10_sot_against'],[3])
        self.assertEqual(self.sm.get_team_state('Chelsea')['last_10_corners_against'],[6])
        self.assertEqual(self.sm.get_team_state('Arsenal')['avg_fouls_against'],12)

    def test_atomic_failure_rolls_back_both_teams_and_result(self):
        before=self.sm.get_team_state('Arsenal')
        original=self.sm._write_team_state
        def fail(team,state):
            if team=='Chelsea':raise RuntimeError('injected failure')
            return original(team,state)
        with patch.object(self.sm,'_write_team_state',side_effect=fail):
            with self.assertRaises(RuntimeError):self.sm.apply_results([self.result])
        self.assertEqual(before,self.sm.get_team_state('Arsenal'))
        self.assertEqual(self.sm._conn.execute('select status from fixtures_26_27 where match_id=1').fetchone()[0],'pending')
        self.assertEqual(self.sm._conn.execute('select count(*) from applied_results').fetchone()[0],0)

    def test_correction_replays_instead_of_double_applying(self):
        self.sm.apply_results([self.result]);self.sm.apply_results([{**self.result,'home_goals':0}])
        e=FeatureEngine();e.start_season('2026/27',{'Arsenal','Chelsea'});e.observe({**self.result,'home_goals':0})
        self.assertEqual(self.sm.get_team_state('Arsenal')['current_elo'],e.teams['Arsenal']['elo'])
        self.assertEqual(len(self.sm.get_team_state('Arsenal')['last_10_goals_for']),1)

    def record(self):
        from scripts.predict_next_gameweek import _features
        from src.data.state_manager import digest
        features=_features(self.sm.get_team_state('Chelsea'),self.sm.get_team_state('Arsenal'),'2099-08-28')
        return dict(feature_values=json.dumps(features),blend_alpha=.5,match_id=2,home_prob=.5,draw_prob=.25,away_prob=.25,predicted_score='1 - 0',model_version='test-version',feature_contract=CONTRACT,feature_hash=digest(features),blend_version='test-blend',state_as_of=datetime.now(timezone.utc).isoformat())

    def test_locks_are_immutable_and_late_locks_rejected(self):
        record=self.record();self.sm.lock_prediction(**record)
        self.sm.lock_prediction(**{**record,'home_prob':.6,'away_prob':.15})
        self.assertEqual(self.sm._conn.execute('select predicted_home_prob from predictions_26_27 where match_id=2').fetchone()[0],.5)
        with self.assertRaises(ValueError):self.sm.lock_prediction(**{**record,'match_id':1})
        with self.assertRaises(ValueError):self.sm.lock_prediction(**{**record,'home_prob':float('nan')})

    def test_dry_run_does_not_migrate_or_write(self):
        from scripts.update_live_state import reconcile
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch('scripts.update_live_state.fetch_results',return_value=[self.result]):
            report=reconcile(self.path,dry_run=True)
        self.assertEqual(report['status'],'dry-run')
        self.assertEqual(before,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_initialization_refuses_existing_database(self):
        from src.db.init_live_db import initialize
        with self.assertRaises(FileExistsError):initialize(self.path,'unused.csv')

    def test_official_schedule_changes_dates_without_replacing_locks(self):
        self.sm.lock_prediction(**self.record())
        original=list(self.sm._conn.execute('SELECT predicted_home_prob,locked_at FROM predictions_26_27 WHERE match_id=2').fetchone())
        rows=[]
        for home,away,gw,day in [('Arsenal','Chelsea',1,'2026-08-21'),('Chelsea','Arsenal',10,'2026-11-08')]:
            rows.append(dict(home_team=home,away_team=away,gameweek=gw,match_date=day,kickoff_at=day+'T16:30:00+00:00',official_match_id=str(gw),schedule_source='premier-league-fpl',schedule_observed_at=datetime.now(timezone.utc).isoformat(),schedule_status='scheduled'))
        self.sm.apply_schedule(rows)
        self.assertEqual(self.sm._conn.execute('SELECT gameweek FROM fixtures_26_27 WHERE match_id=2').fetchone()[0],10)
        self.assertEqual(original,list(self.sm._conn.execute('SELECT predicted_home_prob,locked_at FROM predictions_26_27 WHERE match_id=2').fetchone()))

    def test_failed_collection_is_recorded_and_raised(self):
        from scripts.update_live_state import reconcile
        with patch('scripts.update_live_state.fetch_official_schedule',side_effect=RuntimeError('source unavailable')):
            with self.assertRaises(RuntimeError):reconcile(self.path)
        self.assertEqual(self.sm._conn.execute('SELECT status FROM ingestion_runs ORDER BY run_id DESC LIMIT 1').fetchone()[0],'failed')

    def test_auto_selection_skips_postponed_matches(self):
        from scripts.predict_next_gameweek import get_auto_gameweek
        with self.sm._conn:self.sm._conn.execute("UPDATE fixtures_26_27 SET schedule_status='postponed' WHERE match_id=2")
        self.assertIsNone(get_auto_gameweek(self.path))

    def test_cross_season_results_rejected(self):
        with self.assertRaises(ValueError):self.sm.apply_results([{**self.result,'match_date':'2025-08-21'}])


class EvaluationTests(unittest.TestCase):
    def test_tracker_explains_pending_forecasts_and_missing_quotes(self):
        from src.evaluation import tracker_coverage
        row=dict(status='pending',home_goals=None,away_goals=None,match_date='2099-11-01',locked_at='2026-11-04T00:00:00+00:00',prediction_status='eligible',feature_contract=CONTRACT,model_version='v2',feature_hash='hash',blend_version='blend',state_as_of='2026-11-03T00:00:00+00:00')
        for prefix in ('predicted','ndc','xgb'):
            row.update({f'{prefix}_home_prob':.5,f'{prefix}_draw_prob':.25,f'{prefix}_away_prob':.25})
        legacy={**row,'status':'completed','prediction_status':'quarantined','home_goals':2,'away_goals':1}
        coverage=tracker_coverage(pd.DataFrame([legacy,row,row]))
        self.assertEqual(coverage,dict(historical_completed=1,verified_pending=2,verified_completed=0,matched_completed=0))
        completed={**row,'status':'completed','home_goals':2,'away_goals':1}
        coverage=tracker_coverage(pd.DataFrame([legacy,row,completed]))
        self.assertEqual(coverage['verified_completed'],1)
        self.assertEqual(coverage['matched_completed'],0)

    def test_three_scores_parse_individually(self):
        self.assertIn('2 - 1',score_candidates('1 - 0 • 2 - 1 • 0 - 0'))

    def test_all_models_share_cohort_and_legacy_excluded(self):
        row=dict(status='completed',home_goals=2,away_goals=1,match_date='2026-11-01',locked_at='2026-10-30T00:00:00+00:00',prediction_status='eligible',feature_contract=CONTRACT,model_version='v2',feature_hash='hash',blend_version='blend',state_as_of='2026-10-29T00:00:00+00:00',B365H=2,B365D=3,B365A=4,market_home_prob=.5,market_draw_prob=.25,market_away_prob=.25,market_observed_at='2026-10-29T00:00:00+00:00')
        for prefix in ('predicted','ndc','xgb'):
            row.update({f'{prefix}_home_prob':.5,f'{prefix}_draw_prob':.25,f'{prefix}_away_prob':.25})
        rows=[row,{**row,'prediction_status':'quarantined'},{**row,'market_home_prob':np.nan}]
        metrics=model_metrics(pd.DataFrame(rows))
        self.assertEqual({v['total'] for v in metrics.values()},{1})
        self.assertFalse(eligible({**row,'locked_at':'2026-11-02T00:00:00+00:00'}))
        no_quote=pd.DataFrame([{**row,'market_home_prob':np.nan}])
        self.assertEqual({v['total'] for v in model_metrics(no_quote,matched_market=False).values()},{1})
        self.assertEqual({v['total'] for v in model_metrics(no_quote).values()},{0})

    def test_post_match_odds_do_not_become_pre_match_snapshots(self):
        from src.evaluation import market_triplet
        row=dict(feature_contract=CONTRACT,locked_at='2026-10-30T00:00:00+00:00',market_home_prob=.5,market_draw_prob=.25,market_away_prob=.25,market_observed_at='2026-11-02T00:00:00+00:00',B365H=2,B365D=3,B365A=4)
        self.assertIsNone(market_triplet(row))


class AuditorTests(unittest.TestCase):
    def test_auditor_detects_state_and_forecast_tampering(self):
        from scripts.audit_project import audit
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tampered.db'
            with closing(sqlite3.connect(ROOT/'data/live/epl_2627.db')) as source, closing(sqlite3.connect(path)) as dest:
                source.backup(dest)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("UPDATE live_team_states SET current_elo=current_elo+1 WHERE team_name='Arsenal'")
                row=conn.execute("SELECT match_id FROM predictions_26_27 WHERE prediction_status='eligible' LIMIT 1").fetchone()
                if row is None:self.skipTest('No verified forecast exists in this checkout')
                conn.execute("UPDATE predictions_26_27 SET predicted_score='99 - 99' WHERE match_id=?",(row[0],))
            report=audit(path)
            codes={finding['code'] for finding in report['findings']}
            self.assertIn('state_replay',codes)
            self.assertIn('forecast_payload',codes)
            self.assertFalse(report['passed'])


if __name__=='__main__':unittest.main()
