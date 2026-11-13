"""Read-only audit of result/state integrity, model provenance and locked forecasts.

python scripts/audit_project.py --strict --output reports/latest_audit.json
Legacy quarantined forecasts are reported separately; they cannot earn accuracy.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse
from contextlib import closing
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from models.neural_dixon_coles import predict_scoreline_grid,parse_grid_outputs
from scripts.predict_next_gameweek import model_manifest,run_ndc_inference,run_xgb_inference
from src.model_registry import load_archived_assets
from src.data.state_manager import RESULT_COLUMNS,digest
from src.evaluation import model_metrics,eligible,score_candidates
from src.features.prematch import CONTRACT,FeatureEngine,NDC_FEATURES,XGB_BASE_FEATURES,replay_history


def audit(db_path=ROOT/'data/live/epl_2627.db'):
    findings=[]
    def finding(severity,code,message):findings.append({'severity':severity,'code':code,'message':message})
    with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)) as conn:
        conn.row_factory=sqlite3.Row
        integrity=conn.execute('PRAGMA integrity_check').fetchone()[0]
        foreign=[list(r) for r in conn.execute('PRAGMA foreign_key_check')]
        f=pd.read_sql_query('SELECT * FROM fixtures_26_27',conn)
        p=pd.read_sql_query('SELECT * FROM predictions_26_27',conn)
        states={r['team_name']:dict(r) for r in conn.execute('SELECT * FROM live_team_states')}
        ledger={r['match_id']:dict(r) for r in conn.execute('SELECT * FROM applied_results')}
        metadata=dict(conn.execute('SELECT key,value FROM pipeline_metadata'))
        run=conn.execute('SELECT observed_at,status,summary FROM ingestion_runs ORDER BY run_id DESC LIMIT 1').fetchone()
        legacy=[list(r) for r in conn.execute("SELECT match_id,predicted_home_prob,predicted_draw_prob,predicted_away_prob,predicted_score,ndc_lambda,ndc_mu,ndc_rho,locked_at,ndc_home_prob,ndc_draw_prob,ndc_away_prob,xgb_home_prob,xgb_draw_prob,xgb_away_prob FROM predictions_26_27 WHERE model_version='legacy-unverified' ORDER BY match_id")]
    if integrity!='ok' or foreign:finding('critical','database_integrity','SQLite integrity or foreign keys failed')
    if f.duplicated(['home_team','away_team']).any():finding('critical','duplicate_fixture','Duplicate canonical season fixtures')
    active=set(f.home_team)|set(f.away_team)
    if set(states)!=active:finding('high','state_coverage','Live state does not cover exactly the scheduled teams')
    complete=f[f.status=='completed'].copy()
    if complete[['home_goals','away_goals']].isna().any().any():finding('critical','missing_result','Completed fixtures lack scores')
    now=datetime.now(timezone.utc)
    overdue=f[(f.status=='pending') & (f.schedule_status!='postponed') & (f.match_date<=(now-timedelta(days=2)).date().isoformat())]
    if len(overdue):finding('high','overdue_results',f'{len(overdue)} overdue fixtures remain pending')
    if not run or run['status']!='ok' or (now-datetime.fromisoformat(run['observed_at']))>timedelta(hours=36):
        finding('medium','ingestion_freshness','Latest successful ingestion is missing, failed, or older than 36 hours')
    stats=['home_shots_on_target','away_shots_on_target','home_shots','away_shots','home_corners','away_corners','home_fouls','away_fouls']
    missing={key:int(complete[key].isna().sum()) for key in stats}
    if any(missing.values()):finding('medium','observed_stats_missing',f'Missing observations remain explicit: {missing}')
    try:
        engine=FeatureEngine(json.loads(metadata['base_engine']))
        baseline=replay_history(pd.read_csv(ROOT/'data/raw/epl_10_seasons_raw.csv'),before_season='2026/27')
        baseline.start_season('2026/27',active)
        if digest(baseline.snapshot()) != digest(engine.snapshot()):
            finding('critical','baseline_replay','Season seed differs from full prior-season replay')
        for row in complete.sort_values(['match_date','home_team','away_team']).to_dict('records'):
            observation={field:row.get(column) for field,column in RESULT_COLUMNS.items()}
            observation={k:(None if pd.isna(v) else v) for k,v in observation.items()}
            observation.update(home_team=row['home_team'],away_team=row['away_team'])
            engine.observe(observation)
            expected=digest({k:v for k,v in observation.items() if k!='observed_at'})
            if row['match_id'] not in ledger or ledger[row['match_id']]['result_hash']!=expected:
                finding('critical','result_ledger_hash',f"Applied-result hash mismatch for {row['match_id']}")
        for team in active:
            expected=engine.teams[team]
            stored=json.loads(states[team]['engine_state'])
            if expected!=stored or abs(expected['elo']-states[team]['current_elo'])>1e-8:
                finding('critical','state_replay',f'{team} state differs from chronological replay')
        if len(ledger)!=len(complete):finding('critical','result_ledger_count','Applied result ledger count differs from completed fixtures')
    except Exception as exc:finding('critical','state_replay_error',str(exc))
    if metadata.get('legacy_forecast_hash')!=digest(legacy):finding('high','legacy_forecasts_changed','Original forecast payloads changed after migration')
    try:
        manifest=model_manifest()
        features=pd.read_csv(ROOT/'data/processed/epl_model_features.csv')
        if not features.Feature_Contract.eq(CONTRACT).all():raise ValueError('Historical dataset feature contract mismatch')
        if not np.isfinite(features[NDC_FEATURES+XGB_BASE_FEATURES].values).all():raise ValueError('Non-finite model features')
        if hashlib.sha256((ROOT/'data/processed/epl_model_features.csv').read_bytes()).hexdigest()!=manifest['data_sha256']:
            raise ValueError('Historical feature data changed since model training')
        if hashlib.sha256((ROOT/'data/raw/epl_10_seasons_raw.csv').read_bytes()).hexdigest()!=manifest['raw_data_sha256']:
            raise ValueError('Historical raw dataset changed since model training')
        if set(NDC_FEATURES)&{'Home_GoalsScored','Away_GoalsScored','FTHG','FTAG','B365H','B365D','B365A'}:
            raise ValueError('Forbidden target/market columns in neural inputs')
    except Exception as exc:
        manifest=None;finding('critical','model_bundle',str(exc))
    merged=f.merge(p,on='match_id',how='inner')
    checked=0
    assets_by_version={}
    for row in merged.to_dict('records'):
        if row.get('prediction_status')!='eligible':continue
        checked+=1
        if not eligible(row):finding('high','forecast_timing',f"Forecast {row['match_id']} fails pre-kickoff provenance")
        try:
            vector=json.loads(row['feature_values'])
            if digest(vector)!=row['feature_hash'] or set(vector)!=(set(NDC_FEATURES)|set(XGB_BASE_FEATURES)):
                raise ValueError('Input vector/hash mismatch')
            stamp=json.loads(row['model_manifest'])
            if stamp['feature_contract']!=CONTRACT or stamp['model_version']!=row['model_version']:
                raise ValueError('Model manifest mismatch')
            archived=json.loads((ROOT/'models/versions'/row['model_version']/'manifest.json').read_text())
            if stamp != archived:
                raise ValueError('Locked manifest differs from the immutable archived manifest')
            config=json.loads((ROOT/'models/versions'/row['model_version']/'models/bin/blend_weights.json').read_text())
            if not np.isfinite(row['blend_alpha']) or not 0<=row['blend_alpha']<=1 or abs(row['blend_alpha']-config['ndc_weight'])>1e-12:
                raise ValueError('Forecast blend weight differs from its recorded model version')
            if row['blend_version']!=stamp['artifacts']['models/bin/blend_weights.json'][:16]:
                raise ValueError('Forecast blend version differs from its archived manifest')
            if row['model_version'] not in assets_by_version:
                assets_by_version[row['model_version']]=load_archived_assets(row['model_version'])
            ndc_model,scaler,names,xgb_model=assets_by_version[row['model_version']]
            neural=run_ndc_inference(ndc_model,scaler,np.array([[vector[k] for k in names]],dtype='float32'))
            xgb_actual=run_xgb_inference(xgb_model,np.array([[vector[k] for k in XGB_BASE_FEATURES]],dtype='float32'))
            if max(abs(neural[name]-row[column]) for name,column in (('lambda','ndc_lambda'),('mu','ndc_mu'),('rho','ndc_rho')))>1e-5:
                raise ValueError('Saved neural outputs differ from model inference on the locked input vector')
            if max(abs(actual-row[f'xgb_{side}_prob']) for actual,side in zip(xgb_actual,('home','draw','away')))>1e-6:
                raise ValueError('Saved XGBoost outputs differ from model inference on the locked input vector')
            grid=predict_scoreline_grid(row['ndc_lambda'],row['ndc_mu'],row['ndc_rho'],max_goals=12)
            parsed=parse_grid_outputs(grid,top_k=3)
            pn=np.array([parsed[k]/100 for k in ('home_win','draw','away_win')])
            stored=np.array([row[f'ndc_{side}_prob'] for side in ('home','draw','away')])
            px=np.array([row[f'xgb_{side}_prob'] for side in ('home','draw','away')])
            final=np.array([row[f'predicted_{side}_prob'] for side in ('home','draw','away')])
            if any(not np.isfinite(v).all() or (v<0).any() or (v>1).any() or not np.isclose(v.sum(),1,atol=1e-6) for v in (stored,px,final)):
                raise ValueError('Invalid probability distribution')
            blended=row['blend_alpha']*stored+(1-row['blend_alpha'])*px
            if np.max(abs(pn-stored))>1e-7 or np.max(abs(blended-final))>1e-7:
                raise ValueError('Probabilities disagree with saved parameters/blend')
            if score_candidates(row['predicted_score'])!=[r['scoreline'] for r in parsed['top_scorelines']]:
                raise ValueError('Scorelines disagree with saved parameters')
        except Exception as exc:finding('high','forecast_payload',f"Forecast {row['match_id']}: {exc}")
    missing_quotes = int(sum(r.get('prediction_status')=='eligible' and pd.isna(r.get('market_observed_at')) for r in merged.to_dict('records')))
    if missing_quotes:
        finding('low','market_snapshot_coverage',f'{missing_quotes} eligible forecasts have no contemporaneous bookmaker quote; excluded from the common-cohort market race')
    legacy_count=int(p.prediction_status.eq('quarantined').sum())
    if legacy_count:finding('info','legacy_quarantine',f'{legacy_count} preserved legacy forecasts are excluded from prospective accuracy')
    unknown_kickoffs=int(f.loc[f.status=='pending','kickoff_at'].isna().sum())
    if unknown_kickoffs:finding('low','future_kickoff_coverage',f'{unknown_kickoffs} future fixtures lack verified kickoff times; locks stop at start of match day')
    legacy_schedule=int(f.schedule_source.eq('legacy-import').sum())
    if legacy_schedule:finding('medium','schedule_provenance',f'{legacy_schedule} round assignments were imported from the old schedule; verify against an authoritative full-season schedule')
    if not manifest or manifest['training']['test_season']!='2025/26':finding('high','test_protocol','Expected fixed final test season unavailable')
    return {'audited_at':now.isoformat(),'feature_contract':CONTRACT,'sqlite_integrity':integrity,'fixtures':len(f),'completed':len(complete),
            'pending':int(f.status.eq('pending').sum()),'overdue':len(overdue),'missing_observed_stats':missing,
            'state_replay_checked_teams':len(states),'applied_results':len(ledger),'locked_forecasts':len(p),
            'verified_forecasts_checked':checked,'legacy_quarantined':legacy_count,'legacy_payloads_unchanged':metadata.get('legacy_forecast_hash')==digest(legacy),
            'model_version':manifest['model_version'] if manifest else None,'common_cohort_metrics':model_metrics(merged),
            'model_only_metrics':model_metrics(merged, matched_market=False),
            'findings':findings,'passed':not any(r['severity'] in ('critical','high','medium') for r in findings)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--db',type=Path,default=ROOT/'data/live/epl_2627.db');parser.add_argument('--output',type=Path);parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    try:report=audit(args.db)
    except Exception as exc:report={'passed':False,'findings':[{'severity':'critical','code':'audit_failed','message':str(exc)}]}
    result=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(result)
    else:print(result)
    if args.strict and not report['passed']:sys.exit(1)
