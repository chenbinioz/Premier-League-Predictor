"""Generate immutable, versioned pre-kickoff forecasts from the persisted schedule."""
from __future__ import annotations
from contextlib import closing
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(variable,'1')
import argparse
from datetime import datetime,timedelta,timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import torch
from models.neural_dixon_coles import NeuralDixonColes,predict_scoreline_grid,parse_grid_outputs
from src.data.state_manager import TeamStateManager,digest
from src.features.prematch import FeatureEngine,CONTRACT,NDC_FEATURES,XGB_BASE_FEATURES,normalized_market
DEFAULT_DB_PATH=ROOT/'data/live/epl_2627.db'
XGB_FEATURE_COLS=XGB_BASE_FEATURES


def _verified_manifest(manifest_text):
    manifest=json.loads(manifest_text)
    identity=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='model_version'},sort_keys=True).encode()).hexdigest()[:16]
    if identity!=manifest['model_version']:raise ValueError('Model manifest identity mismatch')
    if manifest['feature_contract']!=CONTRACT:raise ValueError('Model feature contract mismatch; retrain models')
    for package,version in manifest['environment'].items():
        if importlib.metadata.version(package)!=version:raise ValueError(f'{package} version mismatch: install requirements.txt or retrain')
    for relative,expected in {**manifest['artifacts'],**manifest['code_sha256']}.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise ValueError(f'Model bundle file changed: {relative}; retrain the coherent bundle')
    return manifest


def model_manifest():
    path=ROOT/'models/bin/manifest.json'
    if not path.exists():raise ValueError('Verified model bundle missing; run scripts/train_models.py')
    return _verified_manifest(path.read_text())


def _load_ndc():
    model_manifest()
    checkpoint=torch.load(ROOT/'models/bin/neural_dixon_coles.pt',map_location='cpu',weights_only=True)
    names=json.loads((ROOT/'models/bin/ndc_feature_names.json').read_text())
    if names!=NDC_FEATURES or checkpoint.get('feature_contract')!=CONTRACT:raise ValueError('Unsafe neural feature contract')
    model=NeuralDixonColes(len(names));model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,joblib.load(ROOT/'models/bin/ndc_scaler.joblib'),names


def _load_xgb():
    model_manifest()
    return joblib.load(ROOT/'models/calibrated_xgb_outcome.pkl')


def _features(home_state,away_state,match_date=None):
    if match_date is None:
        match_date=(datetime.now(timezone.utc).date()+timedelta(days=1)).isoformat()
    def decode(state):
        value=state.get('engine_state')
        if not value:raise ValueError('Team state requires v2 chronological rebuild')
        return json.loads(value) if isinstance(value,str) else value
    engine=FeatureEngine({'season':'2026/27','active':['home','away'],'teams':{'home':decode(home_state),'away':decode(away_state)}})
    return engine.features('home','away',match_date)


def build_ndc_feature_vector(home_state,away_state,feature_names,*unused,match_date=None,**kwargs):
    if feature_names!=NDC_FEATURES:raise ValueError('Leaked or unknown feature names are not accepted')
    features=_features(home_state,away_state,match_date)
    return np.array([[features[k] for k in feature_names]],dtype='float32')


def build_xgb_feature_vector(home_state,away_state,*unused,match_date=None,**kwargs):
    features=_features(home_state,away_state,match_date)
    return np.array([[features[k] for k in XGB_FEATURE_COLS]],dtype='float32')


def run_ndc_inference(model,scaler,feature_vec):
    values=pd.DataFrame(feature_vec,columns=NDC_FEATURES)
    values=scaler.transform(values).astype('float32')
    with torch.no_grad():lam,mu,rho=model(torch.from_numpy(values))
    l,m,r=float(lam.item()),float(mu.item()),float(rho.item())
    grid=predict_scoreline_grid(l,m,r,max_goals=12)
    return {'lambda':l,'mu':m,'rho':r,'grid':grid,'parsed':parse_grid_outputs(grid,top_k=3)}


def run_xgb_inference(model,feature_vec):
    if list(model.feature_names_in_)!=XGB_FEATURE_COLS:raise ValueError('Primary XGBoost must be market-free')
    p=model.predict_proba(pd.DataFrame(feature_vec,columns=XGB_FEATURE_COLS))[0]
    return float(p[2]),float(p[1]),float(p[0])


def _blend_probs(nh,nd,na,xh,xd,xa,alpha=None):
    if alpha is None:alpha=json.loads((ROOT/'models/bin/blend_weights.json').read_text())['ndc_weight']
    if not 0<=alpha<=1:raise ValueError('Blend weight must be in [0,1]')
    p=float(alpha)*np.array([nh,nd,na])+(1-float(alpha))*np.array([xh,xd,xa])
    p/=p.sum()
    return tuple(float(v) for v in p)


def snapshot(home_state,away_state,match_date=None,assets=None):
    ndc_model,scaler,names,xgb=assets or (*_load_ndc(),_load_xgb())
    nv=build_ndc_feature_vector(home_state,away_state,names,match_date=match_date)
    xv=build_xgb_feature_vector(home_state,away_state,match_date=match_date)
    n=run_ndc_inference(ndc_model,scaler,nv)
    np_=tuple(n['parsed'][k]/100 for k in ('home_win','draw','away_win'))
    xp=run_xgb_inference(xgb,xv)
    return n,np_,xp,_blend_probs(*np_,*xp),digest(_features(home_state,away_state,match_date))


def get_auto_gameweek(db_path):
    now=datetime.now(timezone.utc)
    with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)) as conn, conn:
        conn.row_factory=sqlite3.Row
        pending=[dict(r) for r in conn.execute("SELECT * FROM fixtures_26_27 WHERE status='pending' AND schedule_status!='postponed' ORDER BY match_date")]
    for row in pending:
        cutoff=datetime.fromisoformat(row['kickoff_at']) if row.get('kickoff_at') else datetime.fromisoformat(row['match_date']).replace(tzinfo=timezone.utc)
        if cutoff>now:return row['gameweek']
    return None


def predict_gameweek(gameweek,db_path,dry_run=False,no_fetch=False,blend=None,force=False):
    if force:raise ValueError('Overwriting locked forecasts is disabled; use a simulation instead')
    if blend is not None:raise ValueError('Production locks must use the validated blend configuration')
    manifest=model_manifest();assets=(*_load_ndc(),_load_xgb())
    with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)) as conn, conn:
        conn.row_factory=sqlite3.Row
        contract=conn.execute("SELECT value FROM pipeline_metadata WHERE key='feature_contract'").fetchone()
        if not contract or contract[0]!=CONTRACT:raise ValueError('Live state has not been rebuilt with the trained contract')
        latest=conn.execute('SELECT observed_at,status FROM ingestion_runs ORDER BY run_id DESC LIMIT 1').fetchone()
        if not latest or latest['status']!='ok' or datetime.now(timezone.utc)-datetime.fromisoformat(latest['observed_at'])>timedelta(hours=36):
            raise ValueError('A recent successful result/schedule reconciliation is required before locking forecasts')
        # Do not publish forecasts from an incomplete results state.
        overdue=conn.execute("SELECT COUNT(*) FROM fixtures_26_27 WHERE status='pending' AND schedule_status!='postponed' AND match_date < ?",((datetime.now(timezone.utc)-timedelta(days=2)).date().isoformat(),)).fetchone()[0]
        if overdue:raise ValueError('Overdue fixtures remain; reconcile results before prediction')
        locked={r[0] for r in conn.execute('SELECT match_id FROM predictions_26_27')}
        fixtures=[dict(r) for r in conn.execute("SELECT * FROM fixtures_26_27 WHERE gameweek=? AND status='pending' AND schedule_status!='postponed' ORDER BY match_date,match_id",(gameweek,))]
        states={r['team_name']:dict(r) for r in conn.execute('SELECT * FROM live_team_states')}
    predictions=[]
    for fix in fixtures:
        if fix['match_id'] in locked:continue
        now=datetime.now(timezone.utc)
        cutoff=datetime.fromisoformat(fix['kickoff_at']) if fix.get('kickoff_at') else datetime.fromisoformat(fix['match_date']).replace(tzinfo=timezone.utc)
        if now>=cutoff:continue
        hs,aws=states[fix['home_team']],states[fix['away_team']]
        n,np_,xp,p,feature_hash=snapshot(hs,aws,fix['match_date'],assets)
        record=dict(match_id=fix['match_id'],home_prob=p[0],draw_prob=p[1],away_prob=p[2],
                    predicted_score=' • '.join(s['scoreline'] for s in n['parsed']['top_scorelines']),
                    ndc_home_prob=np_[0],ndc_draw_prob=np_[1],ndc_away_prob=np_[2],xgb_home_prob=xp[0],xgb_draw_prob=xp[1],xgb_away_prob=xp[2],
                    ndc_lambda=n['lambda'],ndc_mu=n['mu'],ndc_rho=n['rho'],model_version=manifest['model_version'],feature_contract=CONTRACT,
                    blend_version=manifest['artifacts']['models/bin/blend_weights.json'][:16],feature_hash=feature_hash,
                    state_as_of=max(hs['last_updated'],aws['last_updated']),feature_values=json.dumps(_features(hs,aws,fix['match_date']),sort_keys=True),
                    blend_alpha=json.loads((ROOT/'models/bin/blend_weights.json').read_text())['ndc_weight'],model_manifest=json.dumps(manifest,sort_keys=True))
        market=normalized_market([fix.get(k) for k in ('B365H','B365D','B365A')])
        if market and fix.get('odds_observed_at') and datetime.fromisoformat(fix['odds_observed_at']) <= now:
            record.update(market_home_prob=market[0],market_draw_prob=market[1],market_away_prob=market[2],market_observed_at=fix['odds_observed_at'])
        predictions.append(record)
    if not dry_run and predictions:
        with TeamStateManager(db_path) as sm:
            for prediction in predictions:sm.lock_prediction(**prediction)
    print(json.dumps({'gameweek':gameweek,'dry_run':dry_run,'forecasts':len(predictions),'model_version':manifest['model_version']},indent=2))
    return predictions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=DEFAULT_DB_PATH);parser.add_argument('--gameweek','-g',type=int)
    parser.add_argument('--auto',action='store_true');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--no-fetch',action='store_true');parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    gw=get_auto_gameweek(args.db) if args.auto else args.gameweek
    if gw is None:
        if not args.auto:parser.error('Specify --auto or --gameweek')
        print('No future pending gameweek')
    else:predict_gameweek(gw,args.db,args.dry_run,args.no_fetch,force=args.force)
