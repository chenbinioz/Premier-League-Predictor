"""Eligibility and common-cohort evaluation shared by UI and audit."""
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from src.features.prematch import normalized_market, CONTRACT


def score_candidates(value):
    if value is None or pd.isna(value):
        return []
    return [s.strip() for s in re.split(r'\s*[•|/]\s*',str(value)) if s.strip()][:3]


def eligible(row):
    if row.get('prediction_status')!='eligible' or row.get('feature_contract')!=CONTRACT:
        return False
    try:
        locked=pd.Timestamp(row['locked_at'])
        cutoff=pd.Timestamp(row.get('kickoff_at')) if pd.notna(row.get('kickoff_at')) else pd.Timestamp(row['match_date'],tz='UTC')
        if locked.tzinfo is None or cutoff.tzinfo is None or locked>=cutoff:return False
        as_of=pd.Timestamp(row.get('state_as_of'))
        if as_of.tzinfo is None or as_of>locked:return False
        return all(pd.notna(row.get(k)) for k in ('model_version','feature_hash','blend_version','state_as_of'))
    except (TypeError,ValueError,KeyError):
        return False


def market_triplet(row):
    if row.get('feature_contract') == CONTRACT:
        values=[row.get(f'market_{side}_prob') for side in ('home','draw','away')]
        try:
            observed=pd.Timestamp(row['market_observed_at'])
            locked=pd.Timestamp(row['locked_at'])
            if observed.tzinfo is None or observed>locked:return None
            p=np.array(values,dtype=float)
            if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-6):return None
            return p.tolist()
        except (TypeError,ValueError,KeyError):return None
    return normalized_market([row.get(k) for k in ('B365H','B365D','B365A')])


def common_cohort(frame, require_market=True):
    if frame.empty:return frame.copy()
    mask=frame.apply(eligible,axis=1)
    mask &= frame.status.eq('completed')
    for prefix in ('predicted','ndc','xgb'):
        cols=[f'{prefix}_{s}_prob' for s in ('home','draw','away')]
        if not set(cols).issubset(frame):return frame.iloc[:0].copy()
        p=frame[cols].to_numpy(dtype=float)
        mask &= np.isfinite(p).all(axis=1) & (p>=0).all(axis=1) & (p<=1).all(axis=1) & np.isclose(p.sum(axis=1),1,atol=1e-6)
    if require_market:
        mask &= frame.apply(lambda r:market_triplet(r) is not None,axis=1)
    return frame[mask].copy()


def tracker_coverage(frame):
    """Explain an empty leaderboard without confusing it with missing results."""
    if frame.empty:
        return {'historical_completed':0,'verified_pending':0,'verified_completed':0,'matched_completed':0}
    verified=frame.apply(eligible,axis=1)
    completed=frame.status.eq('completed')
    return {
        'historical_completed':int((completed & ~verified).sum()),
        'verified_pending':int((verified & frame.status.eq('pending')).sum()),
        'verified_completed':len(common_cohort(frame,require_market=False)),
        'matched_completed':len(common_cohort(frame)),
    }


def model_metrics(frame, matched_market=True):
    common=common_cohort(frame, require_market=matched_market)
    output={}
    models=[('blend','predicted'),('ndc','ndc'),('xgb','xgb')]
    if matched_market:
        models.append(('bookie',None))
    for name,prefix in models:
        if common.empty:
            output[name]={'hits':0,'total':0,'acc':None,'loss':None}
            continue
        p=(np.array([market_triplet(r) for r in common.to_dict('records')]) if prefix is None else
           common[[f'{prefix}_{s}_prob' for s in ('home','draw','away')]].to_numpy(dtype=float))
        p=p/p.sum(axis=1,keepdims=True)
        y=np.where(common.home_goals>common.away_goals,0,np.where(common.home_goals==common.away_goals,1,2))
        hits=int((p.argmax(axis=1)==y).sum())
        output[name]={'hits':hits,'total':len(y),'acc':100*hits/len(y),'loss':float(log_loss(y,p,labels=[0,1,2]))}
    return output
