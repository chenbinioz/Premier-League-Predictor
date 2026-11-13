"""Train reproducible, pre-match-only models with disjoint seasonal stages.

Train <=2022/23; calibration/early stopping 2023/24; blend 2024/25;
untouched evaluation 2025/26. No test-driven tuning or automatic promotion.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score,log_loss
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from xgboost import XGBClassifier
from models.neural_dixon_coles import NeuralDixonColes,DixonColesLoss,predict_scoreline_grid,parse_grid_outputs
from src.features.build_features import build_features
from src.features.prematch import CONTRACT,NDC_FEATURES,XGB_BASE_FEATURES,MARKET_FEATURES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(frame):return np.where(frame.FTHG>frame.FTAG,2,np.where(frame.FTHG==frame.FTAG,1,0))


def score(y,p):
    p=np.array(p,dtype=float,copy=True);p/=p.sum(axis=1,keepdims=True)
    return {'n':len(y),'accuracy':float(accuracy_score(y,2-p[:,::-1].argmax(axis=1))),
            'log_loss':float(log_loss(y,p,labels=[0,1,2])),
            'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))}


def ndc_probs(model,scaler,frame):
    values=scaler.transform(frame[NDC_FEATURES]).astype('float32')
    with torch.no_grad():lam,mu,rho=model(torch.from_numpy(values))
    out=[]
    for l,m,r in zip(lam.numpy(),mu.numpy(),rho.numpy()):
        p=parse_grid_outputs(predict_scoreline_grid(float(l),float(m),float(r),max_goals=12))
        out.append([p['away_win']/100,p['draw']/100,p['home_win']/100])
    return np.array(out)


def train(epochs=50):
    if epochs < 1:raise ValueError("Training requires at least one epoch")
    from src.model_registry import archive_bundle
    archive_bundle()
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(1)
    df=build_features().sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if not np.isfinite(df[NDC_FEATURES].values).all():raise ValueError('Pre-match contract must produce finite features')
    train_df=df[df.Season<='2022/23'];cal=df[df.Season=='2023/24'];val=df[df.Season=='2024/25'];test=df[df.Season=='2025/26']
    if any(len(d)==0 for d in (train_df,cal,val,test)):raise ValueError('All four temporal stages are required')
    out=ROOT/'models/bin';out.mkdir(exist_ok=True,parents=True)
    scaler=StandardScaler().fit(train_df[NDC_FEATURES])
    xt=torch.tensor(scaler.transform(train_df[NDC_FEATURES]),dtype=torch.float32)
    xc=torch.tensor(scaler.transform(cal[NDC_FEATURES]),dtype=torch.float32)
    yh=torch.tensor(train_df.FTHG.values,dtype=torch.float32);ya=torch.tensor(train_df.FTAG.values,dtype=torch.float32)
    ch=torch.tensor(cal.FTHG.values,dtype=torch.float32);ca=torch.tensor(cal.FTAG.values,dtype=torch.float32)
    model=NeuralDixonColes(len(NDC_FEATURES));loss_fn=DixonColesLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');best_state=None;patience=0;best_epoch=0
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(xt)).split(64):
            optimizer.zero_grad();loss=loss_fn(*model(xt[indices]),yh[indices],ya[indices]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():validation=float(loss_fn(*model(xc),ch,ca))
        if validation<best:
            best=validation;best_epoch=epoch+1;best_state={k:v.clone() for k,v in model.state_dict().items()};patience=0
        else:patience+=1
        if patience>=8:break
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':model.state_dict(),'n_features':len(NDC_FEATURES),'feature_contract':CONTRACT,
                'best_val_nll':best,'best_epoch':best_epoch,'train_end':str(train_df.Date.max()),'calibration_season':'2023/24'},out/'neural_dixon_coles.pt')
    joblib.dump(scaler,out/'ndc_scaler.joblib');(out/'ndc_feature_names.json').write_text(json.dumps(NDC_FEATURES,indent=2)+'\n')
    classifiers={}
    for name,features,path in [('market_free',XGB_BASE_FEATURES,ROOT/'models/calibrated_xgb_outcome.pkl'),
                               ('market_aware',XGB_BASE_FEATURES+MARKET_FEATURES,out/'market_xgb.joblib')]:
        fit=train_df.dropna(subset=features);calibration=cal.dropna(subset=features)
        base=XGBClassifier(n_estimators=120,max_depth=2,learning_rate=.015,reg_alpha=1.5,reg_lambda=2.5,subsample=.75,colsample_bytree=.75,objective='multi:softprob',num_class=3,random_state=42,n_jobs=1)
        base.fit(fit[features],labels(fit))
        # Calibrator sees only later matches, unweighted natural outcome frequencies.
        classifier=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid')
        classifier.fit(calibration[features],labels(calibration))
        joblib.dump(classifier,path);classifiers[name]=(classifier,features)
    p_nv=ndc_probs(model,scaler,val);p_xv=classifiers['market_free'][0].predict_proba(val[XGB_BASE_FEATURES])
    fitted=minimize_scalar(lambda a:log_loss(labels(val),a*p_nv+(1-a)*p_xv,labels=[0,1,2]),bounds=(0,1),method='bounded')
    alpha=float(fitted.x)
    blend={'feature_contract':CONTRACT,'method':'scalar-validation-log-loss','ndc_weight':alpha,'xgb_weight':1-alpha,
           'train_end':str(train_df.Date.max()),'calibration_season':'2023/24','validation_season':'2024/25','test_season':'2025/26'}
    (out/'blend_weights.json').write_text(json.dumps(blend,indent=2)+'\n')
    report={'feature_contract':CONTRACT,'splits':{'train':len(train_df),'calibration':len(cal),'blend_validation':len(val),'test':len(test)},'ndc_best_epoch':best_epoch,'evaluation':{}}
    for name,subset in [('validation',val),('test',test)]:
        y=labels(subset);pn=ndc_probs(model,scaler,subset);px=classifiers['market_free'][0].predict_proba(subset[XGB_BASE_FEATURES])
        evaluated={'ndc':score(y,pn),'xgb_market_free':score(y,px),'blend':score(y,alpha*pn+(1-alpha)*px)}
        common=subset.dropna(subset=MARKET_FEATURES)
        market=common[['Bookie_Prob_A','Bookie_Prob_D','Bookie_Prob_H']].values
        evaluated['bookmaker']=score(labels(common),market)
        evaluated['xgb_market_aware']=score(labels(common),classifiers['market_aware'][0].predict_proba(common[XGB_BASE_FEATURES+MARKET_FEATURES]))
        report['evaluation'][name]=evaluated
    from scripts.train_static_poisson import fit_static_poisson
    joblib.dump(fit_static_poisson(train_df),ROOT/'models/dc_mle_params.pkl')
    artifacts=['models/dc_mle_params.pkl','models/calibrated_xgb_outcome.pkl','models/bin/market_xgb.joblib','models/bin/neural_dixon_coles.pt','models/bin/ndc_scaler.joblib','models/bin/ndc_feature_names.json','models/bin/blend_weights.json']
    manifest={'feature_contract':CONTRACT,'environment':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','xgboost','joblib','torch')},
              'data_sha256':sha(ROOT/'data/processed/epl_model_features.csv'),'raw_data_sha256':sha(ROOT/'data/raw/epl_10_seasons_raw.csv'),
              'code_sha256':{p:sha(ROOT/p) for p in ('src/features/prematch.py','models/neural_dixon_coles.py','scripts/train_models.py','scripts/train_static_poisson.py','src/features/build_features.py','scripts/predict_next_gameweek.py','src/model_registry.py')},
              'artifacts':{p:sha(ROOT/p) for p in artifacts},'training':blend,'test_evaluated_once_per_run':True}
    manifest['model_version']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive_bundle()
    (ROOT/'reports/training_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=50);args=parser.parse_args();train(args.epochs)
