"""Compatibility entry point: retrain the coherent v2 bundle, never mix generations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.features.prematch import XGB_BASE_FEATURES,CONTRACT
FEATURE_COLS=XGB_BASE_FEATURES


def prepare_features(df_raw):
    from scripts.train_models import labels
    df=df_raw.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if 'Feature_Contract' not in df or not (df.Feature_Contract==CONTRACT).all():
        raise ValueError('Rebuild pre-match v2 features before training')
    return df[FEATURE_COLS].copy(),__import__('pandas').Series(labels(df),index=df.index)


if __name__=='__main__':
    from scripts.train_models import train
    train()
