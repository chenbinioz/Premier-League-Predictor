"""Retain model versions so immutable forecasts remain reproducible after retraining."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def archive_bundle():
    manifest_path=ROOT/'models/bin/manifest.json'
    if not manifest_path.exists():
        return
    manifest=json.loads(manifest_path.read_text())
    parent=ROOT/'models/versions';parent.mkdir(exist_ok=True)
    destination=parent/manifest['model_version']
    if destination.exists():
        return
    with tempfile.TemporaryDirectory(dir=parent,prefix='.archive-') as temporary:
        stage=Path(temporary)/'bundle';stage.mkdir()
        for relative,expected in {**manifest['artifacts'],**manifest['code_sha256']}.items():
            source=ROOT/relative
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                # Old weights remain usable; edited source can be recovered from
                # its recorded hash/Git history, never misrepresented as original.
                if relative in manifest['artifacts']:
                    raise ValueError(f'Cannot archive corrupted model artifact: {relative}')
                continue
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        shutil.copyfile(manifest_path,stage/'manifest.json')
        stage.rename(destination)


def load_archived_assets(version):
    """Load only checksum-verified weights, never execute archived source files."""
    if len(version)!=16 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('Invalid model version')
    base=ROOT/'models/versions'/version
    manifest=json.loads((base/'manifest.json').read_text())
    identity=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='model_version'},sort_keys=True).encode()).hexdigest()[:16]
    if identity!=version or manifest['model_version']!=version:
        raise ValueError('Archived manifest identity mismatch')
    for package,expected in manifest['environment'].items():
        if importlib.metadata.version(package)!=expected:
            raise ValueError(f'Archived model requires {package} {expected}')
    for relative,expected in manifest['artifacts'].items():
        if hashlib.sha256((base/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Archived artifact checksum mismatch')
    architecture='models/neural_dixon_coles.py'
    if hashlib.sha256((ROOT/architecture).read_bytes()).hexdigest()!=manifest['code_sha256'][architecture]:
        raise ValueError('Archived model requires its original architecture/environment')
    import joblib
    import torch
    from models.neural_dixon_coles import NeuralDixonColes
    checkpoint=torch.load(base/'models/bin/neural_dixon_coles.pt',map_location='cpu',weights_only=True)
    names=json.loads((base/'models/bin/ndc_feature_names.json').read_text())
    model=NeuralDixonColes(len(names));model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,joblib.load(base/'models/bin/ndc_scaler.joblib'),names,joblib.load(base/'models/calibrated_xgb_outcome.pkl')
