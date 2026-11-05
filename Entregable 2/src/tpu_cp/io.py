from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat

def load_any(path):
    path=Path(path)
    if path.suffix.lower()=='.csv': return pd.read_csv(path)
    if path.suffix.lower()=='.parquet': return pd.read_parquet(path)
    if path.suffix.lower()!='.mat': raise ValueError(f'Formato no admitido: {path}')
    try:
        return {k:v for k,v in loadmat(path, squeeze_me=True, struct_as_record=False).items() if not k.startswith('__')}
    except NotImplementedError:
        import mat73
        return mat73.loadmat(path)

def choose(mapping, aliases, required=True):
    for key in aliases:
        if key in mapping: return mapping[key]
    if required: raise KeyError(f'No se encontró ninguno de: {aliases}')
    return None

def scalar(v):
    a=np.asarray(v).squeeze()
    if a.size!=1: raise ValueError('Se esperaba un escalar')
    return a.item()
