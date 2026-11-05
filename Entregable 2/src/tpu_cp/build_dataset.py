import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from .io import load_any, choose, scalar

def aliases(cfg,k): return cfg['aliases'].get(k,[k])
def mat_to_frame(path,cfg):
    m=load_any(path); cp=np.asarray(choose(m,aliases(cfg,'cp')),dtype=float)
    if cp.ndim==1: cp=cp[:,None]
    # Convención: filas temporales, columnas sensores. Si hay menos filas que columnas, se transpone.
    if cp.shape[0] < cp.shape[1]: cp=cp.T
    n=cp.shape[1]
    out={}
    for k in ['H','B','D','theta']:
        v=choose(m,aliases(cfg,k)); arr=np.asarray(v).squeeze()
        out[k]=np.repeat(arr.item(),n) if arr.size==1 else arr
    for k in ['X','Y','Z']:
        out[k]=np.asarray(choose(m,aliases(cfg,k))).squeeze()
    sid=choose(m,aliases(cfg,'sensor_id'),False)
    gid=choose(m,aliases(cfg,'geometry_id'),False)
    out['sensor_id']=np.arange(n) if sid is None else np.asarray(sid).squeeze()
    out['geometry_id']=path.stem if gid is None else str(np.asarray(gid).squeeze())
    out['experiment_id']=path.stem
    out['H_B']=np.asarray(out['H'],float)/np.asarray(out['B'],float)
    out['cp_mean']=np.nanmean(cp,axis=0)
    out['cp_rms']=np.nanstd(cp,axis=0,ddof=1)
    out['cp_min']=np.nanmin(cp,axis=0)
    out['cp_max']=np.nanmax(cp,axis=0)
    frame=pd.DataFrame(out)
    if len(frame)!=n: raise ValueError(f'Dimensiones incompatibles en {path}')
    return frame

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--config',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    frames=[]
    for f in Path(a.data_dir).rglob('*'):
        if f.name==Path(a.out).name: continue
        if f.suffix.lower() in {'.csv','.parquet'}:
            frames.append(load_any(f))
        elif f.suffix.lower()=='.mat': frames.append(mat_to_frame(f,cfg))
    if not frames: raise FileNotFoundError('No hay archivos .mat, .csv o .parquet en data/')
    df=pd.concat(frames,ignore_index=True)
    needed=cfg['features']+cfg['targets']+[cfg['group_column']]
    missing=[c for c in needed if c not in df]
    if missing: raise KeyError(f'Columnas faltantes: {missing}')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_parquet(a.out,index=False)
    print(f'{len(df):,} filas -> {a.out}')
if __name__=='__main__': main()
