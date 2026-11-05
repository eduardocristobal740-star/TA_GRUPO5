import argparse, json
from pathlib import Path
import numpy as np
from .io import load_any

def describe(v):
    if hasattr(v,'shape'): return {'type':type(v).__name__, 'shape':list(v.shape), 'dtype':str(getattr(v,'dtype',''))}
    return {'type':type(v).__name__}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    inv={}
    for f in Path(a.data_dir).rglob('*'):
        if f.suffix.lower()=='.mat':
            try: inv[str(f)]={k:describe(v) for k,v in load_any(f).items()}
            except Exception as e: inv[str(f)]={'error':str(e)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(inv,indent=2),encoding='utf-8')
if __name__=='__main__': main()
