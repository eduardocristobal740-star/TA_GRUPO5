import argparse, json, yaml, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def metric(y,p): return {'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5,'r2':r2_score(y,p)}
def stability(model,X,feature_names):
    rows=[]; base=model.predict(X)
    for col,deltas in {'theta':[-5,-1,1,5], 'X':[-.01,-.005,.005,.01], 'Y':[-.01,-.005,.005,.01], 'Z':[-.01,-.005,.005,.01]}.items():
        if col not in feature_names: continue
        j=feature_names.index(col)
        scale=1.0 if col=='theta' else max(float(np.nanmax(X[:,j])-np.nanmin(X[:,j])),1e-12)
        for d in deltas:
            xp=X.copy(); xp[:,j]+=d*scale if col!='theta' else d
            diff=np.abs(model.predict(xp)-base)
            rows.append({'feature':col,'perturbation':d,'mean_abs_change':float(np.mean(diff)),'p95_abs_change':float(np.quantile(diff,.95))})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--config',required=True); p.add_argument('--out-dir',required=True); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); df=pd.read_parquet(a.data).dropna(subset=cfg['targets'])
    feats=cfg['features']; X=df[feats].to_numpy(float); groups=df[cfg['group_column']].astype(str).to_numpy()
    splitter=GroupShuffleSplit(n_splits=1,test_size=cfg.get('test_size',.2),random_state=cfg.get('random_state',42))
    tr,te=next(splitter.split(X,groups=groups)); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    all_metrics={}
    for target in cfg['targets']:
        y=df[target].to_numpy(float)
        candidates={
          'dummy':DummyRegressor(strategy='median'),
          'rf':RandomForestRegressor(n_estimators=400,min_samples_leaf=2,n_jobs=-1,random_state=42),
          'extra_trees':ExtraTreesRegressor(n_estimators=400,min_samples_leaf=2,n_jobs=-1,random_state=42)}
        cv=GroupKFold(n_splits=min(cfg.get('cv_splits',5),len(np.unique(groups[tr]))))
        scores={}
        for name,reg in candidates.items():
            pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('model',reg)])
            scores[name]=float(-cross_val_score(pipe,X[tr],y[tr],groups=groups[tr],cv=cv,scoring='neg_mean_absolute_error',n_jobs=-1).mean())
        best=min(scores,key=scores.get); model=Pipeline([('impute',SimpleImputer(strategy='median')),('model',candidates[best])]).fit(X[tr],y[tr])
        pred=model.predict(X[te]); metrics=metric(y[te],pred); metrics.update({'selected':best,'cv_mae':scores})
        imp=permutation_importance(model,X[te],y[te],scoring='neg_mean_absolute_error',n_repeats=15,random_state=42,n_jobs=-1)
        pd.DataFrame({'feature':feats,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False).to_csv(out/f'importance_{target}.csv',index=False)
        pd.DataFrame({'observed':y[te],'predicted':pred,'group':groups[te]}).to_csv(out/f'predictions_{target}.csv',index=False)
        pd.DataFrame(stability(model,X[te].copy(),feats)).to_csv(out/f'stability_{target}.csv',index=False)
        joblib.dump(model,out/f'{target}.joblib'); all_metrics[target]=metrics
    (out/'metrics.json').write_text(json.dumps(all_metrics,indent=2),encoding='utf-8')
if __name__=='__main__': main()
