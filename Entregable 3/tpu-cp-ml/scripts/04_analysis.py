"""Paso 4: selección, figuras de resultados, importancia (hold-out) y estabilidad PCS."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp import plotting as P
from tpu_cp.config import PROCESSED_DIR, RANDOM_STATE, TAB_DIR, TARGETS
from tpu_cp.evaluation import metrics
from tpu_cp.features import PHYSICS_FEATURES, add_features
from tpu_cp.models import make_pipeline

df = add_features(pd.read_csv(PROCESSED_DIR / "tpu_cp_table.csv"))
summary = pd.read_csv(TAB_DIR / "resumen_modelos.csv")
pred = pd.read_csv(TAB_DIR / "predicciones_cv.csv"); folds = pd.read_csv(TAB_DIR / "metricas_por_pliegue.csv")

# Mejor modelo por objetivo (excluye la mediana). Nota: seleccionar y reportar con el mismo CV es optimista.
cand = summary[summary.modelo != "Dummy (mediana)"]
best = cand.sort_values("MAE").groupby("objetivo", as_index=False).first()
base = summary[summary.modelo == "Dummy (mediana)"].drop_duplicates("objetivo")[["objetivo", "MAE", "RMSE"]]
comp = best.merge(base, on="objetivo", suffixes=("", "_base"))
comp["mejora_MAE_pct"] = 100 * (1 - comp.MAE / comp.MAE_base); comp["mejora_RMSE_pct"] = 100 * (1 - comp.RMSE / comp.RMSE_base)
comp.to_csv(TAB_DIR / "mejor_por_objetivo.csv", index=False)
print(comp[["objetivo", "modelo", "variables", "MAE", "RMSE", "R2", "mejora_MAE_pct", "mejora_RMSE_pct"]].round(4))


# --- Importancia por permutación en el pliegue RESERVADO (modelo ExtraTrees, variables físicas) ---
imps = []
for t in TARGETS:
    for tr, te in LeaveOneGroupOut().split(df, df[t], df.theta):
        pipe = make_pipeline(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
                             PHYSICS_FEATURES).fit(df.iloc[tr][PHYSICS_FEATURES], df.iloc[tr][t])
        r = permutation_importance(pipe, df.iloc[te][PHYSICS_FEATURES], df.iloc[te][t], n_repeats=10,
                                   scoring="neg_mean_absolute_error", random_state=RANDOM_STATE)
        imps.append(pd.DataFrame({"objetivo": t, "theta": df.iloc[te].theta.iloc[0], "variable": PHYSICS_FEATURES,
                                  "delta_MAE": r.importances_mean}))
imp = pd.concat(imps).groupby(["objetivo", "variable"], as_index=False).delta_MAE.mean()
imp.to_csv(TAB_DIR / "importancia_permutacion_holdout.csv", index=False)

# --- Estabilidad PCS: bootstrap de sensores, semillas y ruido en el objetivo (ExtraTrees, variables físicas) ---
rng = np.random.default_rng(RANDOM_STATE); rows = []
def run(t, mode, seed):
    ys, ps = [], []
    for tr, te in LeaveOneGroupOut().split(df, df[t], df.theta):
        d_tr = df.iloc[tr]; y = d_tr[t].to_numpy().copy()
        if mode == "bootstrap_sensores":
            ids = rng.choice(d_tr.sensor_id.unique(), size=d_tr.sensor_id.nunique(), replace=True)
            d_tr = pd.concat([d_tr[d_tr.sensor_id == i] for i in ids]); y = d_tr[t].to_numpy()
        elif mode == "ruido_5pct":
            y = y + rng.normal(0, 0.05 * df[t].std(), len(y))
        m = make_pipeline(ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=-1),
                          PHYSICS_FEATURES).fit(d_tr[PHYSICS_FEATURES], y)
        ys.append(df.iloc[te][t].to_numpy()); ps.append(m.predict(df.iloc[te][PHYSICS_FEATURES]))
    return metrics(np.concatenate(ys), np.concatenate(ps))
for t in TARGETS:
    for mode, B in [("semillas", 8), ("bootstrap_sensores", 12), ("ruido_5pct", 8)]:
        for b in range(B):
            rows.append({"objetivo": t, "perturbacion": mode, "rep": b, **run(t, mode, RANDOM_STATE + b)})
stab = pd.DataFrame(rows); stab.to_csv(TAB_DIR / "estabilidad_replicas.csv", index=False)
agg = stab.groupby(["objetivo", "perturbacion"]).agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), R2=("R2", "mean"), R2_sd=("R2", "std")).reset_index()
agg.to_csv(TAB_DIR / "estabilidad_resumen.csv", index=False); print(agg.round(4))
