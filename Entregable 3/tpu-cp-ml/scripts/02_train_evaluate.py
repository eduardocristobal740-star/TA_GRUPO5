"""Paso 2: compara modelos y conjuntos de variables con validación angular.

Uso:  python scripts/02_train_evaluate.py [--tune]
Salidas (reports/tables): predicciones_cv.csv, metricas_por_pliegue.csv, resumen_modelos.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp.config import PROCESSED_DIR, TAB_DIR, TARGETS
from tpu_cp.evaluation import fold_metrics, summarize
from tpu_cp.features import BASE_FEATURES, PHYSICS_FEATURES, add_features
from tpu_cp.models import PARAM_GRIDS, base_estimators
from tpu_cp.validation import hybrid_cv, logo_cv, sensor_interpolation_cv

FEATURE_SETS = {"base": BASE_FEATURES, "fisica": PHYSICS_FEATURES}


def main(tune: bool) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = add_features(pd.read_csv(PROCESSED_DIR / "tpu_cp_table.csv"))
    preds, folds, rows = [], [], []

    def register(pred, target, model, fset):
        pred = pred.assign(objetivo=target, modelo=model, variables=fset)
        fm = fold_metrics(pred).assign(objetivo=target, modelo=model, variables=fset)
        preds.append(pred); folds.append(fm)
        rows.append({"objetivo": target, "modelo": model, "variables": fset, **summarize(fm, pred)})

    for target in TARGETS:
        # Baselines por sensor (no usan variables; usan la historia angular del sensor)
        for kind, label in [("linear", "Interp. angular lineal"), ("pchip", "Interp. angular PCHIP")]:
            register(sensor_interpolation_cv(df, target, kind), target, label, "sensor")
        for fset, feats in FEATURE_SETS.items():
            for name, est in base_estimators().items():
                t0 = time.time()
                register(logo_cv(df, target, est, feats, name), target, name, fset)
                if tune and fset == "fisica" and name in PARAM_GRIDS:
                    register(logo_cv(df, target, est, feats, name, tune=True), target, name + " (ajustado)", fset)
                if name in ("ExtraTrees", "HistGradientBoosting"):   # híbrido: variables + interpolación del sensor
                    register(hybrid_cv(df, target, est, feats), target, "Híbrido " + name, fset)
                print(f"{target:8s} {fset:7s} {name:22s} {time.time()-t0:5.1f}s", flush=True)

    pd.concat(preds).to_csv(TAB_DIR / "predicciones_cv.csv", index=False)
    pd.concat(folds).to_csv(TAB_DIR / "metricas_por_pliegue.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB_DIR / "resumen_modelos.csv", index=False)
    print(summary.sort_values(["objetivo", "MAE"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", action="store_true")
    main(ap.parse_args().tune)
