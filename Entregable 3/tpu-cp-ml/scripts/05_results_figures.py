"""Paso 5: figuras de resultados a partir de las tablas ya calculadas."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp import plotting as P
from tpu_cp.config import PROCESSED_DIR, TAB_DIR
from tpu_cp.features import add_features

df = add_features(pd.read_csv(PROCESSED_DIR / "tpu_cp_table.csv"))
summary = pd.read_csv(TAB_DIR / "resumen_modelos.csv"); pred = pd.read_csv(TAB_DIR / "predicciones_cv.csv")
folds = pd.read_csv(TAB_DIR / "metricas_por_pliegue.csv"); best = pd.read_csv(TAB_DIR / "mejor_por_objetivo.csv")
P.fig_comparison(summary); P.fig_obs_pred(pred, best); P.fig_by_angle(folds, best); P.fig_error_map(df, pred, best)
print("Figuras de resultados OK")
