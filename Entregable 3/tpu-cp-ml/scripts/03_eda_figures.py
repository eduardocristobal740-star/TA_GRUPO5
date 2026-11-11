"""Paso 3: figuras exploratorias (no dependen de los modelos)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp import plotting as P
from tpu_cp.config import PROCESSED_DIR, TAB_DIR

df = pd.read_csv(PROCESSED_DIR / "tpu_cp_table.csv")
P.fig_heatmaps(df); P.fig_distributions(df); P.fig_net_maps(df, "cp_mean"); P.fig_net_maps(df, "cp_min")
df.groupby("theta")[["cp_mean", "cp_rms", "cp_min", "cp_max"]].mean().round(4).to_csv(TAB_DIR / "promedio_por_angulo.csv")
df.groupby("surface")[["cp_mean", "cp_rms", "cp_min", "cp_max"]].agg(["mean", "min", "max"]).round(3).to_csv(TAB_DIR / "estadisticos_por_superficie.csv")
print("EDA OK")
