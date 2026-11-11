"""Métricas por pliegue y resumen agregado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (max_error, mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "MedAE": median_absolute_error(y, p),
            "ErrMax": max_error(y, p), "Sesgo": float(np.mean(p - y))}


def fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Métricas por ángulo reservado a partir de las predicciones fuera de muestra."""
    return pd.DataFrame([{"theta": th, **metrics(g.observado, g.predicho)}
                         for th, g in pred.groupby("theta")])


def summarize(fm: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Resumen: media±desv. entre pliegues, métricas agrupadas y por tipo de pliegue."""
    interior = fm[fm.theta.isin([15, 30, 45, 60, 75])]; borde = fm[fm.theta.isin([0, 90])]
    pooled = metrics(pred.observado, pred.predicho)
    return {"MAE": fm.MAE.mean(), "MAE_sd": fm.MAE.std(), "RMSE": fm.RMSE.mean(), "RMSE_sd": fm.RMSE.std(),
            "R2": fm.R2.mean(), "R2_sd": fm.R2.std(), "R2_agrupado": pooled["R2"],
            "ErrMax": fm.ErrMax.max(), "Sesgo": fm.Sesgo.mean(),
            "MAE_interior": interior.MAE.mean(), "MAE_borde": borde.MAE.mean()}
