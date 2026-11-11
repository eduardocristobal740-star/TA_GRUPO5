"""Validación angular (LeaveOneGroupOut por theta) para modelos y baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import LeaveOneGroupOut

from .models import PARAM_GRIDS, make_pipeline, make_tuned


def _interp_one(theta_tr, y_tr, theta_te, kind):
    """Interpola/extrapola y(theta) de UN sensor hasta theta_te."""
    order = np.argsort(theta_tr); th, yy = theta_tr[order], y_tr[order]
    if kind == "pchip":
        return float(PchipInterpolator(th, yy, extrapolate=True)(theta_te))
    # lineal con extrapolación lineal usando los dos puntos extremos más cercanos
    if theta_te < th[0]:
        i, j = 0, 1
    elif theta_te > th[-1]:
        i, j = -2, -1
    else:
        return float(np.interp(theta_te, th, yy))
    return float(yy[i] + (yy[j] - yy[i]) * (theta_te - th[i]) / (th[j] - th[i]))


def sensor_interpolation_cv(df: pd.DataFrame, target: str, kind: str) -> pd.DataFrame:
    """Baseline 'consciente del sensor': predice cada sensor a partir de SU PROPIA
    respuesta en los ángulos de entrenamiento (interpolación angular)."""
    out = []
    for th_te in sorted(df.theta.unique()):
        tr, te = df[df.theta != th_te], df[df.theta == th_te]
        piv = tr.pivot(index="sensor_id", columns="theta", values=target)
        pred = [_interp_one(piv.columns.to_numpy(float), piv.loc[s].to_numpy(float), th_te, kind)
                for s in te.sensor_id]
        out.append(pd.DataFrame({"theta": th_te, "sensor_id": te.sensor_id.to_numpy(),
                                 "observado": te[target].to_numpy(), "predicho": pred}))
    return pd.concat(out, ignore_index=True)


def logo_cv(df: pd.DataFrame, target: str, estimator, features: list[str], name: str,
            tune: bool = False) -> pd.DataFrame:
    """LeaveOneGroupOut por ángulo. Devuelve predicciones fuera de muestra."""
    X, y, g = df[features], df[target], df["theta"]
    out = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if tune and name in PARAM_GRIDS:
            model = make_tuned(estimator, PARAM_GRIDS[name], features)
            model.fit(X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
        else:
            model = make_pipeline(estimator, features).fit(X.iloc[tr], y.iloc[tr])
        out.append(pd.DataFrame({"theta": g.iloc[te].to_numpy(), "sensor_id": df.sensor_id.iloc[te].to_numpy(),
                                 "observado": y.iloc[te].to_numpy(), "predicho": model.predict(X.iloc[te])}))
    return pd.concat(out, ignore_index=True)


def _interp_matrix(piv: pd.DataFrame, sensors, theta_te: float, kind: str) -> np.ndarray:
    """Predicción de interpolación angular para los sensores dados en theta_te."""
    th = piv.columns.to_numpy(float)
    return np.array([_interp_one(th, piv.loc[s].to_numpy(float), theta_te, kind) for s in sensors])


def hybrid_cv(df: pd.DataFrame, target: str, estimator, features: list[str], residual: bool = False,
              kind: str = "linear") -> pd.DataFrame:
    """Modelo híbrido: variables geométricas + interpolación angular del propio sensor.

    Para evitar fuga, la variable de interpolación de una fila de entrenamiento con
    ángulo a se calcula SIN usar el ángulo a (interpolación 'leave-one-angle-out'
    dentro del entrenamiento), imitando lo que ocurre en la fila de prueba.
    Si residual=True el árbol aprende y - interpolación y se suma de nuevo al predecir.
    """
    out = []
    for th_te in sorted(df.theta.unique()):
        tr, te = df[df.theta != th_te].copy(), df[df.theta == th_te].copy()
        piv_tr = tr.pivot(index="sensor_id", columns="theta", values=target)
        tr["interp"] = np.nan
        for a in piv_tr.columns:
            m = tr.theta == a
            tr.loc[m, "interp"] = _interp_matrix(piv_tr.drop(columns=a), tr.loc[m, "sensor_id"], a, kind)
        te["interp"] = _interp_matrix(piv_tr, te.sensor_id, th_te, kind)
        feats = features + ["interp"]
        y_tr = tr[target] - tr["interp"] if residual else tr[target]
        model = make_pipeline(estimator, feats).fit(tr[feats], y_tr)
        pred = model.predict(te[feats]) + (te["interp"].to_numpy() if residual else 0.0)
        out.append(pd.DataFrame({"theta": th_te, "sensor_id": te.sensor_id.to_numpy(),
                                 "observado": te[target].to_numpy(), "predicho": pred}))
    return pd.concat(out, ignore_index=True)
