"""Ingeniería de variables físicamente motivadas para los puntos de presión.

Convención (supuesto de modelado, documentado en el artículo): las coordenadas
(x, y) de los MAT están en un desarrollo plano ("net") de la envolvente:
superficies 1 y 3 = hastiales (x = -/+20 en el plano), 2 y 4 = muros largos
(y = -/+12), 5 y 6 = faldones de cubierta (y<0 y y>0, cumbrera en y = 0).
El ángulo theta se asume medido de modo que el viento sopla hacia +x a 0 grados
(muro 1 a barlovento) y hacia +y a 90 grados (muro 2 a barlovento); esto es
consistente con el signo de Cp medio observado en los extremos 0 y 90 grados
(Cp>0 en el muro a barlovento) y se verifica en el notebook de EDA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF = 8.0                       # semiancho/semiprofundidad del edificio [m] (B = D = 16 m)
PITCH = np.deg2rad(5.0)          # pendiente de cubierta
WALL_SURFACES = (1, 2, 3, 4)
ROOF_SURFACES = (5, 6)

BASE_FEATURES = ["x", "y", "surface", "theta_sin", "theta_cos"]
PHYSICS_FEATURES = BASE_FEATURES + [
    "is_roof", "px", "py", "ridge_dist", "edge_dist", "corner_dist",
    "wind_incidence", "windward", "fetch", "crosswind", "theta",
]
CATEGORICAL = ["surface"]


def _plan_coordinates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Proyecta cada punto a su posición en planta (px, py), en metros."""
    px, py = df["x"].copy(), df["y"].copy()           # cubierta: ya son coordenadas de planta
    px[df.surface == 1], py[df.surface == 1] = -HALF, df.y[df.surface == 1]
    px[df.surface == 3], py[df.surface == 3] = +HALF, df.y[df.surface == 3]
    px[df.surface == 2], py[df.surface == 2] = df.x[df.surface == 2], -HALF
    px[df.surface == 4], py[df.surface == 4] = df.x[df.surface == 4], +HALF
    return px, py


def _wind_incidence(df: pd.DataFrame) -> np.ndarray:
    """Coseno entre la normal exterior de la superficie y el viento incidente.

    Muros: n1=-x, n2=-y, n3=+x, n4=+y con viento w=(cos t, sin t) => +cos, +sin, -cos, -sin.
    Cubierta: normal casi vertical; solo su componente horizontal (sin 5 grados)
    distingue el faldón a barlovento del de sotavento.
    """
    t = np.deg2rad(df["theta"].to_numpy())
    c, s = np.cos(t), np.sin(t)
    table = {1: c, 2: s, 3: -c, 4: -s, 5: np.sin(PITCH) * s, 6: -np.sin(PITCH) * s}
    out = np.zeros(len(df))
    for surf, val in table.items():
        m = (df["surface"] == surf).to_numpy()
        out[m] = val[m]
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con todas las variables derivadas añadidas."""
    d = df.copy()
    t = np.deg2rad(d["theta"])
    d["theta_sin"], d["theta_cos"] = np.sin(t), np.cos(t)
    d["is_roof"] = d["surface"].isin(ROOF_SURFACES).astype(int)

    d["px"], d["py"] = _plan_coordinates(d)
    d["ridge_dist"] = d["py"].abs()                               # distancia a la cumbrera (y=0)
    d["edge_dist"] = np.minimum(HALF - d["px"].abs(), HALF - d["py"].abs())
    # Distancia a la esquina más cercana de la planta (zona de vórtices cónicos)
    d["corner_dist"] = np.hypot(HALF - d["px"].abs(), HALF - d["py"].abs())

    d["wind_incidence"] = _wind_incidence(d)
    d["windward"] = (d["wind_incidence"] > 0).astype(int)
    # Distancia recorrida por el flujo desde la esquina de barlovento (en planta)
    # y coordenada transversal al viento, ambas en metros.
    d["fetch"] = d["px"] * np.cos(t) + d["py"] * np.sin(t) + HALF * (np.cos(t) + np.sin(t))
    d["crosswind"] = -d["px"] * np.sin(t) + d["py"] * np.cos(t)
    return d
