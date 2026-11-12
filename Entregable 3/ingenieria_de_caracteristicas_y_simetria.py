"""Módulo de ingeniería de características físicas y aumento por simetría biaxial (TPU)."""

from __future__ import annotations
import numpy as np
import pandas as pd

SURFACE_NORMALS = {
    1: (1.0, 0.0, 0.0),                        # Muro hastial 1
    2: (0.0, 1.0, 0.0),                        # Muro longitudinal 2
    3: (-1.0, 0.0, 0.0),                       # Muro hastial 3
    4: (0.0, -1.0, 0.0),                       # Muro longitudinal 4
    5: (0.0, np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0))),   # Faldón cubierta 5
    6: (0.0, -np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0))),  # Faldón cubierta 6
}


def add_aerodynamic_features(df: pd.DataFrame, width_b: float = 16.0, depth_d: float = 16.0) -> pd.DataFrame:
    """Agrega distancias relativas al borde de ataque (fetch), incidencia y singularidades."""
    out = df.copy()
    half_b, half_d = width_b / 2.0, depth_d / 2.0

    rad = np.deg2rad(out["angle_deg"].to_numpy(dtype=float))
    out["sin_theta"] = np.sin(rad)
    out["cos_theta"] = np.cos(rad)
    out["is_roof"] = out["surface"].isin([5, 6]).astype(int)

    # Distancias euclidianas a cumbrera, arista perimetral y esquina más cercana
    px = out["px"].to_numpy(dtype=float)
    py = out["py"].to_numpy(dtype=float)
    out["dist_ridge"] = np.where(out["is_roof"] == 1, np.abs(py), half_b)
    out["dist_edge"] = np.minimum(half_d - np.abs(px), half_b - np.abs(py))
    out["dist_corner"] = np.hypot(half_d - np.abs(px), half_b - np.abs(py))

    # Coseno de incidencia entre normal exterior de superficie y dirección del viento
    inc = np.zeros(len(out), dtype=float)
    for surf_id, (nx, ny, _) in SURFACE_NORMALS.items():
        mask = out["surface"] == surf_id
        inc[mask] = nx * out.loc[mask, "cos_theta"] + ny * out.loc[mask, "sin_theta"]
    out["wind_incidence"] = inc
    out["is_windward"] = (out["wind_incidence"] > 0.0).astype(int)

    # Fetch: distancia recorrida por el flujo desde la arista de barlovento
    proj_along = px * out["cos_theta"] + py * out["sin_theta"]
    max_proj = half_d * np.abs(out["cos_theta"]) + half_b * np.abs(out["sin_theta"])
    out["fetch"] = np.clip(max_proj - proj_along, 0.0, np.hypot(width_b, depth_d))
    out["cross_flow_coord"] = -px * out["sin_theta"] + py * out["cos_theta"]

    # Factor de pico empírico observado (para modelo indirecto de extremos)
    if {"cp_min", "cp_mean", "cp_rms"}.issubset(out.columns):
        out["peak_factor_min"] = np.abs(out["cp_min"] - out["cp_mean"]) / np.maximum(out["cp_rms"], 1e-4)

    return out


def augment_by_biaxial_symmetry(df_train: pd.DataFrame) -> pd.DataFrame:
    """Duplica el conjunto de entrenamiento reflejando coordenadas respecto al eje de cumbrera."""
    base = df_train.copy()
    reflected = df_train.copy()

    # Reflexión respecto al eje X (paralelo a la cumbrera): py -> -py, intercambio superficies 2<->4 y 5<->6
    reflected["py"] = -reflected["py"]
    swap_map = {1: 1, 3: 3, 2: 4, 4: 2, 5: 6, 6: 5}
    reflected["surface"] = reflected["surface"].map(swap_map)
    reflected["angle_deg"] = (360.0 - reflected["angle_deg"]) % 360.0

    reflected = add_aerodynamic_features(reflected)
    return pd.concat([base, reflected], ignore_index=True)