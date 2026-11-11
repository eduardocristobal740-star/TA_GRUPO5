"""Lectura de los archivos MAT de TPU y construcción de la tabla sensor-ángulo.

Cada archivo contiene, para un ángulo de viento, la serie temporal de Cp
(9000 muestras x 128 sensores) y las coordenadas de los sensores. Se resume
cada serie en cuatro estadísticos, de modo que 1 fila = 1 sensor en 1 ángulo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import GEOMETRY_KEYS, MAT_PATTERN, RAW_DIR


def load_mat_file(path: Path) -> dict:
    """Carga un archivo MAT (formato v5) y devuelve sus variables como dict."""
    raw = loadmat(str(path))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def summarize_series(cp: np.ndarray) -> dict[str, np.ndarray]:
    """Estadísticos por sensor de una matriz Cp (tiempo x sensores).

    * cp_mean : media temporal.
    * cp_rms  : desviación estándar poblacional (RMS de la fluctuación).
    * cp_min / cp_max : extremos de la serie, que ya viene promediada
      en ventanas móviles de 0.2 s a escala real (ver Readme del MAT).
    """
    cp = cp.astype(np.float64)
    return {
        "cp_mean": cp.mean(axis=0),
        "cp_rms": cp.std(axis=0, ddof=0),
        "cp_min": cp.min(axis=0),
        "cp_max": cp.max(axis=0),
    }


def build_dataset(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Construye la tabla consolidada y un reporte de control de calidad."""
    files = sorted(raw_dir.glob(MAT_PATTERN))
    if not files:
        raise FileNotFoundError(f"No hay archivos {MAT_PATTERN} en {raw_dir}")

    frames, geometries, qc = [], [], {"archivos": len(files), "valores_nan": 0,
                                      "valores_inf": 0, "angulo_nombre_distinto": []}
    for path in files:
        mat = load_mat_file(path)
        cp = mat["Wind_pressure_coefficients"]          # (tiempo, sensores)
        loc = mat["Location_of_measured_points"]        # (4, sensores)
        theta = float(mat["Wind_direction_angle"].item())

        if f"_{int(theta)}.mat" not in path.name:
            qc["angulo_nombre_distinto"].append(path.name)
        qc["valores_nan"] += int(np.isnan(cp).sum())
        qc["valores_inf"] += int(np.isinf(cp).sum())
        geometries.append(tuple(float(mat[k].item()) for k in GEOMETRY_KEYS))

        stats = summarize_series(cp)
        frames.append(pd.DataFrame({
            "theta": theta,
            "sensor_id": loc[2].astype(int),
            "surface": loc[3].astype(int),
            "x": loc[0].astype(float),
            "y": loc[1].astype(float),
            "n_muestras": cp.shape[0],
            **stats,
        }))

    df = pd.concat(frames, ignore_index=True).sort_values(["theta", "sensor_id"])
    qc.update({
        "geometrias_distintas": len(set(geometries)),
        "geometria": dict(zip(GEOMETRY_KEYS, geometries[0])),
        "filas": len(df),
        "sensores_por_angulo": df.groupby("theta").size().to_dict(),
        "duplicados_sensor_angulo": int(df.duplicated(["sensor_id", "theta"]).sum()),
        "angulos": sorted(df.theta.unique().tolist()),
    })
    return df.reset_index(drop=True), qc
