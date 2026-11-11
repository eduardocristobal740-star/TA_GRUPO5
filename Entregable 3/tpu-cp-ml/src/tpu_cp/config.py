"""Constantes y rutas del proyecto (única fuente de verdad)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
MAT_PATTERN = "time_series_of_point_wind_pressure_*.mat"

# Variables objetivo: estadísticos de la serie temporal de Cp de cada sensor.
TARGETS = ["cp_mean", "cp_rms", "cp_min", "cp_max"]
TARGET_LABELS = {
    "cp_mean": r"$C_{p,mean}$",
    "cp_rms": r"$C_{p,rms}$",
    "cp_min": r"$C_{p,min}$",
    "cp_max": r"$C_{p,max}$",
}

# Metadatos que deben ser idénticos en todos los archivos (misma geometría).
GEOMETRY_KEYS = ["Building_height", "Building_breadth", "Building_depth",
                 "Roof_pitch", "Sample_frequency", "Sample_period"]
