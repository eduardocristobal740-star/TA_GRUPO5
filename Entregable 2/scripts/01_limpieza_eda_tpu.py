"""Limpieza y EDA inicial de la base TPU para el caso gable roof seleccionado.

Estructura esperada del proyecto:
TA_GRUPO5/
  data/raw/time_series_of_point_wind_pressure_0.mat
  data/raw/time_series_of_point_wind_pressure_15.mat
  ...
  scripts/01_limpieza_eda_tpu.py

Ejecutar desde la carpeta raíz del repositorio:
    python scripts/01_limpieza_eda_tpu.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
ANGLES = [0, 15, 30, 45, 60, 75, 90]


def read_mat(path: Path, expected_angle: int):
    """Lee un archivo MATLAB y devuelve Cp, ubicaciones y metadatos básicos."""
    data = loadmat(path, squeeze_me=True, chars_as_strings=True)
    cp = np.asarray(data["Wind_pressure_coefficients"], dtype=float)
    locations = np.asarray(data["Location_of_measured_points"], dtype=float)

    # Las ubicaciones deben tener 4 filas: x, y, número de punto y superficie.
    if cp.ndim != 2 or locations.shape != (4, cp.shape[1]):
        raise ValueError(f"Estructura inesperada en {path.name}: Cp={cp.shape}, ubicaciones={locations.shape}")

    angle = int(np.asarray(data["Wind_direction_angle"]).item())
    if angle != expected_angle:
        raise ValueError(f"El archivo {path.name} indica {angle}°, pero se esperaba {expected_angle}°.")

    metadata = {
        "roof_type": str(np.asarray(data["Roof_type"]).item()),
        "building_height_m_from_mat": float(np.asarray(data["Building_height"]).item()),
        "building_breadth_m": float(np.asarray(data["Building_breadth"]).item()),
        "building_depth_m": float(np.asarray(data["Building_depth"]).item()),
        "roof_pitch_deg": float(np.asarray(data["Roof_pitch"]).item()),
        "sample_frequency_hz": float(np.asarray(data["Sample_frequency"]).item()),
        "sample_period_s": float(np.asarray(data["Sample_period"]).item()),
        "wind_direction_deg": angle,
    }
    return cp, locations, metadata


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    reference_locations = None

    for angle in ANGLES:
        file_path = DATA_DIR / f"time_series_of_point_wind_pressure_{angle}.mat"
        if not file_path.exists():
            raise FileNotFoundError(f"No encuentro este archivo: {file_path}")

        cp, locations, meta = read_mat(file_path, angle)
        if reference_locations is None:
            reference_locations = locations.copy()
        elif not np.array_equal(locations, reference_locations):
            raise ValueError(f"Las ubicaciones de puntos no coinciden en {file_path.name}.")

        print(f"{file_path.name}: {cp.shape[0]} registros × {cp.shape[1]} puntos; faltantes={np.isnan(cp).sum()}")

        for j in range(cp.shape[1]):
            series = cp[:, j]
            mean_cp = float(np.mean(series))
            fluctuation = series - mean_cp
            rows.append({
                **meta,
                "height_to_breadth": meta["building_height_m_from_mat"] / meta["building_breadth_m"],
                "depth_to_breadth": meta["building_depth_m"] / meta["building_breadth_m"],
                "tap_x_coordinate_as_stored": float(locations[0, j]),
                "tap_y_coordinate_as_stored": float(locations[1, j]),
                "point_id": int(locations[2, j]),
                "surface_id": int(locations[3, j]),
                "cp_mean_10min": mean_cp,
                # RMS de la fluctuación: raíz de la media de (Cp - Cp medio)^2.
                "cp_rms_fluctuation_10min": float(np.sqrt(np.mean(fluctuation ** 2))),
                # RMS total respecto a cero, incluido como estadística distinta.
                "cp_rms_about_zero_10min": float(np.sqrt(np.mean(series ** 2))),
                # Extremos de la serie TPU, que está promediada en intervalos de 0.2 s.
                "cp_min_0p2s_moving_average": float(np.min(series)),
                "cp_max_0p2s_moving_average": float(np.max(series)),
                "n_time_samples": int(len(series)),
            })

    df = pd.DataFrame(rows).sort_values(["wind_direction_deg", "point_id"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "TPU_Gable_0-90_tabla_por_punto.csv", index=False)

    print("\nResumen de limpieza")
    print("Filas consolidadas:", len(df))
    print("Puntos únicos:", df["point_id"].nunique())
    print("Direcciones:", sorted(df["wind_direction_deg"].unique().tolist()))
    print("Valores faltantes por columna:\n", df.isna().sum().to_string())
    print("\nResumen de las estadísticas Cp:\n", df[[
        "cp_mean_10min", "cp_rms_fluctuation_10min",
        "cp_min_0p2s_moving_average", "cp_max_0p2s_moving_average"
    ]].describe().round(4).to_string())

    # Gráfico 1: Cp medio por dirección.
    mean_by_angle = df.groupby("wind_direction_deg")["cp_mean_10min"].mean()
    ax = mean_by_angle.plot(kind="bar", color="#2878B5", figsize=(8, 4))
    ax.set_title("Coeficiente medio de presión por dirección del viento")
    ax.set_xlabel("Dirección del viento (°)")
    ax.set_ylabel("Cp medio")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cp_medio_por_direccion.png", dpi=180)
    plt.close()

    # Gráfico 2: distribución de Cp medio en las tomas, agrupada por ángulo.
    angle_groups = [df.loc[df["wind_direction_deg"] == angle, "cp_mean_10min"] for angle in ANGLES]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(angle_groups, showfliers=False)
    ax.set_xticks(range(1, len(ANGLES) + 1), [str(angle) for angle in ANGLES])
    ax.set_title("Distribución del Cp medio entre tomas")
    ax.set_xlabel("Dirección del viento (°)")
    ax.set_ylabel("Cp medio por punto")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distribucion_cp_medio_por_direccion.png", dpi=180)
    plt.close(fig)

    # Correlación solo entre variables numéricas que sí varían en este caso.
    candidate_cols = [
        "height_to_breadth", "depth_to_breadth", "roof_pitch_deg",
        "wind_direction_deg", "tap_x_coordinate_as_stored", "tap_y_coordinate_as_stored",
        "cp_mean_10min", "cp_rms_fluctuation_10min"
    ]
    variable_cols = [col for col in candidate_cols if df[col].nunique(dropna=True) > 1]
    corr = df[variable_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(variable_cols)), variable_cols, rotation=60, ha="right")
    ax.set_yticks(range(len(variable_cols)), variable_cols)
    ax.set_title("Correlación de variables numéricas")
    fig.colorbar(image, ax=ax, label="Correlación")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlacion_variables.png", dpi=180)
    plt.close(fig)

    print(f"\nArchivos guardados en: {OUTPUT_DIR}")
    print("Nota: estos siete archivos corresponden a una sola geometría. No permiten evaluar generalización a edificios con otras dimensiones.")
    print("Nota: los MAT incluyen X e Y y el identificador de superficie, pero no una coordenada Z explícita.")


if __name__ == "__main__":
    main()
