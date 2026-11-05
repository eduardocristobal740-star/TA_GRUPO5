import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Genera gráficos EDA del conjunto TPU")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="reports/figures/eda")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_parquet(args.data)
    features = [c for c in cfg["features"] if c in df.columns]
    targets = [c for c in cfg["targets"] if c in df.columns]
    numeric = features + targets

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Datos faltantes
    missing = (df[numeric].isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(missing.index, missing.values)
    plt.ylabel("Datos faltantes (%)")
    plt.xlabel("Variable")
    plt.xticks(rotation=45, ha="right")
    plt.title("Porcentaje de datos faltantes")
    savefig(out / "01_datos_faltantes.png")

    # 2. Distribuciones de objetivos
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        values = df[target].dropna()
        ax.hist(values, bins=50)
        ax.set_title(f"Distribución de {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Frecuencia")
    for ax in axes[len(targets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_distribuciones_objetivos.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 3. Matriz de correlación
    corr = df[numeric].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlación de Spearman")
    fig.colorbar(image, ax=ax, label="Coeficiente")
    fig.tight_layout()
    fig.savefig(out / "03_correlacion_spearman.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 4. Cp frente al ángulo del viento
    if "theta" in df.columns:
        grouped = df.groupby("theta")[targets].agg(["mean", "std"]).sort_index()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        for ax, target in zip(axes, targets):
            x = grouped.index.to_numpy(dtype=float)
            mean = grouped[(target, "mean")].to_numpy(dtype=float)
            std = grouped[(target, "std")].fillna(0).to_numpy(dtype=float)
            ax.plot(x, mean, marker="o", markersize=3)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{target} según dirección del viento")
            ax.set_xlabel("Ángulo θ (°)")
            ax.set_ylabel(target)
        for ax in axes[len(targets):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "04_objetivos_vs_angulo.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 5. Cobertura espacial X-Z coloreada por Cp mínimo
    if {"X", "Z", "cp_min"}.issubset(df.columns):
        sample = df[["X", "Z", "cp_min"]].dropna()
        if len(sample) > 30000:
            sample = sample.sample(30000, random_state=42)
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(sample["X"], sample["Z"], c=sample["cp_min"], s=9, cmap="coolwarm")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title("Cobertura espacial de sensores y Cp mínimo")
        plt.colorbar(scatter, label="Cp mínimo")
        savefig(out / "05_mapa_espacial_cp_min.png")

    # 6. Cantidad de observaciones por grupo
    group_col = cfg["group_column"]
    if group_col in df.columns:
        counts = df[group_col].astype(str).value_counts().head(30).sort_values()
        plt.figure(figsize=(10, 7))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Número de observaciones")
        plt.ylabel(group_col)
        plt.title("Observaciones por grupo, 30 grupos principales")
        savefig(out / "06_observaciones_por_grupo.png")

    summary = df[numeric].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary.to_csv(out.parent / "resumen_estadistico.csv")
    print(f"EDA guardado en: {out}")


if __name__ == "__main__":
    main()
