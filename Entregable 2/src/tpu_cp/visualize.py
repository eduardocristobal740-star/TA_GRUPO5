import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualiza resultados e importancia")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="reports/figures/modelos")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))

    # 1. Comparación MAE y RMSE
    metric_df = pd.DataFrame(metrics).T
    x = np.arange(len(metric_df))
    width = 0.36
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, metric_df["mae"], width, label="MAE")
    plt.bar(x + width / 2, metric_df["rmse"], width, label="RMSE")
    plt.xticks(x, metric_df.index, rotation=30, ha="right")
    plt.ylabel("Error")
    plt.title("Desempeño en el conjunto de prueba agrupado")
    plt.legend()
    savefig(out / "01_metricas_prueba.png")

    # 2. Importancia por permutación
    importance_files = sorted(models_dir.glob("importance_*.csv"))
    if importance_files:
        fig, axes = plt.subplots(len(importance_files), 1, figsize=(10, 4 * len(importance_files)))
        axes = np.atleast_1d(axes)
        for ax, file in zip(axes, importance_files):
            target = file.stem.replace("importance_", "")
            imp = pd.read_csv(file).sort_values("importance_mean", ascending=True)
            ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"])
            ax.set_xlabel("Aumento del MAE al permutar la variable")
            ax.set_title(f"Importancia por permutación: {target}")
        fig.tight_layout()
        fig.savefig(out / "02_importancia_permutacion.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 3. Observado frente a predicho
    prediction_files = sorted(models_dir.glob("predictions_*.csv"))
    if prediction_files:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes = axes.ravel()
        for ax, file in zip(axes, prediction_files):
            target = file.stem.replace("predictions_", "")
            pred = pd.read_csv(file)
            if len(pred) > 20000:
                pred = pred.sample(20000, random_state=42)
            ax.scatter(pred["observed"], pred["predicted"], alpha=0.35, s=10)
            low = min(pred["observed"].min(), pred["predicted"].min())
            high = max(pred["observed"].max(), pred["predicted"].max())
            ax.plot([low, high], [low, high], linestyle="--", color="black")
            ax.set_xlabel("Observado")
            ax.set_ylabel("Predicho")
            ax.set_title(target)
        for ax in axes[len(prediction_files):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "03_observado_vs_predicho.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 4. Estabilidad ante perturbaciones
    stability_files = sorted(models_dir.glob("stability_*.csv"))
    if stability_files:
        frames = []
        for file in stability_files:
            frame = pd.read_csv(file)
            frame["target"] = file.stem.replace("stability_", "")
            frames.append(frame)
        stability = pd.concat(frames, ignore_index=True)
        labels = stability["feature"] + " " + stability["perturbation"].astype(str)
        stability = stability.assign(label=labels)
        pivot = stability.pivot_table(index="label", columns="target", values="p95_abs_change")
        pivot.plot(kind="bar", figsize=(12, 7))
        plt.ylabel("Percentil 95 del cambio absoluto")
        plt.xlabel("Perturbación")
        plt.title("Estabilidad de las predicciones")
        plt.legend(title="Objetivo")
        savefig(out / "04_estabilidad_perturbaciones.png")

    print(f"Visualizaciones guardadas en: {out}")


if __name__ == "__main__":
    main()
