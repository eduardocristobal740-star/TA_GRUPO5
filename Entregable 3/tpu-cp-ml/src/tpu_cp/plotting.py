"""Figuras para el artículo (estilo sobrio, paleta Okabe-Ito apta para daltonismo)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, TARGET_LABELS, TARGETS

PAL = {"azul": "#0072B2", "naranja": "#E69F00", "verde": "#009E73", "rojo": "#D55E00",
       "gris": "#7F7F7F", "celeste": "#56B4E9", "purpura": "#CC79A7"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png"); plt.close(fig)


def fig_heatmaps(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, t, cmap in zip(axs, ["cp_mean", "cp_min"], ["RdBu_r", "viridis"]):
        piv = df.pivot_table(index="surface", columns="theta", values=t)
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto",
                       **({"vmin": -0.7, "vmax": 0.7} if t == "cp_mean" else {}))
        ax.set_xticks(range(piv.shape[1]), piv.columns.astype(int)); ax.set_yticks(range(6), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set(xlabel=r"$\theta$ (°)", ylabel="Superficie" if t == "cp_mean" else "", title=f"Media por superficie: {TARGET_LABELS[t]}"); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig01_heatmaps_superficie_angulo")


def fig_distributions(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9))
    for ax, t in zip(axs, TARGETS):
        ax.hist(df[t], bins=35, color=PAL["azul"]); ax.set_title(TARGET_LABELS[t]); ax.set_yticks([])
    save(fig, "fig02_distribuciones")


def fig_net_maps(df: pd.DataFrame, target: str = "cp_mean", angles=(0, 45, 90)) -> None:
    fig, axs = plt.subplots(1, len(angles), figsize=(7.2, 2.3), sharex=True, sharey=True)
    vmin, vmax = df[target].quantile([.02, .98]) if target != "cp_mean" else (-0.8, 0.8)
    for ax, a in zip(axs, angles):
        d = df[df.theta == a]
        sc = ax.scatter(d.x, d.y, c=d[target], s=22, cmap="RdBu_r" if target == "cp_mean" else "viridis",
                        vmin=vmin, vmax=vmax, marker="s")
        ax.set_title(rf"$\theta={a}°$"); ax.set_aspect("equal"); ax.grid(False)
    axs[0].set_ylabel("y (plano desarrollado)"); [ax.set_xlabel("x") for ax in axs]
    fig.colorbar(sc, ax=axs, fraction=0.02, label=TARGET_LABELS[target])
    save(fig, f"fig03_mapa_{target}")


def fig_comparison(summary: pd.DataFrame, top: int = 7) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, t in zip(axs.ravel(), TARGETS):
        s = summary[summary.objetivo == t].copy()
        s["etq"] = s.modelo + np.where(s.variables.isin(["base", "fisica"]), " [" + s.variables + "]", "")
        s = s.sort_values("MAE").head(top)[::-1]
        colors = [PAL["naranja"] if m.startswith("Interp") else PAL["verde"] if m.startswith("Híbrido")
                  else PAL["azul"] for m in s.modelo]
        ax.barh(s.etq, s.MAE, xerr=s.MAE_sd, color=colors, capsize=2, error_kw={"lw": .6})
        ax.set_title(TARGET_LABELS[t]); ax.set_xlabel("MAE"); ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig04_comparacion_mae")


def fig_obs_pred(pred: pd.DataFrame, best: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True)
    for ax, t in zip(axs, TARGETS):
        b = best[best.objetivo == t].iloc[0]
        q = pred[(pred.objetivo == t) & (pred.modelo == b.modelo) & (pred.variables == b.variables)]
        ax.scatter(q.observado, q.predicho, s=5, alpha=.5, color=PAL["azul"], lw=0)
        lo, hi = q.observado.min(), q.observado.max(); ax.plot([lo, hi], [lo, hi], color=PAL["rojo"], lw=.8)
        ax.set_title(f"{TARGET_LABELS[t]}", fontsize=8); ax.set_xlabel("Observado")
    axs[0].set_ylabel("Predicho")
    save(fig, "fig05_observado_vs_predicho")


def fig_by_angle(folds: pd.DataFrame, best: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), sharex=True, constrained_layout=True)
    for ax, t in zip(axs, TARGETS):
        b = best[best.objetivo == t].iloc[0]
        for (m, v, lab, c, ls) in [(b.modelo, b.variables, "Mejor modelo", PAL["verde"], "-"),
                                   ("Interp. angular lineal", "sensor", "Interp. lineal", PAL["naranja"], "--"),
                                   ("Dummy (mediana)", "base", "Mediana", PAL["gris"], ":")]:
            q = folds[(folds.objetivo == t) & (folds.modelo == m) & (folds.variables == v)].sort_values("theta")
            ax.plot(q.theta, q.MAE, ls, color=c, marker="o", ms=3, label=lab)
        ax.set_title(TARGET_LABELS[t]); ax.set_xlabel(r"$\theta$ reservado (°)")
    axs[0].set_ylabel("MAE"); axs[0].legend(fontsize=5.5, frameon=False)
    save(fig, "fig06_mae_por_angulo")


def fig_error_map(df: pd.DataFrame, pred: pd.DataFrame, best: pd.DataFrame, target="cp_min", angle=45) -> None:
    b = best[best.objetivo == target].iloc[0]
    q = pred[(pred.objetivo == target) & (pred.modelo == b.modelo) & (pred.variables == b.variables)
             & (pred.theta == angle)].merge(df[["sensor_id", "x", "y"]].drop_duplicates(), on="sensor_id")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharex=True, sharey=True)
    lo, hi = q.observado.quantile(.02), q.observado.quantile(.98)
    for ax, col, ttl in zip(axs, ["observado", "predicho", "err"], ["Observado", "Predicho", "Error (pred. - obs.)"]):
        q["err"] = q.predicho - q.observado
        kw = dict(vmin=lo, vmax=hi, cmap="viridis") if col != "err" else dict(vmin=-1, vmax=1, cmap="RdBu_r")
        sc = ax.scatter(q.x, q.y, c=q[col], s=20, marker="s", **kw); ax.set_title(ttl); ax.set_aspect("equal"); ax.grid(False)
        fig.colorbar(sc, ax=ax, fraction=0.046)
    save(fig, "fig07_mapa_error")
