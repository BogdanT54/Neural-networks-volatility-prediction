"""Generarea graficelor cerute in proiect (toate produse din cod).

Folosim backend non-interactiv (Agg) ca scripturile sa ruleze si fara display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nyse_vol import config

_MODEL_COLORS = {
    "LSTM": "#e74c3c",
    "LSTM (HPO)": "#c0392b",
    "Attention": "#e67e22",
    "Attention (HPO)": "#d35400",
    "GARCH": "#27ae60",
    "Naiv": "#95a5a6",
}


def _model_color(name: str) -> str:
    return _MODEL_COLORS.get(name, "#3498db")


def plot_loss_curves(history: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validare")
    ax.set_xlabel("epoca")
    ax.set_ylabel("MSE (log-volatilitate standardizata)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_vs_true(dates, y_true, y_pred, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_true, label="real", linewidth=1, color="#2980b9")
    ax.plot(dates, y_pred, label="prezis", linewidth=1, alpha=0.8, color="#e74c3c")
    ax.set_xlabel("data")
    ax.set_ylabel("volatilitate")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_attention_heatmap(weights: np.ndarray, title: str, out_path: Path):
    """weights: (n_outputs, window) ponderi de atentie pentru un exemplu."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(weights, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("pozitie in fereastra de intrare (zile)")
    ax.set_ylabel("orizont")
    ax.set_yticks(range(len(config.HORIZONS)))
    ax.set_yticklabels([f"h={h}" for h in config.HORIZONS])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="pondere atentie")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(metrics_by_model: dict, metric: str, out_path: Path):
    """Bar chart comparativ pentru o metrica, pe orizonturi, intre modele."""
    horizons = config.HORIZONS
    models = list(metrics_by_model.keys())
    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        vals = [metrics_by_model[m].get(h, np.nan) for h in horizons]
        ax.bar(x + i * width, vals, width, label=m, color=_model_color(m), alpha=0.85)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"h={h} zile" for h in horizons])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparatie modele dupa {metric.upper()} (set de test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Functii noi pentru selectia de stock-uri (pasul 06)
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_models_h1(dates, actuals, preds_dict: dict, symbol: str, out_path: Path):
    """Compara toate modelele pe h=1 pentru un singur simbol.

    preds_dict: {"LSTM": array, "GARCH": array, ...}
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actuals, label="Real", linewidth=1.5, color="#2980b9", zorder=5)
    styles = ["-", "--", "-.", ":"]
    for i, (name, pred) in enumerate(preds_dict.items()):
        mask = np.isfinite(pred)
        if not mask.any():
            continue
        ax.plot(dates[mask], pred[mask], label=name,
                linewidth=1.1, alpha=0.85,
                color=_model_color(name),
                linestyle=styles[i % len(styles)])
    ax.set_xlabel("Data")
    ax.set_ylabel("Volatilitate")
    ax.set_title(f"[{symbol}] Volatilitate reala vs. prezisa — toate modelele (h=1 zi)")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_all_horizons(dates, actuals_by_h: dict, preds_by_h: dict,
                      symbol: str, model_name: str, out_path: Path):
    """4 subploturi (2x2) cu volatilitate reala vs prezisa pentru h=1,5,10,20.

    actuals_by_h: {1: array, 5: array, 10: array, 20: array}
    preds_by_h:   {1: array, 5: array, 10: array, 20: array}
    """
    horizons = config.HORIZONS
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    axes = axes.flatten()
    for ax, h in zip(axes, horizons):
        true_h = actuals_by_h.get(h)
        pred_h = preds_by_h.get(h)
        if true_h is None or pred_h is None:
            ax.set_visible(False)
            continue
        mask = np.isfinite(pred_h)
        ax.plot(dates[mask], true_h[mask], label="Real", linewidth=1.2, color="#2980b9")
        ax.plot(dates[mask], pred_h[mask], label=model_name, linewidth=1.1,
                alpha=0.85, color=_model_color(model_name))
        ax.set_title(f"Orizont h={h} {'zi' if h == 1 else 'zile'}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Volatilitate")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        fig.autofmt_xdate()
    fig.suptitle(f"[{symbol}] Predictii {model_name} — toate orizonturile", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_metrics_per_stock(metrics_rows: list[dict], symbol: str, out_path: Path):
    """Bar chart cu RMSE si MAE per model, pentru un singur simbol.

    metrics_rows: [{"model": "LSTM", "horizon": 1, "rmse": ..., "mae": ...}, ...]
    Afiseaza doar h=1 pentru claritate.
    """
    import pandas as pd
    df = pd.DataFrame(metrics_rows)
    df_h1 = df[df["horizon"] == 1].copy()
    if df_h1.empty:
        return

    models = df_h1["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, df_h1["rmse"], width, label="RMSE",
                    color=[_model_color(m) for m in models], alpha=0.85)
    bars2 = ax2.bar(x + width / 2, df_h1["mae"], width, label="MAE",
                    color=[_model_color(m) for m in models], alpha=0.55, hatch="//")

    ax1.set_ylabel("RMSE", color="#333")
    ax2.set_ylabel("MAE", color="#666")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title(f"[{symbol}] Comparatie metrici modele pe set de test (h=1 zi)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
