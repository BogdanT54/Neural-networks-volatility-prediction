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
    ax.plot(dates, y_true, label="real", linewidth=1)
    ax.plot(dates, y_pred, label="prezis", linewidth=1, alpha=0.8)
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
    """Bar chart comparativ pentru o metrica, pe orizonturi, intre modele.

    metrics_by_model: {model_name: {horizon: value}}
    """
    horizons = config.HORIZONS
    models = list(metrics_by_model.keys())
    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(models):
        vals = [metrics_by_model[m].get(h, np.nan) for h in horizons]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparatie modele dupa {metric.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
