"""Generarea graficelor cerute in proiect (toate produse din cod).

Folosim backend non-interactiv (Agg) ca scripturile sa ruleze si fara display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from nyse_vol import config

# Paleta de culori distincte per model
_MODEL_COLORS = {
    "LSTM":             "#e74c3c",
    "LSTM (HPO)":       "#c0392b",
    "Attention":        "#e67e22",
    "Attention (HPO)":  "#d35400",
    "GARCH":            "#1a7a4b",
    "Naiv":             "#7f8c8d",
}
_ACTUAL_COLOR  = "#2471a3"    # albastru pentru seria reala
_TREND_COLOR   = "#154360"    # albastru inchis pentru media mobila

_LS = {  # line styles per model
    "LSTM":             "-",
    "LSTM (HPO)":       "-",
    "Attention":        "--",
    "Attention (HPO)":  "--",
    "GARCH":            "-.",
    "Naiv":             ":",
}


def _model_color(name: str) -> str:
    return _MODEL_COLORS.get(name, "#3498db")


def _style_ax(ax):
    """Aplica stilul comun (grid, spine-uri, background)."""
    ax.set_facecolor("#f8f9fa")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.5f}", offset=3):
    """Adauga etichete numerice deasupra fiecarei bare."""
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax.annotate(
                fmt.format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Loss curves (antrenare)
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_ax(ax)

    train_l = history["train_loss"]
    val_l   = history["val_loss"]
    epochs  = range(1, len(train_l) + 1)

    ax.plot(epochs, train_l, label="Train",    linewidth=2, color=_ACTUAL_COLOR)
    ax.plot(epochs, val_l,   label="Validare", linewidth=2, color="#c0392b")

    best_ep  = int(np.argmin(val_l)) + 1
    best_val = float(min(val_l))
    ax.axvline(best_ep, color="#27ae60", linestyle="--", alpha=0.8, linewidth=1.5)
    ax.scatter([best_ep], [best_val], color="#27ae60", s=80, zorder=6,
               label=f"Best val={best_val:.4f} (ep. {best_ep})")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("MSE (log-vol. standardizata)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Pred vs. True simplu (pas 05, per simbol de exemplu)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_true(dates, y_true, y_pred, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _style_ax(ax)

    ax.fill_between(dates, y_true, alpha=0.12, color=_ACTUAL_COLOR)
    ax.plot(dates, y_true, label="Volatilitate reala",
            linewidth=1.2, color=_ACTUAL_COLOR, alpha=0.9)
    # trend 21-zile
    roll = pd.Series(y_true).rolling(21, center=True, min_periods=5).mean().values
    ax.plot(dates, roll, linewidth=1.8, color=_TREND_COLOR, alpha=0.55,
            linestyle="-", label="Medie mobila 21 zile")
    ax.plot(dates, y_pred, label="Prezis", linewidth=1.5, alpha=0.85, color="#c0392b")

    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Volatilitate zilnica (σ)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap atentie
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(weights: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(weights, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Pozitie in fereastra de intrare (zile)", fontsize=10)
    ax.set_ylabel("Orizont de predictie", fontsize=10)
    ax.set_yticks(range(len(config.HORIZONS)))
    ax.set_yticklabels([f"h={h}" for h in config.HORIZONS])
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Pondere atentie")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Comparatie globala modele (bar chart RMSE / QLIKE) — pas 05
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(metrics_by_model: dict, metric: str, out_path: Path):
    """Bar chart comparativ pentru o metrica, pe orizonturi, intre modele.

    Bare etichetate cu valori, cel mai bun model evidentiat cu contur auriu.
    """
    horizons = config.HORIZONS
    models   = list(metrics_by_model.keys())
    n_models = len(models)
    x     = np.arange(len(horizons))
    width = 0.75 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_ax(ax)

    # Gasim cel mai bun model per orizont (min)
    best_per_h = {}
    for h in horizons:
        best_m, best_v = None, float("inf")
        for m in models:
            v = metrics_by_model[m].get(h, np.nan)
            if np.isfinite(v) and v < best_v:
                best_v, best_m = v, m
        best_per_h[h] = best_m

    for i, m in enumerate(models):
        vals  = [metrics_by_model[m].get(h, np.nan) for h in horizons]
        offset = x + i * width - (n_models - 1) * width / 2
        bars  = ax.bar(offset, vals, width, label=m,
                       color=_model_color(m), alpha=0.88,
                       edgecolor="white", linewidth=0.6)
        _bar_labels(ax, bars, fmt="{:.5f}")

        # contur auriu pe cel mai bun
        for bar, h in zip(bars, horizons):
            if best_per_h.get(h) == m:
                bar.set_edgecolor("#f39c12")
                bar.set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"h={h} {'zi' if h==1 else 'zile'}" for h in horizons], fontsize=11
    )
    ax.set_xlabel("Orizont de predictie", fontsize=11)

    metric_full = {
        "rmse":  "RMSE  (Root Mean Squared Error)",
        "qlike": "QLIKE  (Quasi-Likelihood Loss)",
    }.get(metric.lower(), metric.upper())

    ax.set_ylabel(metric_full, fontsize=11)
    ax.set_title(
        f"Comparatie modele — {metric.upper()}  ·  Set de test (2022–2026)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9, edgecolor="#cccccc")
    ax.text(
        0.5, -0.14,
        "↓ Mai mic = mai bun   |   Contur portocaliu = cel mai bun per orizont"
        "   |   Unitati: volatilitate zilnica absoluta (σ)",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Toate modelele pe h=1 pentru un simbol (pas 06)
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_models_h1(dates, actuals, preds_dict: dict, symbol: str, out_path: Path):
    """Compara toate modelele pe h=1 pentru un singur simbol.

    preds_dict: {"LSTM (HPO)": array, "GARCH": array, ...}
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    _style_ax(ax)

    # Aria umbrita pentru volatilitatea reala
    ax.fill_between(dates, actuals, alpha=0.10, color=_ACTUAL_COLOR)

    # Linia reala (prominenta)
    ax.plot(dates, actuals, label="Volatilitate reala",
            linewidth=1.2, color=_ACTUAL_COLOR, alpha=0.90, zorder=5)

    # Media mobila 21-zile (trend)
    roll = pd.Series(actuals).rolling(21, center=True, min_periods=5).mean().values
    ax.plot(dates, roll, linewidth=2.0, color=_TREND_COLOR, alpha=0.55,
            linestyle="-", label="Trend 21 zile (real)", zorder=6)

    # Predictii modele
    for name, pred in preds_dict.items():
        mask = np.isfinite(pred)
        if not mask.any():
            continue
        lw = 1.8 if "HPO" in name else 1.3
        ax.plot(dates[mask], pred[mask], label=name,
                linewidth=lw, alpha=0.85,
                color=_model_color(name),
                linestyle=_LS.get(name, "-"),
                zorder=4)

    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Volatilitate zilnica (σ)", fontsize=11)
    ax.set_title(
        f"[{symbol}]  Volatilitate reala vs. prezisa — h=1 zi\n"
        f"Modele: {', '.join(preds_dict.keys())}  ·  Test 2022–2026",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Toate orizonturile pentru cel mai bun model (pas 06)
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_horizons(dates, actuals_by_h: dict, preds_by_h: dict,
                      symbol: str, model_name: str, out_path: Path):
    """4 subploturi (2x2) cu volatilitate reala vs prezisa pentru h=1,5,10,20."""
    horizons = config.HORIZONS
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, h in zip(axes, horizons):
        true_h = actuals_by_h.get(h)
        pred_h = preds_by_h.get(h)
        if true_h is None or pred_h is None:
            ax.set_visible(False)
            continue

        _style_ax(ax)
        mask = np.isfinite(pred_h) & np.isfinite(true_h)

        ax.fill_between(dates[mask], true_h[mask], alpha=0.10, color=_ACTUAL_COLOR)
        ax.plot(dates[mask], true_h[mask],
                label="Real", linewidth=1.2, color=_ACTUAL_COLOR, alpha=0.90)
        ax.plot(dates[mask], pred_h[mask],
                label=model_name, linewidth=1.8, alpha=0.85,
                color=_model_color(model_name), linestyle="-")

        # RMSE in titlul subplotului
        rmse = float(np.sqrt(np.nanmean((true_h[mask] - pred_h[mask]) ** 2)))
        unit = "zi" if h == 1 else "zile"
        ax.set_title(f"h={h} {unit}  ·  RMSE = {rmse:.5f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Data", fontsize=9)
        ax.set_ylabel("Volatilitate (σ)", fontsize=9)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"[{symbol}]  {model_name} — Predictii pe toate orizonturile\n"
        f"Test 2022–2026  ·  h = zile bursiere in viitor",
        fontsize=13, fontweight="bold",
    )
    fig.autofmt_xdate()
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Metrici per stock (pas 06)
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrics_per_stock(metrics_rows: list[dict], symbol: str, out_path: Path):
    """Subploturi separate pentru RMSE, MAE, R², QLIKE — h=1 — per simbol.

    Cel mai bun model per metrica: contur portocaliu.
    """
    df    = pd.DataFrame(metrics_rows)
    df_h1 = df[df["horizon"] == 1].copy()
    if df_h1.empty:
        return

    metric_cfg = [
        ("rmse",  "RMSE",  "mai mic = mai bun", False),
        ("mae",   "MAE",   "mai mic = mai bun", False),
        ("r2",    "R²",    "mai mare = mai bun", True),
        ("qlike", "QLIKE", "mai mic = mai bun", False),
    ]
    avail = [(c, lbl, note, higher) for c, lbl, note, higher in metric_cfg
             if c in df_h1.columns]
    if not avail:
        return

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    # Ordine: cel mai bun RMSE primul
    order_col = "rmse" if "rmse" in df_h1.columns else avail[0][0]
    df_h1 = df_h1.sort_values(order_col).reset_index(drop=True)
    models = df_h1["model"].tolist()
    colors = [_model_color(m) for m in models]
    x      = np.arange(len(models))

    for ax, (col, lbl, note, higher_better) in zip(axes, avail):
        _style_ax(ax)
        vals  = df_h1[col].to_numpy(dtype=float)
        bars  = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Evidentiaza cel mai bun
        best_idx = int(np.nanargmax(vals)) if higher_better else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor("#f39c12")
        bars[best_idx].set_linewidth(2.8)

        _bar_labels(ax, bars, fmt="{:.5f}")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=9.5)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(f"{lbl}\n({note})", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"[{symbol}]  Comparatie metrici modele — h=1 zi  ·  Set de test 2022–2026\n"
        f"Contur portocaliu = cel mai bun per metrica",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
