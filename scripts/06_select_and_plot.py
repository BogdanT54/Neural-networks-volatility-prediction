"""Pas 6: selectie interactiva de stock-uri si generare grafice detaliate.

Utilizatorul alege un subset de simboluri NYSE pentru care se genereaza:
  1. Volatilitate reala vs. prezisa — toate modelele pe h=1
  2. Toate orizonturile (h=1,5,10,20) pentru cel mai bun model HPO
  3. Bar chart cu metrici (RMSE, MAE) per model
  4. Tabel comparativ in consola + CSV

Se poate rula:
  python 06_select_and_plot.py --symbols JPM GS BAC
  python 06_select_and_plot.py                         (selectie interactiva)
"""
from __future__ import annotations

import argparse
import json
import textwrap

import numpy as np
import pandas as pd
import torch

from _common import config, get_features, print_banner, print_data_summary, set_seed
from nyse_vol.config import ModelConfig
from nyse_vol.data import dataset as ds
from nyse_vol.eval import metrics as M
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.trainer import predict


# ──────────────────────────────────────────────────────────────────────────────
# Incarcare modele
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(cls, splits, name, model_cfg=None):
    path = config.MODELS_DIR / f"{name}.pt"
    if not path.exists():
        return None, name
    cfg = model_cfg or ModelConfig()
    model = cls(splits.X_train.shape[-1], splits.y_train.shape[-1], cfg)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()
    return model, name


def _load_hpo_model(cls, splits, model_type: str):
    config_path = config.METRICS_DIR / f"best_hpo_{model_type}_config.json"
    model_path = config.MODELS_DIR / f"{model_type}_best_hpo.pt"
    if not model_path.exists() or not config_path.exists():
        return None, None
    with open(config_path) as f:
        cfg_dict = json.load(f)
    model_cfg = ModelConfig(**cfg_dict["model_cfg"])
    model = cls(splits.X_train.shape[-1], splits.y_train.shape[-1], model_cfg)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()
    return model, model_type


def _nn_pred_vol(model, splits):
    _, _, test_loader = ds.make_loaders(splits)
    pred_std = predict(model, test_loader, config.DEVICE)
    pred_log = ds.inverse_target(pred_std, splits)
    return np.exp(pred_log)


def _load_all_models(splits):
    """Incarca toate modelele disponibile. Prefera HPO fata de referinta."""
    models = {}

    lstm_hpo, _ = _load_hpo_model(LSTMRegressor, splits, "lstm")
    if lstm_hpo is not None:
        models["LSTM (HPO)"] = _nn_pred_vol(lstm_hpo, splits)
        print("  [OK] LSTM (HPO) incarcat")
    else:
        lstm_ref, _ = _load_model(LSTMRegressor, splits, "lstm")
        if lstm_ref is not None:
            models["LSTM"] = _nn_pred_vol(lstm_ref, splits)
            print("  [OK] LSTM referinta incarcat (fara HPO)")

    attn_hpo, _ = _load_hpo_model(Seq2SeqAttention, splits, "attention")
    if attn_hpo is not None:
        models["Attention (HPO)"] = _nn_pred_vol(attn_hpo, splits)
        print("  [OK] Attention (HPO) incarcat")
    else:
        attn_ref, _ = _load_model(Seq2SeqAttention, splits, "attention")
        if attn_ref is not None:
            models["Attention"] = _nn_pred_vol(attn_ref, splits)
            print("  [OK] Attention referinta incarcat (fara HPO)")

    return models


# ──────────────────────────────────────────────────────────────────────────────
# Selectie interactiva de simboluri
# ──────────────────────────────────────────────────────────────────────────────

def _select_symbols(cli_symbols: list[str] | None, available: list[str]) -> list[str]:
    """Returneaza lista de simboluri selectate, validata."""
    all_sym = sorted(available)

    if cli_symbols:
        invalid = [s for s in cli_symbols if s.upper() not in all_sym]
        if invalid:
            print(f"\n  [ATENTIE] Simboluri necunoscute (ignorate): {invalid}")
        chosen = [s.upper() for s in cli_symbols if s.upper() in all_sym]
        if not chosen:
            print("  Niciun simbol valid specificat — se folosesc toate.")
            return all_sym
        return chosen

    # ── Mod interactiv ──
    print("\nStock-uri disponibile in setul de date:")
    chunks = [all_sym[i:i + 10] for i in range(0, len(all_sym), 10)]
    for chunk in chunks:
        print("  " + "  ".join(f"{s:<5}" for s in chunk))

    print(f"\nTotal disponibile: {len(all_sym)} simboluri")
    print("Introdu ticker-ele separate prin spatii (ex: AA JPM GS BAC).")
    print("Apasa ENTER fara text pentru a selecta TOATE simbolurile.")
    print()

    while True:
        raw = input("  Simboluri alese: ").strip()
        if not raw:
            print(f"  -> Selectate toate cele {len(all_sym)} simboluri.")
            return all_sym
        chosen = [s.upper() for s in raw.split()]
        invalid = [s for s in chosen if s not in all_sym]
        valid = [s for s in chosen if s in all_sym]
        if invalid:
            print(f"  [EROARE] Simboluri necunoscute: {invalid}")
            print(f"  Alege din: {all_sym}")
            continue
        if not valid:
            print("  Lista goala. Incearca din nou.")
            continue
        print(f"  -> Selectate: {valid}")
        return valid


# ──────────────────────────────────────────────────────────────────────────────
# Generare grafice per simbol
# ──────────────────────────────────────────────────────────────────────────────

def _plots_for_symbol(sym, meta, true_vol, preds, garch_preds, prev_vol_arr,
                      best_model_name):
    """Genereaza toate graficele pentru un simbol si returneaza randurile de metrici."""
    idx = meta.index[meta["Symbol"] == sym].to_numpy()
    if len(idx) == 0:
        print(f"  [{sym}] Nu exista date in setul de test — sarind peste.")
        return []

    order = np.argsort(meta.loc[idx, "Date"].to_numpy())
    idx = idx[order]
    dates = meta.loc[idx, "Date"].to_numpy()
    pv_sym = prev_vol_arr[idx]

    actuals_by_h = {h: true_vol[idx, j] for j, h in enumerate(config.HORIZONS)}

    # ── Graf 1: toate modelele pe h=1 ──
    preds_h1 = {}
    for name, pv_arr in preds.items():
        preds_h1[name] = pv_arr[idx, 0]
    if garch_preds is not None:
        preds_h1["GARCH"] = garch_preds[idx, 0]
    preds_h1["Naiv"] = pv_sym

    out1 = config.PLOTS_DIR / f"stock_{sym}_h1_toate_modelele.png"
    plots.plot_all_models_h1(dates, actuals_by_h[1], preds_h1, sym, out1)
    print(f"    -> {out1.name}")

    # ── Graf 2: toate orizonturile pentru cel mai bun model ──
    if best_model_name and best_model_name in preds:
        best_pv = preds[best_model_name]
        preds_by_h = {h: best_pv[idx, j] for j, h in enumerate(config.HORIZONS)}
        out2 = config.PLOTS_DIR / f"stock_{sym}_toate_orizonturile.png"
        plots.plot_all_horizons(dates, actuals_by_h, preds_by_h, sym,
                                best_model_name, out2)
        print(f"    -> {out2.name}")

    # ── Graf 3: metrici per model (bar chart) ──
    # Toate array-urile de predictii trebuie sa fie in spatiul global de indici
    # (marime n_test_total, nu per-simbol) pentru ca idx contine indici globali.
    metrics_rows = []
    all_preds_for_metrics = dict(preds)
    if garch_preds is not None:
        all_preds_for_metrics["GARCH"] = garch_preds
    # Naiv (persistence) in spatiu global: prev_vol_arr are n_test_total randuri
    all_preds_for_metrics["Naiv"] = np.repeat(
        prev_vol_arr[:, None], len(config.HORIZONS), axis=1
    )

    for name, pv_arr in all_preds_for_metrics.items():
        for j, h in enumerate(config.HORIZONS):
            yt = actuals_by_h[h]
            yp = pv_arr[idx, j] if pv_arr.ndim == 2 else pv_arr[idx]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() == 0:
                continue
            pv_prev = pv_sym[mask] if name != "Naiv" else None
            met = M.all_metrics(yt[mask], yp[mask], pv_prev)
            metrics_rows.append({"Symbol": sym, "model": name, "horizon": h, **met})

    if metrics_rows:
        out3 = config.PLOTS_DIR / f"stock_{sym}_metrici.png"
        plots.plot_metrics_per_stock(metrics_rows, sym, out3)
        print(f"    -> {out3.name}")

    return metrics_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Simboluri NYSE (ex: --symbols JPM GS BAC). "
                         "Omite pentru selectie interactiva.")
    args = ap.parse_args()

    print_banner(6, 6, "SELECTIE STOCK-URI SI GRAFICE FINALE", [
        "Ce face:",
        "  • Arata lista de stock-uri disponibile",
        "  • Permite selectia unui subset pentru grafice detaliate",
        "  • Genereaza per stock ales:",
        "      Graf 1: Volatilitate reala vs. prezisa (h=1, toate modelele)",
        "      Graf 2: Toate orizonturile h=1,5,10,20 (cel mai bun model)",
        "      Graf 3: Bar chart metrici RMSE/MAE per model",
        "  • Afiseaza si salveaza tabel comparativ CSV",
        "",
        "Modele folosite (in ordine de prioritate):",
        "  1. Modele HPO (lstm_best_hpo.pt, attention_best_hpo.pt) — de la pasul 4",
        "  2. Modele de referinta (lstm.pt, attention.pt) — daca HPO nu exista",
        "  + GARCH baseline + Modelul Naiv (persistence)",
    ])

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    # ── Context: stocuri disponibile pentru grafice ──
    print_data_summary(feats, splits)

    y_true_log = ds.inverse_target(splits.y_test.numpy(), splits)
    true_vol = np.exp(y_true_log)
    meta = splits.meta_test.copy()

    base = feats[["Symbol", "Date", "vol_garman_klass"]].rename(
        columns={"vol_garman_klass": "prev_vol"})
    meta = meta.merge(base, on=["Symbol", "Date"], how="left")
    prev_vol_arr = meta["prev_vol"].to_numpy()

    # ── Incarcare modele ──
    print("\nIncarcare modele disponibile:")
    preds = _load_all_models(splits)

    garch_preds = None
    garch_path = config.METRICS_DIR / "garch_forecasts.pkl"
    if garch_path.exists():
        fc = pd.read_pickle(garch_path)
        m = meta.merge(fc, on=["Symbol", "Date"], how="left")
        garch_cols = [f"vol_garch_h{h}" for h in config.HORIZONS]
        garch_preds = m[garch_cols].to_numpy()
        print("  [OK] GARCH forecasts incarcate")
    else:
        print("  [!!] Lipsesc forecasturile GARCH")

    print(f"\nModele disponibile pentru grafice: "
          f"{list(preds.keys()) + (['GARCH'] if garch_preds is not None else []) + ['Naiv']}")

    # ── Cel mai bun model pentru graficul cu toate orizonturile ──
    preferred = ["LSTM (HPO)", "Attention (HPO)", "LSTM", "Attention"]
    best_model_name = next((m for m in preferred if m in preds), None)
    if best_model_name:
        print(f"\nModel principal pentru graficul cu toate orizonturile: {best_model_name}")

    # ── Selectie simboluri ──
    available_syms = sorted(meta["Symbol"].unique().tolist())
    chosen_syms = _select_symbols(args.symbols, available_syms)

    # ── Generare grafice ──
    all_metrics_rows = []
    print(f"\nGenerez grafice pentru {len(chosen_syms)} simboluri alese...\n")
    for sym in chosen_syms:
        print(f"  [{sym}]")
        rows = _plots_for_symbol(sym, meta, true_vol, preds, garch_preds,
                                 prev_vol_arr, best_model_name)
        all_metrics_rows.extend(rows)

    if not all_metrics_rows:
        print("\nNiciun grafic generat (date insuficiente).")
        return

    # ── Tabel comparativ in consola ──
    results_df = pd.DataFrame(all_metrics_rows)
    results_h1 = results_df[results_df["horizon"] == 1].copy()

    print(f"\n{'=' * 72}")
    print("  REZULTATE FINALE — STOCK-URI ALESE (h=1 zi, set de test)")
    print(f"{'─' * 72}")
    display_cols = ["Symbol", "model", "rmse", "mae", "r2", "qlike"]
    avail = [c for c in display_cols if c in results_h1.columns]
    fmt = results_h1[avail].copy()
    for col in ["rmse", "mae", "r2", "qlike"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.5f}")
    print(fmt.to_string(index=False))
    print(f"{'=' * 72}")

    # ── Tabel complet pe toate orizonturile ──
    out_csv = config.METRICS_DIR / "results_selected_stocks.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Tabel complet (toate orizonturile) salvat: {out_csv}")
    print(f"  Grafice salvate in: {config.PLOTS_DIR}")
    print(f"  Fisiere generate:")
    for sym in chosen_syms:
        print(f"    stock_{sym}_h1_toate_modelele.png")
        if best_model_name:
            print(f"    stock_{sym}_toate_orizonturile.png")
        print(f"    stock_{sym}_metrici.png")


if __name__ == "__main__":
    main()
