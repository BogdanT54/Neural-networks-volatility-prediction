"""Pas 5: evalueaza si compara GARCH vs LSTM vs Attention (+ baseline naiv).

Produce:
- un tabel de metrici per model si orizont (CSV + afisare),
- grafice comparative (bar charts RMSE/QLIKE),
- grafice predictie vs realitate pentru un simbol,
toate in scara volatilitatii (nu logaritmica).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import config, get_features, set_seed
from nyse_vol.data import dataset as ds
from nyse_vol.data.features import TARGET_COLS
from nyse_vol.eval import metrics as M
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.trainer import predict


def _load_model(cls, splits, name):
    path = config.MODELS_DIR / f"{name}.pt"
    if not path.exists():
        return None
    model = cls(splits.X_train.shape[-1], splits.y_train.shape[-1])
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()
    return model


def _nn_pred_vol(model, splits):
    """Predictii NN in scara volatilitatii: (N, n_horizons)."""
    _, _, test_loader = ds.make_loaders(splits)
    pred_std = predict(model, test_loader, config.DEVICE)
    pred_log = ds.inverse_target(pred_std, splits)
    return np.exp(pred_log)


def main():
    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    # tinte reale pe test, in scara volatilitatii, aliniate cu meta_test
    y_true_log = ds.inverse_target(splits.y_test.numpy(), splits)
    true_vol = np.exp(y_true_log)                          # (N, H)
    meta = splits.meta_test.copy()

    # nivel anterior (volatilitate zilnica curenta) pentru acuratete directionala
    base = feats[["Symbol", "Date", "vol_garman_klass"]].rename(
        columns={"vol_garman_klass": "prev_vol"})
    meta = meta.merge(base, on=["Symbol", "Date"], how="left")
    prev_vol = meta["prev_vol"].to_numpy()

    preds = {}

    # --- modele NN ---
    lstm = _load_model(LSTMRegressor, splits, "lstm")
    if lstm is not None:
        preds["LSTM"] = _nn_pred_vol(lstm, splits)
    attn = _load_model(Seq2SeqAttention, splits, "attention")
    if attn is not None:
        preds["Attention"] = _nn_pred_vol(attn, splits)

    # --- baseline naiv (persistenta): vol viitoare = vol zilnica curenta ---
    preds["Naiv"] = np.repeat(prev_vol[:, None], len(config.HORIZONS), axis=1)

    # --- GARCH (din forecasturile salvate la pasul 2), aliniat pe (Symbol,Date) ---
    garch_path = config.METRICS_DIR / "garch_forecasts.pkl"
    if garch_path.exists():
        fc = pd.read_pickle(garch_path)
        m = meta.merge(fc, on=["Symbol", "Date"], how="left")
        garch_cols = [f"vol_garch_h{h}" for h in config.HORIZONS]
        preds["GARCH"] = m[garch_cols].to_numpy()
    else:
        print("Atentie: lipsesc forecasturile GARCH (ruleaza scripts/02_train_garch.py).")

    # --- tabel de metrici per model x orizont ---
    rows = []
    rmse_by_model, qlike_by_model = {}, {}
    for name, pv in preds.items():
        rmse_by_model[name], qlike_by_model[name] = {}, {}
        for j, h in enumerate(config.HORIZONS):
            yt, yp = true_vol[:, j], pv[:, j]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() == 0:
                continue
            met = M.all_metrics(yt[mask], yp[mask], prev_vol[mask] if name != "Naiv" else None)
            rows.append({"model": name, "horizon": h, **met})
            rmse_by_model[name][h] = met["rmse"]
            qlike_by_model[name][h] = met["qlike"]

    table = pd.DataFrame(rows)
    out_csv = config.METRICS_DIR / "comparison.csv"
    table.to_csv(out_csv, index=False)
    print("\n=== Comparatie modele (test) ===")
    print(table.to_string(index=False))
    print(f"\nTabel salvat in: {out_csv}")

    # --- grafice comparative ---
    plots.plot_model_comparison(rmse_by_model, "rmse", config.PLOTS_DIR / "compare_rmse.png")
    plots.plot_model_comparison(qlike_by_model, "qlike", config.PLOTS_DIR / "compare_qlike.png")

    # --- predictie vs realitate pentru un simbol, orizont h=1 ---
    sym = meta["Symbol"].value_counts().index[0]
    idx = meta.index[meta["Symbol"] == sym].to_numpy()
    order = np.argsort(meta.loc[idx, "Date"].to_numpy())
    idx = idx[order]
    dates = meta.loc[idx, "Date"].to_numpy()
    for name, pv in preds.items():
        if not np.isfinite(pv[idx, 0]).any():
            continue
        plots.plot_pred_vs_true(
            dates, true_vol[idx, 0], pv[idx, 0],
            f"{name}: volatilitate prezisa vs reala — {sym} (h=1)",
            config.PLOTS_DIR / f"pred_vs_true_{name.lower()}_{sym}.png")

    print(f"Grafice salvate in: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
