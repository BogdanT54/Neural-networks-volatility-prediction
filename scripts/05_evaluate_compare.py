"""Pas 5: evalueaza si compara toate modelele (GARCH, LSTM, Attention, Naiv).

Produce:
- un tabel de metrici per model si orizont (CSV + afisare),
- grafice comparative (bar charts RMSE/QLIKE),
- grafice predictie vs realitate pentru un simbol,
toate in scara volatilitatii (nu logaritmica).

Foloseste modelele HPO (din pasul 4) daca exista, altfel modelele de referinta.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from _common import config, get_features, print_banner, print_data_summary, set_seed
from nyse_vol.config import ModelConfig
from nyse_vol.data import dataset as ds
from nyse_vol.data.features import TARGET_COLS
from nyse_vol.eval import metrics as M
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.trainer import predict


def _load_model(cls, splits, name, model_cfg=None):
    """Incarca un model din checkpoint. Daca model_cfg e None, foloseste ModelConfig()."""
    path = config.MODELS_DIR / f"{name}.pt"
    if not path.exists():
        return None
    cfg = model_cfg or ModelConfig()
    model = cls(splits.X_train.shape[-1], splits.y_train.shape[-1], cfg)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()
    return model


def _load_hpo_model(cls, splits, model_type: str):
    """Incarca modelul reantrenat dupa HPO, cu config-ul sau specific."""
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
    return model, cfg_dict


def _nn_pred_vol(model, splits):
    """Predictii NN in scara volatilitatii: (N, n_horizons)."""
    _, _, test_loader = ds.make_loaders(splits)
    pred_std = predict(model, test_loader, config.DEVICE)
    pred_log = ds.inverse_target(pred_std, splits)
    return np.exp(pred_log)


def _print_metrics_explanation():
    print("""
  METRICI FOLOSITE:
  ─────────────────────────────────────────────────────────────
  RMSE (Root Mean Squared Error)
    Eroarea medie patratica — in aceleasi unitati ca volatilitatea.
    Penalizeaza mai mult erorile mari. Mai mic = mai bun.

  MAE (Mean Absolute Error)
    Eroarea medie absoluta. Mai robusta la outlieri decat RMSE.
    Mai mic = mai bun.

  R² (Coefficient of Determination)
    Cat din variatia volatilitatii reale este explicata de model.
    1.0 = predictie perfecta | 0.0 = la fel cu media | <0 = mai rau.

  QLIKE (Quasi-Likelihood)
    Metrica asimetrica specifica volatilitatii: penalizeaza mai mult
    SUBESTIMAREA decat supraestimarea. Importanta in managementul
    riscului: e mai periculos sa crezi ca piata e calma cand nu e.
    Mai mic = mai bun.

  Dir.Acc (Directional Accuracy)
    Procentul de zile in care modelul a prezis corect DIRECTIA
    schimbarii volatilitatii (creste sau scade fata de ziua precedenta).
    0.5 = aleator | >0.5 = are valoare predictiva.

  MODELUL NAIV (Persistence Baseline):
    Cel mai simplu predictor posibil: presupune ca volatilitatea
    de maine va fi IDENTICA cu volatilitatea de azi.
    Formula: pred_naiv(t+h) = vol_realizata(t)  pentru orice h.

    De ce il includem?
    → Orice model serios TREBUIE sa bata acest baseline trivial.
    → Daca LSTM nu il bate, modelul nu aduce valoare.
    → Volatilitatea e mean-reverting, deci Naivul e surprinzator
      de competitiv pe orizonturi scurte (h=1 zi).
    → Este limita inferioara a utilitatii unui model.
  ─────────────────────────────────────────────────────────────
""")


def main():
    print_banner(5, 6, "EVALUARE SI COMPARARE MODELE", [
        "Ce face:",
        "  • Incarca toate modelele antrenate (GARCH, LSTM, Attention, Naiv)",
        "  • Prefera modelele HPO (pasul 4) fata de cele de referinta (pasul 3)",
        "  • Calculeaza metrici pe setul de TEST (> 2021-12-31)",
        "  • Genereaza bar charts comparative si grafice pred vs. real",
        "",
        "Rezultate: artifacts/metrics/comparison.csv",
        "Grafice:   artifacts/plots/compare_*.png",
        "Urmator:   python 06_select_and_plot.py  (grafice per stock ales)",
    ])

    _print_metrics_explanation()

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    # ── Context: ce stocuri sunt evaluate, pe ce perioada de test ──
    print_data_summary(feats, splits)

    y_true_log = ds.inverse_target(splits.y_test.numpy(), splits)
    true_vol = np.exp(y_true_log)
    meta = splits.meta_test.copy()

    base = feats[["Symbol", "Date", "vol_garman_klass"]].rename(
        columns={"vol_garman_klass": "prev_vol"})
    meta = meta.merge(base, on=["Symbol", "Date"], how="left")
    prev_vol = meta["prev_vol"].to_numpy()

    preds = {}
    model_labels = {}

    # ── Modele HPO (prioritate) ──
    lstm_hpo, lstm_hpo_cfg = _load_hpo_model(LSTMRegressor, splits, "lstm")
    if lstm_hpo is not None:
        preds["LSTM (HPO)"] = _nn_pred_vol(lstm_hpo, splits)
        model_labels["LSTM (HPO)"] = lstm_hpo_cfg
        print(f"  [OK] LSTM HPO incarcat  (best_val={lstm_hpo_cfg.get('best_val_retrain', '?'):.4f})")
    else:
        lstm_ref = _load_model(LSTMRegressor, splits, "lstm")
        if lstm_ref is not None:
            preds["LSTM"] = _nn_pred_vol(lstm_ref, splits)
            print("  [OK] LSTM referinta incarcat (HPO nu exista — ruleaza pasul 4)")

    attn_hpo, attn_hpo_cfg = _load_hpo_model(Seq2SeqAttention, splits, "attention")
    if attn_hpo is not None:
        preds["Attention (HPO)"] = _nn_pred_vol(attn_hpo, splits)
        model_labels["Attention (HPO)"] = attn_hpo_cfg
        print(f"  [OK] Attention HPO incarcat  (best_val={attn_hpo_cfg.get('best_val_retrain', '?'):.4f})")
    else:
        attn_ref = _load_model(Seq2SeqAttention, splits, "attention")
        if attn_ref is not None:
            preds["Attention"] = _nn_pred_vol(attn_ref, splits)
            print("  [OK] Attention referinta incarcat (HPO nu exista — ruleaza pasul 4)")

    # ── Baseline Naiv (persistence) ──
    preds["Naiv"] = np.repeat(prev_vol[:, None], len(config.HORIZONS), axis=1)
    print("  [OK] Modelul Naiv (persistence) configurat")

    # ── GARCH ──
    garch_path = config.METRICS_DIR / "garch_forecasts.pkl"
    if garch_path.exists():
        fc = pd.read_pickle(garch_path)
        m = meta.merge(fc, on=["Symbol", "Date"], how="left")
        garch_cols = [f"vol_garch_h{h}" for h in config.HORIZONS]
        preds["GARCH"] = m[garch_cols].to_numpy()
        print("  [OK] Forecasts GARCH incarcate")
    else:
        print("  [!!] Lipsesc forecasturile GARCH — ruleaza scripts/02_train_garch.py")

    # ── Calcul metrici ──
    print(f"\n{'─' * 62}")
    print("  Calcul metrici pe setul de test...")
    print(f"{'─' * 62}")

    rows = []
    rmse_by_model, qlike_by_model = {}, {}
    for name, pv in preds.items():
        rmse_by_model[name], qlike_by_model[name] = {}, {}
        for j, h in enumerate(config.HORIZONS):
            yt, yp = true_vol[:, j], pv[:, j]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() == 0:
                continue
            pv_prev = prev_vol[mask] if name != "Naiv" else None
            met = M.all_metrics(yt[mask], yp[mask], pv_prev)
            rows.append({"model": name, "horizon": h, **met})
            rmse_by_model[name][h] = met["rmse"]
            qlike_by_model[name][h] = met["qlike"]

    table = pd.DataFrame(rows)
    out_csv = config.METRICS_DIR / "comparison.csv"
    table.to_csv(out_csv, index=False)

    print(f"\n{'=' * 62}")
    print("  COMPARATIE MODELE — SET DE TEST")
    print(f"{'─' * 62}")
    print(table.to_string(index=False))
    print(f"{'─' * 62}")
    print(f"  Tabel salvat: {out_csv}")
    print(f"{'=' * 62}")

    # ── Grafice comparative (bar charts RMSE/QLIKE) ──
    plots.plot_model_comparison(rmse_by_model, "rmse",
                                config.PLOTS_COMPARATII_DIR / "compare_rmse.png")
    plots.plot_model_comparison(qlike_by_model, "qlike",
                                config.PLOTS_COMPARATII_DIR / "compare_qlike.png")

    # ── Grafice pred vs real pentru simbolul cel mai frecvent ──
    sym = meta["Symbol"].value_counts().index[0]
    idx = meta.index[meta["Symbol"] == sym].to_numpy()
    order = np.argsort(meta.loc[idx, "Date"].to_numpy())
    idx = idx[order]
    dates = meta.loc[idx, "Date"].to_numpy()
    for name, pv in preds.items():
        if not np.isfinite(pv[idx, 0]).any():
            continue
        fname = f"pred_vs_true_{name.lower().replace(' ', '_')}_{sym}.png"
        plots.plot_pred_vs_true(
            dates, true_vol[idx, 0], pv[idx, 0],
            f"{name}: volatilitate prezisa vs reala — {sym} (h=1)",
            config.PLOTS_PREDICTII_DIR / fname)

    print(f"\n  Grafice comparatii: {config.PLOTS_COMPARATII_DIR}")
    print(f"    compare_rmse.png, compare_qlike.png")
    print(f"  Grafice predictii:  {config.PLOTS_PREDICTII_DIR}")
    print(f"    pred_vs_true_*_{sym}.png  (simbol exemplu)")
    print(f"\n  Pentru grafice detaliate per stock ales:")
    print(f"    python 06_select_and_plot.py --symbols {sym} JPM GS")
    print(f"    python 06_select_and_plot.py  (selectie interactiva)")


if __name__ == "__main__":
    main()
