"""Pas 3: antreneaza modelele NN de referinta (LSTM si Encoder-Decoder + Attention).

Foloseste configuratia implicita din config. Salveaza checkpoint-uri, istoricul
de antrenare si curbele de loss. Acestea sunt modele de REFERINTA — pentru
configuratia optima ruleaza pasul 4 (HPO).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from _common import config, get_features, print_banner, print_data_summary, set_seed
from nyse_vol.config import ModelConfig, TrainConfig
from nyse_vol.data import dataset as ds
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.trainer import train_model


def _save(model, name, history):
    torch.save(model.state_dict(), config.MODELS_DIR / f"{name}.pt")
    with open(config.METRICS_DIR / f"{name}_history.json", "w") as f:
        json.dump({k: v for k, v in history.items() if k != "config"}
                  | {"config": history["config"]}, f, indent=2)
    plots.plot_loss_curves(history, f"Curbe de loss — {name}",
                           config.PLOTS_ANTRENARE_DIR / f"{name}_loss.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", choices=["lstm", "attention", "both"], default="both")
    args = ap.parse_args()

    print_banner(3, 6, "ANTRENARE MODELE NEURONALE (REFERINTA)", [
        "Ce face:",
        "  • Antreneaza LSTM si/sau Encoder-Decoder cu Attention",
        "  • Foloseste hiperparametri IMPLICITI din config.py",
        "  • Salveaza checkpoint-urile (lstm.pt, attention.pt)",
        "",
        "Arhitectura LSTM:",
        f"  Intrare: fereastra de {config.WINDOW} zile x 9 features (standardizate)",
        "  Straturi: 2 LSTM stacked, hidden_size=64, dropout=0.2",
        "  Iesire:   4 valori = volatilitate prezisa pentru h=1,5,10,20 zile",
        "",
        "Arhitectura Attention (Encoder-Decoder):",
        "  Encoder LSTM consuma fereastra de intrare",
        "  Decoder genereaza auto-regresiv cate un orizont pe rand",
        "  Attention (Luong): decoderului ii 'permite sa priveasca inapoi'",
        "  la pozitiile cele mai relevante din fereastra de intrare",
        "",
        "Scalare date (anti-leakage):",
        "  Features: StandardScaler fit DOAR pe TRAIN, aplicat pe toate split-urile",
        "  Tinte:    Standardizare fit DOAR pe TRAIN (medie si std din train)",
        "  Motivul:  Modelul nu 'vede' informatii din viitor in timpul antrenarii",
        "",
        "Early stopping: oprire daca val_loss nu scade 6 epoci consecutive",
        "LR scheduler:   ReduceLROnPlateau (factor=0.5, patience=2 epoci)",
        "",
        "NOTA: Acesta este modelul de referinta cu hiperparametri impliciti.",
        "      Ruleaza pasul 04 (HPO) pentru a gasi configuratia optima,",
        "      care va fi folosita la evaluarea si graficele finale.",
        "",
        "Rezultat: artifacts/models/lstm.pt + attention.pt",
        "Urmator:  python 04_run_hpo.py --model lstm --trials 10 --epochs 15",
    ])

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    # ── Context complet: stocuri, date, split-uri ──
    print_data_summary(feats, splits)

    n_feat = splits.X_train.shape[-1]
    n_out  = splits.y_train.shape[-1]
    print(f"\n  Fiecare fereastra de intrare: {config.WINDOW} zile × {n_feat} features → {n_out} orizonturi prezise\n")

    base_cfg = TrainConfig()
    if args.epochs:
        base_cfg = replace(base_cfg, epochs=args.epochs)

    if args.model in ("lstm", "both"):
        print(f"\n{'─' * 62}")
        print("  LSTM — model de referinta")
        print(f"{'─' * 62}")
        tl, vl, _ = ds.make_loaders(splits, base_cfg.batch_size)
        model = LSTMRegressor(n_feat, n_out, ModelConfig())
        model, hist = train_model(model, tl, vl, base_cfg, device=config.DEVICE)
        _save(model, "lstm", hist)
        print(f"  Salvat: {config.MODELS_DIR / 'lstm.pt'}  "
              f"(best_val={hist['best_val']:.4f})")

    if args.model in ("attention", "both"):
        print(f"\n{'─' * 62}")
        print("  Encoder-Decoder + Attention — model de referinta")
        print(f"{'─' * 62}")
        tl, vl, test_l = ds.make_loaders(splits, base_cfg.batch_size)
        model = Seq2SeqAttention(n_feat, n_out, ModelConfig())
        model, hist = train_model(model, tl, vl, base_cfg, device=config.DEVICE)
        _save(model, "attention", hist)
        print(f"  Salvat: {config.MODELS_DIR / 'attention.pt'}  "
              f"(best_val={hist['best_val']:.4f})")

        model.return_attention = True
        model.eval()
        with torch.no_grad():
            xb = splits.X_test[:1].to(config.DEVICE)
            _, attn = model(xb)
        plots.plot_attention_heatmap(
            attn[0].cpu().numpy(), "Ponderi de atentie (exemplu test)",
            config.PLOTS_ANTRENARE_DIR / "attention_heatmap.png")
        model.return_attention = False

    print(f"\n{'=' * 62}")
    print(f"  Modele salvate in:  {config.MODELS_DIR}")
    print(f"  Grafice salvate in: {config.PLOTS_ANTRENARE_DIR}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
