"""Pas 3: antreneaza modelele NN (LSTM baseline si Encoder-Decoder + Attention).

Foloseste configuratia implicita din config. Salveaza checkpoint-uri, istoricul
de antrenare si curbele de loss. Pentru modelul cu atentie salveaza si un heatmap
de atentie pe un exemplu din test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from _common import config, get_features, set_seed
from nyse_vol.config import ModelConfig, TrainConfig
from nyse_vol.data import dataset as ds
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.trainer import train_model


def _save(model, name, history):
    torch.save(model.state_dict(), config.MODELS_DIR / f"{name}.pt")
    with open(config.METRICS_DIR / f"{name}_history.json", "w") as f:
        json.dump({k: v for k, v in history.items() if k != "config"} | {"config": history["config"]}, f, indent=2)
    plots.plot_loss_curves(history, f"Curbe de loss — {name}",
                           config.PLOTS_DIR / f"{name}_loss.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", choices=["lstm", "attention", "both"], default="both")
    args = ap.parse_args()

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)
    print(f"Ferestre: train={len(splits.X_train)}, val={len(splits.X_val)}, "
          f"test={len(splits.X_test)} | features={splits.X_train.shape[-1]} "
          f"outputs={splits.y_train.shape[-1]}")

    base_cfg = TrainConfig()
    if args.epochs:
        base_cfg = replace(base_cfg, epochs=args.epochs)

    if args.model in ("lstm", "both"):
        print("\n=== LSTM baseline ===")
        tl, vl, _ = ds.make_loaders(splits, base_cfg.batch_size)
        model = LSTMRegressor(splits.X_train.shape[-1], splits.y_train.shape[-1], ModelConfig())
        model, hist = train_model(model, tl, vl, base_cfg, device=config.DEVICE)
        _save(model, "lstm", hist)

    if args.model in ("attention", "both"):
        print("\n=== Encoder-Decoder + Attention ===")
        tl, vl, test_l = ds.make_loaders(splits, base_cfg.batch_size)
        model = Seq2SeqAttention(splits.X_train.shape[-1], splits.y_train.shape[-1], ModelConfig())
        model, hist = train_model(model, tl, vl, base_cfg, device=config.DEVICE)
        _save(model, "attention", hist)

        # heatmap de atentie pe primul batch din test
        model.return_attention = True
        model.eval()
        with torch.no_grad():
            xb = splits.X_test[:1].to(config.DEVICE)
            _, attn = model(xb)
        plots.plot_attention_heatmap(
            attn[0].cpu().numpy(), "Ponderi de atentie (exemplu test)",
            config.PLOTS_DIR / "attention_heatmap.png")
        model.return_attention = False

    print(f"\nArtefacte salvate in: {config.MODELS_DIR} si {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
