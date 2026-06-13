"""Pas 4: cautare de hiperparametri (random search) + reantrenare cu config optim.

Logheaza fiecare configuratie incercata intr-un CSV, raporteaza cele mai bune
configuratii si retreneaza modelul ales cu numarul complet de epoci.
Modelul reantrenat este cel folosit la evaluare si grafice (pasii 5 si 6).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import torch

from _common import config, get_features, print_banner, set_seed
from nyse_vol.config import ModelConfig, TrainConfig
from nyse_vol.data import dataset as ds
from nyse_vol.eval import plots
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.hpo import best_configs, random_search
from nyse_vol.train.trainer import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10,
                    help="epoci per trial HPO (mai putine = HPO mai rapid)")
    ap.add_argument("--retrain-epochs", type=int, default=None,
                    help="epoci pentru reantrenarea finala (implicit: config.TrainConfig.epochs)")
    ap.add_argument("--model", choices=["lstm", "attention"], default="lstm")
    args = ap.parse_args()

    factory = LSTMRegressor if args.model == "lstm" else Seq2SeqAttention
    retrain_epochs = args.retrain_epochs or TrainConfig().epochs

    print_banner(4, 6, f"OPTIMIZARE HIPERPARAMETRI — {args.model.upper()}", [
        "Ce face:",
        f"  • Ruleaza {args.trials} trial-uri cu configuratii aleatoare",
        f"  • Fiecare trial: {args.epochs} epoci (mai putine decat antrenarea finala)",
        "  • Evalueaza pe setul de VALIDARE (niciodata pe test!)",
        "  • Selecteaza configuratia cu cel mai mic val_loss",
        f"  • Retreneaza cu configuratia optima ({retrain_epochs} epoci complete)",
        f"  • Salveaza: artifacts/models/{args.model}_best_hpo.pt",
        "",
        "Spatiul de cautare:",
        f"  hidden_size:    {config.SEARCH_SPACE.hidden_size}",
        f"  num_layers:     {config.SEARCH_SPACE.num_layers}",
        f"  dropout:        {config.SEARCH_SPACE.dropout}",
        f"  bidirectional:  {config.SEARCH_SPACE.bidirectional}",
        f"  head_activation:{config.SEARCH_SPACE.head_activation}",
        f"  optimizer:      {config.SEARCH_SPACE.optimizer}",
        f"  lr:             {config.SEARCH_SPACE.lr}",
        f"  weight_decay:   {config.SEARCH_SPACE.weight_decay}",
        "",
        "De ce HPO pe validare si nu pe test?",
        "  Setul de test este INTOTDEAUNA secret pana la evaluarea finala.",
        "  Daca am alege hiperparametrii pe test, am 'trisа' — modelul",
        "  ar fi suprainvatat pe test si nu ar generaliza pe date noi.",
        "",
        f"Rezultat: artifacts/models/{args.model}_best_hpo.pt",
        "Urmator:  python 05_evaluate_compare.py",
    ])

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    log_path = config.METRICS_DIR / f"hpo_{args.model}.csv"
    print(f"Incep cautarea: {args.trials} trials x {args.epochs} epoci...\n")

    df = random_search(
        factory, splits, ds.make_loaders,
        n_trials=args.trials, epochs=args.epochs,
        device=config.DEVICE, seed=config.SEED, log_path=log_path,
    )

    print(f"\n{'=' * 62}")
    print(f"  TOP 5 CONFIGURATII (dupa val_loss):")
    print(f"{'─' * 62}")
    cols = ["trial", "best_val", "rmse_val_h1", "hidden_size",
            "num_layers", "dropout", "optimizer", "lr"]
    avail_cols = [c for c in cols if c in df.columns]
    print(df.head(5)[avail_cols].to_string(index=False))
    print(f"{'─' * 62}")
    print(f"  Log complet: {log_path}")
    print(f"{'=' * 62}")

    # ── Reantrenare cu configuratia optima ──
    model_cfg, train_cfg_hpo = best_configs(df)
    full_cfg = replace(
        TrainConfig(
            optimizer=train_cfg_hpo.optimizer,
            lr=train_cfg_hpo.lr,
            weight_decay=train_cfg_hpo.weight_decay,
        ),
        epochs=retrain_epochs,
    )

    bidir_str = "bidirectional" if model_cfg.bidirectional else "unidirectional"
    print(f"\n{'─' * 62}")
    print(f"  REANTRENARE COMPLETA cu configuratia optima:")
    print(f"  Model:       {args.model.upper()} ({bidir_str})")
    print(f"  hidden_size: {model_cfg.hidden_size}  |  num_layers: {model_cfg.num_layers}")
    print(f"  dropout:     {model_cfg.dropout}  |  activation: {model_cfg.head_activation}")
    print(f"  optimizer:   {full_cfg.optimizer}  |  lr: {full_cfg.lr}  "
          f"|  weight_decay: {full_cfg.weight_decay}")
    print(f"  Epoci:       {retrain_epochs} (antrenare completa)")
    print(f"{'─' * 62}\n")

    n_features = splits.X_train.shape[-1]
    n_outputs = splits.y_train.shape[-1]
    tl, vl, _ = ds.make_loaders(splits, full_cfg.batch_size)
    model = factory(n_features, n_outputs, model_cfg)
    model, history = train_model(model, tl, vl, full_cfg, device=config.DEVICE)

    model_name = f"{args.model}_best_hpo"
    model_path = config.MODELS_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)

    config_dict = {
        "model_type": args.model,
        "model_cfg": asdict(model_cfg),
        "train_cfg": {
            "optimizer": full_cfg.optimizer,
            "lr": full_cfg.lr,
            "weight_decay": full_cfg.weight_decay,
            "epochs": retrain_epochs,
        },
        "hpo_trials": args.trials,
        "hpo_epochs_per_trial": args.epochs,
        "best_val_hpo": float(df.iloc[0]["best_val"]),
        "best_val_retrain": float(history["best_val"]),
    }
    config_path = config.METRICS_DIR / f"best_hpo_{args.model}_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    plots.plot_loss_curves(
        history, f"Curbe de loss — {args.model.upper()} (HPO reantrenat)",
        config.PLOTS_DIR / f"{args.model}_best_hpo_loss.png",
    )

    print(f"\n{'=' * 62}")
    print(f"  REANTRENARE COMPLETA")
    print(f"  best_val (HPO trial):     {df.iloc[0]['best_val']:.4f}")
    print(f"  best_val (reantrenat):    {history['best_val']:.4f}")
    print(f"  Model salvat:   {model_path}")
    print(f"  Config salvat:  {config_path}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
