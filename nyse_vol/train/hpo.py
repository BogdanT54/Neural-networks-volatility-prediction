"""Cautare de hiperparametri (random search) pentru modelele NN.

Esantioneaza configuratii din spatiul definit in config.SEARCH_SPACE, antreneaza
fiecare model pe partitia de train, evalueaza pe validare si logheaza rezultatele
intr-un CSV. Selecteaza cea mai buna configuratie dupa pierderea de validare.

Acopera explicit cerinta profesorului de imbunatatire iterativa: variaza
optimizatorul, numarul de straturi (stacked), bidirectionalitatea, dimensiunea
ascunsa, functia de activare, dropout-ul, weight decay si tipul de atentie.
"""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pandas as pd

from nyse_vol import config
from nyse_vol.config import ModelConfig, TrainConfig
from nyse_vol.train.trainer import predict, train_model


def sample_config(rng: random.Random):
    s = config.SEARCH_SPACE
    model_cfg = ModelConfig(
        hidden_size=rng.choice(s.hidden_size),
        num_layers=rng.choice(s.num_layers),
        dropout=rng.choice(s.dropout),
        bidirectional=rng.choice(s.bidirectional),
        head_activation=rng.choice(s.head_activation),
        attention=rng.choice(s.attention),
    )
    train_cfg = TrainConfig(
        optimizer=rng.choice(s.optimizer),
        lr=rng.choice(s.lr),
        weight_decay=rng.choice(s.weight_decay),
    )
    return model_cfg, train_cfg


def _val_metrics_h1(model, splits, make_loaders, device):
    """Calculeaza RMSE si MAE pe val pentru orizontul h=1, in scara originala."""
    from nyse_vol.data.dataset import inverse_target
    _, val_loader, _ = make_loaders(splits)
    pred_std = predict(model, val_loader, device)
    pred_log = inverse_target(pred_std, splits)
    pred_vol = np.exp(pred_log[:, 0])

    true_log = inverse_target(splits.y_val.numpy(), splits)
    true_vol = np.exp(true_log[:, 0])

    mask = np.isfinite(true_vol) & np.isfinite(pred_vol)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((true_vol[mask] - pred_vol[mask]) ** 2)))
    mae = float(np.mean(np.abs(true_vol[mask] - pred_vol[mask])))
    return rmse, mae


def random_search(model_factory, splits, make_loaders, n_trials: int = 10,
                  epochs: int = 15, device: str = "cpu", seed: int = 42,
                  log_path=None) -> pd.DataFrame:
    """Ruleaza random search si intoarce un DataFrame cu rezultatele sortate.

    `model_factory(n_features, n_outputs, model_cfg)` construieste modelul.
    """
    rng = random.Random(seed)
    n_features = splits.X_train.shape[-1]
    n_outputs = splits.y_train.shape[-1]
    results = []
    print_every = max(1, epochs // 3)

    for trial in range(n_trials):
        model_cfg, train_cfg = sample_config(rng)
        train_cfg = replace(train_cfg, epochs=epochs)
        train_loader, val_loader, _ = make_loaders(splits, train_cfg.batch_size)
        model = model_factory(n_features, n_outputs, model_cfg)

        bidir_str = "bidir" if model_cfg.bidirectional else "unidir"
        print(f"\n[Trial {trial + 1}/{n_trials}] "
              f"hidden={model_cfg.hidden_size} | layers={model_cfg.num_layers} | "
              f"dropout={model_cfg.dropout} | {bidir_str} | "
              f"act={model_cfg.head_activation} | "
              f"opt={train_cfg.optimizer} | lr={train_cfg.lr}")

        _, history = train_model(model, train_loader, val_loader, train_cfg,
                                 device=device, verbose=True, print_every=print_every)

        val_rmse, val_mae = _val_metrics_h1(model, splits, make_loaders, device)
        print(f"  → best_val={history['best_val']:.4f} | "
              f"RMSE_val(h=1)={val_rmse:.5f} | MAE_val(h=1)={val_mae:.5f}")

        row = {
            "trial": trial + 1,
            "best_val": history["best_val"],
            "rmse_val_h1": val_rmse,
            "mae_val_h1": val_mae,
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "bidirectional": model_cfg.bidirectional,
            "head_activation": model_cfg.head_activation,
            "attention": model_cfg.attention,
            "optimizer": train_cfg.optimizer,
            "lr": train_cfg.lr,
            "weight_decay": train_cfg.weight_decay,
        }
        results.append(row)

    df = pd.DataFrame(results).sort_values("best_val").reset_index(drop=True)
    if log_path is not None:
        df.to_csv(log_path, index=False)
    return df


def best_configs(df: pd.DataFrame):
    """Reconstruieste (ModelConfig, TrainConfig) din cel mai bun rand."""
    r = df.iloc[0]
    model_cfg = ModelConfig(
        hidden_size=int(r["hidden_size"]), num_layers=int(r["num_layers"]),
        dropout=float(r["dropout"]), bidirectional=bool(r["bidirectional"]),
        head_activation=str(r["head_activation"]), attention=str(r["attention"]),
    )
    train_cfg = TrainConfig(
        optimizer=str(r["optimizer"]), lr=float(r["lr"]),
        weight_decay=float(r["weight_decay"]),
    )
    return model_cfg, train_cfg
