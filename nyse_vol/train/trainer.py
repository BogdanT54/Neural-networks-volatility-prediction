"""Bucla de antrenare pentru modelele NN, cu early stopping si grad clipping.

Antrenarea se face pe MSE in scara tintei standardizate (log-volatilitate).
Functia `evaluate` intoarce pierderea medie, iar `predict` colecteaza predictiile
pe un loader intreg.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from nyse_vol.config import TrainConfig

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd": lambda params, **kw: torch.optim.SGD(params, momentum=0.9, **kw),
}


def make_optimizer(model, cfg: TrainConfig):
    opt_cls = _OPTIMIZERS[cfg.optimizer]
    return opt_cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += loss_fn(pred, yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def predict(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            outs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outs)


def train_model(model, train_loader, val_loader, cfg: TrainConfig | None = None,
                device: str = "cpu", verbose: bool = True):
    """Antreneaza modelul si intoarce (model, history) cu cel mai bun checkpoint."""
    cfg = cfg or TrainConfig()
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = make_optimizer(model, cfg)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
                 if cfg.lr_scheduler else None)

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item() * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose:
            print(f"  epoch {epoch + 1:02d}/{cfg.epochs} "
                  f"train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}")

        if patience >= cfg.early_stop_patience:
            if verbose:
                print(f"  early stopping la epoca {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best_val
    history["config"] = asdict(cfg)
    return model, history
