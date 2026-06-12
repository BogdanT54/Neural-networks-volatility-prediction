"""Construirea ferestrelor glisante, split cronologic, scalare si DataLoader-e.

Transforma DataFrame-ul cu trasaturi (per simbol, pe zi) in tensori cu ferestre
de lungime `WINDOW`, alaturi de vectorul tinta multi-orizont. Split-ul este
cronologic (pe data ultimei zile din fereastra) pentru a evita scurgerea de
informatie din viitor in trecut. Scalarea (standardizare) este fit-uita doar pe
partitia de train.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nyse_vol import config
from nyse_vol.data.features import FEATURE_COLS, TARGET_COLS


@dataclass
class Splits:
    """Containere cu tensori pentru cele trei partitii + scaler-ul folosit."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_scaler: StandardScaler
    target_mean: np.ndarray
    target_std: np.ndarray
    meta_test: pd.DataFrame  # Symbol + Date pentru ultima zi a fiecarei ferestre din test


def _windows_for_symbol(g: pd.DataFrame, window: int):
    """Genereaza ferestre (X) si tinte (y) pentru un simbol, plus meta (Symbol,Date)."""
    g = g.sort_values("Date")
    feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = g[TARGET_COLS].to_numpy(dtype=np.float32)
    dates = g["Date"].to_numpy()
    sym = g["Symbol"].iloc[0]

    n = len(g)
    Xs, ys, end_dates = [], [], []
    for end in range(window - 1, n):
        start = end - window + 1
        Xs.append(feats[start:end + 1])
        ys.append(targets[end])
        end_dates.append(dates[end])
    if not Xs:
        return None
    meta = pd.DataFrame({"Symbol": sym, "Date": end_dates})
    return np.stack(Xs), np.stack(ys), meta


def build_splits(feat_df: pd.DataFrame, window: int | None = None) -> Splits:
    window = window or config.WINDOW
    train_end = pd.Timestamp(config.TRAIN_END)
    val_end = pd.Timestamp(config.VAL_END)

    X_parts, y_parts, meta_parts = [], [], []
    for _, g in feat_df.groupby("Symbol"):
        res = _windows_for_symbol(g, window)
        if res is None:
            continue
        X, y, meta = res
        X_parts.append(X)
        y_parts.append(y)
        meta_parts.append(meta)

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    meta = pd.concat(meta_parts, ignore_index=True)

    end_dates = meta["Date"].values
    train_mask = end_dates <= np.datetime64(train_end)
    val_mask = (end_dates > np.datetime64(train_end)) & (end_dates <= np.datetime64(val_end))
    test_mask = end_dates > np.datetime64(val_end)

    # --- scalare trasaturi (fit doar pe train) ---
    n_feat = X.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_feat))

    def scale_X(arr):
        flat = arr.reshape(-1, n_feat)
        return scaler.transform(flat).reshape(arr.shape).astype(np.float32)

    # --- standardizare tinte (fit doar pe train) ---
    t_mean = y[train_mask].mean(axis=0)
    t_std = y[train_mask].std(axis=0) + 1e-8

    def scale_y(arr):
        return ((arr - t_mean) / t_std).astype(np.float32)

    to_t = lambda a: torch.from_numpy(a)
    return Splits(
        X_train=to_t(scale_X(X[train_mask])), y_train=to_t(scale_y(y[train_mask])),
        X_val=to_t(scale_X(X[val_mask])), y_val=to_t(scale_y(y[val_mask])),
        X_test=to_t(scale_X(X[test_mask])), y_test=to_t(scale_y(y[test_mask])),
        feature_scaler=scaler, target_mean=t_mean, target_std=t_std,
        meta_test=meta[test_mask].reset_index(drop=True),
    )


def make_loaders(splits: Splits, batch_size: int | None = None):
    bs = batch_size or config.TrainConfig().batch_size
    train = DataLoader(TensorDataset(splits.X_train, splits.y_train),
                       batch_size=bs, shuffle=True, drop_last=False)
    val = DataLoader(TensorDataset(splits.X_val, splits.y_val),
                     batch_size=bs, shuffle=False)
    test = DataLoader(TensorDataset(splits.X_test, splits.y_test),
                      batch_size=bs, shuffle=False)
    return train, val, test


def inverse_target(scaled: np.ndarray, splits: Splits) -> np.ndarray:
    """Readuce tintele standardizate in scara log-volatilitatii originale."""
    return scaled * splits.target_std + splits.target_mean
