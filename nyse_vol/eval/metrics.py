"""Metrici de evaluare pentru predictia volatilitatii.

Toate metricile lucreaza pe volatilitatea in scara originala (nu logaritmica),
exceptand acolo unde se specifica. QLIKE este metrica standard pentru volatilitate
si penalizeaza asimetric subestimarea.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def qlike(y_true_vol, y_pred_vol):
    """QLIKE pe variante: mean( v_true/v_pred - log(v_true/v_pred) - 1 )."""
    vt = np.clip(y_true_vol, 1e-8, None) ** 2
    vp = np.clip(y_pred_vol, 1e-8, None) ** 2
    ratio = vt / vp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def directional_accuracy(y_true_vol, y_pred_vol, prev_vol):
    """Acuratetea directiei schimbarii de volatilitate fata de nivelul anterior."""
    true_dir = np.sign(y_true_vol - prev_vol)
    pred_dir = np.sign(y_pred_vol - prev_vol)
    valid = true_dir != 0
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(true_dir[valid] == pred_dir[valid]))


def all_metrics(y_true_vol, y_pred_vol, prev_vol=None) -> dict:
    out = {
        "rmse": rmse(y_true_vol, y_pred_vol),
        "mae": mae(y_true_vol, y_pred_vol),
        "r2": r2(y_true_vol, y_pred_vol),
        "qlike": qlike(y_true_vol, y_pred_vol),
    }
    if prev_vol is not None:
        out["dir_acc"] = directional_accuracy(y_true_vol, y_pred_vol, prev_vol)
    return out
