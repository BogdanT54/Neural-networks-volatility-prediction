"""Construirea trasaturilor (features) si a tintelor pentru predictia volatilitatii.

Pentru fiecare simbol calculam, pe baza OHLCV:
- log-randament zilnic
- log-range (High/Low)
- log-volum (standardizat ulterior)
- cei patru/cinci estimatori de volatilitate zilnica (intrari multivariate)
- ziua din saptamana (ciclic, sin/cos)

Tinta multi-step: volatilitatea realizata medie pe urmatoarele h zile, pentru
fiecare orizont din config.HORIZONS, in scara logaritmica (mai stabila).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyse_vol import config
from nyse_vol.data import volatility as vol

# coloanele de trasaturi produse (in afara de eventuale extra)
FEATURE_COLS = [
    "log_ret", "log_range", "log_volume",
    "vol_parkinson", "vol_garman_klass", "vol_rogers_satchell", "vol_close_to_close",
    "dow_sin", "dow_cos",
]


def _per_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    o, h, l, c, v = g["Open"], g["High"], g["Low"], g["Close"], g["Volume"]

    g["log_ret"] = np.log(c / c.shift(1))
    g["log_range"] = np.log(h / l)
    g["log_volume"] = np.log(v.clip(lower=1))

    # estimatori de volatilitate zilnica (radacina variantei)
    g["vol_parkinson"] = np.sqrt(vol.parkinson(h, l).clip(lower=0))
    g["vol_garman_klass"] = np.sqrt(vol.garman_klass(o, h, l, c).clip(lower=0))
    g["vol_rogers_satchell"] = np.sqrt(vol.rogers_satchell(o, h, l, c).clip(lower=0))
    g["vol_close_to_close"] = np.sqrt(vol.close_to_close(c).clip(lower=0))

    dow = g["Date"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # --- tinte: volatilitate realizata medie pe urmatoarele h zile ---
    daily_var = vol.daily_variance(g, config.TARGET_ESTIMATOR).clip(lower=0)
    for hz in config.HORIZONS:
        # media variantei pe ferestrele viitoare [t+1, t+h], apoi radacina
        fwd_var = (daily_var.shift(-1)
                   .rolling(hz).mean()
                   .shift(-(hz - 1)))
        g[f"target_h{hz}"] = np.log(np.sqrt(fwd_var).clip(lower=1e-8))
    return g


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Aplica calculul de trasaturi si tinte pe fiecare simbol din panel."""
    parts = []
    for sym, g in panel.groupby("Symbol"):
        gf = _per_symbol_features(g)
        gf["Symbol"] = sym
        parts.append(gf)
    out = pd.concat(parts, ignore_index=True)
    target_cols = [f"target_h{h}" for h in config.HORIZONS]
    out = out.dropna(subset=FEATURE_COLS + target_cols).reset_index(drop=True)
    return out


TARGET_COLS = [f"target_h{h}" for h in config.HORIZONS]
