"""Baseline clasic GARCH pentru predictia volatilitatii.

Pentru fiecare simbol fitam un model GARCH(p,q) pe log-randamentele zilnice
(scalate la procente, conform conventiei bibliotecii `arch`) si producem
forecasturi de varianta pe orizont, convertite la volatilitate medie pe
orizontul cerut. Rezultatul este aliniat la aceeasi tinta ca modelele NN, pentru
comparatie corecta.

Evaluarea pe test se face walk-forward cu fereastra expanding si re-fit periodic
(la fiecare `refit_every` zile), astfel incat forecasturile variaza in timp si nu
folosesc informatie din viitor. Iterarea pe ordine (p,q) si pe distributia
reziduurilor (normal vs Student-t) reflecta cerinta de imbunatatire iterativa.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nyse_vol import config

try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:                       # pragma: no cover
    _HAS_ARCH = False


def _fit(returns_pct: pd.Series, p, q, dist):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(returns_pct, mean="Constant", vol="GARCH",
                        p=p, q=q, dist=dist, rescale=False)
        return am.fit(disp="off")


def garch_forecast_panel(panel: pd.DataFrame, test_start: str,
                         order=(1, 1), dist="normal", refit_every: int = 21) -> pd.DataFrame:
    """Forecasturi GARCH walk-forward pe test, per simbol si orizont.

    Intoarce un DataFrame cu Symbol, Date si vol_garch_h<H> (volatilitate zilnica
    medie pe orizont), comparabila direct cu tinta NN.
    """
    if not _HAS_ARCH:
        raise ImportError("Pachetul `arch` nu este instalat (pip install arch).")

    test_start = pd.Timestamp(test_start)
    p, q = order
    max_h = max(config.HORIZONS)
    rows = []

    for sym, g in panel.groupby("Symbol"):
        g = g.sort_values("Date")
        ret = (np.log(g["Close"] / g["Close"].shift(1)).dropna() * 100.0)
        ret.index = pd.DatetimeIndex(g["Date"].iloc[1:].values)
        if len(ret) < config.MIN_OBS_PER_SYMBOL:
            continue

        test_pos = np.where(ret.index >= test_start)[0]
        if len(test_pos) == 0 or test_pos[0] < 100:
            continue

        res = None
        for k, pos in enumerate(test_pos):
            # re-fit periodic pe fereastra expanding pana la ziua curenta (exclusiv)
            if res is None or k % refit_every == 0:
                hist = ret.iloc[:pos]
                try:
                    res = _fit(hist, p, q, dist)
                except Exception:
                    res = None
                    continue
            if res is None:
                continue
            try:
                fc = res.forecast(horizon=max_h, reindex=False)
                var_path = fc.variance.values[-1]      # (max_h,) in procente^2
            except Exception:
                continue
            row = {"Symbol": sym, "Date": ret.index[pos]}
            for h in config.HORIZONS:
                mean_var = var_path[:h].mean() / (100.0 ** 2)
                row[f"vol_garch_h{h}"] = float(np.sqrt(max(mean_var, 0.0)))
            rows.append(row)

    return pd.DataFrame(rows)
