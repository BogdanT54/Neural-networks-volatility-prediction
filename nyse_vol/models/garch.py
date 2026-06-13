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


# --------------------------------------------------------------------------- #
# Teste de stationaritate
# --------------------------------------------------------------------------- #

def stationarity_tests(panel: pd.DataFrame) -> dict[str, dict]:
    """Ruleaza ADF, KPSS si Phillips-Perron pe log-randamentele din panel.

    Parametri
    ---------
    panel : DataFrame Silver cu coloanele Symbol, Date, Close.

    Returneaza
    ----------
    dict  {simbol -> {adf, kpss, pp}} cu statisticile si p-valorile.

    Note
    ----
    ADF  : H0 = nestationar (radacina unitara). p < 0.05 → stationara.
    KPSS : H0 = stationara.                     p < 0.05 → nestationara.
    PP   : H0 = nestationar (ca ADF, mai robust). p < 0.05 → stationara.
    Log-randamentele sunt aproape intotdeauna stationare; testele confirma
    ca GARCH este aplicabil fara diferentiere suplimentara.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.stattools import kpss as _kpss
        _HAS_STATSMODELS = True
    except ImportError:
        _HAS_STATSMODELS = False

    results: dict[str, dict] = {}

    for sym, g in panel.groupby("Symbol"):
        g = g.sort_values("Date")
        ret = (np.log(g["Close"] / g["Close"].shift(1)).dropna() * 100.0)
        if len(ret) < 50:
            continue

        r: dict = {}

        if _HAS_STATSMODELS:
            # ADF
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_stat, adf_p, _, _, _, _ = adfuller(ret, autolag="AIC")
            r["adf"] = {
                "stat": float(adf_stat),
                "p": float(adf_p),
                "stationary": bool(adf_p < 0.05),
            }

            # KPSS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k_stat, k_p, _, _ = _kpss(ret, regression="c", nlags="auto")
            r["kpss"] = {
                "stat": float(k_stat),
                "p": float(k_p),
                "stationary": bool(k_p >= 0.05),
            }
        else:
            r["adf"] = r["kpss"] = None

        # Phillips-Perron (din arch)
        try:
            from arch.unitroot import PhillipsPerron
            pp = PhillipsPerron(ret)
            r["pp"] = {
                "stat": float(pp.stat),
                "p": float(pp.pvalue),
                "stationary": bool(pp.pvalue < 0.05),
            }
        except Exception:
            r["pp"] = None

        results[sym] = r

    return results


def _fit(returns_pct: pd.Series, p, q, dist):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(returns_pct, mean="Constant", vol="GARCH",
                        p=p, q=q, dist=dist, rescale=False)
        return am.fit(disp="off")


def garch_forecast_panel(panel: pd.DataFrame, test_start: str,
                         order=(1, 1), dist="normal", refit_every: int = 21) -> pd.DataFrame:
    """Forecasturi GARCH walk-forward pe test, per simbol si orizont.

    La fiecare refit, generam un forecast multi-pas de lungime (refit_every + max_h)
    din care extragem zilnic valoarea corespunzatoare pozitiei in ciclul de refit.
    Astfel forecasturile variaza zilnic (decay GARCH spre varianta neconditionata)
    in loc sa fie constante intre refitturi (efect de treapta).

    Intoarce un DataFrame cu Symbol, Date si vol_garch_h<H> (volatilitate zilnica
    medie pe orizont), comparabila direct cu tinta NN.
    """
    if not _HAS_ARCH:
        raise ImportError("Pachetul `arch` nu este instalat (pip install arch).")

    test_start = pd.Timestamp(test_start)
    p, q = order
    max_h = max(config.HORIZONS)
    # budget de forecast per refit: suficient pentru toate zilele + orizontul maxim
    horizon_budget = refit_every + max_h
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

        last_refit_k: int | None = None
        fc_cache: np.ndarray | None = None  # variante precomputate per ciclu

        for k, pos in enumerate(test_pos):
            # refit periodic pe fereastra expanding
            if last_refit_k is None or (k - last_refit_k) >= refit_every:
                hist = ret.iloc[:pos]
                try:
                    res = _fit(hist, p, q, dist)
                    # forecast multi-pas: fc_cache[d] = varianta d pasi inainte
                    # de la ultimul punct din hist (adica de la ziua de refit)
                    fc_obj = res.forecast(horizon=horizon_budget, reindex=False)
                    fc_cache = fc_obj.variance.values[-1]  # shape: (horizon_budget,)
                    last_refit_k = k
                except Exception:
                    fc_cache = None
                    last_refit_k = k
                    continue

            if fc_cache is None:
                continue

            # d = zile scurse de la ultimul refit (0 = ziua refitului)
            d = k - last_refit_k
            if d >= len(fc_cache):
                continue

            row = {"Symbol": sym, "Date": ret.index[pos]}
            for h in config.HORIZONS:
                # media variantei pe urmatoarele h zile, plecand de la ziua d
                end_idx = min(d + h, len(fc_cache))
                mean_var = fc_cache[d:end_idx].mean() / (100.0 ** 2)
                row[f"vol_garch_h{h}"] = float(np.sqrt(max(mean_var, 0.0)))
            rows.append(row)

    return pd.DataFrame(rows)


def garch_model_stats(panel: pd.DataFrame, train_end: str,
                      order=(1, 1), dist: str = "normal") -> dict:
    """AIC, BIC, Log-Likelihood in-sample pe datele de train+val, per simbol.

    Fitam GARCH o singura data pe toata fereastra istorica (pana la train_end)
    si returnam mediile metricilor de fit pentru compararea formala a ordinelor.
    Aceste metrici masoara cat de bine modelul explica volatilitatea in-sample,
    spre deosebire de RMSE/MAE/QLIKE care masoara calitatea predictiei out-of-sample.

    Note
    ----
    AIC  = -2 * LogLik + 2 * k          (mai mic = mai bun, penalizeaza complexitatea)
    BIC  = -2 * LogLik + k * log(n)     (penalizeaza mai mult parametrii in plus)
    Valorile sunt medii peste toate simbolurile din panel.
    """
    if not _HAS_ARCH:
        raise ImportError("Pachetul `arch` nu este instalat.")

    train_end_ts = pd.Timestamp(train_end)
    p, q = order
    aic_vals, bic_vals, loglik_vals = [], [], []

    for sym, g in panel.groupby("Symbol"):
        g = g.sort_values("Date")
        hist = g[g["Date"] <= train_end_ts]
        ret = (np.log(hist["Close"] / hist["Close"].shift(1)).dropna() * 100.0)
        if len(ret) < 100:
            continue
        try:
            res = _fit(ret, p, q, dist)
            aic_vals.append(res.aic)
            bic_vals.append(res.bic)
            loglik_vals.append(res.loglikelihood)
        except Exception:
            continue

    if not aic_vals:
        return {"aic": float("nan"), "bic": float("nan"), "loglik": float("nan")}
    return {
        "aic":    float(np.mean(aic_vals)),
        "bic":    float(np.mean(bic_vals)),
        "loglik": float(np.mean(loglik_vals)),
    }
