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

    Dupa fiecare refit, parametrii (omega, alpha, beta) sunt fixati iar varianta
    conditionala h_t este actualizata ZILNIC cu randamentul observat:
        h_{t+1} = omega + alpha * r_t^2 + beta * h_t
    Astfel predictiile variaza zilnic (nu in trepte constante intre refitturi).

    Intoarce un DataFrame cu Symbol, Date si vol_garch_h<H> (volatilitate zilnica
    medie pe orizont), comparabila direct cu tinta NN.
    """
    if not _HAS_ARCH:
        raise ImportError("Pachetul `arch` nu este instalat (pip install arch).")

    test_start_ts = pd.Timestamp(test_start)
    p, q = order
    max_h = max(config.HORIZONS)
    rows = []

    for sym, g in panel.groupby("Symbol"):
        g = g.sort_values("Date")
        ret = (np.log(g["Close"] / g["Close"].shift(1)).dropna() * 100.0)
        ret.index = pd.DatetimeIndex(g["Date"].iloc[1:].values)
        if len(ret) < config.MIN_OBS_PER_SYMBOL:
            continue

        test_pos = np.where(ret.index >= test_start_ts)[0]
        if len(test_pos) == 0 or test_pos[0] < 100:
            continue

        # Starea GARCH actualizata zilnic
        omega: float | None = None
        alpha_coeffs: list[float] = []
        beta_coeffs:  list[float] = []
        past_h:  list[float] = []   # q h-uri recente (newest first)
        past_r2: list[float] = []   # p-1 r^2-uri recente (newest first)
        last_refit_k: int | None = None

        for k, pos in enumerate(test_pos):
            # ── Refit periodic pe fereastra expanding ──
            if last_refit_k is None or (k - last_refit_k) >= refit_every:
                hist = ret.iloc[:pos]
                try:
                    res = _fit(hist, p, q, dist)
                    pm = res.params
                    omega = float(pm["omega"])
                    alpha_coeffs = [float(pm.get(f"alpha[{i+1}]", 0.0)) for i in range(p)]
                    beta_coeffs  = [float(pm.get(f"beta[{i+1}]",  0.0)) for i in range(q)]

                    # Initializam starea din valorile fittate
                    cv = res.conditional_volatility.values   # sigma_t shape (T,)
                    past_h  = [float(cv[-(j+1)])**2 for j in range(min(q, len(cv)))]
                    past_r2 = [float(ret.iloc[pos-1-j])**2 for j in range(min(p-1, pos))]

                    # Avanseaza starea cu ultimul randament de antrenare
                    # astfel incat past_h[0] = h_{pos} (nu h_{pos-1})
                    r_adv  = float(ret.iloc[pos - 1])
                    r2_adv = r_adv ** 2
                    r2_in  = [r2_adv] + past_r2
                    h_adv  = omega
                    h_adv += sum(a * r for a, r in zip(alpha_coeffs, r2_in[:p]))
                    h_adv += sum(b * h for b, h in zip(beta_coeffs,  past_h[:q]))
                    past_r2 = ([r2_adv] + past_r2)[:p - 1]
                    past_h  = ([h_adv]  + past_h )[:q]

                    last_refit_k = k
                except Exception:
                    omega = None
                    last_refit_k = k
                    continue

            if omega is None:
                continue

            # ── Actualizare zilnica: h_{pos+1} = f(r_{pos} observat) ──
            r_t  = float(ret.iloc[pos])
            r2_t = r_t ** 2
            r2_in = [r2_t] + past_r2
            h_next = omega
            h_next += sum(a * r for a, r in zip(alpha_coeffs, r2_in[:p]))
            h_next += sum(b * h for b, h in zip(beta_coeffs,  past_h[:q]))

            # Stare pentru ziua urmatoare
            past_r2 = ([r2_t] + past_r2)[:p - 1]
            past_h  = ([h_next] + past_h )[:q]

            # ── Forecast multi-pas din h_{pos+1} ──
            # h_{t+j} = omega + (sum_alpha + sum_beta) * h_{t+j-1}  (j >= 2)
            alpha_s = sum(alpha_coeffs)
            beta_s  = sum(beta_coeffs)
            forecast_h: list[float] = [h_next]
            for _ in range(1, max_h):
                forecast_h.append(omega + (alpha_s + beta_s) * forecast_h[-1])

            # ── Volatilitate medie per orizont (din variante la scara %) ──
            row = {"Symbol": sym, "Date": ret.index[pos]}
            for h in config.HORIZONS:
                mean_var = float(np.mean(forecast_h[:h])) / (100.0 ** 2)
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
