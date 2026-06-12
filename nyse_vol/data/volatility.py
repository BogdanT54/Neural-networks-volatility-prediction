"""Estimatori de volatilitate range-based si close-to-close.

Implementeaza cele patru estimatoare clasice descrise in materialul atasat
(Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) plus estimatorul
close-to-close pe randamente. Fiecare functie intoarce o *varianta zilnica*
(volatilitate la patrat) estimata din OHLC; radacina patrata da volatilitatea.

Conventii:
- intrarile sunt serii pandas aliniate pe acelasi index (o valoare pe zi);
- toate logaritmurile sunt naturale;
- pentru estimatorii pe o singura zi, "media pe T perioade" din formule devine
  contributia unei singure zile (T = 1), iar agregarea pe ferestre se face
  ulterior prin medie (vezi rolling_volatility).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)


def _logs(open_, high, low, close):
    """Randamente logaritmice uzuale folosite de estimatori."""
    hl = np.log(high / low)          # log(High/Low)
    co = np.log(close / open_)       # log(Close/Open)
    ho = np.log(high / open_)        # log(High/Open)
    lo = np.log(low / open_)         # log(Low/Open)
    return hl, co, ho, lo


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Varianta Parkinson (1980), pe o singura zi.

    sigma^2 = (1 / (4 ln 2)) * [ln(High/Low)]^2
    """
    hl = np.log(high / low)
    return (1.0 / (4.0 * _LN2)) * hl.pow(2)


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series,
                 close: pd.Series) -> pd.Series:
    """Varianta Garman-Klass (1980), pe o singura zi.

    sigma^2 = 0.5 [ln(H/L)]^2 - (2 ln 2 - 1) [ln(C/O)]^2
    """
    hl, co, _, _ = _logs(open_, high, low, close)
    return 0.5 * hl.pow(2) - (2.0 * _LN2 - 1.0) * co.pow(2)


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    """Varianta Rogers-Satchell (1991), robusta la drift, pe o singura zi.

    sigma^2 = ln(H/O) ln(H/C) + ln(L/O) ln(L/C)
    """
    ho = np.log(high / open_)
    hc = np.log(high / close)
    lo = np.log(low / open_)
    lc = np.log(low / close)
    return ho * hc + lo * lc


def close_to_close(close: pd.Series) -> pd.Series:
    """Varianta close-to-close, pe o singura zi, ca patratul randamentului log.

    Estimatorul clasic foloseste media pe o fereastra; aici intoarcem
    contributia zilnica r_t^2, iar agregarea pe fereastra se face in
    rolling_volatility.
    """
    r = np.log(close / close.shift(1))
    return r.pow(2)


def yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series,
               close: pd.Series, window: int = 20, k: float | None = None) -> pd.Series:
    """Varianta Yang-Zhang (2000): combina overnight, open-to-close si Rogers-Satchell.

    Spre deosebire de ceilalti, Yang-Zhang necesita o fereastra pentru a estima
    componentele de varianta a randamentelor overnight si open-to-close, deci
    intoarce direct o serie pe fereastra mobila `window`.

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_open_to_close + (1-k) * sigma^2_RS
    """
    o = np.log(open_ / close.shift(1))   # overnight: ln(O_t / C_{t-1})
    c = np.log(close / open_)            # open-to-close: ln(C_t / O_t)
    rs = rogers_satchell(open_, high, low, close)

    n = window
    if k is None:
        # constanta care minimizeaza varianta estimatorului (Yang-Zhang)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_o = o.rolling(n).var(ddof=1)
    var_c = c.rolling(n).var(ddof=1)
    var_rs = rs.rolling(n).mean()
    return var_o + k * var_c + (1.0 - k) * var_rs


# Estimatorii care produc o varianta zilnica (agregabila prin medie).
_DAILY_ESTIMATORS = {
    "parkinson": lambda df: parkinson(df["High"], df["Low"]),
    "garman_klass": lambda df: garman_klass(df["Open"], df["High"], df["Low"], df["Close"]),
    "rogers_satchell": lambda df: rogers_satchell(df["Open"], df["High"], df["Low"], df["Close"]),
    "close_to_close": lambda df: close_to_close(df["Close"]),
}


def daily_variance(df: pd.DataFrame, estimator: str) -> pd.Series:
    """Varianta zilnica pentru un estimator dat, dintr-un DataFrame OHLC.

    Pentru yang_zhang, care e definit pe fereastra, intoarce direct varianta
    pe fereastra implicita.
    """
    if estimator == "yang_zhang":
        return yang_zhang(df["Open"], df["High"], df["Low"], df["Close"])
    if estimator not in _DAILY_ESTIMATORS:
        raise ValueError(f"Estimator necunoscut: {estimator!r}")
    return _DAILY_ESTIMATORS[estimator](df)


def daily_volatility(df: pd.DataFrame, estimator: str) -> pd.Series:
    """Volatilitatea zilnica (radacina variantei), curatata de valori invalide."""
    var = daily_variance(df, estimator)
    var = var.clip(lower=0)          # variantele estimate pot deveni usor negative
    return np.sqrt(var)


def rolling_volatility(df: pd.DataFrame, estimator: str, window: int) -> pd.Series:
    """Volatilitate realizata pe o fereastra de `window` zile.

    Pentru estimatorii zilnici, varianta pe fereastra e media variantelor
    zilnice; pentru yang_zhang folosim fereastra ceruta direct.
    """
    if estimator == "yang_zhang":
        var = yang_zhang(df["Open"], df["High"], df["Low"], df["Close"], window=window)
    else:
        var = daily_variance(df, estimator).rolling(window).mean()
    return np.sqrt(var.clip(lower=0))


ALL_ESTIMATORS = ["parkinson", "garman_klass", "rogers_satchell",
                  "close_to_close", "yang_zhang"]
