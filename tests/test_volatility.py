"""Teste numerice pentru estimatorii de volatilitate.

Ruleaza cu: python -m pytest tests/ -q  (sau direct: python tests/test_volatility.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyse_vol.data import volatility as vol


def _series(vals):
    return pd.Series(vals, dtype=float)


def test_parkinson_known_value():
    # High=110, Low=100 -> ln(1.1)=0.0953102; sigma^2 = ln^2 / (4 ln2)
    h, l = _series([110.0]), _series([100.0])
    expected = (np.log(1.1) ** 2) / (4 * np.log(2))
    assert np.isclose(vol.parkinson(h, l).iloc[0], expected)


def test_garman_klass_no_range_is_close_term():
    # daca High=Low=Open=Close, ln-urile sunt 0 -> varianta 0
    s = _series([50.0])
    gk = vol.garman_klass(s, s, s, s).iloc[0]
    assert np.isclose(gk, 0.0)


def test_rogers_satchell_zero_when_flat():
    s = _series([50.0])
    rs = vol.rogers_satchell(s, s, s, s).iloc[0]
    assert np.isclose(rs, 0.0)


def test_estimators_nonnegative_after_sqrt():
    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    for est in ["parkinson", "garman_klass", "rogers_satchell", "close_to_close"]:
        v = vol.daily_volatility(df, est)
        assert (v.dropna() >= 0).all(), est


def test_yang_zhang_runs_on_window():
    rng = np.random.default_rng(1)
    n = 100
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    yz = vol.rolling_volatility(df, "yang_zhang", window=20)
    assert yz.notna().sum() > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} teste au trecut.")
