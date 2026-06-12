"""Generator de date OHLCV sintetice in formatul real NYSE.

Reproduce structura exacta a datelor reale astfel incat restul pipeline-ului sa
ruleze identic cu si fara datele reale:
- zip-uri an cu an `NYSE_<YYYY>.zip`
- in fiecare zip, cate un fisier zilnic `NYSE_<YYYYMMDD>.csv`
- header `Symbol,Date,Open,High,Low,Close,Volume`, Date in format `d-Mon-yy`

Preturile urmeaza o miscare browniana geometrica cu volatilitate stocastica
(proces de tip GARCH simplificat), astfel incat estimatorii de volatilitate sa
aiba structura temporala invatabila.
"""
from __future__ import annotations

import calendar
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_MONTHS = {i: calendar.month_abbr[i] for i in range(1, 13)}


def _fmt_date(ts: pd.Timestamp) -> str:
    """Formateaza ca in datele reale: '2-Jan-01' (d-Mon-yy, fara zero in fata)."""
    return f"{ts.day}-{_MONTHS[ts.month]}-{ts.strftime('%y')}"


def _simulate_symbol(dates: pd.DatetimeIndex, rng: np.random.Generator,
                     start_price: float) -> pd.DataFrame:
    """Simuleaza OHLCV pentru un simbol pe un set de zile de tranzactionare."""
    n = len(dates)
    # Volatilitate stocastica: log-vol urmeaza un AR(1) -> grupuri de volatilitate.
    log_vol = np.zeros(n)
    log_vol[0] = np.log(0.015)
    for t in range(1, n):
        log_vol[t] = 0.95 * log_vol[t - 1] + 0.05 * np.log(0.015) + 0.15 * rng.standard_normal()
    sigma = np.exp(log_vol)                       # vol zilnica intraday

    drift = 0.0002
    close = np.empty(n)
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    prev_close = start_price
    for t in range(n):
        # gap overnight mic
        op = prev_close * np.exp(rng.normal(0, sigma[t] * 0.3))
        # traseu intraday aproximat prin cateva incremente brownian
        steps = rng.normal(drift, sigma[t], size=8).cumsum()
        path = op * np.exp(steps)
        hi = max(op, path.max())
        lo = min(op, path.min())
        cl = path[-1]
        open_[t], high[t], low[t], close[t] = op, hi, lo, cl
        prev_close = cl

    volume = rng.lognormal(mean=13.5, sigma=0.6, size=n).astype(np.int64)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


def generate(out_dir: Path, symbols: list[str], start: str = "2005-01-01",
             end: str = "2024-12-31", seed: int = 42) -> Path:
    """Genereaza zip-uri an cu an cu CSV-uri zilnice in `out_dir`.

    Intoarce `out_dir`. Daca zip-urile exista deja, nu le regenereaza.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # zile de tranzactionare ~ zile lucratoare
    all_days = pd.bdate_range(start, end)

    # preturi de pornire per simbol
    start_prices = {s: float(rng.uniform(15, 120)) for s in symbols}

    # serii complete per simbol, apoi le "feliem" pe zi
    panel = {}
    for s in symbols:
        panel[s] = _simulate_symbol(all_days, rng, start_prices[s])

    years = sorted({d.year for d in all_days})
    for year in years:
        zip_path = out_dir / f"NYSE_{year}.zip"
        if zip_path.exists():
            continue
        year_days = all_days[all_days.year == year]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for day in year_days:
                rows = []
                for s in symbols:
                    r = panel[s].loc[day]
                    rows.append({
                        "Symbol": s,
                        "Date": _fmt_date(day),
                        "Open": round(float(r["Open"]), 4),
                        "High": round(float(r["High"]), 4),
                        "Low": round(float(r["Low"]), 4),
                        "Close": round(float(r["Close"]), 4),
                        "Volume": int(r["Volume"]),
                    })
                df = pd.DataFrame(rows, columns=["Symbol", "Date", "Open", "High",
                                                 "Low", "Close", "Volume"])
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                fname = f"NYSE_{day.strftime('%Y%m%d')}.csv"
                zf.writestr(fname, buf.getvalue())
    return out_dir


if __name__ == "__main__":
    from nyse_vol import config

    path = generate(config.DATA_DIR, config.SYMBOLS)
    print(f"Date sintetice generate in: {path}")
