"""Incarcarea datelor NYSE din zip-uri an cu an in format "long".

Citeste `NYSE_<YYYY>.zip` -> fisiere zilnice `NYSE_<YYYYMMDD>.csv` si le
concateneaza intr-un singur DataFrame cu coloanele:
    Symbol, Date (datetime), Open, High, Low, Close, Volume

Data este derivata *autoritar* din numele fisierului (`NYSE_YYYYMMDD.csv`),
nu din coloana Date (care e in format ambiguu `d-Mon-yy`).

Curatare aplicata:
- elimina randuri cu preturi non-pozitive sau NaN;
- elimina zilele nelichide (Open=High=Low=Close) si Volume=0, care strica
  estimatorii de volatilitate (log-range = 0).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from nyse_vol import config

_DATE_RE = re.compile(r"NYSE_(\d{8})\.csv$", re.IGNORECASE)
_COLS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def _read_daily_csv(raw: bytes, date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = date
    return df[_COLS]


def load_year(zip_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un zip de an, optional filtrand pe simboluri."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _DATE_RE.search(name)
            if not m:
                continue
            date = pd.to_datetime(m.group(1), format="%Y%m%d")
            df = _read_daily_csv(zf.read(name), date)
            if symbols is not None:
                df = df[df["Symbol"].isin(symbols)]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(frames, ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # zile nelichide: nicio miscare intraday (range nul) sau volum zero
    flat = (df["High"] == df["Low"])
    df = df[~flat & (df["Volume"] > 0)]
    return df


def load_panel(data_dir: Path | None = None, symbols=None,
               start_year: int | None = None, end_year: int | None = None,
               auto_generate: bool = True) -> pd.DataFrame:
    """Incarca panelul complet (toate zip-urile) intr-un DataFrame "long".

    Daca nu exista zip-uri si `auto_generate` e True, genereaza date sintetice.
    Rezultatul e sortat dupa (Symbol, Date) si curatat.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    symbols = set(symbols or config.SYMBOLS)

    zips = sorted(data_dir.glob("NYSE_*.zip"))
    if not zips and auto_generate:
        from nyse_vol.data import sample_generator
        sample_generator.generate(data_dir, sorted(symbols))
        zips = sorted(data_dir.glob("NYSE_*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"Nu exista zip-uri NYSE in {data_dir}. Seteaza DATA_DIR sau "
            f"permite generarea sintetica (auto_generate=True)."
        )

    frames = []
    for zp in zips:
        m = re.search(r"NYSE_(\d{4})", zp.name)
        year = int(m.group(1)) if m else None
        if year is not None:
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
        frames.append(load_year(zp, symbols))

    panel = pd.concat(frames, ignore_index=True)
    panel = _clean(panel)
    panel = panel.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # pastreaza doar simbolurile cu suficiente observatii
    counts = panel.groupby("Symbol")["Date"].transform("size")
    panel = panel[counts >= config.MIN_OBS_PER_SYMBOL].reset_index(drop=True)
    return panel


def load_symbols_description(txt_path: Path) -> pd.DataFrame:
    """Citeste fisierul Symbols_NYSE (tab-separat: Symbol<TAB>Description)."""
    return pd.read_csv(txt_path, sep="\t", names=["Symbol", "Description"],
                       header=0, dtype=str).dropna(subset=["Symbol"])


def select_liquid_symbols(panel: pd.DataFrame, top_n: int = 30) -> list[str]:
    """Selecteaza cele mai lichide `top_n` simboluri dupa volumul median."""
    med = panel.groupby("Symbol")["Volume"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()
