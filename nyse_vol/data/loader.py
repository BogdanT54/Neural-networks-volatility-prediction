"""Incarcarea datelor NYSE in format "long".

Suporta doua organizari ale datelor de intrare:
- **Directoare** (cum extrage Kaggle un dataset): `NYSE_2001/`, `NYSE_2002/`, ...
  fiecare continand fisiere zilnice `NYSE_YYYYMMDD.csv`.
- **Arhive zip** an cu an: `NYSE_2001.zip`, ... fiecare continand acelasi tip de
  fisiere zilnice.

In ambele cazuri rezulta un DataFrame cu coloanele:
    Symbol, Date (datetime), Open, High, Low, Close, Volume

Data este derivata *autoritar* din numele fisierului (`NYSE_YYYYMMDD.csv`),
nu din coloana Date (care e in format ambiguu `d-Mon-yy`).

Curatare aplicata:
- elimina randuri cu preturi non-pozitive sau NaN;
- elimina zilele nelichide (High=Low) si Volume=0, care strica estimatorii de
  volatilitate (log-range = 0).
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
_NUM_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _date_from_name(name: str) -> pd.Timestamp | None:
    m = _DATE_RE.search(name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def _read_daily(buf, date: pd.Timestamp, symbols: set[str] | None) -> pd.DataFrame:
    """Citeste un fisier zilnic (cale sau buffer) si optional filtreaza simbolurile."""
    df = pd.read_csv(buf)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        return pd.DataFrame(columns=_COLS)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    if symbols is not None:
        df = df[df["Symbol"].isin(symbols)]
    df["Date"] = date
    keep = [c for c in _COLS if c in df.columns]
    return df[keep]


def load_year_zip(zip_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un zip de an, optional filtrand pe simboluri."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            date = _date_from_name(name)
            if date is None:
                continue
            frames.append(_read_daily(io.BytesIO(zf.read(name)), date, symbols))
    if not frames:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(frames, ignore_index=True)


def _in_year_range(date: pd.Timestamp, start_year, end_year) -> bool:
    if start_year and date.year < start_year:
        return False
    if end_year and date.year > end_year:
        return False
    return True


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in _NUM_COLS:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(",", "", regex=False)  # ex. "1,617,800"
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # zile nelichide: nicio miscare intraday (range nul) sau volum zero
    flat = (df["High"] == df["Low"])
    df = df[~flat & (df["Volume"].fillna(0) > 0)]
    return df


def _find_sources(data_dir: Path):
    """Gaseste fisierele CSV zilnice si arhivele zip sub data_dir (recursiv)."""
    csv_files = [p for p in data_dir.rglob("NYSE_*.csv") if _DATE_RE.search(p.name)]
    zip_files = sorted(data_dir.rglob("NYSE_*.zip"))
    return sorted(csv_files), zip_files


def load_panel(data_dir: Path | None = None, symbols=None,
               start_year: int | None = None, end_year: int | None = None,
               auto_generate: bool = True) -> pd.DataFrame:
    """Incarca panelul complet intr-un DataFrame "long".

    Cauta date reale (directoare cu CSV-uri zilnice SAU zip-uri an cu an) sub
    `data_dir`. Daca nu gaseste si `auto_generate` e True, genereaza date
    sintetice intr-un director scriibil (config.SAMPLE_DIR) si incarca de acolo
    — astfel nu se scrie niciodata in DATA_DIR (care pe Kaggle e read-only).
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    symbols = set(symbols) if symbols is not None else set(config.SYMBOLS)

    csv_files, zip_files = ([], [])
    if data_dir.exists():
        csv_files, zip_files = _find_sources(data_dir)

    if not csv_files and not zip_files:
        if not auto_generate:
            raise FileNotFoundError(
                f"Nu am gasit date NYSE in {data_dir} (nici directoare NYSE_YYYY/ "
                f"cu CSV-uri zilnice, nici zip-uri NYSE_YYYY.zip)."
            )
        from nyse_vol.data import sample_generator
        sample_generator.generate(config.SAMPLE_DIR, sorted(symbols))
        return load_panel(config.SAMPLE_DIR, symbols, start_year, end_year,
                          auto_generate=False)

    frames = []
    # 1) fisiere CSV zilnice (directoare extrase, ex. Kaggle)
    for csv_path in csv_files:
        date = _date_from_name(csv_path.name)
        if date is None or not _in_year_range(date, start_year, end_year):
            continue
        frames.append(_read_daily(csv_path, date, symbols))
    # 2) arhive zip an cu an
    for zp in zip_files:
        m = re.search(r"NYSE_(\d{4})", zp.name)
        year = int(m.group(1)) if m else None
        if year is not None and not _in_year_range(pd.Timestamp(year=year, month=6, day=1),
                                                   start_year, end_year):
            continue
        frames.append(load_year_zip(zp, symbols))

    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        raise ValueError(
            "Datele s-au gasit, dar niciun rand nu a ramas dupa filtrarea pe "
            f"simboluri/ani. Verifica config.SYMBOLS (ex.: {sorted(symbols)[:5]}...) "
            "si intervalul de ani."
        )
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
