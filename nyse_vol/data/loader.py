"""Bronze layer: incarcare bruta a datelor NYSE din directoare sau zip-uri.

Structura principala (date reale Kaggle):
    DATA_DIR/NYSE_2001/NYSE_20010102.csv
    DATA_DIR/NYSE_2001/NYSE_20010103.csv
    ...
    DATA_DIR/NYSE_2025/NYSE_20251231.csv

Fallback (date sintetice generate local):
    DATA_DIR/NYSE_2001.zip  (contine NYSE_20010102.csv etc.)

NOTA: Acest modul face MINIMUL necesar — citeste CSV-urile si le concateneaza.
Curatarea, validarea si transformarile sunt responsabilitatea stratului Silver
(nyse_vol/data/silver.py).

Data este derivata *autoritar* din numele fisierului (`NYSE_YYYYMMDD.csv`),
nu din coloana Date (care poate fi in format ambiguu `d-Mon-yy`).
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from nyse_vol import config

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"NYSE_(\d{8})\.csv$", re.IGNORECASE)
_YEAR_DIR_RE = re.compile(r"^NYSE_(\d{4})", re.IGNORECASE)
_COLS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def _read_daily_csv(raw: bytes, date: pd.Timestamp) -> pd.DataFrame:
    """Citeste un fisier CSV zilnic si returneaza un DataFrame cu coloanele standard."""
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = date
    missing_cols = [c for c in _COLS if c not in df.columns and c != "Date"]
    if missing_cols:
        logger.warning("  [%s] Coloane lipsa: %s", date.date(), missing_cols)
    available = [c for c in _COLS if c in df.columns]
    return df[available]


def load_year_dir(year_dir: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un director de an (Bronze)."""
    csv_files = sorted(f for f in year_dir.iterdir() if _DATE_RE.search(f.name))
    frames = []
    skipped = 0
    for csv_file in csv_files:
        m = _DATE_RE.search(csv_file.name)
        date = pd.to_datetime(m.group(1), format="%Y%m%d")
        try:
            df = _read_daily_csv(csv_file.read_bytes(), date)
        except Exception as exc:
            logger.warning("  Eroare la citirea %s: %s", csv_file.name, exc)
            skipped += 1
            continue
        if symbols is not None:
            df = df[df["Symbol"].isin(symbols)]
        frames.append(df)
    if skipped:
        logger.warning("  %d fisiere sarite din cauza erorilor in %s.", skipped, year_dir.name)
    if not frames:
        return pd.DataFrame(columns=_COLS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("  %s: %d zile, %d randuri brute", year_dir.name, len(csv_files), len(result))
    return result


def load_year(zip_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un zip de an (Bronze — fallback)."""
    frames = []
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if _DATE_RE.search(n)]
        for name in csv_names:
            m = _DATE_RE.search(name)
            date = pd.to_datetime(m.group(1), format="%Y%m%d")
            try:
                df = _read_daily_csv(zf.read(name), date)
            except Exception as exc:
                logger.warning("  Eroare la citirea %s: %s", name, exc)
                skipped += 1
                continue
            if symbols is not None:
                df = df[df["Symbol"].isin(symbols)]
            frames.append(df)
    if skipped:
        logger.warning("  %d fisiere sarite in %s.", skipped, zip_path.name)
    if not frames:
        return pd.DataFrame(columns=_COLS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("  %s: %d zile, %d randuri brute", zip_path.stem, len(csv_names), len(result))
    return result


def load_bronze(data_dir: Path | None = None, symbols=None,
                start_year: int | None = None, end_year: int | None = None,
                auto_generate: bool = True) -> pd.DataFrame:
    """Incarca datele NYSE brute (stratul Bronze).

    Cauta mai intai directoare de an (NYSE_YYYY*/), apoi zip-uri (NYSE_YYYY.zip).
    Daca nu exista niciuna si `auto_generate` e True, genereaza date sintetice.
    Returneaza un DataFrame "long" neprocesat — curatarea e in stratul Silver.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    symbols_set = set(symbols or config.SYMBOLS)
    logger.info("=== Bronze layer: incarcare date NYSE ===")
    logger.info("  Sursa: %s", data_dir)
    logger.info("  Simboluri: %d (%s)", len(symbols_set), ", ".join(sorted(symbols_set)))

    year_dirs = []
    zips = []
    if data_dir.exists():
        year_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and _YEAR_DIR_RE.match(d.name)
        )
        zips = sorted(data_dir.glob("NYSE_*.zip"))

    frames = []

    if year_dirs:
        logger.info("  Structura: %d directoare de an (%s ... %s)",
                    len(year_dirs), year_dirs[0].name, year_dirs[-1].name)
        for year_dir in year_dirs:
            m = _YEAR_DIR_RE.match(year_dir.name)
            year = int(m.group(1)) if m else None
            if year is not None:
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
            frames.append(load_year_dir(year_dir, symbols_set))

    elif zips:
        logger.info("  Structura: %d fisiere ZIP", len(zips))
        for zp in zips:
            m = re.search(r"NYSE_(\d{4})", zp.name)
            year = int(m.group(1)) if m else None
            if year is not None:
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
            frames.append(load_year(zp, symbols_set))

    else:
        if auto_generate:
            logger.info("  Nu s-au gasit date NYSE — generez date sintetice...")
            from nyse_vol.data import sample_generator
            sample_generator.generate(data_dir, sorted(symbols_set))
            zips = sorted(data_dir.glob("NYSE_*.zip"))
            frames = [load_year(zp, symbols_set) for zp in zips]
        else:
            raise FileNotFoundError(
                f"Nu exista date NYSE in {data_dir}. "
                f"Seteaza DATA_DIR la directorul cu folderele NYSE_YYYY/."
            )

    bronze = pd.concat(frames, ignore_index=True)
    logger.info("=== Bronze complet: %d randuri brute ===", len(bronze))
    return bronze


# Alias pentru compatibilitate cu codul vechi
def load_panel(data_dir=None, symbols=None, start_year=None, end_year=None,
               auto_generate=True) -> pd.DataFrame:
    """Alias backward-compatible pentru load_bronze()."""
    return load_bronze(data_dir, symbols, start_year, end_year, auto_generate)


def load_symbols_description(txt_path: Path) -> pd.DataFrame:
    """Citeste fisierul Symbols_NYSE (tab-separat: Symbol<TAB>Description)."""
    return pd.read_csv(txt_path, sep="\t", names=["Symbol", "Description"],
                       header=0, dtype=str).dropna(subset=["Symbol"])


def select_liquid_symbols(panel: pd.DataFrame, top_n: int = 30) -> list[str]:
    """Selecteaza cele mai lichide `top_n` simboluri dupa volumul median."""
    med = panel.groupby("Symbol")["Volume"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()
