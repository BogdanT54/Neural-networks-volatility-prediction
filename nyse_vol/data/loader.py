"""Incarcarea datelor NYSE din directoare an cu an (sau zip-uri ca fallback).

Structura principala (date reale Kaggle):
    DATA_DIR/NYSE_2001/NYSE_20010102.csv
    DATA_DIR/NYSE_2001/NYSE_20010103.csv
    ...
    DATA_DIR/NYSE_2025/NYSE_20251231.csv

Fallback (date sintetice generate local):
    DATA_DIR/NYSE_2001.zip  (contine NYSE_20010102.csv etc.)

Data este derivata *autoritar* din numele fisierului (`NYSE_YYYYMMDD.csv`),
nu din coloana Date (care poate fi in format ambiguu `d-Mon-yy`).

Curatare aplicata:
- elimina randuri cu preturi non-pozitive sau NaN;
- elimina zilele nelichide (Open=High=Low=Close) si Volume=0, care strica
  estimatorii de volatilitate (log-range = 0).
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
_YEAR_ZIP_RE = re.compile(r"^NYSE_(\d{4})\.zip$", re.IGNORECASE)
_COLS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def _read_daily_csv(raw: bytes, date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = date
    missing_cols = [c for c in _COLS if c not in df.columns and c != "Date"]
    if missing_cols:
        logger.warning("  [%s] Coloane lipsa in CSV: %s", date.date(), missing_cols)
    return df[_COLS]


def load_year_dir(year_dir: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un director de an, optional filtrand pe simboluri."""
    csv_files = sorted(f for f in year_dir.iterdir() if _DATE_RE.search(f.name))
    frames = []
    skipped = 0
    for csv_file in csv_files:
        m = _DATE_RE.search(csv_file.name)
        date = pd.to_datetime(m.group(1), format="%Y%m%d")
        try:
            df = _read_daily_csv(csv_file.read_bytes(), date)
        except Exception as exc:
            logger.warning("  Eroare la citirea %s: %s — sarind peste.", csv_file.name, exc)
            skipped += 1
            continue
        if symbols is not None:
            df = df[df["Symbol"].isin(symbols)]
        frames.append(df)
    if skipped:
        logger.warning("  %d fisiere zilnice sarite din cauza erorilor.", skipped)
    if not frames:
        logger.warning("  Niciun fisier valid gasit in %s.", year_dir.name)
        return pd.DataFrame(columns=_COLS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("  %s: %d zile, %d randuri", year_dir.name, len(csv_files), len(result))
    return result


def load_year(zip_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un zip de an, optional filtrand pe simboluri."""
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
                logger.warning("  Eroare la citirea %s: %s — sarind peste.", name, exc)
                skipped += 1
                continue
            if symbols is not None:
                df = df[df["Symbol"].isin(symbols)]
            frames.append(df)
    if skipped:
        logger.warning("  %d fisiere zilnice sarite din cauza erorilor.", skipped)
    if not frames:
        logger.warning("  Niciun fisier valid gasit in %s.", zip_path.name)
        return pd.DataFrame(columns=_COLS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("  %s: %d zile, %d randuri", zip_path.stem, len(csv_names), len(result))
    return result


def _reindex_and_ffill(panel: pd.DataFrame, max_ffill: int) -> pd.DataFrame:
    """Reindexeaza fiecare simbol la calendarul bursier complet si aplica forward-fill.

    Detecteaza zilele de tranzactionare in care un simbol lipseste complet
    (suspendat, eroare la provider etc.) si le umple cu ultimul pret cunoscut,
    dar numai daca golul e de cel mult `max_ffill` zile consecutive.
    Goluri mai lungi sunt lasate ca NaN si eliminate ulterior.
    """
    trading_days = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    logger.info(
        "Reindexare la calendarul bursier: %d zile de tranzactionare, "
        "forward-fill maxim %d zile consecutive.",
        len(trading_days), max_ffill
    )

    parts = []
    total_missing = 0
    total_filled = 0
    total_dropped_gap = 0

    for sym, g in panel.groupby("Symbol"):
        g_idx = g.set_index("Date")[ohlcv_cols].reindex(trading_days)

        n_missing = int(g_idx[ohlcv_cols[0]].isna().sum())
        g_filled = g_idx.ffill(limit=max_ffill)
        n_still_nan = int(g_filled[ohlcv_cols[0]].isna().sum())
        n_filled = n_missing - n_still_nan
        n_dropped = n_still_nan

        if n_dropped > 0:
            logger.debug("  [%s] %d zile lipsa: %d ffill, %d eliminate (gap > %d).",
                         sym, n_missing, n_filled, n_dropped, max_ffill)

        g_filled["Symbol"] = sym
        g_filled = (g_filled.dropna(subset=ohlcv_cols)
                    .reset_index()
                    .rename(columns={"index": "Date"}))
        parts.append(g_filled)
        total_missing += n_missing
        total_filled += n_filled
        total_dropped_gap += n_dropped

    logger.info(
        "Reindexare finalizata: %d zile lipsa detectate | "
        "%d umplute (ffill) | %d eliminate (gap > %d zile).",
        total_missing, total_filled, total_dropped_gap, max_ffill
    )
    return pd.concat(parts, ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nan_mask = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    df = df[~nan_mask]

    neg_mask = ~(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    df = df[~neg_mask]

    flat = df["High"] == df["Low"]
    zero_vol = df["Volume"] <= 0
    df = df[~flat & ~zero_vol]

    n_end = len(df)
    eliminated = n_start - n_end
    if eliminated:
        logger.info("  Curatare: %d randuri eliminate (NaN, preturi invalide, zile nelichide).",
                    eliminated)
    return df


def load_panel(data_dir: Path | None = None, symbols=None,
               start_year: int | None = None, end_year: int | None = None,
               auto_generate: bool = True) -> pd.DataFrame:
    """Incarca panelul complet intr-un DataFrame "long".

    Cauta mai intai directoare de an (NYSE_YYYY*/), apoi zip-uri (NYSE_YYYY.zip).
    Daca nu exista niciuna si `auto_generate` e True, genereaza date sintetice.
    Rezultatul e sortat dupa (Symbol, Date) si curatat.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    symbols = set(symbols or config.SYMBOLS)
    logger.info("=== Incarcare panel NYSE ===")
    logger.info("Director date: %s", data_dir)
    logger.info("Simboluri cerute: %d (%s ... %s)",
                len(symbols), sorted(symbols)[0], sorted(symbols)[-1])

    # ── Detectare structura date: directoare sau zip-uri ──
    year_dirs = []
    zips = []
    if data_dir.exists():
        year_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and _YEAR_DIR_RE.match(d.name)
        )
        zips = sorted(data_dir.glob("NYSE_*.zip"))

    if year_dirs:
        logger.info("Gasit %d directoare de an (%s ... %s) — citesc CSV-uri zilnice...",
                    len(year_dirs), year_dirs[0].name, year_dirs[-1].name)
        frames = []
        for year_dir in year_dirs:
            m = _YEAR_DIR_RE.match(year_dir.name)
            year = int(m.group(1)) if m else None
            if year is not None:
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
            frames.append(load_year_dir(year_dir, symbols))

    elif zips:
        logger.info("Gasit %d fisiere ZIP — citesc...", len(zips))
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

    else:
        if auto_generate:
            logger.info("Nu s-au gasit date NYSE — generez date sintetice...")
            from nyse_vol.data import sample_generator
            sample_generator.generate(data_dir, sorted(symbols))
            zips = sorted(data_dir.glob("NYSE_*.zip"))
            frames = [load_year(zp, symbols) for zp in zips]
        else:
            raise FileNotFoundError(
                f"Nu exista date NYSE in {data_dir}. "
                f"Seteaza DATA_DIR la directorul cu folderele NYSE_YYYY/ "
                f"sau permite generarea sintetica (auto_generate=True)."
            )

    panel = pd.concat(frames, ignore_index=True)
    logger.info("Date brute combinate: %d randuri — curatare...", len(panel))
    panel = _clean(panel)

    logger.info("Reindexare la calendarul bursier + forward-fill (max %d zile)...",
                config.MAX_FFILL_DAYS)
    panel = _reindex_and_ffill(panel, max_ffill=config.MAX_FFILL_DAYS)
    panel = panel.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    counts = panel.groupby("Symbol")["Date"].transform("size")
    before_filter = panel["Symbol"].nunique()
    panel = panel[counts >= config.MIN_OBS_PER_SYMBOL].reset_index(drop=True)
    after_filter = panel["Symbol"].nunique()
    dropped_syms = before_filter - after_filter
    if dropped_syms:
        logger.warning(
            "  %d simbol(uri) eliminate pentru ca au sub %d observatii.",
            dropped_syms, config.MIN_OBS_PER_SYMBOL
        )
    logger.info(
        "Panel final: %d randuri | %d simboluri | %s -> %s",
        len(panel), after_filter,
        panel["Date"].min().date(), panel["Date"].max().date()
    )
    return panel


def load_symbols_description(txt_path: Path) -> pd.DataFrame:
    """Citeste fisierul Symbols_NYSE (tab-separat: Symbol<TAB>Description)."""
    return pd.read_csv(txt_path, sep="\t", names=["Symbol", "Description"],
                       header=0, dtype=str).dropna(subset=["Symbol"])


def select_liquid_symbols(panel: pd.DataFrame, top_n: int = 30) -> list[str]:
    """Selecteaza cele mai lichide `top_n` simboluri dupa volumul median."""
    med = panel.groupby("Symbol")["Volume"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()
