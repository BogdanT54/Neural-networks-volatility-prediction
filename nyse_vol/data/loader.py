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
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from nyse_vol import config

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"NYSE_(\d{8})\.csv$", re.IGNORECASE)
_COLS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def _read_daily_csv(raw: bytes, date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = date
    # coloane lipsa fata de schema asteptata
    missing_cols = [c for c in _COLS if c not in df.columns and c != "Date"]
    if missing_cols:
        logger.warning("  [%s] Coloane lipsa in CSV: %s", date.date(), missing_cols)
    return df[_COLS]


def load_year(zip_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    """Incarca toate zilele dintr-un zip de an, optional filtrand pe simboluri."""
    logger.info("Procesez fisierul: %s", zip_path.name)
    frames = []
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        csv_names = [n for n in all_names if _DATE_RE.search(n)]
        logger.info("  Gasit %d fisiere zilnice in arhiva (total intrari: %d)",
                    len(csv_names), len(all_names))
        for name in csv_names:
            m = _DATE_RE.search(name)
            date = pd.to_datetime(m.group(1), format="%Y%m%d")
            logger.debug("  Citesc: %s (%s)", name, date.date())
            try:
                df = _read_daily_csv(zf.read(name), date)
            except Exception as exc:
                logger.warning("  Eroare la citirea %s: %s — sarind peste.", name, exc)
                skipped += 1
                continue
            if symbols is not None:
                before = len(df)
                df = df[df["Symbol"].isin(symbols)]
                dropped = before - len(df)
                if dropped:
                    logger.debug("  [%s] %d simboluri filtrate (nu sunt in lista dorita)",
                                 date.date(), dropped)
            frames.append(df)
    if skipped:
        logger.warning("  %d fisiere zilnice sarite din cauza erorilor.", skipped)
    if not frames:
        logger.warning("  Niciun fisier valid gasit in %s.", zip_path.name)
        return pd.DataFrame(columns=_COLS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("  => %d randuri incarcate din %s", len(result), zip_path.name)
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

        if n_missing > 0:
            logger.warning(
                "  [%s] %d zile lipsa din calendar: "
                "%d umplute cu forward-fill (gap <= %d zile), "
                "%d eliminate (gap prea lung).",
                sym, n_missing, n_filled, max_ffill, n_dropped
            )
        else:
            logger.debug("  [%s] Serie completa, nicio zi lipsa.", sym)

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

    # NaN-uri in coloanele de pret
    nan_mask = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan:
        # afiseaza care simboluri au cele mai multe NaN-uri
        top_nan = (df[nan_mask].groupby("Symbol").size()
                   .sort_values(ascending=False).head(5))
        logger.warning(
            "  Valori lipsa (NaN) in coloanele OHLC: %d randuri eliminate.\n"
            "  Top simboluri afectate: %s",
            n_nan, top_nan.to_dict()
        )
    df = df[~nan_mask]

    # preturi non-pozitive
    neg_mask = ~(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    n_neg = neg_mask.sum()
    if n_neg:
        logger.warning("  Preturi <= 0: %d randuri eliminate.", n_neg)
    df = df[~neg_mask]

    # zile nelichide: range nul sau volum zero
    flat = df["High"] == df["Low"]
    zero_vol = df["Volume"] <= 0
    n_flat = flat.sum()
    n_zvol = (~flat & zero_vol).sum()
    if n_flat:
        logger.warning(
            "  Zile nelichide (High=Low): %d randuri eliminate "
            "(log-range=0 ar strica estimatorii de volatilitate).", n_flat
        )
    if n_zvol:
        logger.warning("  Volum zero (cu range valid): %d randuri eliminate.", n_zvol)

    df = df[~flat & ~zero_vol]

    n_end = len(df)
    logger.info("  Curatare: %d -> %d randuri (%d eliminate total).",
                n_start, n_end, n_start - n_end)
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
    logger.info("=== Incarcare panel NYSE ===")
    logger.info("Director date: %s", data_dir)
    logger.info("Simboluri cerute: %d (%s ... %s)",
                len(symbols), sorted(symbols)[0], sorted(symbols)[-1])

    zips = sorted(data_dir.glob("NYSE_*.zip"))
    if not zips and auto_generate:
        logger.info("Nu s-au gasit zip-uri NYSE — generez date sintetice...")
        from nyse_vol.data import sample_generator
        sample_generator.generate(data_dir, sorted(symbols))
        zips = sorted(data_dir.glob("NYSE_*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"Nu exista zip-uri NYSE in {data_dir}. Seteaza DATA_DIR sau "
            f"permite generarea sintetica (auto_generate=True)."
        )

    logger.info("Gasit %d fisiere ZIP: %s", len(zips), [z.name for z in zips])

    frames = []
    for zp in zips:
        m = re.search(r"NYSE_(\d{4})", zp.name)
        year = int(m.group(1)) if m else None
        if year is not None:
            if start_year and year < start_year:
                logger.debug("Sar peste %s (inainte de %d).", zp.name, start_year)
                continue
            if end_year and year > end_year:
                logger.debug("Sar peste %s (dupa %d).", zp.name, end_year)
                continue
        frames.append(load_year(zp, symbols))

    logger.info("--- Curatare globala a panelului ---")
    panel = pd.concat(frames, ignore_index=True)
    logger.info("--- Curatare date originale (inainte de reindexare) ---")
    panel = _clean(panel)

    # Reindexare la calendarul bursier + forward-fill pentru zile lipsa.
    # Se face DUPA curatare: zilele flat/corupte din datele originale sunt
    # eliminate inainte; zilele flat rezultate din forward-fill sunt pastrate
    # intentionat (reprezinta absenta informatiei, semnal valid pentru LSTM).
    logger.info("--- Reindexare si forward-fill ---")
    panel = _reindex_and_ffill(panel, max_ffill=config.MAX_FFILL_DAYS)
    panel = panel.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # pastreaza doar simbolurile cu suficiente observatii
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
