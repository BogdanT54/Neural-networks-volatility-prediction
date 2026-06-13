"""Silver layer: validare, curatare si interpolare a datelor NYSE.

Transforma datele Bronze (brute, necuratate) in date Silver (validate, curate):

  1. Dtype enforcement  — Symbol: str, Date: datetime64[ns], OHLCV: float64, Volume: int64
  2. Symbol coverage    — verifica ce simboluri cerute exista in date
  3. Weekend removal    — NYSE nu tranzactioneaza sambata/duminica
  4. Price integrity    — High >= max(Open,Close), Low <= min(Open,Close), toate > 0
  5. Deduplication      — elimina duplicate (Symbol, Date), pastreaza primul
  6. Reindexare + interpolare liniara (max MAX_INTERP_DAYS zile consecutive)
  7. Flat days removal  — High == Low (range nul, strica estimatorii de volatilitate)
  8. Min observations   — simboluri cu < MIN_OBS_PER_SYMBOL randuri sunt excluse
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nyse_vol import config

logger = logging.getLogger(__name__)

_OHLCV_FLOAT = ["Open", "High", "Low", "Close"]
_ALL_NUMERIC = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Raport Silver
# --------------------------------------------------------------------------- #

@dataclass
class SilverReport:
    """Raport de calitate generat de stratul Silver."""
    requested_symbols: list[str] = field(default_factory=list)
    found_symbols: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)

    rows_bronze: int = 0
    rows_after_dtype: int = 0
    rows_after_weekend: int = 0
    rows_after_integrity: int = 0
    rows_after_dedup: int = 0
    rows_after_interp: int = 0
    rows_after_flat: int = 0
    rows_silver: int = 0

    integrity_violations: dict = field(default_factory=dict)
    duplicates_removed: int = 0
    weekend_rows_removed: int = 0
    gaps_interpolated: dict = field(default_factory=dict)   # sym -> nr. zile interpolate
    flat_days_removed: int = 0
    symbols_dropped_min_obs: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  RAPORT SILVER — CALITATE DATE")
        print(sep)

        # Symbol coverage
        found_pct = 100 * len(self.found_symbols) / max(len(self.requested_symbols), 1)
        print(f"  Simboluri cerute:  {len(self.requested_symbols)}")
        print(f"  Gasite in date:    {len(self.found_symbols)} ({found_pct:.0f}%)")
        if self.missing_symbols:
            print(f"  LIPSA din date:    {self.missing_symbols}")

        # Row counts through pipeline
        print(f"\n  Randuri per etapa:")
        print(f"    Bronze (brut):             {self.rows_bronze:>8,}")
        if self.weekend_rows_removed:
            print(f"    Dupa eliminare weekends:   {self.rows_after_weekend:>8,}  "
                  f"(-{self.weekend_rows_removed:,})")
        if self.integrity_violations:
            total_viol = sum(self.integrity_violations.values())
            print(f"    Dupa integritate preturi:  {self.rows_after_integrity:>8,}  "
                  f"(-{total_viol:,})")
        if self.duplicates_removed:
            print(f"    Dupa deduplicare:          {self.rows_after_dedup:>8,}  "
                  f"(-{self.duplicates_removed:,})")

        interp_total = sum(self.gaps_interpolated.values())
        if interp_total:
            print(f"    Dupa interpolare:          {self.rows_after_interp:>8,}  "
                  f"(+{interp_total:,} zile interpolate linear)")
        if self.flat_days_removed:
            print(f"    Dupa zile flat (H=L):      {self.rows_after_flat:>8,}  "
                  f"(-{self.flat_days_removed:,})")
        print(f"    Silver (final):            {self.rows_silver:>8,}")

        # Integrity violations detail
        if self.integrity_violations:
            print(f"\n  Violari integritate preturi eliminate:")
            for check, cnt in self.integrity_violations.items():
                print(f"    {check}: {cnt:,} randuri")

        # Interpolation detail (only if gaps exist)
        if self.gaps_interpolated:
            syms_with_gaps = {s: n for s, n in self.gaps_interpolated.items() if n > 0}
            if syms_with_gaps:
                print(f"\n  Zile interpolate per simbol (max {config.MAX_INTERP_DAYS} zile gap):")
                for sym, n in sorted(syms_with_gaps.items(), key=lambda x: -x[1])[:10]:
                    print(f"    [{sym}] {n} zile")
                if len(syms_with_gaps) > 10:
                    print(f"    ... si alte {len(syms_with_gaps) - 10} simboluri")

        if self.symbols_dropped_min_obs:
            print(f"\n  Simboluri excluse (< {config.MIN_OBS_PER_SYMBOL} obs): "
                  f"{self.symbols_dropped_min_obs}")

        print(sep)


# --------------------------------------------------------------------------- #
# Etape individuale
# --------------------------------------------------------------------------- #

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Castare la tipuri corecte: Symbol str, Date datetime64, OHLCV float64, Volume int64."""
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in _OHLCV_FLOAT:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(np.int64)

    # Elimina randuri cu Date invalida (NaT)
    n_bad_date = df["Date"].isna().sum()
    if n_bad_date:
        logger.warning("  [dtype] %d randuri cu Date invalida eliminate.", n_bad_date)
        df = df.dropna(subset=["Date"])

    return df.reset_index(drop=True)


def _check_symbol_coverage(df: pd.DataFrame,
                           requested: list[str]) -> tuple[list[str], list[str]]:
    """Returneaza (found, missing) pentru simbolurile cerute."""
    present = set(df["Symbol"].unique())
    found = [s for s in requested if s in present]
    missing = [s for s in requested if s not in present]
    return found, missing


def _remove_weekends(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina date care cad in weekend (NYSE nu tranzactioneaza)."""
    is_weekend = df["Date"].dt.dayofweek >= 5
    n = int(is_weekend.sum())
    return df[~is_weekend].reset_index(drop=True), n


def _check_price_integrity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Verifica si elimina randuri cu preturi imposibile.

    Verificari:
      - OHLC > 0 (preturi pozitive)
      - High >= Low (relatia fundamentala)
      - High >= Open si High >= Close
      - Low  <= Open si Low  <= Close
    """
    violations: dict[str, int] = {}
    mask_keep = pd.Series(True, index=df.index)

    # Preturi non-pozitive (NaN sau <= 0)
    non_pos = ~(df[_OHLCV_FLOAT] > 0).all(axis=1)
    violations["OHLC non-pozitiv sau NaN"] = int(non_pos.sum())
    mask_keep &= ~non_pos

    # High < Low — imposibil
    high_lt_low = df["High"] < df["Low"]
    violations["High < Low"] = int(high_lt_low.sum())
    mask_keep &= ~high_lt_low

    # High sub Open sau Close — imposibil
    high_lt_open = df["High"] < df["Open"]
    high_lt_close = df["High"] < df["Close"]
    bad_high = high_lt_open | high_lt_close
    violations["High < Open sau Close"] = int(bad_high.sum())
    mask_keep &= ~bad_high

    # Low peste Open sau Close — imposibil
    low_gt_open = df["Low"] > df["Open"]
    low_gt_close = df["Low"] > df["Close"]
    bad_low = low_gt_open | low_gt_close
    violations["Low > Open sau Close"] = int(bad_low.sum())
    mask_keep &= ~bad_low

    # Volume negativ
    neg_vol = df["Volume"] < 0
    violations["Volume negativ"] = int(neg_vol.sum())
    mask_keep &= ~neg_vol

    # Pastreaza doar violarile cu count > 0
    violations = {k: v for k, v in violations.items() if v > 0}
    return df[mask_keep].reset_index(drop=True), violations


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina duplicate (Symbol, Date) — pastreaza primul rand."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["Symbol", "Date"], keep="first")
    return df.reset_index(drop=True), n_before - len(df)


def _reindex_and_interpolate(df: pd.DataFrame,
                              max_gap: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reindexeaza la calendarul bursier complet si aplica interpolare liniara.

    Goluri de pana la `max_gap` zile consecutive sunt completate prin interpolare
    liniara (2 zile inapoi, 2 zile in fata). Goluri mai mari raman NaN si sunt
    eliminate ulterior.
    """
    trading_days = pd.DatetimeIndex(sorted(df["Date"].unique()))
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    gaps_filled: dict[str, int] = {}

    parts = []
    for sym, g in df.groupby("Symbol"):
        g_idx = g.set_index("Date")[ohlcv_cols].reindex(trading_days)
        n_missing_before = int(g_idx["Close"].isna().sum())

        if n_missing_before > 0:
            g_idx = g_idx.interpolate(
                method="linear",
                limit=max_gap,
                limit_direction="both",
                limit_area="inside",
            )
            # Volume trebuie sa fie intreg dupa interpolare
            g_idx["Volume"] = g_idx["Volume"].round().astype("Int64")

        n_missing_after = int(g_idx["Close"].isna().sum())
        filled = n_missing_before - n_missing_after
        gaps_filled[sym] = filled

        g_idx["Symbol"] = sym
        g_clean = (g_idx.dropna(subset=ohlcv_cols)
                   .reset_index()
                   .rename(columns={"index": "Date"}))
        g_clean["Volume"] = g_clean["Volume"].astype(np.int64)
        parts.append(g_clean)

    result = pd.concat(parts, ignore_index=True)
    return result, gaps_filled


def _remove_flat_days(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina zile cu High == Low (range nul — strica estimatorii de volatilitate)."""
    flat = df["High"] == df["Low"]
    n = int(flat.sum())
    return df[~flat].reset_index(drop=True), n


def _filter_min_obs(df: pd.DataFrame,
                    min_obs: int) -> tuple[pd.DataFrame, list[str]]:
    """Elimina simbolurile cu mai putine de `min_obs` observatii."""
    counts = df.groupby("Symbol")["Date"].transform("count")
    dropped = df.loc[counts < min_obs, "Symbol"].unique().tolist()
    return df[counts >= min_obs].reset_index(drop=True), dropped


# --------------------------------------------------------------------------- #
# Orchestrator principal
# --------------------------------------------------------------------------- #

def stage(bronze: pd.DataFrame,
          requested_symbols: list[str] | None = None) -> tuple[pd.DataFrame, SilverReport]:
    """Bronze → Silver: toate verificarile de integritate si transformarile.

    Parameters
    ----------
    bronze : DataFrame din stratul Bronze (Symbol, Date, Open, High, Low, Close, Volume)
    requested_symbols : lista de simboluri cerute de utilizator

    Returns
    -------
    silver : DataFrame curatat, validat, cu tipuri corecte
    report : SilverReport cu detalii despre toate transformarile aplicate
    """
    report = SilverReport(
        requested_symbols=list(requested_symbols or []),
        rows_bronze=len(bronze),
    )
    logger.info("=== Silver layer: %d randuri bronze de procesat ===", len(bronze))

    # 1. Dtype enforcement
    df = _enforce_dtypes(bronze)
    report.rows_after_dtype = len(df)
    logger.info("  [1/7] Dtype enforcement: %d randuri", len(df))

    # 2. Symbol coverage check
    if requested_symbols:
        found, missing = _check_symbol_coverage(df, requested_symbols)
        report.found_symbols = found
        report.missing_symbols = missing
        if missing:
            logger.warning("  [2/7] Simboluri LIPSA din date: %s", missing)
        else:
            logger.info("  [2/7] Coverage: toate cele %d simboluri gasite.", len(found))
        # Filtreaza doar simbolurile cerute
        df = df[df["Symbol"].isin(found)].reset_index(drop=True)
    else:
        report.found_symbols = df["Symbol"].unique().tolist()
        logger.info("  [2/7] Coverage: %d simboluri in date.", len(report.found_symbols))

    # 3. Weekend removal
    df, n_weekend = _remove_weekends(df)
    report.weekend_rows_removed = n_weekend
    report.rows_after_weekend = len(df)
    if n_weekend:
        logger.warning("  [3/7] Weekend: %d randuri eliminate.", n_weekend)
    else:
        logger.info("  [3/7] Weekend: niciun rand de weekend gasit.")

    # 4. Price integrity
    df, violations = _check_price_integrity(df)
    report.integrity_violations = violations
    report.rows_after_integrity = len(df)
    total_viol = sum(violations.values())
    if total_viol:
        logger.warning("  [4/7] Integritate preturi: %d randuri eliminate. Detalii: %s",
                       total_viol, violations)
    else:
        logger.info("  [4/7] Integritate preturi: OK.")

    # 5. Deduplication
    df, n_dup = _deduplicate(df)
    report.duplicates_removed = n_dup
    report.rows_after_dedup = len(df)
    if n_dup:
        logger.warning("  [5/7] Deduplicare: %d duplicate (Symbol, Date) eliminate.", n_dup)
    else:
        logger.info("  [5/7] Deduplicare: niciun duplicat.")

    # 6. Reindexare + interpolare liniara
    max_gap = config.MAX_INTERP_DAYS
    df, gaps = _reindex_and_interpolate(df, max_gap=max_gap)
    report.gaps_interpolated = gaps
    report.rows_after_interp = len(df)
    total_interp = sum(gaps.values())
    logger.info("  [6/7] Interpolare liniara (max %d zile): %d zile completate.",
                max_gap, total_interp)

    # 7. Zile flat (High == Low)
    df, n_flat = _remove_flat_days(df)
    report.flat_days_removed = n_flat
    report.rows_after_flat = len(df)
    if n_flat:
        logger.info("  [7a] Zile flat (High=Low): %d eliminate.", n_flat)

    # 8. Filtrare simboluri cu prea putine observatii
    df, dropped = _filter_min_obs(df, config.MIN_OBS_PER_SYMBOL)
    report.symbols_dropped_min_obs = dropped
    if dropped:
        logger.warning("  [7b] Simboluri excluse (< %d obs): %s",
                       config.MIN_OBS_PER_SYMBOL, dropped)

    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    report.rows_silver = len(df)
    logger.info("=== Silver layer complet: %d randuri | %d simboluri ===",
                len(df), df["Symbol"].nunique())
    return df, report
