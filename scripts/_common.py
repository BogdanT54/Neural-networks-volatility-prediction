"""Utilitare comune pentru scripturi: seed, cache pe trei straturi, banner.

Pipeline medallion:
  Bronze → Silver → Gold
  loader  → silver  → features

La fiecare load din cache se verifica daca simbolurile din cache corespund
cu SYMBOLS curent. Daca nu, se afiseaza un avertisment clar si cache-ul
NU este folosit — modelele nu vor fi antrenate pe stocuri gresite.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyse_vol import config  # noqa: E402
from nyse_vol.data import features as feat_mod  # noqa: E402
from nyse_vol.data import loader as loader_mod  # noqa: E402
from nyse_vol.data import silver as silver_mod  # noqa: E402

logger = logging.getLogger(__name__)

_SEP = "=" * 64

BRONZE_CACHE   = config.PROCESSED_DIR / "bronze.pkl"
SILVER_CACHE   = config.PROCESSED_DIR / "panel.pkl"
FEATURES_CACHE = config.PROCESSED_DIR / "features.pkl"
METADATA_FILE  = config.PROCESSED_DIR / "pipeline_metadata.json"


def print_banner(step: int, total: int, title: str, lines: list | None = None) -> None:
    print(f"\n{_SEP}")
    print(f"  PAS {step}/{total} — {title}")
    if lines:
        print(f"  {'-' * 60}")
        for line in lines:
            print(f"  {line}")
    print(f"{_SEP}\n")


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Metadata cache — verifica consistenta simboluri
# --------------------------------------------------------------------------- #

def _symbols_fingerprint() -> str:
    """Hash scurt al listei curente de simboluri — detecteaza schimbari."""
    key = ",".join(sorted(config.SYMBOLS))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _save_metadata(stage: str, df) -> None:
    """Salveaza metadata dupa fiecare build de cache."""
    import pandas as pd
    meta: dict = {}
    if METADATA_FILE.exists():
        try:
            meta = json.loads(METADATA_FILE.read_text())
        except Exception:
            meta = {}
    meta[stage] = {
        "symbols": sorted(df["Symbol"].unique().tolist()),
        "symbols_fingerprint": _symbols_fingerprint(),
        "n_rows": len(df),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    METADATA_FILE.write_text(json.dumps(meta, indent=2))


def _validate_cache(stage: str, cache_path: Path) -> bool:
    """Verifica daca cache-ul exista si a fost construit pentru SYMBOLS curent.

    Returns
    -------
    True  — cache valid, poate fi folosit
    False — cache outdated sau lipsa, trebuie rebuild
    """
    if not cache_path.exists():
        return False

    if not METADATA_FILE.exists():
        logger.warning(
            "Cache %s gasit dar fara metadata (posibil build anterior noii versiuni). "
            "Foloseste --force in 01_prepare_data.py pentru a valida.",
            stage
        )
        return True  # permite cu avertisment

    try:
        meta = json.loads(METADATA_FILE.read_text())
    except Exception:
        return True  # metadata corupta, permite

    if stage not in meta:
        return True

    saved_fp   = meta[stage].get("symbols_fingerprint", "")
    saved_syms = meta[stage].get("symbols", [])
    current_fp = _symbols_fingerprint()

    if saved_fp == current_fp:
        # Cache valid — afiseaza simbolurile pentru transparenta
        logger.info(
            "Cache [%s] valid: %d simboluri → %s",
            stage, len(saved_syms), saved_syms
        )
        return True

    # Cache outdated — afiseaza ce s-a schimbat
    current_syms = sorted(config.SYMBOLS)
    added   = [s for s in current_syms if s not in saved_syms]
    removed = [s for s in saved_syms   if s not in current_syms]

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  ⚠️  CACHE OUTDATED — SIMBOLURI DIFERITE!")
    print(sep)
    print(f"  Cache [{stage}] construit pentru: {saved_syms}")
    print(f"  SYMBOLS curent:                   {current_syms}")
    if added:
        print(f"  Adaugate (nu sunt in cache):      {added}")
    if removed:
        print(f"  Eliminate (sunt in cache extra):  {removed}")
    print(f"\n  Modelele AR FI ANTRENATE PE STOCURI GRESITE daca s-ar")
    print(f"  folosi cache-ul vechi. Cache ignorat — rebuild necesar.")
    print(f"\n  Solutie: ruleaza din nou celula de preprocesare:")
    print(f"    !python 01_prepare_data.py --force")
    print(f"{sep}\n")
    return False


# --------------------------------------------------------------------------- #
# Sumar explicativ — afisat in fiecare script inainte de antrenare
# --------------------------------------------------------------------------- #

def print_data_summary(feats, splits=None) -> None:
    """Afiseaza context complet despre datele si split-urile incarcate.

    Apelat la inceputul fiecarui script (02-06) dupa get_features() si
    optional dupa build_splits() pentru a confirma exact ce stocuri,
    ce date si ce se prezice in acel pas al pipeline-ului.
    """
    import pandas as pd

    symbols     = sorted(feats["Symbol"].unique())
    sym_counts  = feats.groupby("Symbol").size()
    sym_dates   = feats.groupby("Symbol")["Date"].agg(["min", "max"])
    total_rows  = len(feats)

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  CONTEXT DATE — ce se antreneaza / evalueaza")
    print(sep)
    print(f"  Sursa:    {FEATURES_CACHE}")
    print(f"  Stocuri ({len(symbols)}):  {', '.join(symbols)}")
    print(f"  Perioada: {feats['Date'].min().date()}  →  {feats['Date'].max().date()}")
    print(f"  Total:    {total_rows:,} randuri  ({total_rows // len(symbols):,} zile medii/stoc)")

    print(f"\n  Zile per stoc:")
    for sym in symbols:
        cnt  = sym_counts[sym]
        dmin = sym_dates.loc[sym, "min"].date()
        dmax = sym_dates.loc[sym, "max"].date()
        print(f"    {sym:<6}  {cnt:>5,} zile   {dmin} → {dmax}")

    print(f"\n  Ce se prezice (tinte):")
    print(f"    Estimator: {config.TARGET_ESTIMATOR}  (volatilitate zilnica Garman-Klass)")
    for h in config.HORIZONS:
        unit = "zi" if h == 1 else "zile"
        print(f"    h={h:>2}  →  media vol. realizate in urmatoarele {h} {unit} bursiere")

    if splits is not None:
        mt    = splits.meta_train
        mv    = splits.meta_val
        mtest = splits.meta_test

        print(f"\n  Split-uri (strict cronologice — fara scurgere de informatii):")
        print(f"    TRAIN  ≤{config.TRAIN_END}: "
              f"{len(splits.X_train):>7,} ferestre  "
              f"({mt['Date'].min().date()} → {mt['Date'].max().date()})")
        print(f"    VAL    {config.TRAIN_END[:4]}–{config.VAL_END[:4]}:       "
              f"{len(splits.X_val):>7,} ferestre  "
              f"({mv['Date'].min().date()} → {mv['Date'].max().date()})")
        print(f"    TEST   >{config.VAL_END}: "
              f"{len(splits.X_test):>7,} ferestre  "
              f"({mtest['Date'].min().date()} → {mtest['Date'].max().date()})"
              f"  ← evaluare finala")

        test_counts = mtest.groupby("Symbol").size()
        print(f"\n  Ferestre TEST per stoc:")
        for sym in symbols:
            cnt = test_counts.get(sym, 0)
            print(f"    {sym:<6}  {cnt:>5,} ferestre")

    print(sep)


# --------------------------------------------------------------------------- #
# Bronze
# --------------------------------------------------------------------------- #

def get_bronze(force: bool = False):
    import pandas as pd
    if not force and _validate_cache("bronze", BRONZE_CACHE):
        return pd.read_pickle(BRONZE_CACHE)
    logger.info("Bronze: citesc fisierele brute NYSE...")
    bronze = loader_mod.load_bronze()
    bronze.to_pickle(BRONZE_CACHE)
    _save_metadata("bronze", bronze)
    logger.info("Bronze salvat: %d randuri brute.", len(bronze))
    return bronze


# --------------------------------------------------------------------------- #
# Silver  (Bronze + validare + interpolare)
# --------------------------------------------------------------------------- #

def get_panel(force: bool = False):
    """Silver panel = Bronze + toate verificarile de integritate + interpolare."""
    import pandas as pd
    if not force and _validate_cache("silver", SILVER_CACHE):
        return pd.read_pickle(SILVER_CACHE)

    bronze = get_bronze(force=False)  # foloseste cache-ul bronze (proaspat construit)
    logger.info("Silver: validare, curatare, interpolare...")
    silver, report = silver_mod.stage(bronze, requested_symbols=list(config.SYMBOLS))
    report.print_summary()
    silver.to_pickle(SILVER_CACHE)
    _save_metadata("silver", silver)
    logger.info("Silver: %d randuri | %d simboluri.", len(silver), silver["Symbol"].nunique())
    return silver


# --------------------------------------------------------------------------- #
# Gold  (Silver + features + tinte)
# --------------------------------------------------------------------------- #

def get_features(force: bool = False):
    import pandas as pd
    if not force and _validate_cache("gold", FEATURES_CACHE):
        return pd.read_pickle(FEATURES_CACHE)
    logger.info("Gold: construiesc features si tinte...")
    panel = get_panel(force=False)  # foloseste cache-ul silver (proaspat construit)
    feats = feat_mod.build_features(panel)
    feats.to_pickle(FEATURES_CACHE)
    _save_metadata("gold", feats)
    logger.info("Gold: %d randuri finale.", len(feats))
    return feats
