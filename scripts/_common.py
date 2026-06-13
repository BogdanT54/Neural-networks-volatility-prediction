"""Utilitare comune pentru scripturi: seed, cache pe trei straturi, banner.

Pipeline medallion:
  Bronze → Silver → Gold
  loader  → silver  → features
"""
from __future__ import annotations

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

BRONZE_CACHE  = config.PROCESSED_DIR / "bronze.pkl"
SILVER_CACHE  = config.PROCESSED_DIR / "panel.pkl"    # "panel" = silver
FEATURES_CACHE = config.PROCESSED_DIR / "features.pkl"


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
# Bronze
# --------------------------------------------------------------------------- #

def get_bronze(force: bool = False):
    import pandas as pd
    if BRONZE_CACHE.exists() and not force:
        logger.info("Bronze din cache: %s", BRONZE_CACHE)
        return pd.read_pickle(BRONZE_CACHE)
    logger.info("Bronze: citesc fisierele brute NYSE...")
    bronze = loader_mod.load_bronze()
    bronze.to_pickle(BRONZE_CACHE)
    logger.info("Bronze salvat: %s (%d randuri)", BRONZE_CACHE, len(bronze))
    return bronze


# --------------------------------------------------------------------------- #
# Silver  (Bronze + validare + interpolare)
# --------------------------------------------------------------------------- #

def get_panel(force: bool = False):
    """Silver panel = Bronze + toate verificarile de integritate + interpolare.

    Alias „panel" pastrat pentru compatibilitate cu scripturile existente.
    """
    import pandas as pd
    if SILVER_CACHE.exists() and not force:
        logger.info("Silver din cache: %s", SILVER_CACHE)
        return pd.read_pickle(SILVER_CACHE)

    bronze = get_bronze(force=force)
    logger.info("Silver: validare, curatare, interpolare...")
    silver, report = silver_mod.stage(bronze, requested_symbols=list(config.SYMBOLS))
    report.print_summary()
    silver.to_pickle(SILVER_CACHE)
    logger.info("Silver salvat: %s (%d randuri | %d simboluri)",
                SILVER_CACHE, len(silver), silver["Symbol"].nunique())
    return silver


# --------------------------------------------------------------------------- #
# Gold  (Silver + features + tinte)
# --------------------------------------------------------------------------- #

def get_features(force: bool = False):
    import pandas as pd
    if FEATURES_CACHE.exists() and not force:
        logger.info("Gold (features) din cache: %s", FEATURES_CACHE)
        return pd.read_pickle(FEATURES_CACHE)
    logger.info("Gold: construiesc features si tinte...")
    panel = get_panel(force=force)
    feats = feat_mod.build_features(panel)
    feats.to_pickle(FEATURES_CACHE)
    logger.info("Gold salvat: %s (%d randuri)", FEATURES_CACHE, len(feats))
    return feats
