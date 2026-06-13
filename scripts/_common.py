"""Utilitare comune pentru scripturi: seed, cache pentru date procesate, banner."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# permite rularea scripturilor direct (python scripts/01_...py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyse_vol import config  # noqa: E402
from nyse_vol.data import features as feat_mod  # noqa: E402
from nyse_vol.data import loader as loader_mod  # noqa: E402

logger = logging.getLogger(__name__)

_SEP = "=" * 64


def print_banner(step: int, total: int, title: str, lines: list | None = None) -> None:
    """Afiseaza un banner vizibil in consola pentru un pas al pipeline-ului."""
    print(f"\n{_SEP}")
    print(f"  PAS {step}/{total} — {title}")
    if lines:
        print(f"  {'-' * 60}")
        for line in lines:
            print(f"  {line}")
    print(f"{_SEP}\n")


FEATURES_CACHE = config.PROCESSED_DIR / "features.pkl"
PANEL_CACHE = config.PROCESSED_DIR / "panel.pkl"


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_panel(force: bool = False):
    import pandas as pd
    if PANEL_CACHE.exists() and not force:
        logger.info("Panel incarcat din cache: %s (foloseste --force pentru a reface)", PANEL_CACHE)
        return pd.read_pickle(PANEL_CACHE)
    logger.info("Cache inexistent sau --force activ — procesez datele brute.")
    panel = loader_mod.load_panel()
    panel.to_pickle(PANEL_CACHE)
    logger.info("Panel salvat in cache: %s", PANEL_CACHE)
    return panel


def get_features(force: bool = False):
    import pandas as pd
    if FEATURES_CACHE.exists() and not force:
        logger.info("Features incarcate din cache: %s (foloseste --force pentru a reface)", FEATURES_CACHE)
        return pd.read_pickle(FEATURES_CACHE)
    logger.info("Cache inexistent sau --force activ — construiesc features.")
    panel = get_panel(force=force)
    feats = feat_mod.build_features(panel)
    feats.to_pickle(FEATURES_CACHE)
    logger.info("Features salvate in cache: %s", FEATURES_CACHE)
    return feats
