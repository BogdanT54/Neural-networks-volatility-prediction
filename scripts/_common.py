"""Utilitare comune pentru scripturi: seed, cache pentru date procesate."""
from __future__ import annotations

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

FEATURES_CACHE = config.PROCESSED_DIR / "features.pkl"
PANEL_CACHE = config.PROCESSED_DIR / "panel.pkl"


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_panel(force: bool = False):
    import pandas as pd
    if PANEL_CACHE.exists() and not force:
        return pd.read_pickle(PANEL_CACHE)
    panel = loader_mod.load_panel()
    panel.to_pickle(PANEL_CACHE)
    return panel


def get_features(force: bool = False):
    import pandas as pd
    if FEATURES_CACHE.exists() and not force:
        return pd.read_pickle(FEATURES_CACHE)
    panel = get_panel(force=force)
    feats = feat_mod.build_features(panel)
    feats.to_pickle(FEATURES_CACHE)
    return feats
