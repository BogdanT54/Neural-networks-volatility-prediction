"""Pas 1: incarca/genereaza datele, calculeaza trasaturi si tinte, salveaza cache.

Daca DATA_DIR nu contine zip-uri NYSE, se genereaza automat date sintetice in
formatul real, astfel incat pipeline-ul sa ruleze imediat.
"""
from __future__ import annotations

import argparse

from _common import config, get_features, get_panel, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconstruieste cache-ul")
    args = ap.parse_args()

    set_seed()
    print(f"DATA_DIR = {config.DATA_DIR}")
    panel = get_panel(force=args.force)
    print(f"Panel: {len(panel):,} randuri, {panel['Symbol'].nunique()} simboluri, "
          f"{panel['Date'].min().date()} -> {panel['Date'].max().date()}")

    feats = get_features(force=args.force)
    print(f"Trasaturi: {len(feats):,} randuri dupa eliminarea NaN.")
    print(f"Tinta: estimator={config.TARGET_ESTIMATOR}, orizonturi={config.HORIZONS}")
    print(f"Salvat in: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
