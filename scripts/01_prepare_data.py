"""Pas 1: incarca/genereaza datele, calculeaza trasaturi si tinte, salveaza cache.

Daca DATA_DIR nu contine zip-uri NYSE, se genereaza automat date sintetice in
formatul real, astfel incat pipeline-ul sa ruleze imediat.
"""
from __future__ import annotations

import argparse
import logging
import sys

from _common import config, get_features, get_panel, set_seed


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # reducere zgomot de la biblioteci externe
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconstruieste cache-ul")
    ap.add_argument("--verbose", action="store_true", help="afiseaza log-uri DEBUG")
    args = ap.parse_args()

    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    set_seed()
    log.info("DATA_DIR = %s", config.DATA_DIR)
    log.info("PROCESSED_DIR = %s", config.PROCESSED_DIR)
    log.info("Estimator tinta: %s | Orizonturi: %s",
             config.TARGET_ESTIMATOR, config.HORIZONS)

    log.info("--- Pasul 1/2: Incarcare panel ---")
    panel = get_panel(force=args.force)
    log.info(
        "Panel incarcat: %d randuri | %d simboluri | %s -> %s",
        len(panel), panel["Symbol"].nunique(),
        panel["Date"].min().date(), panel["Date"].max().date()
    )

    log.info("--- Pasul 2/2: Construire features ---")
    feats = get_features(force=args.force)
    log.info("Features finale: %d randuri dupa eliminarea NaN.", len(feats))
    log.info("Salvat in: %s", config.PROCESSED_DIR)
    log.info("=== Preprocesare completa ===")


if __name__ == "__main__":
    main()
