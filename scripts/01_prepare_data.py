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


def _export_csv(panel, feats, out_dir, log) -> None:
    """Exporta CSV-uri inspectabile in artifacts/processed/export/."""
    import numpy as np

    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Panel complet dupa curatare + ffill
    panel_path = export_dir / "panel_preprocessed.csv"
    panel.to_csv(panel_path, index=False, date_format="%Y-%m-%d")
    log.info("  Exportat: %s (%d randuri)", panel_path.name, len(panel))

    # 2. Features complete (toate simbolurile)
    feats_path = export_dir / "features_all.csv"
    feats.to_csv(feats_path, index=False, date_format="%Y-%m-%d", float_format="%.6f")
    log.info("  Exportat: %s (%d randuri)", feats_path.name, len(feats))

    # 3. Sample per simbol: primele si ultimele 5 randuri din fiecare simbol,
    #    util pentru a vedea rapid efectul ffill la capetele seriei
    sample_parts = []
    for sym, g in feats.groupby("Symbol"):
        sample_parts.append(g.head(5))
        sample_parts.append(g.tail(5))
    import pandas as pd
    sample = pd.concat(sample_parts).sort_values(["Symbol", "Date"])
    sample_path = export_dir / "features_sample_per_symbol.csv"
    sample.to_csv(sample_path, index=False, date_format="%Y-%m-%d", float_format="%.6f")
    log.info("  Exportat: %s (%d randuri, cap+coada per simbol)", sample_path.name, len(sample))

    # 4. Statistici descriptive per simbol
    from nyse_vol.data.features import FEATURE_COLS, TARGET_COLS
    stats = feats.groupby("Symbol")[FEATURE_COLS + TARGET_COLS].describe().round(4)
    stats_path = export_dir / "features_stats_per_symbol.csv"
    stats.to_csv(stats_path)
    log.info("  Exportat: %s (statistici descriptive per simbol)", stats_path.name)

    log.info("Toate fisierele exportate in: %s", export_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconstruieste cache-ul")
    ap.add_argument("--verbose", action="store_true", help="afiseaza log-uri DEBUG")
    ap.add_argument("--export", action="store_true",
                    help="salveaza CSV-uri inspectabile in artifacts/processed/export/")
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

    if args.export:
        log.info("--- Export CSV-uri inspectabile ---")
        _export_csv(panel, feats, config.PROCESSED_DIR, log)

    log.info("=== Preprocesare completa ===")


if __name__ == "__main__":
    main()
