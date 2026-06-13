"""Pas 1: incarca/genereaza datele, calculeaza trasaturi si tinte, salveaza cache.

Daca DATA_DIR nu contine zip-uri NYSE, se genereaza automat date sintetice in
formatul real, astfel incat pipeline-ul sa ruleze imediat.
"""
from __future__ import annotations

import argparse
import logging
import sys

from _common import config, get_features, get_panel, print_banner, set_seed


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # loader-ul are prea multe mesaje DEBUG per fisier — pastram doar INFO
    logging.getLogger("nyse_vol.data.loader").setLevel(logging.INFO)


def _print_csv_schema():
    sep = "-" * 62
    print(f"\n{sep}")
    print("  STRUCTURA FISIERELOR CSV EXPORTATE")
    print(sep)
    print("""
  panel_preprocessed.csv
  ───────────────────────────────────────────────────────────
  Un rand = o zi de tranzactionare per simbol.
  Coloane:
    Symbol  : ticker-ul actiunii (ex: "JPM", "GS", "BAC")
    Date    : data (YYYY-MM-DD)
    Open    : pretul la deschiderea zilei
    High    : pretul maxim intraday
    Low     : pretul minim intraday
    Close   : pretul la inchidere
    Volume  : numarul de actiuni tranzactionate

  features_all.csv
  ───────────────────────────────────────────────────────────
  Un rand = o zi per simbol (dupa eliminarea NaN-urilor).
  Contine toate coloanele de mai sus PLUS:

  FEATURES — intrarile pentru modelele neuronale (9 total):
    log_ret            : log(Close_azi / Close_ieri)
                         Randamentul zilnic in scara logaritmica.
                         Stabil si stationar (vs. pretul brut).

    log_range          : log(High / Low)
                         Amplitudinea intraday. Cu cat e mai mare,
                         cu atat piata a fost mai volatila in ziua
                         respectiva.

    log_volume         : log(Volume)
                         Volumul comprimat logaritmic. Volumul mare
                         precede adesea miscari mari de pret.

    vol_parkinson      : sqrt(1/(4*ln2) * log(H/L)^2)
                         Estimator de volatilitate care foloseste
                         doar High si Low. Eficient, dar ignora
                         directia si gap-urile overnight.

    vol_garman_klass   : formula OHLC (Open, High, Low, Close)
                         Cel mai precis estimator range-based.
                         Folosit si ca TINTA principala a modelului.

    vol_rogers_satchell: formula OHLC robusta la drift si gap-uri
                         Utila pentru actiuni cu gap-uri overnight
                         frecvente.

    vol_close_to_close : sqrt(log_ret^2) = |log_ret|
                         Estimator simplu bazat doar pe randamentul
                         zilnic. Cel mai zgomotos, dar usor de
                         calculat.

    dow_sin / dow_cos  : sin(2*pi*DOW/5) si cos(2*pi*DOW/5)
                         Ziua saptamanii (0=Luni, 4=Vineri) codata
                         ciclic. Modelul invata efecte de calendar
                         (ex: volatilitate mai mare Luni dimineata).

  TINTE — ce prezice modelul (4 orizonturi):
    target_h1  : volatilitatea medie realizata in ziua urmatoare
    target_h5  : volatilitatea medie realizata in urm. 5 zile
    target_h10 : volatilitatea medie realizata in urm. 10 zile
    target_h20 : volatilitatea medie realizata in urm. 20 zile

    Toate tintele sunt in SCALA LOGARITMICA (log-volatilitate).
    Motivul: distributia log-volatilitatii e aproape normala,
    ceea ce face antrenarea mai stabila numeric.
    La evaluare, convertim inapoi cu exp() la scala originala.

  features_sample_per_symbol.csv
  ───────────────────────────────────────────────────────────
  Primele si ultimele 5 randuri din fiecare simbol.
  Util pentru a verifica rapid efectul forward-fill la capetele
  seriei si ca datele arata corect.

  features_stats_per_symbol.csv
  ───────────────────────────────────────────────────────────
  Statistici descriptive (medie, std, min, max, quartile)
  per simbol pentru toate features si tintele.
  Util pentru a detecta simboluri cu distributii anormale.
""")
    print(sep)


def _export_csv(panel, feats, out_dir, log) -> None:
    """Exporta CSV-uri inspectabile in artifacts/processed/export/."""
    import numpy as np

    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    panel_path = export_dir / "panel_preprocessed.csv"
    panel.to_csv(panel_path, index=False, date_format="%Y-%m-%d")
    log.info("  Exportat: %s (%d randuri)", panel_path.name, len(panel))

    feats_path = export_dir / "features_all.csv"
    feats.to_csv(feats_path, index=False, date_format="%Y-%m-%d", float_format="%.6f")
    log.info("  Exportat: %s (%d randuri)", feats_path.name, len(feats))

    sample_parts = []
    for sym, g in feats.groupby("Symbol"):
        sample_parts.append(g.head(5))
        sample_parts.append(g.tail(5))
    import pandas as pd
    sample = pd.concat(sample_parts).sort_values(["Symbol", "Date"])
    sample_path = export_dir / "features_sample_per_symbol.csv"
    sample.to_csv(sample_path, index=False, date_format="%Y-%m-%d", float_format="%.6f")
    log.info("  Exportat: %s (%d randuri, cap+coada per simbol)", sample_path.name, len(sample))

    from nyse_vol.data.features import FEATURE_COLS, TARGET_COLS
    stats = feats.groupby("Symbol")[FEATURE_COLS + TARGET_COLS].describe().round(4)
    stats_path = export_dir / "features_stats_per_symbol.csv"
    stats.to_csv(stats_path)
    log.info("  Exportat: %s (statistici descriptive per simbol)", stats_path.name)

    log.info("Toate fisierele exportate in: %s", export_dir)
    _print_csv_schema()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconstruieste cache-ul")
    ap.add_argument("--verbose", action="store_true", help="afiseaza log-uri DEBUG")
    ap.add_argument("--export", action="store_true",
                    help="salveaza CSV-uri inspectabile in artifacts/processed/export/")
    args = ap.parse_args()

    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    print_banner(1, 6, "PREPROCESARE DATE NYSE 2001–2026", [
        "Ce face:",
        "  1. Incarca datele brute NYSE din directoare an cu an (NYSE_YYYY/)",
        "     Fallback: fisiere ZIP (NYSE_YYYY.zip) sau date sintetice.",
        "  2. Curata datele: elimina preturi invalide, zile nelichide",
        "  3. Reindexeaza la calendarul bursier + forward-fill (max 3 zile)",
        "  4. Calculeaza 9 features tehnice per simbol per zi",
        "  5. Calculeaza tintele de volatilitate pentru h=1,5,10,20 zile",
        "  6. Salveaza cache-ul pentru pasi urmatori",
        "",
        "Rezultat: artifacts/processed/panel.pkl + features.pkl",
        "Urmator:  python 02_train_garch.py  (baseline GARCH)",
    ])

    set_seed()
    log.info("DATA_DIR = %s", config.DATA_DIR)
    log.info("PROCESSED_DIR = %s", config.PROCESSED_DIR)
    log.info("Estimator tinta: %s | Orizonturi: %s",
             config.TARGET_ESTIMATOR, config.HORIZONS)
    log.info("Split: Train <= %s | Val <= %s | Test > %s",
             config.TRAIN_END, config.VAL_END, config.VAL_END)

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
