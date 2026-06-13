"""Pas 2: baseline clasic GARCH pe partitia de test (walk-forward).

Itereaza optional pe mai multe ordine (p,q) si distributii, alegand configuratia
cu cea mai mica eroare fata de tinta, si salveaza forecasturile pentru comparatie.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from _common import config, get_features, get_panel, print_banner, set_seed
from nyse_vol.data.features import TARGET_COLS
from nyse_vol.models.garch import garch_forecast_panel


def _true_targets(feats: pd.DataFrame) -> pd.DataFrame:
    """Tintele reale (volatilitate, nu log) per (Symbol, Date) pe orizonturi."""
    cols = ["Symbol", "Date"] + TARGET_COLS
    df = feats[cols].copy()
    for h, tc in zip(config.HORIZONS, TARGET_COLS):
        df[f"true_vol_h{h}"] = np.exp(df[tc])
    return df.drop(columns=TARGET_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true",
                    help="itereaza pe GARCH_ORDERS x GARCH_DISTS")
    ap.add_argument("--refit-every", type=int, default=21)
    args = ap.parse_args()

    print_banner(2, 6, "BASELINE CLASIC — GARCH (walk-forward)", [
        "Ce face:",
        "  • Antreneaza modele GARCH pe datele de train+val",
        "  • Evalueaza walk-forward pe setul de test (> 2021-12-31)",
        "  • La fiecare 21 zile bursiere, refac fit-ul pe date extinse",
        "  • Selecteaza configuratia cu cel mai mic RMSE(h=1)",
        "",
        "De ce GARCH?",
        "  GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)",
        "  este standardul industrial pentru modelarea volatilitatii.",
        "  Il folosim ca BASELINE — modelele NN trebuie sa il depaseasca.",
        "",
        f"  Ordine testate: {config.GARCH_ORDERS if args.search else '[(1,1)]'}",
        f"  Distributii: {config.GARCH_DISTS if args.search else '[normal]'}",
        f"  Refit la fiecare: {args.refit_every} zile bursiere",
        "",
        "Rezultat: artifacts/metrics/garch_forecasts.pkl",
        "Urmator:  python 03_train_nn.py  (modele neuronale)",
    ])

    set_seed()
    panel = get_panel()
    feats = get_features()
    truth = _true_targets(feats)

    test_start = feats[feats["Date"] > config.VAL_END]["Date"].min()
    test_end = feats["Date"].max()
    n_test_days = feats[feats["Date"] > config.VAL_END]["Date"].nunique()
    print(f"Perioada de test: {test_start.date()} -> {test_end.date()} "
          f"({n_test_days} zile bursiere)")

    if args.search:
        combos = list(itertools.product(config.GARCH_ORDERS, config.GARCH_DISTS))
        print(f"Cautare pe {len(combos)} configuratii GARCH...\n")
    else:
        combos = [((1, 1), "normal")]
        print("Folosind configuratia implicita GARCH(1,1) dist=normal.\n"
              "Adauga --search pentru a cauta configuratia optima.\n")

    best = None
    for order, dist in combos:
        print(f"  GARCH{order} dist={dist} ...")
        fc = garch_forecast_panel(panel, config.VAL_END, order=order, dist=dist,
                                  refit_every=args.refit_every)
        if fc.empty:
            print("    (niciun forecast valid — sarind peste)")
            continue
        merged = truth.merge(fc, on=["Symbol", "Date"], how="inner")
        err = np.sqrt(np.mean((merged["true_vol_h1"] - merged["vol_garch_h1"]) ** 2))
        n_pts = len(merged)
        print(f"    RMSE(h=1)={err:.5f} pe {n_pts} puncte de test")
        if best is None or err < best[0]:
            best = (err, order, dist, fc)

    if best is None:
        raise SystemExit("Niciun forecast GARCH valid.")

    _, order, dist, fc = best
    fc.to_pickle(config.METRICS_DIR / "garch_forecasts.pkl")
    with open(config.METRICS_DIR / "garch_best.txt", "w") as f:
        f.write(f"order={order} dist={dist}\n")

    print(f"\n{'=' * 62}")
    print(f"  Cea mai buna configuratie GARCH: order={order} dist={dist}")
    print(f"  Forecasturi salvate in: {config.METRICS_DIR / 'garch_forecasts.pkl'}")
    print(f"  Coloane forecast: vol_garch_h1, vol_garch_h5, vol_garch_h10, vol_garch_h20")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
