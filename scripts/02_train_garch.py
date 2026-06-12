"""Pas 2: baseline clasic GARCH pe partitia de test (walk-forward).

Itereaza optional pe mai multe ordine (p,q) si distributii, alegand configuratia
cu cea mai mica eroare fata de tinta, si salveaza forecasturile pentru comparatie.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from _common import config, get_features, get_panel, set_seed
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

    set_seed()
    panel = get_panel()
    feats = get_features()
    truth = _true_targets(feats)

    if args.search:
        combos = list(itertools.product(config.GARCH_ORDERS, config.GARCH_DISTS))
    else:
        combos = [((1, 1), "normal")]

    best = None
    for order, dist in combos:
        print(f"GARCH{order} dist={dist} ...")
        fc = garch_forecast_panel(panel, config.VAL_END, order=order, dist=dist,
                                  refit_every=args.refit_every)
        if fc.empty:
            print("  (fara forecasturi)")
            continue
        merged = truth.merge(fc, on=["Symbol", "Date"], how="inner")
        # eroare medie RMSE pe orizontul cel mai scurt, pentru selectie
        err = np.sqrt(np.mean((merged["true_vol_h1"] - merged["vol_garch_h1"]) ** 2))
        print(f"  RMSE(h=1)={err:.5f} pe {len(merged)} puncte")
        if best is None or err < best[0]:
            best = (err, order, dist, fc)

    if best is None:
        raise SystemExit("Niciun forecast GARCH valid.")

    _, order, dist, fc = best
    fc.to_pickle(config.METRICS_DIR / "garch_forecasts.pkl")
    with open(config.METRICS_DIR / "garch_best.txt", "w") as f:
        f.write(f"order={order} dist={dist}\n")
    print(f"\nCea mai buna configuratie GARCH: order={order} dist={dist}")
    print(f"Forecasturi salvate in: {config.METRICS_DIR / 'garch_forecasts.pkl'}")


if __name__ == "__main__":
    main()
