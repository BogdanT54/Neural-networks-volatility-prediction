"""Pas 2: baseline clasic GARCH pe partitia de test (walk-forward).

Testeaza stationaritatea log-randamentelor (ADF, KPSS, PP) per simbol,
itereaza pe toate ordinele (1,1),(1,2),(2,1),(2,2) si distributiile
normal/Student-t, alege configuratia optima dupa RMSE(h=1) si salveaza
forecasturile pentru comparatie cu modelele NN.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from _common import config, get_features, get_panel, print_banner, print_data_summary, set_seed
from nyse_vol.data.features import TARGET_COLS
from nyse_vol.models.garch import garch_forecast_panel, stationarity_tests


def _true_targets(feats: pd.DataFrame) -> pd.DataFrame:
    """Tintele reale (volatilitate, nu log) per (Symbol, Date) pe orizonturi."""
    cols = ["Symbol", "Date"] + TARGET_COLS
    df = feats[cols].copy()
    for h, tc in zip(config.HORIZONS, TARGET_COLS):
        df[f"true_vol_h{h}"] = np.exp(df[tc])
    return df.drop(columns=TARGET_COLS)


def _fmt_p(p: float, test: str = "adf") -> str:
    """Formateaza p-value cu notatii speciale pentru margini."""
    if test == "kpss":
        # statsmodels KPSS truncheaza p la intervalul [0.01, 0.10]
        if p >= 0.10:
            return " >0.10"
        if p <= 0.01:
            return " <0.01"
        return f"{p:>6.4f}"
    # ADF si PP: p-valori reale, pot fi extrem de mici
    if p < 0.0001:
        return "<.0001"
    return f"{p:>6.4f}"


def _print_stationarity(results: dict) -> None:
    """Afiseaza tabelul cu rezultatele testelor de stationaritate.

    KPSS: H0 = stationara (nivel constant). p >= 0.05 → stationara (nu respingem H0).
          p-value trunchiat la [0.01, 0.10] in statsmodels — afisat ca >0.10 / <0.01.
    ADF:  H0 = nestationar. p < 0.05 → stationara.
    PP:   H0 = nestationar. p < 0.05 → stationara.
    """
    n = len(results)
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  TESTE DE STATIONARITATE PE LOG-RANDAMENTE ({n} stocuri)")
    print(f"  {'Stoc':<7}  {'ADF stat':>9}  {'p-ADF':>6}  {'KPSS stat':>9}  "
          f"{'p-KPSS':>6}  {'PP stat':>9}  {'p-PP':>6}  Verdict")
    print(sep)

    all_stationary = True
    for sym, r in sorted(results.items()):
        adf  = r.get("adf")
        kpss = r.get("kpss")
        pp   = r.get("pp")

        adf_str  = (f"{adf['stat']:>9.3f}  {_fmt_p(adf['p'], 'adf'):>6}"
                    if adf else f"{'N/A':>9}  {'N/A':>6}")
        kpss_str = (f"{kpss['stat']:>9.3f}  {_fmt_p(kpss['p'], 'kpss'):>6}"
                    if kpss else f"{'N/A':>9}  {'N/A':>6}")
        pp_str   = (f"{pp['stat']:>9.3f}  {_fmt_p(pp['p'], 'pp'):>6}"
                    if pp else f"{'N/A':>9}  {'N/A':>6}")

        adf_ok  = adf["stationary"]  if adf  else True
        kpss_ok = kpss["stationary"] if kpss else True
        pp_ok   = pp["stationary"]   if pp   else True

        is_ok = adf_ok and kpss_ok and pp_ok
        if not is_ok:
            all_stationary = False
        verdict = "STATIONARA ✓" if is_ok else "ATENTIE ⚠"

        print(f"  {sym:<7}  {adf_str}  {kpss_str}  {pp_str}  {verdict}")

    print(sep)
    if all_stationary:
        print("  Concluzie: TOATE seriile sunt STATIONARE — GARCH aplicabil direct.")
    else:
        print("  ATENTIE: unele serii pot necesita diferentiere suplimentara.")
    print(f"\n  Interpretare:")
    print("    ADF  : H0=nestationar.  p<0.05 → respingem H0 → STATIONARA")
    print("    KPSS : H0=stationara.   p>=0.05 → nu respingem H0 → STATIONARA")
    print("           (p trunchiat la [0.01, 0.10] in statsmodels — >0.10 = foarte stationar)")
    print("    PP   : H0=nestationar.  p<0.05 → respingem H0 → STATIONARA")
    print(f"{sep}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every", type=int, default=21,
                    help="refit GARCH la fiecare N zile bursiere (implicit 21)")
    args = ap.parse_args()

    combos = list(itertools.product(config.GARCH_ORDERS, config.GARCH_DISTS))

    print_banner(2, 6, "BASELINE CLASIC — GARCH (walk-forward)", [
        "Ce face:",
        "  • Calculeaza log-randamente si testeaza stationaritatea (ADF, KPSS, PP)",
        "  • Aplica GARCH pe log-randamente (scalate la procente, conv. arch)",
        "  • Evalueaza walk-forward pe setul de test (> 2021-12-31)",
        f"  • La fiecare {args.refit_every} zile bursiere, reface fit-ul pe date extinse",
        f"  • Testeaza {len(combos)} configuratii: 4 ordine x 2 distributii",
        "  • Selecteaza configuratia cu cel mai mic RMSE(h=1)",
        "",
        "Ordine testate:      GARCH(1,1), GARCH(1,2), GARCH(2,1), GARCH(2,2)",
        "Distributii testate: normal (Gaussian), Student-t",
        "",
        "De ce log-randamente?",
        "  log(Close_t / Close_{t-1}) * 100 — stationare, simetrice,",
        "  standard in literatura de volatilitate si in biblioteca arch.",
        "",
        "De ce GARCH?",
        "  GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)",
        "  este standardul industrial pentru modelarea volatilitatii.",
        "  Il folosim ca BASELINE — modelele NN trebuie sa il depaseasca.",
        "",
        "Rezultat: artifacts/metrics/garch_forecasts.pkl",
        "Urmator:  python 03_train_nn.py  (modele neuronale)",
    ])

    set_seed()
    panel = get_panel()
    feats = get_features()
    truth = _true_targets(feats)

    # ── Context: ce stocuri, ce date ──
    print_data_summary(feats)

    feats_test = feats[feats["Date"] > config.VAL_END]
    test_start  = feats_test["Date"].min()
    test_end    = feats_test["Date"].max()
    n_test_days = feats_test["Date"].nunique()
    n_test_sym  = feats_test["Symbol"].nunique()
    print(f"Perioada de test: {test_start.date()} → {test_end.date()}  "
          f"({n_test_days} zile bursiere × {n_test_sym} stocuri = "
          f"{len(feats_test):,} puncte de evaluat)")

    all_syms = sorted(panel["Symbol"].unique().tolist())
    print(f"\n  Stocuri in panel ({len(all_syms)}): {', '.join(all_syms)}")
    print(f"  Teste de stationaritate + GARCH se aplica pe FIECARE stoc.")

    # ── Teste de stationaritate ──
    print("\nTestez stationaritatea log-randamentelor per simbol...")
    stat_results = stationarity_tests(panel)
    _print_stationarity(stat_results)

    # ── Cautare configuratie optima GARCH ──
    print(f"Testez {len(combos)} configuratii GARCH pe {len(all_syms)} stocuri...\n")
    print(f"  GARCH = walk-forward per stoc: fit pe date istorice, forecast pe test.")
    print(f"  Cel mai bun config (RMSE h=1 agregat) → salvat in garch_forecasts.pkl.\n")
    print(f"  {'Configuratie':<22}  {'RMSE(h=1)':>10}  {'Puncte test':>12}")
    print(f"  {'─' * 22}  {'─' * 10}  {'─' * 12}")

    best = None
    for order, dist in combos:
        label = f"GARCH{order} dist={dist}"
        fc = garch_forecast_panel(panel, config.VAL_END, order=order, dist=dist,
                                  refit_every=args.refit_every)
        if fc.empty:
            print(f"  {label:<22}  {'—':>10}  niciun forecast valid")
            continue
        merged = truth.merge(fc, on=["Symbol", "Date"], how="inner")
        err = np.sqrt(np.mean((merged["true_vol_h1"] - merged["vol_garch_h1"]) ** 2))
        n_pts = len(merged)
        marker = "  ← cel mai bun" if (best is None or err < best[0]) else ""
        print(f"  {label:<22}  {err:>10.5f}  {n_pts:>12,}{marker}")
        if best is None or err < best[0]:
            best = (err, order, dist, fc)

    if best is None:
        raise SystemExit("Niciun forecast GARCH valid.")

    _, order, dist, fc = best
    fc.to_pickle(config.METRICS_DIR / "garch_forecasts.pkl")
    with open(config.METRICS_DIR / "garch_best.txt", "w") as f:
        f.write(f"order={order} dist={dist}\n")

    n_syms_saved = fc["Symbol"].nunique()
    n_rows_saved = len(fc)
    print(f"\n{'=' * 62}")
    print(f"  Configuratia optima: GARCH{order}  dist={dist}")
    print(f"  Forecasturi salvate pentru {n_syms_saved} stocuri ({n_rows_saved:,} zile-stoc)")
    print(f"  Fisier: {config.METRICS_DIR / 'garch_forecasts.pkl'}")
    print(f"  Coloane: vol_garch_h1, vol_garch_h5, vol_garch_h10, vol_garch_h20")
    print(f"  Folosit in: 05_evaluate_compare.py + 06_select_and_plot.py")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
