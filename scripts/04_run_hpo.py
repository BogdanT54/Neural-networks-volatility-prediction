"""Pas 4: cautare de hiperparametri (random search) pentru modelele NN.

Logheaza fiecare configuratie incercata intr-un CSV si raporteaza cele mai bune
configuratii. Acopera cerinta de imbunatatire iterativa (optimizatori, straturi,
bidirectionalitate, activari, dropout, weight decay, tip atentie).
"""
from __future__ import annotations

import argparse

from _common import config, get_features, set_seed
from nyse_vol.data import dataset as ds
from nyse_vol.models.lstm import LSTMRegressor
from nyse_vol.models.seq2seq_attention import Seq2SeqAttention
from nyse_vol.train.hpo import random_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--model", choices=["lstm", "attention"], default="lstm")
    args = ap.parse_args()

    set_seed()
    feats = get_features()
    splits = ds.build_splits(feats)

    factory = LSTMRegressor if args.model == "lstm" else Seq2SeqAttention
    log_path = config.METRICS_DIR / f"hpo_{args.model}.csv"
    df = random_search(factory, splits, ds.make_loaders,
                       n_trials=args.trials, epochs=args.epochs,
                       device=config.DEVICE, seed=config.SEED, log_path=log_path)

    print("\n=== Top 5 configuratii (dupa val loss) ===")
    print(df.head(5).to_string(index=False))
    print(f"\nLog complet: {log_path}")


if __name__ == "__main__":
    main()
