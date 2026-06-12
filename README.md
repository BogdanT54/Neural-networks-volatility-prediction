# Predicția volatilității NYSE cu rețele neuronale

Proiect pentru *Rețele neuronale și tehnici de Deep Learning* (ASE). Scopul:
prezicerea **volatilității** acțiunilor de pe NYSE pe orizonturi multiple
(`1, 5, 10, 20` zile), comparând un model clasic (**GARCH**) cu rețele neuronale
recurente (**LSTM** și **Encoder-Decoder + Attention**), cu accent pe căutarea
iterativă a hiperparametrilor.

## Ideea pe scurt

- **Țintă:** volatilitatea realizată viitoare, estimată din OHLC cu estimatori
  *range-based* (implicit **Garman-Klass**; sunt implementați și Parkinson,
  Rogers-Satchell, Yang-Zhang, close-to-close).
- **Intrări (multivariate):** log-randament, log-range, log-volum, cei patru
  estimatori de volatilitate zilnică, ziua din săptămână (ciclic).
- **Scop:** un model **global** antrenat pe un coș de ~30 de simboluri lichide.
- **Split cronologic** (fără scurgere de informație): train ≤ 2018,
  validare 2019–2021, test > 2021 (configurabil în `nyse_vol/config.py`).

## Structura

```
nyse_vol/
  config.py                 # căi, simboluri, orizonturi, spații de căutare HPO
  data/
    sample_generator.py     # OHLCV sintetic în formatul real NYSE (rulare fără date reale)
    loader.py               # zip an -> CSV zilnice -> DataFrame long, cu curățare
    volatility.py           # Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, close-to-close
    features.py             # trăsături + ținte multi-orizont (log-volatilitate)
    dataset.py              # ferestre glisante, split cronologic, scalare, DataLoader
  models/
    garch.py                # baseline GARCH(p,q) walk-forward (lib. `arch`)
    lstm.py                 # LSTM/GRU stivuit, bidirecțional opțional
    seq2seq_attention.py    # Encoder-Decoder LSTM + atenție Bahdanau/Luong
  train/
    trainer.py              # buclă antrenare, early stopping, grad clipping
    hpo.py                  # random search (optimizatori, straturi, activări, regularizare)
  eval/
    metrics.py              # RMSE, MAE, QLIKE, R², acuratețe direcțională
    plots.py                # curbe loss, pred vs real, heatmap atenție, comparații
scripts/
  01_prepare_data.py        # date + trăsături (generează sintetic dacă lipsesc)
  02_train_garch.py         # baseline clasic
  03_train_nn.py            # antrenează LSTM și Attention
  04_run_hpo.py             # căutare hiperparametri
  05_evaluate_compare.py    # tabel + grafice comparative
tests/
  test_volatility.py        # teste numerice pentru estimatori
```

## Instalare

```bash
pip install -r requirements.txt
```

## Date

Codul citește zip-uri an cu an (`NYSE_2001.zip` … `NYSE_2026.zip`) cu fișiere
zilnice `NYSE_YYYYMMDD.csv` (`Symbol,Date,Open,High,Low,Close,Volume`). Setează
calea cu variabila de mediu `DATA_DIR`:

```bash
export DATA_DIR=/cale/catre/datele_NYSE
```

Dacă `DATA_DIR` nu conține zip-uri, pipeline-ul **generează automat date sintetice**
în același format, ca să poată rula imediat.

## Rulare end-to-end

```bash
python scripts/01_prepare_data.py            # date + trăsături (cache)
python scripts/02_train_garch.py             # baseline GARCH (--search pentru iterație p,q,dist)
python scripts/03_train_nn.py --epochs 30    # LSTM + Attention
python scripts/04_run_hpo.py --trials 10     # căutare hiperparametri (LSTM)
python scripts/05_evaluate_compare.py        # tabel + grafice comparative
```

Artefactele (modele, grafice, metrici) apar în `artifacts/`.

## Note

- Pentru testare rapidă folosește `--epochs 2` și `--trials 3`.
- Datele reale și artefactele NU se comit (vezi `.gitignore`).
