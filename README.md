# NYSE Volatility Prediction with Neural Networks

> Can a sequence model beat GARCH at forecasting stock volatility? This project builds the experiment properly and lets the numbers answer.

Forecasting daily volatility for 30 liquid NYSE stocks over 2001 to 2026, comparing two deep learning architectures (LSTM and Encoder-Decoder with attention) against the econometric standard (GARCH) and a naive persistence baseline, across four forecast horizons.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![arch](https://img.shields.io/badge/arch-GARCH-006400)
![statsmodels](https://img.shields.io/badge/statsmodels-ADF%20%7C%20KPSS%20%7C%20PP-4B8BBE)

---

## Why this project

Volatility forecasting sits at the centre of three practical problems: risk management, option pricing through Black-Scholes, and portfolio allocation under Markowitz. GARCH has been the working standard since the 1980s and it is genuinely hard to beat.

The interesting question is not whether a neural network can fit volatility. It is whether it can beat GARCH once the experiment is set up honestly, with no look-ahead bias, a test set that stays sealed until the end, and a naive baseline that is allowed to be as strong as it really is. Volatility is highly autocorrelated at lag 1, so "tomorrow looks like today" is a surprisingly tough opponent. Any model that cannot beat it adds no practical value.

That constraint shaped the whole design.

## Data

| Item | Value |
|------|-------|
| Universe | 30 liquid NYSE stocks, continuously listed since 2001 |
| Period | 2001 to 2026 |
| Source format | Annual OHLCV CSV files (`Symbol, Date, Open, High, Low, Close, Volume`) |
| Target | Garman-Klass volatility estimator |
| Horizons | 1, 5, 10 and 20 trading days |
| Input window | 60 trading days by 9 features |

The split is strictly chronological, so no information from the future ever reaches the past:

```
Train      2001-01-01 to 2018-12-31    ~18 years
Validation 2019-01-01 to 2021-12-31    ~3 years
Test       2022-01-01 onward           ~4 years, sealed until final evaluation
```

**Why Garman-Klass and not close-to-close?** Garman-Klass uses all four OHLC prices and is 5 to 8 times more efficient, meaning the same precision from a fraction of the data. Close-to-close throws away all intraday movement and badly understates volatility on exactly the days that matter, such as crises and earnings surprises.

## Approach

**Data pipeline (medallion architecture).** Raw CSV goes through Bronze (loaded and type checked), Silver (cleaned, gaps interpolated up to 2 consecutive days, minimum 250 observations per symbol enforced), and Gold (9 features plus 4 targets). Each transition validates before passing on, and each layer caches to Parquet so reruns skip work already done.

**Features.** Daily log return, log range, log volume, four different volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, close-to-close) and a cyclical sine/cosine encoding of the weekday. Multiple volatility estimators are included deliberately because each captures something different: Parkinson uses only the high-low range, Rogers-Satchell is robust to overnight drift, Garman-Klass combines all prices. Together they describe the current volatility regime from several angles.

**Models.**

- **LSTM.** Stacked LSTM, optionally bidirectional, with a dense regression head predicting all four horizons at once.
- **Encoder-Decoder with attention.** The encoder reads the 60 day window, the decoder generates each horizon autoregressively, attending over all 60 encoder states at every step. Both Bahdanau (additive) and Luong (multiplicative) scoring are implemented. The attention weights are exported as a heatmap, which shows which days in the window the model actually leans on for each horizon.
- **GARCH baseline.** Eight configurations tested (orders (1,1), (1,2), (2,1), (2,2) crossed with normal and Student-t), evaluated walk-forward with a refit every 21 trading days and the conditional variance updated daily in between.
- **Naive baseline.** Tomorrow's volatility equals today's. The bar every other model has to clear.

**Guarding against leakage.** The feature scaler and target normalisation are fit on training data only. The input window always ends before the target day. Hyperparameters are selected on validation, never on test.

## Evaluation

Five metrics on the sealed test set, computed in raw volatility scale after inverting the log transform:

| Metric | Reads as |
|--------|----------|
| RMSE | Average squared error, same units as volatility |
| MAE | Average absolute error, more robust to outliers |
| R² | Share of variance explained |
| QLIKE | Asymmetric loss that punishes underestimation harder |
| Dir. Acc | Share of days where the direction of change is right |

**Why QLIKE matters here.** In risk management, underestimating volatility (believing the market is calm when it is not) is far more dangerous than overestimating it. RMSE and MAE are symmetric and treat both errors the same. QLIKE does not, which makes it the metric that reflects the actual cost of being wrong.

## Stack

Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib, `arch` for GARCH, `statsmodels` for the stationarity tests (ADF, KPSS, Phillips-Perron).

## Repository structure

```
nyse_vol/                  Core package
  config.py                Central configuration
  data/                    Volatility estimators, features, windowing, splits
  models/                  LSTM, Seq2Seq with attention, GARCH
  train/                   Training loop, random search HPO
  eval/                    Metrics and plots
scripts/                   Executable pipeline, steps 01 to 06
tests/                     Unit tests for the volatility estimators
docs/METHODOLOGY.md        Full technical reference
```

## How to run

```bash
pip install -r requirements.txt

python scripts/01_prepare_data.py        # preprocess and build features
python scripts/02_train_garch.py         # GARCH baseline, picks best of 8 configs
python scripts/03_train_nn.py --epochs 30
python scripts/04_run_hpo.py --model lstm --trials 10 --epochs 15
python scripts/05_evaluate_compare.py    # all models on the test set
python scripts/06_select_and_plot.py --symbols JPM GS BAC
```

Raw data goes in `data_raw/` as `NYSE_YYYY.csv`. Set `DEVICE=cuda` for GPU, or `SYMBOLS="JPM,GS,BAC"` to run on a subset.

## What I took away from it

Setting up the comparison honestly turned out to be harder and more interesting than building the models. The naive baseline is strong at the one day horizon because volatility is so persistent, and being disciplined about that rather than quietly choosing a weaker baseline is what makes the result mean anything. Sequence models also smooth the signal, which shows up as a small visible lag against the real series. That is inherent to the architecture rather than a bug, and it is why the evaluation leans on aggregate error metrics instead of eyeballing spike alignment.

Full technical detail, including every formula, the HPO search space and the design rationale for each choice, is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
