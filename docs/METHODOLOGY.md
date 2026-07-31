# Methodology

Full technical reference for the NYSE volatility prediction project. The [README](../README.md) covers the high level design; this document covers every formula, parameter and design decision.

## Contents

1. [Objective](#1-objective)
2. [Data and symbols](#2-data-and-symbols)
3. [Pipeline, step by step](#3-pipeline-step-by-step)
4. [Features and targets](#4-features-and-targets)
5. [Model architectures](#5-model-architectures)
6. [Training and hyperparameter optimisation](#6-training-and-hyperparameter-optimisation)
7. [Evaluation and metrics](#7-evaluation-and-metrics)
8. [Generated plots](#8-generated-plots)
9. [Running the pipeline](#9-running-the-pipeline)
10. [Configuration](#10-configuration)
11. [Design notes](#11-design-notes)

---

## 1. Objective

Predict **future volatility** of NYSE stocks across four horizons:

| Horizon | Meaning |
|---------|---------|
| h = 1  | Average realised volatility tomorrow |
| h = 5  | Average volatility over the next 5 trading days (about 1 week) |
| h = 10 | Average volatility over the next 10 trading days (about 2 weeks) |
| h = 20 | Average volatility over the next 20 trading days (about 1 month) |

**Target:** the Garman-Klass estimator of daily volatility, which is more precise than close-to-close because it uses the full OHLC price set.

**Practical motivation:** volatility prediction is central to risk management, option pricing (Black-Scholes) and portfolio allocation (Markowitz).

---

## 2. Data and symbols

### Symbols

30 liquid stocks, continuously listed on the NYSE since 2001:

```
AA   ABT  AEE  AEP  AES  AFL  AIG  AXP  BA   BAC
C    CAT  CL   CVX  DD   DIS  DOW  DUK  EMR  F
GE   GS   HD   IBM  JNJ  JPM  KO   MCD  MMM  MO
```

The list can be overridden through the `SYMBOLS` environment variable (comma separated):

```bash
export SYMBOLS="JPM,GS,BAC,C,AXP"
```

### Raw data format

Annual CSV files in `data_raw/`, named `NYSE_YYYY.csv` (for example `NYSE_2001.csv`), containing the columns `Symbol, Date, Open, High, Low, Close, Volume`.

### Chronological split

```
Train:      2001-01-01  to  2018-12-31   (~18 years)
Validation: 2019-01-01  to  2021-12-31   (~3 years)
Test:       2022-01-01  onward           (~4 years, sealed until final evaluation)
```

The split is **strictly chronological**, so no information from the future reaches the past. There is no look-ahead bias.

---

## 3. Pipeline, step by step

### Medallion architecture (Bronze to Silver to Gold)

Data processing follows a three layer model with validation at every transition:

```
[Bronze]  Raw CSV          -->  full DataFrame, all symbols
    |     validation: columns present, types correct, null values checked
[Silver]  Cleaned data     -->  interpolation of at most 2 consecutive missing days,
    |                           symbol activity checked
    |     validation: minimum 250 observations per symbol
[Gold]    Features+targets -->  9 features and 4 targets, ready for the models
```

The cache is written as Parquet files in `artifacts/processed/`. If the symbol set has not changed, preprocessing steps are skipped on subsequent runs.

### Step 1, preprocessing and feature engineering (`01_prepare_data.py`)

- Loads every CSV file from `data_raw/`
- Validates and cleans the data (Silver): missing values, interpolation of at most 2 consecutive days
- Computes the 9 features and 4 targets (Gold)
- Optional (`--export`): writes `features_all.csv`, `panel_silver.csv` and summary statistics

### Step 2, GARCH baseline (`02_train_garch.py`)

- Runs **stationarity tests** on the log returns of each symbol:
  - ADF (H0 = non-stationary; p < 0.05 means stationary)
  - KPSS (H0 = stationary; p >= 0.05 means stationary)
  - Phillips-Perron (H0 = non-stationary; p < 0.05 means stationary)
- Tests **8 GARCH configurations** (4 orders by 2 distributions):
  - Orders: (1,1), (1,2), (2,1), (2,2)
  - Distributions: normal, Student-t
  - Compared on RMSE, MAE, QLIKE (out of sample) and AIC, BIC (in sample)
- **Walk-forward evaluation** on the test set: refit every 21 trading days on an expanding window. Between refits the conditional variance is updated daily with the observed return: `h_{t+1} = omega + alpha * r_t^2 + beta * h_t`
- Saves the best model as `artifacts/metrics/garch_forecasts.pkl`

### Step 3, reference neural networks (`03_train_nn.py`)

- Trains the LSTM and/or the Encoder-Decoder with attention using **default hyperparameters**
- Saves `artifacts/models/lstm.pt` and `artifacts/models/attention.pt`
- Generates loss curves and the attention heatmap in `plots/antrenare/`

### Step 4, hyperparameter optimisation (`04_run_hpo.py`)

- **Random search** over the space defined in `config.py`
- Evaluates each trial on the **validation** set (the test set stays sealed)
- Retrains the chosen model for the full number of epochs
- Saves `lstm_best_hpo.pt` / `attention_best_hpo.pt` plus a JSON config
- Runs separately per model: `--model lstm` and `--model attention`

### Step 5, comparative evaluation (`05_evaluate_compare.py`)

- Loads every available model (HPO versions take priority over reference versions)
- Computes RMSE, MAE, R², QLIKE and directional accuracy on the test set
- Generates comparative bar charts and predicted versus actual plots
- Saves `artifacts/metrics/comparison.csv`

### Step 6, stock selection and final plots (`06_select_and_plot.py`)

- Allows a subset of symbols to be chosen, interactively or with `--symbols`
- Automatically selects the best model by RMSE at h=1 across the whole test set
- Per chosen stock, generates:
  - A separate subplot per model (actual versus predicted at h=1, with RMSE in the title)
  - 4 subplots (h=1, 5, 10, 20) for the best model by RMSE
  - A bar chart of RMSE, MAE, R² and QLIKE per model
- Saves `artifacts/metrics/results_selected_stocks.csv`

---

## 4. Features and targets

### The 9 input features

Every input window contains **60 days by 9 features**, standardised with a `StandardScaler` fit exclusively on the training data.

| Feature | Formula | Meaning |
|---------|---------|---------|
| `log_ret` | `log(Close_t / Close_{t-1})` | Daily log return |
| `log_range` | `log(High / Low)` | Intraday range, a volatility proxy |
| `log_volume` | `log(Volume)` | Volume, log compressed |
| `vol_parkinson` | `sqrt( [log(H/L)]^2 / (4*ln2) )` | Parkinson (1980), range based |
| `vol_garman_klass` | `sqrt( 0.5*[log(H/L)]^2 - (2*ln2-1)*[log(C/O)]^2 )` | Garman-Klass (1980), full OHLC |
| `vol_rogers_satchell` | `sqrt( log(H/O)*log(H/C) + log(L/O)*log(L/C) )` | Rogers-Satchell (1991), robust to drift |
| `vol_close_to_close` | `sqrt( [log(C_t/C_{t-1})]^2 )` | Simple volatility, squared return |
| `dow_sin` | `sin(2*pi * weekday / 5)` | Intraweek seasonality, cyclical encoding |
| `dow_cos` | `cos(2*pi * weekday / 5)` | Intraweek seasonality, cyclical encoding |

**Why several volatility estimators as features?** Each captures different information. Parkinson uses only the high and low, Rogers-Satchell is robust to overnight drift, and Garman-Klass combines all prices. Together they give the model a multidimensional description of the current volatility regime.

**Why cyclical (sine/cosine) encoding for the weekday?** A linear encoding (Monday=0, Tuesday=1, ..., Friday=4) would not capture that Friday and Monday are neighbours in the weekly cycle. Sine and cosine map the day onto a circle, preserving that continuity.

### The 4 output targets

The model predicts **4 values** per day, one per horizon. From `features.py`:

```python
daily_var = garman_klass(Open, High, Low, Close)   # daily variance

fwd_var = (daily_var.shift(-1)        # shift 1 day into the future
           .rolling(hz).mean()        # mean variance over hz consecutive days
           .shift(-(hz - 1)))         # align the window to start at t+1

target_h{hz}(t) = log( sqrt(fwd_var(t)) )
```

Concretely, for a day `t`:

| Column | Represents | Detailed form |
|--------|-----------|---------------|
| `target_h1` | Log volatility realised **tomorrow** (day t+1) | `log(sigma_GK(t+1))` |
| `target_h5` | Mean log volatility over the **next 5 days** | `log( sqrt( mean(sigma^2_GK(t+1..t+5)) ) )` |
| `target_h10` | Mean over **t+1 to t+10** | Same form |
| `target_h20` | Mean over **t+1 to t+20** | Same form |

**Why the log transform?** Raw volatility is log-normally distributed, with rare but extreme spikes. In log space the distribution is closer to normal, gradients are more stable and MSE converges better. At evaluation time `exp()` is applied to return to the original scale (daily absolute volatility, typically 0.01 to 0.05).

### Anti-leakage measures

All transformations are **fit exclusively on the training set**:

- `StandardScaler` for features: fit on `X_train`, then `transform` applied to `X_val` and `X_test`
- Target normalisation: mean and standard deviation computed on `y_train`, applied to every split
- The input window (`X`) always ends **before** the target day
- HPO optimises on validation, never on test

---

## 5. Model architectures

### LSTM (`nyse_vol/models/lstm.py`)

Stacked LSTM with a regression head:

```
Input: (batch, 60, 9)             # 60 days by 9 features

Stacked LSTM (num_layers layers, optionally bidirectional)
  -> Last hidden state: (batch, hidden_size x [1 or 2])

Dense head: Linear(hidden_size) -> Activation -> Dropout -> Linear(4)
  -> Output: (batch, 4)           # log volatility for h = 1, 5, 10, 20
```

**Defaults:** `hidden_size=64`, `num_layers=2`, `dropout=0.2`, `bidirectional=False`, `head_activation="tanh"`.

### Encoder-Decoder with attention (`nyse_vol/models/seq2seq_attention.py`)

Seq2Seq architecture with an attention mechanism, generating each horizon autoregressively:

```
Input: (batch, 60, 9)

[ENCODER]
Stacked LSTM (optionally bidirectional)
  --> Hidden states for all 60 positions: (batch, 60, enc_dim)
  --> Final state --> linear bridge --> initial decoder state

[DECODER, 4 autoregressive steps, one per horizon]
At each step:
  1. Attention over the 60 encoder states:
       Bahdanau (additive):      score = v^T * tanh(W1*s + W2*h_enc)
       Luong (multiplicative):   score = s^T * W * h_enc
     --> attention weights over the 60 day window (summing to 1)
  2. Context vector = weighted sum of the encoder states
  3. LSTMCell(previous_prediction + context) --> new state
  4. Linear(state + context) --> prediction for the current horizon

Output: (batch, 4)
```

**Advantage over the plain LSTM:** attention lets the decoder focus on the days in the 60 day window that matter most for each specific horizon. The attention heatmap generated in step 3 visualises those weights.

### GARCH baseline

Classical GARCH(p,q) for the conditional variance:

```
h_{t+1} = omega + alpha_1*r_t^2 + ... + alpha_p*r_{t-p+1}^2
                + beta_1*h_t   + ... + beta_q*h_{t-q+1}

vol_garch_h1(t) = sqrt(h_{t+1})
vol_garch_h5(t) = sqrt( mean(h_{t+1}, h_{t+2}, ..., h_{t+5}) )
```

**Walk-forward evaluation with daily updating:** after each refit (every 21 days) the parameters (omega, alpha, beta) are held fixed while `h_t` is updated **daily** with the observed return. This makes the predictions vary naturally day to day rather than moving in steps.

### Naive model (persistence baseline)

```
pred_naive(t + h) = vol_garman_klass(t)    for any h
```

Assumes tomorrow's volatility equals today's. It is surprisingly competitive at h=1 because volatility autocorrelation at lag 1 is high. Any serious model has to beat it to add practical value.

---

## 6. Training and hyperparameter optimisation

### Training loop (`nyse_vol/train/trainer.py`)

- **Loss:** MSE on the standardised targets (normalised log volatility)
- **Optimisers available:** Adam, AdamW, RMSprop, SGD
- **Early stopping:** patience of 6 epochs without improvement in `val_loss`
- **LR scheduler:** `ReduceLROnPlateau` with `factor=0.5`, `patience=2`
- **Gradient clipping:** maximum norm 1.0, for RNN stability
- **Best checkpoint:** restored automatically at the end of training

### HPO, random search (`nyse_vol/train/hpo.py`)

```
For each of N trials:
  1. Sample a random configuration from the search space
  2. Train the model for `--epochs` epochs (default 10)
  3. Evaluate val_loss on the VALIDATION set
  4. Log the configuration and results to CSV

Select the configuration with the lowest val_loss.

Retrain with the optimal configuration for the full number of epochs (default 30).
Save: lstm_best_hpo.pt / attention_best_hpo.pt plus a JSON config
```

**Why optimise on validation and not test?** The test set stays sealed until final evaluation. Choosing hyperparameters on test would be cheating: the model would be overfit to test and would not generalise to new data.

**Reading `best_val`:** it is MSE on the standardised targets. The reference value is 1.0, which is what a model that always predicts the mean would score. `best_val < 1.0` means the model has learned something. Typical good values are 0.3 to 0.6.

### HPO search space

| Parameter | Values |
|-----------|--------|
| `optimizer` | adam, adamw, rmsprop, sgd |
| `lr` | 3e-4, 1e-3, 3e-3 |
| `weight_decay` | 0.0, 1e-5, 1e-4 |
| `hidden_size` | 32, 64, 128 |
| `num_layers` | 1, 2, 3 |
| `dropout` | 0.0, 0.2, 0.4 |
| `bidirectional` | False, True |
| `head_activation` | tanh, relu |
| `attention` (Seq2Seq) | bahdanau, luong |

---

## 7. Evaluation and metrics

All metrics are computed on the test set (after 2021-12-31), in **raw volatility scale** after applying `exp()` to the log targets.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `sqrt(mean((y - y_pred)^2))` | Root mean squared error, same units as volatility. Lower is better. |
| **MAE** | `mean(abs(y - y_pred))` | Mean absolute error, more robust to outliers. Lower is better. |
| **R²** | `1 - SS_res / SS_tot` | Share of variance explained. 1.0 is perfect, 0.0 matches the mean, below 0 is worse than the mean. |
| **QLIKE** | `mean(sigma^2 / h^2 + log(h^2))` | Penalises underestimation asymmetrically. Lower is better. |
| **Dir. Acc** | `mean(sign(dy) == sign(dy_pred))` | Share of days where the direction of change is correct. 0.5 is random, above 0.5 is informative. |

**Why QLIKE?** In risk management, underestimating volatility (believing the market is calm when it is not) is far more dangerous than overestimating it. QLIKE penalises that scenario asymmetrically, unlike RMSE and MAE which are symmetric.

---

## 8. Generated plots

### `artifacts/plots/antrenare/`
- `lstm_loss.png`, `attention_loss.png`: train and validation loss curves for the reference models
- `lstm_best_hpo_loss.png`, `attention_best_hpo_loss.png`: loss curves after HPO
- `attention_heatmap.png`: attention weight heatmap (60 days by 4 horizons)

### `artifacts/plots/comparatii/`
- `compare_rmse.png`: RMSE bar chart per model and horizon (h = 1, 5, 10, 20), best outlined in orange
- `compare_qlike.png`: QLIKE bar chart per model and horizon

### `artifacts/plots/predictii/`
- `pred_vs_true_{model}_{symbol}.png`: actual versus predicted volatility for the most frequent symbol in the test set, with a 21 day trend overlaid

### `artifacts/plots/stocuri/` (from step 6)
- `stock_{SYM}_h1_toate_modelele.png`: a separate subplot per model (actual versus predicted at h=1), each with its RMSE in the title, which is easier to compare than overlaid lines
- `stock_{SYM}_toate_orizonturile.png`: 4 subplots (h = 1, 5, 10, 20) for the best model by RMSE across the whole test set
- `stock_{SYM}_metrici.png`: RMSE, MAE, R² and QLIKE bar charts per model at h=1, best per metric outlined in orange

---

## 9. Running the pipeline

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib arch statsmodels
```

Raw NYSE data must be in `data_raw/NYSE_YYYY.csv`, one file per year.

### Full run (recommended)

```bash
# Step 1: preprocessing and feature engineering
python scripts/01_prepare_data.py

# Optional: export inspectable CSVs (features_all.csv, panel_silver.csv)
python scripts/01_prepare_data.py --export

# Step 2: GARCH baseline (tests 8 combinations, keeps the best)
python scripts/02_train_garch.py

# Step 3: neural networks with default hyperparameters (reference models)
python scripts/03_train_nn.py --epochs 30

# Step 4: HPO, find the optimal configuration and retrain fully
python scripts/04_run_hpo.py --model lstm      --trials 10 --epochs 15
python scripts/04_run_hpo.py --model attention --trials 10 --epochs 15

# Step 5: evaluate and compare all models on the test set
python scripts/05_evaluate_compare.py

# Step 6: detailed plots per stock
python scripts/06_select_and_plot.py --symbols JPM GS BAC
# or interactive selection:
python scripts/06_select_and_plot.py
```

### Quick run (no HPO)

```bash
python scripts/01_prepare_data.py
python scripts/02_train_garch.py
python scripts/03_train_nn.py --epochs 20 --model lstm
python scripts/05_evaluate_compare.py
python scripts/06_select_and_plot.py --symbols JPM GS
```

### Running on a subset of symbols

```bash
export SYMBOLS="JPM,GS,BAC,C"
python scripts/01_prepare_data.py
python scripts/03_train_nn.py
```

### Running on GPU (CUDA)

```bash
export DEVICE=cuda
python scripts/03_train_nn.py
python scripts/04_run_hpo.py --model lstm --trials 20 --epochs 20
```

### Running on Kaggle (2x T4 GPU)

```python
import os
os.environ["DATA_DIR"]      = "/kaggle/input/nyse-data"
os.environ["ARTIFACTS_DIR"] = "/kaggle/working/artifacts"
os.environ["DEVICE"]        = "cuda"
```

```bash
!python scripts/01_prepare_data.py --export
!python scripts/02_train_garch.py
!python scripts/03_train_nn.py --epochs 50
!python scripts/04_run_hpo.py --model lstm      --trials 15 --epochs 20 --retrain-epochs 50
!python scripts/04_run_hpo.py --model attention --trials 15 --epochs 20 --retrain-epochs 50
!python scripts/05_evaluate_compare.py
!python scripts/06_select_and_plot.py --symbols JPM GS BAC C AXP
```

---

## 10. Configuration

Central settings live in `nyse_vol/config.py`:

```python
HORIZONS         = [1, 5, 10, 20]    # prediction horizons (trading days)
WINDOW           = 60                 # input window length (days)
TARGET_ESTIMATOR = "garman_klass"     # estimator used as the target
TRAIN_END        = "2018-12-31"       # training cutoff
VAL_END          = "2021-12-31"       # validation cutoff; test is everything after
SEED             = 42
DEVICE           = "cpu"              # overridable through the DEVICE env var
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data_raw/` | Directory holding the NYSE CSV files |
| `ARTIFACTS_DIR` | `artifacts/` | Directory for generated artifacts |
| `SYMBOLS` | all 30 | Symbol list, comma separated |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

### Requirements

```
torch >= 2.0
numpy
pandas
scikit-learn
matplotlib
arch          # GARCH models
statsmodels   # stationarity tests (ADF, KPSS, PP)
```

---

## 11. Design notes

### Why log volatility as the target?

Raw volatility is log-normally distributed, and its rare but extreme spikes make MSE unstable. In log space the distribution is closer to normal, gradients are more stable and convergence is faster. At evaluation time `exp()` returns predictions to an interpretable scale.

### Why a 60 day window?

60 trading days (about 3 months) captures:

- Volatility autocorrelation, since the GARCH effect propagates over tens of days
- Intra-quarter seasonality, such as quarterly reporting effects
- Volatility regimes, which typically last weeks to months

### Why Garman-Klass as the target rather than close-to-close?

Garman-Klass uses all four OHLC prices and is **5 to 8 times more efficient** than close-to-close, meaning the same level of precision from roughly a fifth of the data. Close-to-close ignores all intraday movement and significantly understates volatility on large move days such as crises and earnings surprises.

### Behaviour of the naive model

The naive model predicts `vol(t+h) = vol_garman_klass(t)` for every h. In the plots it looks almost identical to the actual series, shifted by exactly one day. That is correct behaviour, not a bug: volatility has high autocorrelation at lag 1. If any model has a higher RMSE than the naive model, it adds no practical value.

### The visible lag in the LSTM models

The LSTM and attention models can show a small visible lag against the real series. This is a characteristic of sequence models, which smooth the signal, and it is largely unavoidable. What matters for evaluation is the global RMSE, not exact alignment with individual extreme spikes.
