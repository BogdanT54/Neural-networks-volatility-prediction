"""Configurare centrala a proiectului de predictie a volatilitatii NYSE.

Toate caile, listele de simboluri, orizonturile de predictie si spatiile de
cautare a hiperparametrilor sunt definite aici, astfel incat scripturile sa fie
reproductibile si usor de ajustat fara modificari in cod.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Cai
# --------------------------------------------------------------------------- #
# Radacina proiectului (directorul care contine pachetul nyse_vol/).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directorul cu datele brute NYSE (zip-uri an cu an: NYSE_2001.zip ...).
# Poate fi suprascris din variabila de mediu DATA_DIR.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data_raw"))

# Directorul cu artefacte generate (date procesate, modele, grafice, metrici).
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR    = ARTIFACTS_DIR / "models"
METRICS_DIR   = ARTIFACTS_DIR / "metrics"

# Subdirectoare plots organizate pe tip de continut
PLOTS_DIR           = ARTIFACTS_DIR / "plots"
PLOTS_ANTRENARE_DIR = PLOTS_DIR / "antrenare"    # loss curves, heatmap atentie
PLOTS_COMPARATII_DIR = PLOTS_DIR / "comparatii"  # RMSE/QLIKE bar charts globale
PLOTS_PREDICTII_DIR = PLOTS_DIR / "predictii"    # pred vs real per model+simbol
PLOTS_STOCURI_DIR   = PLOTS_DIR / "stocuri"      # grafice detaliate per stock ales

for _d in (ARTIFACTS_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR,
           PLOTS_DIR, PLOTS_ANTRENARE_DIR, PLOTS_COMPARATII_DIR,
           PLOTS_PREDICTII_DIR, PLOTS_STOCURI_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Universul de simboluri (cos de actiuni lichide)
# --------------------------------------------------------------------------- #
# Simboluri lichide, prezente pe NYSE inca din 2001, folosite pentru modelul
# global. Poate fi suprascris din variabila de mediu SYMBOLS (virgula-separat).
# Exemplu: export SYMBOLS="JPM,GS,BAC,C,AXP"
_ALL_SYMBOLS = [
    "AA", "ABT", "AEE", "AEP", "AES", "AFL", "AIG", "AXP", "BA", "BAC",
    "C", "CAT", "CL", "CVX", "DD", "DIS", "DOW", "DUK", "EMR", "F",
    "GE", "GS", "HD", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MO",
]
_symbols_env = os.environ.get("SYMBOLS", "").strip()
SYMBOLS = (
    [s.strip().upper() for s in _symbols_env.split(",") if s.strip()]
    if _symbols_env else _ALL_SYMBOLS
)

# --------------------------------------------------------------------------- #
# Parametrii seriei de timp
# --------------------------------------------------------------------------- #
# Estimatorul de volatilitate folosit ca tinta principala.
# Optiuni: "parkinson", "garman_klass", "rogers_satchell", "yang_zhang", "close_to_close".
TARGET_ESTIMATOR = "garman_klass"

# Orizonturi de predictie (zile in viitor). Tinta este volatilitatea realizata
# medie pe urmatoarele h zile.
HORIZONS = [1, 5, 10, 20]

# Lungimea ferestrei glisante de intrare (numar de zile istorice).
WINDOW = 60

# Numar minim de observatii valide pe simbol pentru a fi inclus.
MIN_OBS_PER_SYMBOL = 250

# Numar maxim de zile consecutive lipsa pentru care se aplica interpolare liniara.
# Goluri mai lungi (suspendare reala, lipsa date) → randul ramane NaN si e eliminat.
# Interpretare: 2 zile inapoi + 2 zile in fata = goluri de max 2 zile consecutive.
MAX_INTERP_DAYS = 2

# Numar de zile de tranzactionare intr-un an (pentru anualizare optionala).
TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Split cronologic (evita scurgerea de informatie)
# --------------------------------------------------------------------------- #
TRAIN_END = "2018-12-31"   # train: <= aceasta data
VAL_END = "2021-12-31"     # validare: (TRAIN_END, VAL_END]
# test: > VAL_END

# --------------------------------------------------------------------------- #
# Antrenare implicita (modele NN)
# --------------------------------------------------------------------------- #
SEED = 42
DEVICE = os.environ.get("DEVICE", "cpu")


@dataclass
class TrainConfig:
    """Hiperparametri de antrenare folositi de trainer si ca implicit pentru NN."""

    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    optimizer: str = "adam"          # adam | adamw | rmsprop | sgd
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    early_stop_patience: int = 6
    lr_scheduler: bool = True


@dataclass
class ModelConfig:
    """Hiperparametri de arhitectura pentru modelele recurente."""

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    head_activation: str = "tanh"    # tanh | relu
    attention: str = "luong"         # bahdanau (aditiv) | luong (multiplicativ)


# --------------------------------------------------------------------------- #
# Spatiile de cautare a hiperparametrilor (random search)
# --------------------------------------------------------------------------- #
@dataclass
class SearchSpace:
    optimizer: list = field(default_factory=lambda: ["adam", "adamw", "rmsprop", "sgd"])
    lr: list = field(default_factory=lambda: [3e-4, 1e-3, 3e-3])
    hidden_size: list = field(default_factory=lambda: [32, 64, 128])
    num_layers: list = field(default_factory=lambda: [1, 2, 3])
    dropout: list = field(default_factory=lambda: [0.0, 0.2, 0.4])
    bidirectional: list = field(default_factory=lambda: [False, True])
    head_activation: list = field(default_factory=lambda: ["tanh", "relu"])
    weight_decay: list = field(default_factory=lambda: [0.0, 1e-5, 1e-4])
    attention: list = field(default_factory=lambda: ["bahdanau", "luong"])


SEARCH_SPACE = SearchSpace()

# Ordinele GARCH explorate pentru baseline-ul clasic.
GARCH_ORDERS = [(1, 1), (1, 2), (2, 1), (2, 2)]
GARCH_DISTS = ["normal", "t"]
