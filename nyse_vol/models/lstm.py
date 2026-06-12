"""Model LSTM baseline (multi-output) pentru predictia volatilitatii.

Arhitectura: LSTM stivuit (optional bidirectional) peste fereastra de intrare,
urmat de un cap dens care produce un vector cu o valoare per orizont. Reproduce
elementele predate la Cursurile 9-10 (RNN, Stacked RNN, Bidirectional RNN).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nyse_vol.config import ModelConfig

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU}


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        act = _ACT[cfg.head_activation]
        self.head = nn.Sequential(
            nn.Linear(out_dim, cfg.hidden_size),
            act(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # ultima stare ascunsa pe lungimea secventei
        return self.head(last)
