"""Encoder-Decoder LSTM cu mecanism de atentie (Cursul 11).

Encoderul (optional bidirectional) consuma fereastra de intrare si produce toate
starile ascunse. Decoderul genereaza secventa de volatilitati prezise (cate una
per orizont din config.HORIZONS), folosind la fiecare pas un vector de context
calculat prin atentie peste starile encoderului.

Sunt implementate ambele forme de scor de aliniere predate:
- atentie aditiva (Bahdanau): v^T tanh(W1 s + W2 h)
- atentie multiplicativa (Luong, "general"): s^T W h
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nyse_vol.config import ModelConfig


class _Attention(nn.Module):
    def __init__(self, dec_dim: int, enc_dim: int, kind: str = "luong"):
        super().__init__()
        self.kind = kind
        if kind == "bahdanau":
            hidden = dec_dim
            self.W1 = nn.Linear(dec_dim, hidden, bias=False)
            self.W2 = nn.Linear(enc_dim, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)
        elif kind == "luong":
            self.W = nn.Linear(enc_dim, dec_dim, bias=False)
        else:
            raise ValueError(f"Tip de atentie necunoscut: {kind!r}")

    def forward(self, dec_state: torch.Tensor, enc_outputs: torch.Tensor):
        # dec_state: (batch, dec_dim); enc_outputs: (batch, T, enc_dim)
        if self.kind == "bahdanau":
            scores = self.v(torch.tanh(
                self.W1(dec_state).unsqueeze(1) + self.W2(enc_outputs)
            )).squeeze(-1)                       # (batch, T)
        else:  # luong general
            proj = self.W(enc_outputs)           # (batch, T, dec_dim)
            scores = torch.bmm(proj, dec_state.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(scores, dim=1)       # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.n_outputs = n_outputs

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        enc_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        dec_dim = cfg.hidden_size

        self.bridge = nn.Linear(enc_dim, dec_dim)        # comprima starea encoderului
        self.attn = _Attention(dec_dim, enc_dim, cfg.attention)
        # intrarea decoderului: predictia anterioara (1) + context (enc_dim)
        self.dec_cell = nn.LSTMCell(1 + enc_dim, dec_dim)
        self.out = nn.Linear(dec_dim + enc_dim, 1)
        self.return_attention = False

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        enc_outputs, (h_n, c_n) = self.encoder(x)        # enc_outputs: (B,T,enc_dim)

        # stare initiala decoder din ultimul strat encoder (concat directii daca bidir)
        if self.cfg.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            c_last = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        else:
            h_last, c_last = h_n[-1], c_n[-1]
        h = torch.tanh(self.bridge(h_last))
        c = torch.tanh(self.bridge(c_last))

        prev = torch.zeros(batch, 1, device=x.device)    # autoregresiv: pornim de la 0
        preds, attns = [], []
        for _ in range(self.n_outputs):
            context, weights = self.attn(h, enc_outputs)
            h, c = self.dec_cell(torch.cat([prev, context], dim=-1), (h, c))
            y = self.out(torch.cat([h, context], dim=-1))   # (batch, 1)
            preds.append(y)
            attns.append(weights)
            prev = y
        out = torch.cat(preds, dim=1)                    # (batch, n_outputs)
        if self.return_attention:
            return out, torch.stack(attns, dim=1)        # (batch, n_outputs, T)
        return out
