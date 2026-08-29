"""Sequence models.

Two architectures, both taking a (batch, lookback, channels) window plus a
static machine-type index and returning a failure logit.

`TemporalFusionTransformer` is a compact implementation of the Lim et al.
architecture: gated residual networks, a variable selection network, an LSTM
encoder for local processing, static enrichment and interpretable multi-head
attention.  It is not a wrapper around a forecasting library -- it is here so
the model can hand the investigation agent two things no black box gives up:
which *channels* it weighted, and which *timesteps* it attended to.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
class GatedLinearUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(self.drop(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: skip connection with a learned gate, optionally context-conditioned."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 dropout: float = 0.1, d_context: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.glu = GatedLinearUnit(d_hidden, d_out, dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = self.fc2(F.elu(h))
        return self.norm(self.skip(x) + self.glu(h))


class BatchedVariableGRN(nn.Module):
    """One GRN per input channel, evaluated as batched einsums.

    Mathematically identical to a `ModuleList` of `n_vars` GRNs mapping a
    single scalar channel to `d_out`, but the per-channel Python loop is
    replaced by four batched matmuls.

    Measured on this dataset (24 channels, batch 256, 36 steps, CPU), forward
    plus backward for this block alone:

        per-channel ModuleList loop, hidden 48 :  5312 ms
        batched einsum,              hidden 48 :   516 ms   (10.3x)
        batched einsum,              hidden 10 :   192 ms   (27.6x)

    The narrower per-variable hidden width is the second lever: (B, T, V, H) is
    the widest tensor in the model, so H dominates. Together these take a TFT
    epoch from roughly 8 minutes to 45 seconds, which is what makes a 30-epoch
    run practical on CPU.
    """

    def __init__(self, n_vars: int, d_hidden: int, d_out: int,
                 dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        def param(*shape):
            p = nn.Parameter(torch.empty(*shape))
            nn.init.xavier_uniform_(p)
            return p

        self.w1 = param(n_vars, 1, d_hidden)
        self.b1 = nn.Parameter(torch.zeros(n_vars, d_hidden))
        self.w2 = param(n_vars, d_hidden, d_hidden)
        self.b2 = nn.Parameter(torch.zeros(n_vars, d_hidden))
        self.wg = param(n_vars, d_hidden, d_out * 2)
        self.bg = nn.Parameter(torch.zeros(n_vars, d_out * 2))
        self.wskip = param(n_vars, 1, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.unsqueeze(-1)                                    # (B,T,V,1)
        h = torch.einsum("btvi,vih->btvh", u, self.w1) + self.b1
        h = torch.einsum("btvh,vhg->btvg", F.elu(h), self.w2) + self.b2
        g = torch.einsum("btvh,vho->btvo", self.drop(h), self.wg) + self.bg
        a, b = g.chunk(2, dim=-1)
        skip = torch.einsum("btvi,vio->btvo", u, self.wskip)
        return self.norm(skip + a * torch.sigmoid(b))          # (B,T,V,D)


class VariableSelectionNetwork(nn.Module):
    """Learns a softmax weight per input channel, per timestep.

    The weights are the model's own answer to "which sensor mattered here",
    which is what the investigation agent reports as evidence.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1,
                 d_context: int | None = None, d_var_hidden: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        # The per-variable transform is the widest tensor in the model
        # ((B, T, V, H)); a narrower hidden width there costs little accuracy
        # and dominates the epoch time.
        self.per_var = BatchedVariableGRN(
            n_vars, d_var_hidden or max(d_model // 3, 8), d_model, dropout)
        self.selector = GatedResidualNetwork(
            n_vars, d_model, n_vars, dropout, d_context=d_context)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        # x: (B, T, V)
        weights = torch.softmax(self.selector(x, context), dim=-1)   # (B, T, V)
        transformed = self.per_var(x)                                 # (B,T,V,D)
        out = (transformed * weights.unsqueeze(-1)).sum(dim=-2)       # (B, T, D)
        return out, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Shared-value multi-head attention -- heads can be averaged into one
    attention map that is meaningful to read."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, self.d_head * n_heads)
        self.k = nn.Linear(d_model, self.d_head * n_heads)
        self.v = nn.Linear(d_model, self.d_head)      # shared across heads
        self.out = nn.Linear(self.d_head, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).unsqueeze(1)                                    # (B,1,T,dh)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)     # (B,H,T,T)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (self.drop(attn) @ v).mean(dim=1)                       # (B,T,dh)
        return self.out(ctx), attn.mean(dim=1)                        # (B,T,T)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int, n_static: int = 4, d_model: int = 64,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.static = nn.Embedding(n_static, 8)
        self.lstm = nn.LSTM(n_channels, d_model, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(d_model + 8, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor):
        h, _ = self.lstm(x)
        z = torch.cat([h[:, -1], self.static(static)], dim=-1)
        return self.head(z).squeeze(-1)

    def explain(self, x: torch.Tensor, static: torch.Tensor) -> dict:
        """LSTMs expose no native attribution; report occlusion deltas instead."""
        with torch.no_grad():
            base = torch.sigmoid(self(x, static))
            deltas = []
            for c in range(x.shape[-1]):
                xc = x.clone()
                xc[..., c] = 0.0
                deltas.append(base - torch.sigmoid(self(xc, static)))
            imp = torch.stack(deltas, dim=-1)
        return {"channel_importance": imp, "attention": None}


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_channels: int, n_static: int = 4, d_model: int = 32,
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.static_emb = nn.Embedding(n_static, d_model)
        self.static_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        self.vsn = VariableSelectionNetwork(n_channels, d_model, dropout,
                                            d_context=d_model)
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_after_lstm = GatedLinearUnit(d_model, d_model, dropout)
        self.norm_after_lstm = nn.LayerNorm(d_model)

        self.enrich = GatedResidualNetwork(d_model, d_model, d_model, dropout,
                                           d_context=d_model)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.gate_after_attn = GatedLinearUnit(d_model, d_model, dropout)
        self.norm_after_attn = nn.LayerNorm(d_model)
        self.position_wise = GatedResidualNetwork(d_model, d_model, d_model,
                                                  dropout)
        self.head = nn.Linear(d_model, 1)

    def _trunk(self, x: torch.Tensor, static: torch.Tensor):
        s = self.static_grn(self.static_emb(static))          # (B, D)
        sel, weights = self.vsn(x, s.unsqueeze(1).expand(-1, x.shape[1], -1))
        enc, _ = self.encoder(sel)
        enc = self.norm_after_lstm(self.gate_after_lstm(enc) + sel)

        enriched = self.enrich(enc, s.unsqueeze(1).expand(-1, x.shape[1], -1))
        att_out, att_w = self.attn(enriched)
        z = self.norm_after_attn(self.gate_after_attn(att_out) + enriched)
        z = self.position_wise(z)
        return z, weights, att_w

    def forward(self, x: torch.Tensor, static: torch.Tensor):
        z, _, _ = self._trunk(x, static)
        return self.head(z[:, -1]).squeeze(-1)

    def explain(self, x: torch.Tensor, static: torch.Tensor) -> dict:
        """Native interpretability: channel weights and the attention map."""
        with torch.no_grad():
            _, weights, att = self._trunk(x, static)
        return {
            # average the selection weights over the window
            "channel_importance": weights.mean(dim=1),   # (B, V)
            "channel_importance_by_step": weights,       # (B, T, V)
            "attention": att[:, -1, :],                  # (B, T) last-step focus
        }


class EnsembleModel(nn.Module):
    """Weighted probability average.  The weight is *fitted on validation*, not
    assumed -- see `ml/training.fit_ensemble_weight`."""

    def __init__(self, models: list[nn.Module], weights: list[float]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor, static: torch.Tensor):
        probs = torch.stack([torch.sigmoid(m(x, static)) for m in self.models],
                            dim=-1)
        p = (probs * self.w).sum(-1).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))      # back to logit space

    def explain(self, x: torch.Tensor, static: torch.Tensor) -> dict:
        for m in self.models:
            if isinstance(m, TemporalFusionTransformer):
                return m.explain(x, static)
        return self.models[0].explain(x, static)
