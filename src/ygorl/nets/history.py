"""History modules over the event token stream (T4b.2, design I6).

Two interchangeable backbones behind one interface, :class:`HistoryEncoder`::

    summary, state, tokens = module(events, event_mask, state=None)

* ``events`` ``[B, T, 20]`` / ``event_mask`` ``[B, T]``: event tokens in time order, valid rows first
  (exactly the ``events`` / ``event_mask`` arrays of docs/encoding.md).
* ``tokens`` ``[B, T, d]``: causal per-token outputs (row ``t`` depends on rows ``<= t`` and the state only),
  for next-token / auxiliary heads.
* ``summary`` ``[B, d]``: the output at the latest valid token (the state's summary when the chunk has
  none; a fixed vector for an empty history).
* ``state``: what the module carries between chunks (detached). ``state=None`` starts a new history.

Two ways to use it (docs/nets.md):

* **window mode** (training default): each decision's observation already holds the last ``L`` tokens, so
  run every observation with ``state=None``. Samples are independent: PPO minibatches need no BPTT.
* **streaming mode**: feed only the tokens that are new since the previous call, threading ``state``.
  Outputs equal the window-mode outputs over the concatenated stream (exactly for the LSTM; for the
  Transformer while the stream fits in ``history_mem_len``).

:class:`TransformerHistory` is a GTrXL (Parisotto et al. 2020): pre-norm blocks whose residual connections
are GRU-type gates biased towards the identity, TrXL-style per-layer memory of past hidden states, and
ALiBi relative position biases (so outputs do not depend on where a token sits in the window).
:class:`LSTMHistory` is the ablation baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ygorl.nets.config import NetConfig
from ygorl.nets.encoders import EventEmbedding


class HistoryOutput(NamedTuple):
    summary: Tensor  # [B, d]
    state: HistoryState
    tokens: Tensor  # [B, T, d]


@dataclass
class HistoryState:
    count: Tensor  # [B] tokens consumed so far
    last: Tensor  # [B, d] top-layer output at the latest token (before the final norm)

    def tensors(self) -> list[Tensor]:
        out = []
        for f in fields(self):
            v = getattr(self, f.name)
            out.extend(v if isinstance(v, tuple | list) else [v])
        return out

    def index(self, rows: Tensor) -> HistoryState:
        """The state of a subset of the batch (e.g. environments that did not reset)."""
        raise NotImplementedError


@dataclass
class TransformerState(HistoryState):
    memory: tuple[Tensor, ...]  # per layer [B, M, d]: the layer's input at past tokens
    valid: Tensor  # [B, M] bool
    pos: Tensor  # [B, M] absolute token positions

    def index(self, rows: Tensor) -> TransformerState:
        return TransformerState(self.count[rows], self.last[rows], tuple(m[rows] for m in self.memory),
                                self.valid[rows], self.pos[rows])  # fmt: skip


@dataclass
class LSTMState(HistoryState):
    h: Tensor  # [layers, B, d]
    c: Tensor  # [layers, B, d]

    def index(self, rows: Tensor) -> LSTMState:
        return LSTMState(self.count[rows], self.last[rows], self.h[:, rows], self.c[:, rows])


class HistoryEncoder(nn.Module):
    """Common front (event embedding) and back (final norm, summary) of the history backbones."""

    def __init__(self, cfg: NetConfig, embed: EventEmbedding) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.embed = embed
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, events: Tensor, event_mask: Tensor, state: HistoryState | None = None) -> HistoryOutput:
        mask = event_mask.bool()
        raw, last, new_state = self._run(self.embed(events), mask, state)
        return HistoryOutput(self.norm(last), new_state, self.norm(raw))

    def _run(self, x: Tensor, mask: Tensor, state: HistoryState | None) -> tuple[Tensor, Tensor, HistoryState]:
        """-> (top-layer outputs ``[B, T, d]``, output at the latest token ``[B, d]``, detached new state)."""
        raise NotImplementedError

    @staticmethod
    def _last(raw: Tensor, mask: Tensor, previous: Tensor) -> Tensor:
        """Output at the last valid token of each row, ``previous`` where the chunk has none."""
        n = mask.sum(1)
        if raw.shape[1] == 0:
            return previous
        idx = (n - 1).clamp(min=0)
        at = raw[torch.arange(raw.shape[0], device=raw.device), idx]
        return torch.where((n > 0).unsqueeze(-1), at, previous)


class GRUGate(nn.Module):
    """GTrXL gating layer: ``g(x, y) = (1 - z) * x + z * tanh(W_g y + U_g (r * x))``, ``z`` biased shut."""

    def __init__(self, d: int, bias: float) -> None:
        super().__init__()
        self.wy = nn.Linear(d, 3 * d)  # W_r, W_z, W_g
        self.ux = nn.Linear(d, 2 * d, bias=False)  # U_r, U_z
        self.ug = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.wy.bias.zero_()
            self.wy.bias[d : 2 * d] = -bias

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        wr, wz, wg = self.wy(y).chunk(3, -1)
        ur, uz = self.ux(x).chunk(2, -1)
        r = torch.sigmoid(wr + ur)
        z = torch.sigmoid(wz + uz)
        h = torch.tanh(wg + self.ug(r * x))
        return (1 - z) * x + z * h


class GTrXLBlock(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.heads = cfg.n_heads
        self.norm1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.gate1 = GRUGate(d, cfg.gate_bias)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.ReLU(), nn.Linear(cfg.ff_mult * d, d))
        self.gate2 = GRUGate(d, cfg.gate_bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: Tensor, memory: Tensor | None, bias: Tensor) -> Tensor:
        b, t, d = h.shape
        z = self.norm1(h if memory is None else torch.cat([memory, h], 1))  # LN([StopGrad(M), E])
        q = self.q(z[:, -t:]).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = (x.view(b, -1, self.heads, d // self.heads).transpose(1, 2) for x in self.kv(z).chunk(2, -1))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(b, t, d)
        h = self.gate1(h, self.drop(F.relu(self.out(y))))
        return self.gate2(h, self.drop(F.relu(self.ff(self.norm2(h)))))


class TransformerHistory(HistoryEncoder):
    """Causal GTrXL over event tokens (see module docstring)."""

    def __init__(self, cfg: NetConfig, embed: EventEmbedding) -> None:
        super().__init__(cfg, embed)
        self.blocks = nn.ModuleList(GTrXLBlock(cfg) for _ in range(cfg.history_layers))
        self.mem_len = cfg.history_mem_len
        h = cfg.n_heads
        self.register_buffer("slopes", torch.tensor([2 ** (-8 * (i + 1) / h) for i in range(h)]), persistent=False)

    def initial_state(self, batch: int, device: torch.device | None = None) -> TransformerState:
        empty = torch.zeros(batch, 0, self.d_model, device=device)
        return TransformerState(torch.zeros(batch, dtype=torch.long, device=device),
                                torch.zeros(batch, self.d_model, device=device), tuple(empty for _ in self.blocks),
                                torch.zeros(batch, 0, dtype=torch.bool, device=device),
                                torch.zeros(batch, 0, dtype=torch.long, device=device))  # fmt: skip

    def _run(self, x: Tensor, mask: Tensor, state: TransformerState | None):
        b, t, _ = x.shape
        if state is None:
            state = self.initial_state(b, x.device)
        if t == 0:
            return x, state.last, state
        m = state.valid.shape[1]
        pos = state.count.unsqueeze(1) + mask.long().cumsum(1) - 1  # absolute position of each valid token
        kpos = torch.cat([state.pos, pos], 1)
        kvalid = torch.cat([state.valid, mask], 1)
        qidx = torch.arange(t, device=x.device) + m
        kidx = torch.arange(m + t, device=x.device)
        causal = kidx[None, :] <= qidx[:, None]  # [T, S]
        self_key = kidx[None, :] == qidx[:, None]
        allowed = causal & (kvalid.unsqueeze(1) | self_key)  # [B, T, S]; padding queries see themselves
        dist = (pos.unsqueeze(2) - kpos.unsqueeze(1)).clamp(min=0).to(x.dtype)
        bias = -self.slopes.view(1, -1, 1, 1).to(x.dtype) * dist.unsqueeze(1)
        bias = bias.masked_fill(~allowed.unsqueeze(1), float("-inf"))

        h, inputs = x, []
        for i, block in enumerate(self.blocks):
            inputs.append(h)
            h = block(h, None if m == 0 else state.memory[i], bias)
        last = self._last(h, mask, state.last)
        return h, last, self._next_state(state, inputs, kvalid, kpos, mask, last)

    def _next_state(self, state, inputs, kvalid, kpos, mask, last) -> TransformerState:
        keep = min(kvalid.shape[1], self.mem_len)
        # keep the latest ``mem_len`` *valid* slots (padding of ragged chunks is dropped first)
        order = torch.where(kvalid, torch.arange(kvalid.shape[1], device=kvalid.device), -1)
        idx = order.argsort(dim=1, stable=True)[:, kvalid.shape[1] - keep :]
        memory = []
        for i, h in enumerate(inputs):
            full = h if state.memory[i].shape[1] == 0 else torch.cat([state.memory[i], h], 1)
            memory.append(torch.gather(full.detach(), 1, idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])))
        return TransformerState(state.count + mask.sum(1), last.detach(), tuple(memory), torch.gather(kvalid, 1, idx),
                                torch.gather(kpos, 1, idx))  # fmt: skip


class LSTMHistory(HistoryEncoder):
    """LSTM baseline over event tokens (valid rows must come first in each row of the chunk)."""

    def __init__(self, cfg: NetConfig, embed: EventEmbedding) -> None:
        super().__init__(cfg, embed)
        self.layers = cfg.history_layers
        self.lstm = nn.LSTM(cfg.d_model, cfg.d_model, cfg.history_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.history_layers > 1 else 0.0)  # fmt: skip

    def initial_state(self, batch: int, device: torch.device | None = None) -> LSTMState:
        z = torch.zeros(self.layers, batch, self.d_model, device=device)
        return LSTMState(torch.zeros(batch, dtype=torch.long, device=device), z[0], z, z.clone())

    def _run(self, x: Tensor, mask: Tensor, state: LSTMState | None):
        b, t, _ = x.shape
        if state is None:
            state = self.initial_state(b, x.device)
        n = mask.sum(1)
        if t == 0:
            return x, state.last, state
        packed = pack_padded_sequence(x, n.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed, (state.h, state.c))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
        empty = (n == 0).view(1, -1, 1)
        h = torch.where(empty, state.h, h)
        c = torch.where(empty, state.c, c)
        return out, h[-1], LSTMState(state.count + n, h[-1].detach(), h.detach(), c.detach())


def build_history(cfg: NetConfig, embed: EventEmbedding) -> HistoryEncoder | None:
    if cfg.history == "transformer":
        return TransformerHistory(cfg, embed)
    if cfg.history == "lstm":
        return LSTMHistory(cfg, embed)
    return None
