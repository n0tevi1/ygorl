"""Token encoders for the arrays of docs/encoding.md (T4b.1): cards, globals, actions, events.

Normalisation belongs here (the encoders emit raw integers): categorical columns (enums, small counts,
bit indices) are embedded, bit masks are expanded to bits, and large magnitudes (attack / defense, LP,
event values) enter as scaled numbers. Each token is the *sum* of per-part projections, which is the
same as concatenating the parts and applying one linear layer (the ``⊕`` of design 06).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ygorl.cards.cdb import CardVocab
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.events import EVENT_TYPES
from ygorl.nets.config import NetConfig
from ygorl.nets.gemm import tn_mm
from ygorl.nets.text import EFFECT_SLOTS, TextFeatures

LOG_CLAMP = math.log1p(65535)  # the encoders clamp magnitudes to [0, 65535]
CHAINING = EVENT_TYPES.index("chaining") + 1
SYSTEM_STRING_BUCKETS = 2048  # system strings (descriptions below 1 << 20) are hashed into buckets


def bits(x: Tensor, n: int) -> Tensor:
    """``[...]`` int -> ``[..., n]`` float: the low ``n`` bits."""
    return ((x.unsqueeze(-1) >> torch.arange(n, device=x.device)) & 1).float()


def log_scale(x: Tensor) -> Tensor:
    return torch.log1p(x.clamp(min=0).float()) / LOG_CLAMP


SMALL_TABLE = 1024  # tables up to this many rows take the multi-hot matmul backward of _SummedRows


class _SummedRows(torch.autograd.Function):
    """``weight[idx].sum(1)`` for ``idx`` ``[M, C]`` over a small table.

    Forward: ``F.embedding_bag`` (no ``[M, C, d]`` intermediate). Backward: ``counts.T @ grad`` with the multi-hot
    count matrix ``[M, rows]``; the default embedding backward expands the gradient to ``[M, C, d]`` and sorts
    ``M * C`` indices, several times slower on CPU. Same values up to float summation order.
    """

    @staticmethod
    def forward(ctx, idx: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(idx)
        ctx.rows = weight.shape[0]
        return nn.functional.embedding_bag(idx, weight, mode="sum")

    @staticmethod
    def backward(ctx, grad: Tensor):
        (idx,) = ctx.saved_tensors
        counts = grad.new_zeros(idx.shape[0], ctx.rows).scatter_add_(1, idx, grad.new_ones(idx.shape))
        return None, tn_mm(counts, grad)


class CategoricalEmbedding(nn.Module):
    """Sum of one embedding per column; column ``i`` is clamped to ``[0, sizes[i])``."""

    def __init__(self, sizes: Sequence[int], d: int) -> None:
        super().__init__()
        self.register_buffer("offsets", torch.tensor([0, *sizes[:-1]]).cumsum(0), persistent=False)
        self.register_buffer("limits", torch.tensor(sizes) - 1, persistent=False)
        self.table = nn.Embedding(sum(sizes), d)
        nn.init.normal_(self.table.weight, std=len(sizes) ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        idx = torch.minimum(x.clamp(min=0), self.limits) + self.offsets
        flat = idx.reshape(-1, idx.shape[-1])
        weight = self.table.weight
        if weight.shape[0] <= SMALL_TABLE:
            out = _SummedRows.apply(flat, weight)
        else:  # e.g. the action encoder's system-string buckets: a count matrix would be too large
            out = nn.functional.embedding_bag(flat, weight, mode="sum")
        return out.view(*idx.shape[:-1], weight.shape[1])


class CardIdentity(nn.Module):
    """Vocab index -> ``[d]``: known / hidden / none status ⊕ ID embedding ⊕ card text ⊕ mean effect text.

    Every identity source except the status is optional (``NetConfig.id_embedding``, text widths); with
    all of them off a card is described by its structured columns only.
    """

    def __init__(self, cfg: NetConfig, text: TextFeatures | None = None) -> None:
        super().__init__()
        self.vocab_size = cfg.vocab_size
        d = cfg.d_model
        self.status = nn.Embedding(3, d)  # 0 none / padding, 1 hidden (unknown), 2 a known card
        self.id_embedding = nn.Embedding(cfg.vocab_size, d) if cfg.id_embedding else None
        if self.id_embedding is not None:
            nn.init.normal_(self.id_embedding.weight, std=0.02)
        self.card_text_proj = None
        if cfg.card_text_dim:
            table = _table(text, "card", cfg.card_text_dim, cfg.vocab_size)
            self.register_buffer("card_text", torch.as_tensor(table).float(), persistent=False)
            self.card_text_proj = nn.Linear(cfg.card_text_dim, d, bias=False)
        self.effect_text_proj = None
        if cfg.effect_text_dim:
            _table(text, "effect", cfg.effect_text_dim, cfg.vocab_size)
            self.register_buffer("effect_mean", torch.as_tensor(text.card_effect_mean()).float(), persistent=False)
            self.effect_text_proj = nn.Linear(cfg.effect_text_dim, d, bias=False)

    def forward(self, index: Tensor) -> Tensor:
        index = index.clamp(0, self.vocab_size - 1)
        out = self.status(index.clamp(max=CardVocab.FIRST_INDEX))
        if self.id_embedding is not None:
            out = out + self.id_embedding(index)
        if self.card_text_proj is not None:
            out = out + self.card_text_proj(self.card_text[index])
        if self.effect_text_proj is not None:
            out = out + self.effect_text_proj(self.effect_mean[index])
        return out


class EffectText(nn.Module):
    """``(effect card vocab index, string index + 1)`` -> ``[d]`` from the frozen effect-text table."""

    def __init__(self, cfg: NetConfig, text: TextFeatures | None) -> None:
        super().__init__()
        _table(text, "effect", cfg.effect_text_dim, cfg.vocab_size)
        self.vocab_size = cfg.vocab_size
        self.register_buffer("vectors", torch.as_tensor(text.effect).float(), persistent=False)
        self.register_buffer("lookup", torch.as_tensor(text.effect_lookup).long(), persistent=False)
        self.proj = nn.Linear(cfg.effect_text_dim, cfg.d_model, bias=False)

    def forward(self, card: Tensor, slot: Tensor) -> Tensor:
        row = self.lookup[card.clamp(0, self.vocab_size - 1), slot.clamp(0, EFFECT_SLOTS - 1)]
        return self.proj(self.vectors[row])


def _table(text: TextFeatures | None, kind: str, width: int, vocab_size: int):
    table = None if text is None else (text.card if kind == "card" else text.effect)
    if table is None:
        raise ValueError(f"config enables {kind} text (width {width}) but no {kind}-text table was supplied")
    if table.shape[1] != width:
        raise ValueError(f"{kind}-text table has width {table.shape[1]}, config says {width}")
    if text.vocab_size != vocab_size:
        raise ValueError(f"text tables cover {text.vocab_size} vocab entries, config says {vocab_size}")
    return table


class CardEncoder(nn.Module):
    """Card-table rows ``[..., 23]`` -> ``[..., d]`` (structured columns ⊕ identity of column 0)."""

    # card columns 1..8, 10..16, 20..22 (see docs/encoding.md): categorical sizes
    CATEGORICAL = {1: 9, 2: 64, 3: 16, 4: 2, 5: 2, 6: 5, 7: 2, 8: 2, 10: 8, 11: 64, 12: 16, 13: 16, 14: 16, 15: 16,
                   16: 16, 20: 16, 21: 16, 22: 2}  # fmt: skip
    N_NUMERIC = 32 + 9 + 4 + 1  # type bits, link-marker bits, attack / defense (linear + log), counters

    def __init__(self, cfg: NetConfig, identity: CardIdentity) -> None:
        super().__init__()
        self.identity = identity
        self.columns = list(self.CATEGORICAL)
        self.categorical = CategoricalEmbedding(list(self.CATEGORICAL.values()), cfg.d_model)
        self.numeric = nn.Linear(self.N_NUMERIC, cfg.d_model)

    def forward(self, cards: Tensor) -> Tensor:
        atk_def = cards[..., 17:19].float()
        numeric = torch.cat([bits(cards[..., 9], 32), bits(cards[..., 19], 9), atk_def / 5000, log_scale(atk_def),
                             log_scale(cards[..., 20:21])], -1)  # fmt: skip
        return self.identity(cards[..., 0]) + self.categorical(cards[..., self.columns]) + self.numeric(numeric)


class GlobalEncoder(nn.Module):
    """Global vector ``[..., 22]`` -> ``[..., d]``."""

    # viewer, is_first, is_my_turn, turn, phase, chain, decision, substep, augmented_start
    CATEGORICAL = {0: 2, 1: 2, 2: 2, 3: 32, 4: 11, 17: 8, 18: 256, 19: 16, 21: 2}
    N_NUMERIC = 1 + 2 + 10 + 1 + 1 + 1  # turn, LP, zone counts, chain, substep, number of actions

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.columns = list(self.CATEGORICAL)
        self.categorical = CategoricalEmbedding(list(self.CATEGORICAL.values()), cfg.d_model)
        self.numeric = nn.Linear(self.N_NUMERIC, cfg.d_model)

    def forward(self, glob: Tensor) -> Tensor:
        g = glob.float()
        numeric = torch.cat([g[..., 3:4] / 20, g[..., 5:7] / 8000, g[..., 7:17] / 40, g[..., 17:18] / 4,
                             g[..., 19:20] / 8, log_scale(glob[..., 20:21])], -1)  # fmt: skip
        return self.categorical(glob[..., self.columns]) + self.numeric(numeric)


class ActionEncoder(nn.Module):
    """Action rows ``[B, A, 10]`` + contextual card tokens ``[B, N, d]`` -> action embeddings ``[B, A, d]``.

    Column 1 (card-table row + 1) gathers the board Transformer's output for that card, so an action on a
    card sees the card in context; column 2 adds the identity of the card itself (also for declared cards
    that are not on the table), columns 3/4 the frozen effect-text vector of the chosen effect.
    """

    # kind, effect_index, system_string (bucketed), position, zone, value, index
    SIZES = (len(ACTION_KINDS) + 1, EFFECT_SLOTS, SYSTEM_STRING_BUCKETS, 5, 33, 64, 256)

    def __init__(self, cfg: NetConfig, identity: CardIdentity, effect: EffectText | None) -> None:
        super().__init__()
        d = cfg.d_model
        self.identity = identity
        self.effect = effect
        self.categorical = CategoricalEmbedding(list(self.SIZES), d)
        self.numeric = nn.Linear(1, d)
        self.card_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.ReLU(), nn.Linear(cfg.ff_mult * d, d))
        self.out_norm = nn.LayerNorm(d)

    def forward(self, actions: Tensor, cards: Tensor) -> Tensor:
        cat = torch.stack([actions[..., 0], actions[..., 4], actions[..., 5] % SYSTEM_STRING_BUCKETS, actions[..., 6],
                           actions[..., 7], actions[..., 8], actions[..., 9]], -1)  # fmt: skip
        x = self.categorical(cat) + self.numeric(log_scale(actions[..., 8:9])) + self.identity(actions[..., 2])
        if self.effect is not None:
            x = x + self.effect(actions[..., 3], actions[..., 4])
        row = actions[..., 1]
        idx = (row - 1).clamp(0, cards.shape[1] - 1)
        gathered = torch.gather(cards, 1, idx.unsqueeze(-1).expand(-1, -1, cards.shape[-1]))
        x = x + self.card_proj(gathered) * (row > 0).unsqueeze(-1)
        x = x + self.ff(self.norm(x))
        return self.out_norm(x)


class EventEmbedding(nn.Module):
    """Event tokens ``[..., 20]`` -> ``[..., d]`` (docs/encoding.md 「事件 token 流」)."""

    # type, player, from (controller, location, sequence, position), to (same), value1..3 (clamped), turn, phase,
    # my_turn
    CATEGORICAL = {0: len(EVENT_TYPES) + 1, 1: 3, 4: 3, 5: 9, 6: 64, 7: 7, 8: 3, 9: 9, 10: 64, 11: 7, 12: 64, 13: 64,
                   14: 64, 15: 32, 16: 11, 17: 2}  # fmt: skip
    N_NUMERIC = 3 + 32 + 1 + 2  # values (log), value1 bits (reason / trigger masks), turn, LP

    def __init__(self, cfg: NetConfig, identity: CardIdentity, effect: EffectText | None = None) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.identity = identity
        self.effect = effect
        self.columns = list(self.CATEGORICAL)
        self.categorical = CategoricalEmbedding(list(self.CATEGORICAL.values()), d)
        self.numeric = nn.Linear(self.N_NUMERIC, d)
        self.card2_proj = nn.Linear(d, d, bias=False)  # second card (target / effect owner) has its own role

    def forward(self, events: Tensor) -> Tensor:
        e = events.float()
        numeric = torch.cat([log_scale(events[..., 12:15]), bits(events[..., 12], 32), e[..., 15:16] / 20,
                             e[..., 18:20] / 8000], -1)  # fmt: skip
        x = self.categorical(events[..., self.columns]) + self.numeric(numeric)
        x = x + self.identity(events[..., 2]) + self.card2_proj(self.identity(events[..., 3]))
        if self.effect is not None:  # chaining: card2 = effect-string owner, value2 = string index + 1
            chaining = (events[..., 0] == CHAINING).unsqueeze(-1)
            x = x + self.effect(events[..., 3], events[..., 13]) * chaining
        return x


class BoardEncoder(nn.Module):
    """Global token (+ history token) + card tokens -> Transformer -> contextual tokens.

    Padding card rows (``card_index == 0``) are masked out of attention. Returns ``(global_out, cards_out)``.
    """

    def __init__(self, cfg: NetConfig, identity: CardIdentity, with_history: bool) -> None:
        super().__init__()
        d = cfg.d_model
        self.cards = CardEncoder(cfg, identity)
        self.globals = GlobalEncoder(cfg)
        self.history_proj = nn.Linear(d, d) if with_history else None
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff_mult * d, cfg.dropout, batch_first=True,
                                           norm_first=True)  # fmt: skip
        self.transformer = nn.TransformerEncoder(layer, cfg.board_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, cards: Tensor, glob: Tensor, history: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        card_mask = cards[..., 0] > 0
        tokens = [self.globals(glob).unsqueeze(1)]
        if self.history_proj is not None:
            tokens.append(self.history_proj(history).unsqueeze(1))
        n_special = len(tokens)
        x = torch.cat([*tokens, self.cards(cards)], 1)
        keep = torch.cat([card_mask.new_ones(card_mask.shape[0], n_special), card_mask], 1)
        pad = torch.zeros(keep.shape, dtype=x.dtype, device=x.device).masked_fill_(~keep, float("-inf"))[:, None, None]
        for layer in self.transformer.layers:  # pre-norm layers, same math as nn.TransformerEncoderLayer
            x = x + _self_attention(layer, layer.norm1(x), pad)
            x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        x = self.norm(x)
        return x[:, 0], x[:, n_special:], card_mask


def _self_attention(layer: nn.TransformerEncoderLayer, x: Tensor, pad: Tensor) -> Tensor:
    """``layer.self_attn(x, x, x, key_padding_mask)`` (batch first) as one packed projection + SDPA.

    Same operations as the training path of ``F.multi_head_attention_forward``, without its packed-projection
    copies (their backward zero-fills a ``[3, N, B, d]`` buffer per projection); forward outputs are bitwise equal
    to it and eval mode no longer switches to the fused inference kernel.
    """
    attn = layer.self_attn
    b, n, d = x.shape
    heads = attn.num_heads
    qkv = nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias).view(b, n, 3, heads, d // heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    y = nn.functional.scaled_dot_product_attention(q, k, v, pad, attn.dropout if attn.training else 0.0)
    y = nn.functional.linear(y.transpose(1, 2).reshape(b, n, d), attn.out_proj.weight, attn.out_proj.bias)
    return layer.dropout1(y)
