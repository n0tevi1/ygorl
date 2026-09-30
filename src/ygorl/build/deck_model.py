"""Masked deck model: learned add / remove candidates for deck evolution (#150, design 05 §5.3, docs/tuning.md
「掩码卡组模型」).

The hand-written candidate rules of #145 (engine members, addition pool, rule strata) have an exception for every
deck, so the candidates are learned instead: a set transformer over a deck list, trained by fill-in on the whole deck
history (every distinct masterduelmeta list whose cards all map, legal now or not).

- **Tokens**: one per distinct card of the list: frozen card-text vector (``tools/build_text_embeddings.py``) through
  a learned projection + a learned ID embedding (only for cards the training lists contain: an unseen card is its
  text) + a copy-count embedding. No positions: a deck list is a set.
- **Masking** (:meth:`DeckModel.masked_batch`): ``mask_rate`` of a list's cards (at least one) become ``[MASK]``
  tokens: the whole card (its token is replaced) or, for a card run in 2-3 copies with probability ``copy_rate``,
  some of its copies (its token keeps the rest, a ``[MASK]`` token is added). A ``[MASK]`` token predicts the card
  (a softmax over the vocab, logits = the token's output against the cards' own input embeddings, so a card's text
  places it even when it is rare) and, given the card, its masked copies (1-3).
- **Scores**: :meth:`DeckModel.removal_scores` (how out of place each card is: ``-log P(card | the rest)`` with one
  copy of it masked) and :meth:`DeckModel.addition_scores` (``log P(card | the deck)`` of a ``[MASK]`` added to the
  whole deck; a card the deck already runs means one more copy). :func:`ygorl.build.learned.learned_children` turns
  them into evolution children.

The data source is :func:`history_lists` (the raw history through the ``ygorl.data.masterduelmeta`` readers), kept
behind one function so it can switch to a dataset loader later. :func:`split_by_type` holds out whole deck types.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ygorl.cards.ydk import Deck

FORMAT = "ygorl-deck-model"
MAX_COPIES = 3
_CARD, _MASK, _PAD = 0, 1, 2  # token kinds


# ------------------------------------------------------------------ data


def history_lists(source: str | Path, cards: Mapping) -> list[tuple[str, Deck]]:
    """Every distinct deck list of the masterduelmeta history whose cards all map to passwords of ``cards``, as
    ``(deck type, deck)`` (side decks dropped), newest first; distinct by the card counts of main and Extra Deck.
    ``source`` is a deck dataset directory (``tools/build_deck_dataset.py``, ``ygorl.data.deck_dataset``; the Extra
    Deck cards are told apart by card kind) or else the raw history directory (``tools/build_deck_corpus.py
    --fetch``), parsed here."""
    src = Path(source)
    if (src / "decks.npz").is_file():
        from ygorl.data.deck_dataset import load_deck_dataset

        ds = load_deck_dataset(src)
        out = []
        for i in range(len(ds)):
            main: list[int] = []
            extra: list[int] = []
            for c, n in ds.cards(i).items():
                card = cards.get(c)
                (extra if card is not None and card.is_extra_deck else main).extend([c] * n)
            out.append((ds.type_name(i), Deck(main=tuple(main), extra=tuple(extra), name=ds.type_name(i))))
        return out
    from ygorl.data import masterduelmeta as mdm
    from ygorl.data.cardmap import CardMapper
    from ygorl.data.fetch import read_raw

    raw = src
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    records = mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, CardMapper(cards), "")
    out, seen = [], set()
    for r in sorted(records, key=lambda r: r.created, reverse=True):
        key = (tuple(sorted(r.main)), tuple(sorted(r.extra)))
        if r.unmapped or key in seen or not (r.main or r.extra):
            continue
        seen.add(key)
        out.append((r.type, mdm.to_deck(r, r.type)))
    return out


def held_out(deck_type: str, share: float, seed: int = 0) -> bool:
    """Whether ``deck_type`` is in the held-out ``share`` of types (a hash of the name: stable across runs)."""
    h = int(hashlib.sha1(f"{seed}:{deck_type}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def split_by_type(lists: Sequence[tuple[str, Deck]], share: float, seed: int = 0) -> tuple[list, list]:
    """(train, test): every list of a held-out type (:func:`held_out`) goes to test."""
    train, test = [], []
    for t, d in lists:
        (test if held_out(t, share, seed) else train).append((t, d))
    return train, test


def load_text_table(text_dir: str | Path, passwords: Sequence[int]) -> np.ndarray:
    """``card_text.npy`` rows of ``text_dir`` (``tools/build_text_embeddings.py``) aligned to ``passwords`` (a zero
    row for a card without text)."""
    d = Path(text_dir)
    vec = np.load(d / "card_text.npy")
    pws = np.load(d / "card_text_passwords.npy")
    row = {int(p): i for i, p in enumerate(pws)}
    out = np.zeros((len(passwords), vec.shape[1]), dtype=np.float32)
    for i, p in enumerate(passwords):
        j = row.get(int(p))
        if j is not None:
            out[i] = vec[j]
    return out


# ------------------------------------------------------------------ model


@dataclass
class DeckModelConfig:
    dim: int = 256
    layers: int = 3
    heads: int = 4
    ff: int = 512
    dropout: float = 0.1
    mask_rate: float = 0.15  # share of a list's distinct cards masked per training example (at least one)
    copy_rate: float = 0.3  # a masked card with 2-3 copies loses only some copies with this probability
    count_weight: float = 0.5  # weight of the copy-count loss


class DeckModel(nn.Module):
    """Set transformer over a deck list's distinct cards (module docstring). ``passwords`` is the vocab (the
    environment's card pool plus every card of the training lists), ``text`` its frozen text table."""

    def __init__(self, passwords: Sequence[int], text: np.ndarray | torch.Tensor, config: DeckModelConfig | None = None,
                 *, environment: Mapping | None = None) -> None:  # fmt: skip
        super().__init__()
        cfg = self.config = config or DeckModelConfig()
        self.environment = dict(environment or {})
        self.passwords = [int(p) for p in passwords]
        self.index = {p: i for i, p in enumerate(self.passwords)}
        v = len(self.passwords)
        text = torch.as_tensor(np.asarray(text, dtype=np.float32))
        self.register_buffer("text", text.half())  # frozen: saved with the model, never trained
        self.register_buffer("seen", torch.zeros(v, dtype=torch.bool))  # cards with a trained ID embedding
        self.text_proj = nn.Linear(text.shape[1], cfg.dim, bias=False)
        self.id_emb = nn.Embedding(v, cfg.dim)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.card_norm = nn.LayerNorm(cfg.dim)
        self.count_emb = nn.Embedding(MAX_COPIES + 1, cfg.dim)
        self.mask_emb = nn.Parameter(torch.zeros(cfg.dim))
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.query = nn.Linear(cfg.dim, cfg.dim)
        self.count_head = nn.Sequential(nn.Linear(2 * cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, MAX_COPIES))
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(cfg.dim)))

    # -- embeddings
    def cards(self) -> torch.Tensor:
        """[V, dim] input (and output) embedding of every vocab card."""
        ids = self.id_emb.weight * self.seen[:, None].to(self.id_emb.weight.dtype)
        return self.card_norm(self.text_proj(self.text.float()) + ids)

    def forward(self, idx: torch.Tensor, cnt: torch.Tensor, kind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens ``[B, L]``: vocab index, shown copies and kind (card / mask / pad). Returns (per-token outputs
        ``[B, L, dim]``, card embeddings ``[V, dim]``)."""
        emb = self.cards()
        card = emb[idx.clamp(min=0)] + self.count_emb(cnt.clamp(0, MAX_COPIES))
        x = torch.where((kind == _MASK)[..., None], self.mask_emb.expand_as(card), card)
        x = x * (kind != _PAD)[..., None]
        h = self.encoder(x, src_key_padding_mask=kind == _PAD)
        return self.out_norm(h), emb

    def heads(self, h: torch.Tensor, emb: torch.Tensor, card: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(card logits ``[..., V]``, copy-count logits ``[..., 3]`` given the card ``card``) of output tokens ``h``."""
        return self.query(h) @ emb.T * self.scale, self.count_head(torch.cat([h, emb[card]], -1))

    # -- training batches
    def encode(self, decks: Sequence[Deck]) -> tuple[torch.Tensor, torch.Tensor]:
        """(vocab index, copies) ``[N, L]`` of each deck's distinct cards (index -1 / copies 0 pad; cards outside the
        vocab dropped)."""
        rows = [[(self.index[c], min(n, MAX_COPIES)) for c, n in sorted(d.counts().items()) if c in self.index]
                for d in decks]  # fmt: skip
        width = max((len(r) for r in rows), default=0)
        idx = np.full((len(rows), width), -1, dtype=np.int64)
        cnt = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            if r:
                idx[i, : len(r)], cnt[i, : len(r)] = zip(*r, strict=True)
        return torch.from_numpy(idx), torch.from_numpy(cnt)

    def masked_batch(self, idx: torch.Tensor, cnt: torch.Tensor,
                     generator: torch.Generator | None = None) -> tuple[torch.Tensor, ...]:  # fmt: skip
        """A masked training batch of encoded lists: (idx, cnt, kind) of the list's ``L`` tokens followed by the
        ``[MASK]`` slots of masked copies (as many as the row with the most) and the targets of the ``[MASK]`` tokens (card index, copies
        masked; -1 elsewhere)."""
        cfg = self.config
        dev = idx.device
        valid = idx >= 0
        r = torch.rand(idx.shape, device=dev, generator=generator)
        chosen = (r < cfg.mask_rate) & valid
        # at least one masked card per list: the valid token with the lowest draw
        first = torch.where(valid, r, torch.full_like(r, 2.0)).argmin(1)
        chosen[torch.arange(len(idx), device=dev), first] |= valid.any(1)
        partial = chosen & (cnt >= 2) & (torch.rand(idx.shape, device=dev, generator=generator) < cfg.copy_rate)
        k = 1 + (torch.rand(idx.shape, device=dev, generator=generator) * (cnt - 1).clamp(min=1)).long()  # 1..cnt-1
        k = torch.where(partial, k.clamp(max=(cnt - 1).clamp(min=1)), cnt)
        whole = chosen & ~partial
        kind_a = torch.where(valid, torch.where(whole, _MASK, _CARD), _PAD)
        cnt_a = torch.where(partial, cnt - k, torch.where(whole, 0, cnt))
        kind_b = torch.where(partial, _MASK, _PAD)
        tgt_a = torch.where(whole, idx, -1)
        tgt_b = torch.where(partial, idx, -1)
        # the copy-mask slots: each row's first, trimmed to the most any row has
        order = torch.argsort((~partial).to(torch.int8), dim=1, stable=True)
        extra = max(int(partial.sum(1).max()), 1)
        order = order[:, :extra]
        pick = lambda t: t.gather(1, order)  # noqa: E731
        return (torch.cat([idx, pick(idx)], 1), torch.cat([cnt_a, torch.zeros_like(pick(cnt))], 1),
                torch.cat([kind_a, pick(kind_b)], 1), torch.cat([tgt_a, pick(tgt_b)], 1),
                torch.cat([torch.where(whole, k, 0), pick(torch.where(partial, k, 0))], 1))  # fmt: skip

    def loss(self, idx, cnt, kind, target, copies) -> tuple[torch.Tensor, dict[str, float]]:
        h, emb = self(idx, cnt, kind)
        at = target >= 0
        logits, count = self.heads(h[at], emb, target[at])
        card_loss = F.cross_entropy(logits, target[at])
        count_loss = F.cross_entropy(count, copies[at] - 1)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target[at]).float().mean().item()
        return card_loss + self.config.count_weight * count_loss, {"card": card_loss.item(),
                                                                   "count": count_loss.item(), "top1": acc}  # fmt: skip

    # -- scoring
    def _sets(self, sets: Sequence[Sequence[tuple[int, int]]],
              card: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:  # fmt: skip
        """(card log-probabilities ``[B, V]``, copy probabilities ``[B, 3]`` of vocab card ``card[b]``, or None
        without ``card``) of one ``[MASK]`` added to each set of (vocab index, copies) tokens."""
        width = max(len(s) for s in sets) + 1
        idx = np.full((len(sets), width), -1, dtype=np.int64)
        cnt = np.zeros((len(sets), width), dtype=np.int64)
        kind = np.full((len(sets), width), _PAD, dtype=np.int64)
        for i, s in enumerate(sets):
            for j, (c, n) in enumerate(s):
                idx[i, j], cnt[i, j], kind[i, j] = c, n, _CARD
            kind[i, len(s)] = _MASK
        dev = self.text.device
        t = [torch.from_numpy(a).to(dev) for a in (idx, cnt, kind)]
        with torch.no_grad():
            h, emb = self(*t)
            at = t[2] == _MASK
            c = torch.as_tensor(list(card) if card is not None else [0] * len(sets), device=dev)
            logits, count = self.heads(h[at], emb, c)
        probs = torch.softmax(count.float(), -1).cpu() if card is not None else None
        return torch.log_softmax(logits.float(), -1).cpu(), probs

    def _tokens(self, deck: Deck) -> list[tuple[int, int]]:
        return [(self.index[c], min(n, MAX_COPIES)) for c, n in sorted(deck.counts().items()) if c in self.index]

    def removal_scores(self, deck: Deck) -> dict[int, float]:
        """Per distinct card of ``deck``: how out of place one copy of it is, ``-log P(card | the rest)`` with that
        copy masked (higher: more out of place). Cards outside the vocab are left out."""
        was = self.training
        self.eval()
        toks = self._tokens(deck)
        if not toks:
            return {}
        sets = [[t if j != i else (t[0], t[1] - 1) for j, t in enumerate(toks) if j != i or t[1] > 1]
                for i in range(len(toks))]  # fmt: skip
        logp, _ = self._sets(sets)
        self.train(was)
        return {self.passwords[c]: float(-logp[i, c]) for i, (c, _) in enumerate(toks)}

    def addition_scores(self, deck: Deck, pool: Iterable[int]) -> dict[int, float]:
        """Per card of ``pool`` in the vocab: how much it belongs in ``deck``, ``log P(card | deck)`` of a ``[MASK]``
        added to the whole deck (a card the deck already runs: one more copy of it)."""
        was = self.training
        self.eval()
        logp, _ = self._sets([self._tokens(deck)])
        self.train(was)
        return {int(p): float(logp[0, self.index[p]]) for p in pool if p in self.index}

    def fill_in(self, decks: Sequence[Deck], masked: Sequence[int], batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Fill-in evaluation: for each deck, every copy of ``masked[i]`` masked; returns (card log-probabilities
        ``[N, V]`` as float32, the copies predicted for the masked card ``[N]``)."""
        was = self.training
        self.eval()
        lp, cp = [], []
        for s in range(0, len(decks), batch):
            sets = [[t for t in self._tokens(d) if self.passwords[t[0]] != m]
                    for d, m in zip(decks[s : s + batch], masked[s : s + batch], strict=True)]  # fmt: skip
            a, b = self._sets(sets, [self.index[m] for m in masked[s : s + batch]])
            lp.append(a.numpy())
            cp.append(b.argmax(-1).numpy() + 1)
        self.train(was)
        return np.concatenate(lp), np.concatenate(cp)

    # -- files
    def save(self, path: str | Path, meta: Mapping | None = None) -> None:
        torch.save({"format": FORMAT, "version": 1, "config": asdict(self.config), "passwords": self.passwords,
                    "environment": self.environment, "meta": dict(meta or {}),
                    "state": {k: v.cpu() for k, v in self.state_dict().items()}}, path)  # fmt: skip


def load_deck_model(path: str | Path, device: str | torch.device = "cpu") -> DeckModel:
    """A model written by :meth:`DeckModel.save`, in eval mode; ``model.meta`` is what the trainer recorded."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    state = data["state"]
    model = DeckModel(data["passwords"], state["text"].float(), DeckModelConfig(**data["config"]),
                      environment=data["environment"])  # fmt: skip
    model.load_state_dict(state)
    model.meta = data["meta"]
    return model.to(device).eval()


# ------------------------------------------------------------------ training


def train_deck_model(model: DeckModel, lists: Sequence[tuple[str, Deck]], *, epochs: int = 20, batch: int = 256,
                     lr: float = 1e-3, weight_decay: float = 0.01, device: str | torch.device = "cpu", seed: int = 0,
                     log: Callable[[str], None] | None = None, amp: bool | None = None,
                     on_epoch: Callable[[int, DeckModel], None] | None = None) -> DeckModel:  # fmt: skip
    """Train ``model`` by fill-in on ``lists`` (AdamW, one-cycle schedule; bfloat16 autocast when ``amp``, by default
    on CUDA); marks the lists' cards as seen (their ID embeddings train). Batches hold lists of similar size (sorted
    within random chunks of 32 batches) so little of a batch is padding."""
    say = log or (lambda m: None)
    torch.manual_seed(seed)
    model.to(device).train()
    idx, cnt = model.encode([d for _, d in lists])
    present = torch.zeros(len(model.passwords), dtype=torch.bool)
    present[idx[idx >= 0]] = True
    model.seen.copy_(present.to(model.seen.device))
    idx, cnt = idx.to(device), cnt.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * max(1, math.ceil(len(idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator(device=device).manual_seed(seed)
    amp = torch.device(device).type == "cuda" if amp is None else amp
    size = (idx >= 0).sum(1)
    for epoch in range(epochs):
        order = torch.randperm(len(idx), device=device, generator=gen)
        batches = []
        for s in range(0, len(order), batch * 32):
            chunk = order[s : s + batch * 32]
            chunk = chunk[torch.argsort(size[chunk], stable=True)]
            batches += [chunk[k : k + batch] for k in range(0, len(chunk), batch)]
        tot: Counter = Counter()
        n = 0
        for j in torch.randperm(len(batches), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist():
            b = batches[j]
            bi, bc = idx[b], cnt[b]
            width = int(size[b].max())
            parts = model.masked_batch(bi[:, :width], bc[:, :width], generator=gen)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                loss, stats = model.loss(*parts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.update(stats)
            n += 1
        say(f"epoch {epoch + 1}/{epochs}: " + ", ".join(f"{k} {v / n:.3f}" for k, v in sorted(tot.items())))
        if on_epoch is not None:
            on_epoch(epoch, model)
    return model.eval()


# ------------------------------------------------------------------ evaluation


def ranks(scores: np.ndarray, target: np.ndarray, exclude: Sequence[Iterable[int]]) -> np.ndarray:
    """Rank (1 = best) of ``target[i]`` in row ``i`` of ``scores [N, V]`` among the columns not in ``exclude[i]``;
    ties go against the target (a tie counts as ranked above it)."""
    out = np.empty(len(target), dtype=np.int64)
    for i, t in enumerate(target):
        row = scores[i]
        ex = np.fromiter(exclude[i], dtype=np.int64)
        s = row[t]
        above = int((row >= s).sum()) - 1  # without the target itself
        if len(ex):
            above -= int((row[ex] >= s).sum())
        out[i] = above + 1
    return out


def topk(r: np.ndarray, ks: Sequence[int] = (1, 5, 10, 50)) -> dict[str, float]:
    return {f"top{k}": float((r <= k).mean()) for k in ks} | {"mrr": float((1 / r).mean()), "n": int(len(r))}


def fill_in_tasks(lists: Sequence[tuple[str, Deck]], index: Mapping[int, int], seed: int = 0) -> list[tuple[Deck, int]]:
    """One fill-in task per list: a random distinct card of it (seeded), from the cards in ``index``."""
    rng = np.random.default_rng(seed)
    out = []
    for _, d in lists:
        cards = sorted(c for c in d.counts() if c in index)
        if len(cards) >= 2:
            out.append((d, int(cards[rng.integers(len(cards))])))
    return out


__all__ = ["FORMAT", "DeckModel", "DeckModelConfig", "fill_in_tasks", "held_out", "history_lists", "load_deck_model",
           "load_text_table", "ranks", "split_by_type", "topk", "train_deck_model"]  # fmt: skip
