"""Edit-value model: the learned win-rate change of a deck edit, with uncertainty (#151, design 05 §5.3,
docs/tuning.md「改动价值模型」).

The card-value model (:class:`ygorl.build.signals.CardValueModel`) is first order: an edit is worth the cards in
minus the cards out, per deck type. This model reads the edit in the context of the deck through the masked deck
model (:class:`ygorl.build.deck_model.DeckModel`, frozen) and predicts Δ = child − parent win rate against the
environment's opponent mix:

- **Features** (:class:`DeckModelFeatures`): the deck model's pooled representation of the parent ``z(parent)``
  and the change ``z(child) − z(parent)`` (copy-weighted mean of the encoder outputs, no mask), the summed card
  embeddings of the cards in minus the cards out, the masked model's own scores of the edit (log P(card in |
  parent), typicality, support and the combined removal position of the cards out), the number of edits, and the
  label's **age** (``ygorl.build.edit_labels.Clock``; 0 at prediction).
- **Model**: a small MLP (``hidden`` units, dropout, an L2 penalty ``weight_decay`` on the weights, inputs
  standardized) plus ``beta × (s(child) − s(parent))``, where ``s`` is a linear deck-strength head on ``z``. An
  **ensemble** of ``members`` such nets, each fitted on a bootstrap sample of the labels from its own seed: the
  mean is the prediction; the standard deviation is the members' spread combined with the out-of-bag residual
  scale (``sigma0``: the weighted mean of residual² − stderr² of each label over the members that did not see it).
- **Labels** (:mod:`ygorl.build.edit_labels`): clean paired evaluations, each weighted by ``clock.weight / stderr²``
  (older policies count less); and, optionally, training games: an auxiliary loss ``aux_weight × BCE`` of
  P(deck a beats deck b) = σ(s(a) − s(b) ± first-player term) on the same strength head, weighted by the clock.
  The strength head is shared with the Δ prediction through ``beta``: the plentiful, confounded game outcomes act
  as a prior on which decks are strong, the paired labels decide how much of it carries over to an edit.

:func:`cross_validate` gives held-out predictions with whole parent decks left out (and the card-value model's
and the masked model's scores on the same folds, for comparison; ``tools/fit_value_model.py --cv``).
:meth:`EditValueModel.predict` is what the evolution step calls (``ygorl.build.evolve.Lab.value_model``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ygorl.build.edit_labels import Clock, GameData, PairedLabel
from ygorl.build.signals import CardValueModel, Observation, spearman
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import Deck

FORMAT = "ygorl-edit-value-model"
Y_SCALE = 0.02  # targets are fitted in units of 2 pp
SCORES = ("add", "typicality", "support", "removal")  # the masked model's own scores of an edit (summed over edits)
# feature blocks of an edit row, in order: z(child) - z(parent), Σ emb(in) - Σ emb(out), z(parent), the masked
# model's scores, the number of edits, the label's age
BLOCKS = ("dz", "de", "zp", "scores", "size", "age")


def columns(blocks: Sequence[str], dim: int) -> np.ndarray:
    """Column indices of ``blocks`` in an edit row of :func:`edit_features` (deck-model width ``dim``)."""
    width = {"dz": dim, "de": dim, "zp": dim, "scores": len(SCORES), "size": 1, "age": 1}
    out, at = [], 0
    for b in BLOCKS:
        if b in blocks:
            out.extend(range(at, at + width[b]))
        at += width[b]
    return np.array(out, dtype=np.int64)


# ------------------------------------------------------------------ features


def _key(deck: Deck) -> tuple:
    return tuple(sorted(deck.main)), tuple(sorted(deck.extra))


def _child(parent: Deck, edits: Sequence[Edit]) -> Deck:
    deck = parent
    for e in edits:
        deck = apply(deck, e)
    return deck


class DeckModelFeatures:
    """Features of decks and edits from a frozen masked deck model, cached per deck (scoring a parent's support
    takes D² sets: seconds on the CPU, once per parent)."""

    def __init__(self, model, *, batch: int = 128) -> None:
        self.model = model.eval()
        self.batch = batch
        with torch.no_grad():
            self._cards = model.cards().float().cpu().numpy()
        self.dim = self._cards.shape[1]
        self._decks: dict[tuple, np.ndarray] = {}
        self._parents: dict[tuple, dict] = {}

    def card_vectors(self, cards: Sequence[int]) -> np.ndarray:
        out = np.zeros((len(cards), self.dim), dtype=np.float32)
        for i, c in enumerate(cards):
            j = self.model.index.get(int(c))
            if j is not None:
                out[i] = self._cards[j]
        return out

    def deck_vectors(self, decks: Sequence[Deck]) -> np.ndarray:
        """[N, dim]: each deck's copy-weighted mean of the encoder outputs (no mask)."""
        todo = list({_key(d): d for d in decks if _key(d) not in self._decks}.values())
        from ygorl.build.deck_model import _CARD, _PAD

        for s in range(0, len(todo), self.batch):
            chunk = todo[s : s + self.batch]
            idx, cnt = self.model.encode(chunk)
            kind = torch.where(idx >= 0, _CARD, _PAD)
            dev = self.model.text.device
            with torch.no_grad():
                h, _ = self.model(idx.to(dev), cnt.to(dev), kind.to(dev))
            w = cnt.to(h.dtype).to(dev)
            pooled = (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
            for d, v in zip(chunk, pooled.float().cpu().numpy(), strict=True):
                self._decks[_key(d)] = v
        return np.stack([self._decks[_key(d)] for d in decks]) if decks else np.zeros((0, self.dim), np.float32)

    def _parent(self, parent: Deck) -> dict:
        k = _key(parent)
        if k not in self._parents:
            m = self.model
            logp, _ = m._sets([m._tokens(parent)])
            typ = m.typicality_scores(parent)
            sup = m.support_scores(parent)
            comb = m.removal_scores(parent, "combined") if typ else {}
            n = max(len(comb), 1)
            self._parents[k] = {"logp": logp[0].numpy(), "typicality": typ, "support": sup,
                                "removal": {c: v / n for c, v in comb.items()}}  # fmt: skip
        return self._parents[k]

    def edit_scores(self, parent: Deck, edits: Sequence[Edit]) -> np.ndarray:
        """The masked model's scores of an edit, summed over its single edits: log P(card in | parent) (a card
        outside the vocab: the lowest log-probability), typicality, support and combined removal position (D for
        the first card to go, scaled to (0, 1]) of the cards out."""
        p = self._parent(parent)
        floor = float(p["logp"].min())
        add = sum(float(p["logp"][self.model.index[e.into]]) if e.into in self.model.index else floor for e in edits)
        return np.array([add, *(sum(p[k].get(e.out, 0.0) for e in edits) for k in SCORES[1:])], dtype=np.float32)


@dataclass
class EditBatch:
    """Features of a batch of edits: ``x`` [N, F] (the age column last), ``zp`` / ``zc`` [N, dim] (parent and
    child representations, for the strength head)."""

    x: np.ndarray
    zp: np.ndarray
    zc: np.ndarray
    scores: np.ndarray  # [N, len(SCORES)]: the masked model's scores (also inside x)
    cards: list[dict[int, int]]  # per edit: copies in minus copies out of each card (the first-order term)

    def card_matrix(self, cards: Sequence[int]) -> np.ndarray:
        """[N, len(cards)]: the signed copies of ``cards`` each edit swaps (cards not listed are left out)."""
        pos = {c: j for j, c in enumerate(cards)}
        out = np.zeros((len(self.cards), len(cards)), np.float32)
        for i, row in enumerate(self.cards):
            for c, k in row.items():
                if c in pos:
                    out[i, pos[c]] = k
        return out


def edit_features(features, parents: Sequence[Deck], edits: Sequence[Sequence[Edit]],
                  ages: Sequence[float] | None = None) -> EditBatch:  # fmt: skip
    """Feature rows of edits (``edits[i]`` applied in place to ``parents[i]``); ``ages`` in half-lives (0: the
    target policy)."""
    children = [_child(p, e) for p, e in zip(parents, edits, strict=True)]
    zp = features.deck_vectors(list(parents))
    zc = features.deck_vectors(children)
    de = np.stack([features.card_vectors([e.into for e in es]).sum(0) - features.card_vectors([e.out for e in es]).sum(0)
                   for es in edits]) if edits else np.zeros((0, features.dim), np.float32)  # fmt: skip
    scores = np.stack([features.edit_scores(p, es) for p, es in zip(parents, edits, strict=True)]) if edits else \
        np.zeros((0, len(SCORES)), np.float32)  # fmt: skip
    n_edits = np.array([[len(es)] for es in edits], dtype=np.float32).reshape(-1, 1)
    age = np.zeros((len(edits), 1), np.float32) if ages is None else np.asarray(ages, np.float32).reshape(-1, 1)
    x = np.concatenate([zc - zp, de, zp, scores, n_edits, age], 1).astype(np.float32)
    signed = []
    for es in edits:
        row: dict[int, int] = {}
        for e in es:
            row[e.into] = row.get(e.into, 0) + 1
            row[e.out] = row.get(e.out, 0) - 1
        signed.append({c: k for c, k in row.items() if k})
    return EditBatch(x, zp.astype(np.float32), zc.astype(np.float32), scores, signed)


# ------------------------------------------------------------------ model


@dataclass
class ValueModelConfig:
    hidden: int = 16
    dropout: float = 0.2
    weight_decay: float = 3e-2  # L2 penalty on the weights (inputs standardized, targets in units of Y_SCALE)
    members: int = 8
    steps: int = 300
    lr: float = 3e-3
    bootstrap: bool = True
    aux_weight: float = 0.3  # weight of the training-game loss (0: off)
    aux_batch: int = 2048
    sigma_floor: float = 0.005  # the smallest residual scale (win-rate units)
    blocks: tuple[str, ...] = BLOCKS  # the feature blocks the net reads (the strength head always reads z)
    card_terms: bool = False  # a value per card seen in the labels (first order, as CardValueModel's card effect)
    card_decay: float = 1.0  # L2 penalty of the per-card values
    seed: int = 0

    def __post_init__(self) -> None:
        self.blocks = tuple(self.blocks)
        if unknown := set(self.blocks) - set(BLOCKS):
            raise ValueError(f"unknown feature blocks {sorted(unknown)} (of {BLOCKS})")


class _Net(nn.Module):
    def __init__(self, d_in: int, d_z: int, cfg: ValueModelConfig, n_cards: int = 0) -> None:
        super().__init__()
        self.card_decay = cfg.card_decay
        self.cards = nn.Linear(n_cards, 1, bias=False) if n_cards else None
        if self.cards is not None:
            nn.init.zeros_(self.cards.weight)
        if cfg.hidden > 0:
            self.body = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                      nn.Linear(cfg.hidden, 1))  # fmt: skip
        else:  # linear
            self.body = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d_in, 1))
        self.strength = nn.Linear(d_z, 1)
        nn.init.zeros_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)
        self.first = nn.Parameter(torch.zeros(()))  # going-first advantage (logit)
        self.beta = nn.Parameter(torch.zeros(()))  # Δ (in Y_SCALE units) per unit of strength difference

    def delta(self, x, zp, zc, cards=None) -> torch.Tensor:
        out = self.body(x).squeeze(-1) + self.beta * (self.strength(zc) - self.strength(zp)).squeeze(-1)
        if self.cards is not None:
            out = out + self.cards(cards).squeeze(-1)
        return out

    def game_logit(self, za, zb, a_first) -> torch.Tensor:
        return (self.strength(za) - self.strength(zb)).squeeze(-1) + self.first * (2 * a_first - 1)

    def penalty(self) -> torch.Tensor:
        p = sum((m.weight**2).sum() for m in (*self.body, self.strength) if isinstance(m, nn.Linear))
        if self.cards is not None:
            p = p + self.card_decay * (self.cards.weight**2).sum()
        return p


@dataclass
class _Standard:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> _Standard:
        sd = a.std(0)
        return cls(a.mean(0), np.where(sd > 1e-8, sd, 1.0))

    def __call__(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((a - self.mean) / self.std).astype(np.float32))


@dataclass
class _Games:
    """Training games as index arrays into a deck-vector table."""

    z: np.ndarray  # [D, dim]
    a: np.ndarray
    b: np.ndarray
    a_first: np.ndarray
    a_won: np.ndarray
    weight: np.ndarray


def game_arrays(features, games: GameData, clock: Clock) -> _Games | None:
    if not games.games:
        return None
    names = sorted({n for g in games.games for n in (g.a, g.b)})
    pos = {n: i for i, n in enumerate(names)}
    z = features.deck_vectors([games.decks[n] for n in names])
    w = np.array([clock.weight(g.checkpoint) for g in games.games])
    return _Games(z, np.array([pos[g.a] for g in games.games]), np.array([pos[g.b] for g in games.games]),
                  np.array([g.first == 0 for g in games.games], np.float32),
                  np.array([g.winner == 0 for g in games.games], np.float32), w / w.mean())  # fmt: skip


@dataclass
class FitReport:
    labels: int
    games: int
    sigma0: float
    oob_spearman: float
    members: int
    extra: dict = field(default_factory=dict)


class EditValueModel:
    """An ensemble predicting Δ of an edit (module docstring). Built by :func:`fit`; saved with :meth:`save`."""

    def __init__(self, config: ValueModelConfig, nets: list[_Net], xs: _Standard, zs: _Standard, sigma0: float,
                 clock: Clock, features=None, meta: Mapping | None = None,
                 cards: Sequence[int] = ()) -> None:  # fmt: skip
        self.config, self.nets, self.xs, self.zs = config, nets, xs, zs
        self.cards = [int(c) for c in cards]  # the cards with a first-order value (seen in the labels)
        self.sigma0, self.clock, self.features = float(sigma0), clock, features
        self.meta = dict(meta or {})
        self.cols = columns(config.blocks, len(zs.mean))
        for n in self.nets:
            n.eval()

    def predict_batch(self, batch: EditBatch) -> tuple[np.ndarray, np.ndarray]:
        """(mean, sd) in win-rate units: the members' mean; sd = √(members' variance + sigma0²)."""
        if len(batch.x) == 0:
            return np.zeros(0), np.zeros(0)
        x, zp, zc = self.xs(batch.x[:, self.cols]), self.zs(batch.zp), self.zs(batch.zc)
        c = torch.from_numpy(batch.card_matrix(self.cards))
        with torch.no_grad():
            p = np.stack([n.delta(x, zp, zc, c).numpy() for n in self.nets]) * Y_SCALE
        return p.mean(0), np.sqrt(p.var(0) + self.sigma0**2)

    def predict(self, parent: Deck, edits: Sequence[Sequence[Edit]]) -> list[tuple[float, float]]:
        """(mean, sd) of Δ for each edit list applied to ``parent`` under the target policy (age 0)."""
        if not edits:
            return []
        if self.features is None:
            raise ValueError("the value model has no deck-model features attached (load it with a deck model)")
        m, s = self.predict_batch(edit_features(self.features, [parent] * len(edits), edits))
        return [(float(a), float(b)) for a, b in zip(m, s, strict=True)]

    def save(self, path: str | Path) -> None:
        torch.save({"format": FORMAT, "version": 1, "config": asdict(self.config),
                    "nets": [{k: v.cpu() for k, v in n.state_dict().items()} for n in self.nets],
                    "d_in": int(self.xs.mean.shape[0]), "d_z": int(self.zs.mean.shape[0]), "cards": self.cards,
                    "xs": [torch.from_numpy(self.xs.mean), torch.from_numpy(self.xs.std)],
                    "zs": [torch.from_numpy(self.zs.mean), torch.from_numpy(self.zs.std)],
                    "sigma0": self.sigma0, "clock": self.clock.to_dict(), "meta": self.meta}, path)  # fmt: skip


def load_value_model(path: str | Path, deck_model=None) -> EditValueModel:
    """A model written by :meth:`EditValueModel.save`; ``deck_model`` (a :class:`DeckModel`, the one it was fitted
    with: ``meta["deck_model"]`` records its path and sha256) attaches the features :meth:`predict` needs."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    cfg = ValueModelConfig(**d["config"])
    nets = []
    for sd in d["nets"]:
        n = _Net(d["d_in"], d["d_z"], cfg, len(d["cards"]))
        n.load_state_dict(sd)
        nets.append(n)
    xs = _Standard(d["xs"][0].numpy(), d["xs"][1].numpy())
    zs = _Standard(d["zs"][0].numpy(), d["zs"][1].numpy())
    feats = DeckModelFeatures(deck_model) if deck_model is not None else None
    return EditValueModel(cfg, nets, xs, zs, d["sigma0"], Clock.from_dict(d["clock"]), feats, d.get("meta"),
                          d["cards"])  # fmt: skip


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ fitting


def label_batch(features, labels: Sequence[PairedLabel], clock: Clock) -> tuple[EditBatch, np.ndarray, np.ndarray]:
    """(features with each label's age in half-lives, targets, weights clock.weight / stderr²)."""
    ages = [clock.age(lab.checkpoint) / clock.half_life if math.isfinite(clock.half_life) else 0.0 for lab in labels]
    batch = edit_features(features, [lab.parent for lab in labels], [lab.edits for lab in labels], ages)
    y = np.array([lab.diff for lab in labels])
    w = np.array([clock.weight(lab.checkpoint) / lab.stderr**2 for lab in labels])
    return batch, y, w


def _fit_net(net: _Net, x, zp, zc, c, y, w, games: _Games | None, gz, cfg: ValueModelConfig,
             gen: torch.Generator) -> None:  # fmt: skip
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    yt = torch.from_numpy((y / Y_SCALE).astype(np.float32))
    wt = torch.from_numpy((w / w.mean()).astype(np.float32))
    use_games = games is not None and cfg.aux_weight > 0
    if use_games:
        ga, gb = torch.from_numpy(games.a), torch.from_numpy(games.b)
        gf, gy = torch.from_numpy(games.a_first), torch.from_numpy(games.a_won)
        gw = torch.from_numpy(games.weight.astype(np.float32))
    net.train()
    for _ in range(cfg.steps):
        loss = (wt * (net.delta(x, zp, zc, c) - yt) ** 2).mean() + cfg.weight_decay * net.penalty() / max(len(y), 1)
        if use_games:
            k = torch.randint(len(ga), (min(cfg.aux_batch, len(ga)),), generator=gen)
            logit = net.game_logit(gz[ga[k]], gz[gb[k]], gf[k])
            loss = (
                loss
                + cfg.aux_weight * (gw[k] * F.binary_cross_entropy_with_logits(logit, gy[k], reduction="none")).mean()
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()


def fit(features, labels: Sequence[PairedLabel], clock: Clock, config: ValueModelConfig | None = None, *,
        games: GameData | None = None, meta: Mapping | None = None,
        log: Callable[[str], None] | None = None) -> tuple[EditValueModel, FitReport]:  # fmt: skip
    """Fit the ensemble on ``labels`` (and, with ``config.aux_weight`` > 0, the training ``games``) for the target
    policy of ``clock``."""
    cfg = config or ValueModelConfig()
    if not labels:
        raise ValueError("no labels to fit")
    batch, y, w = label_batch(features, labels, clock)
    batch.x = batch.x[:, columns(cfg.blocks, batch.zp.shape[1])]
    g = game_arrays(features, games, clock) if games is not None else None
    xs = _Standard.fit(batch.x)
    zall = np.concatenate([batch.zp, batch.zc] + ([g.z] if g is not None else []))
    zs = _Standard.fit(zall)
    x, zp, zc = xs(batch.x), zs(batch.zp), zs(batch.zc)
    cards = sorted({k for row in batch.cards for k in row}) if cfg.card_terms else []
    c = torch.from_numpy(batch.card_matrix(cards))
    gz = zs(g.z) if g is not None else None
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = len(y)
    nets, oob = [], [[] for _ in range(n)]
    for m in range(cfg.members):
        idx = rng.integers(0, n, n) if cfg.bootstrap and cfg.members > 1 else np.arange(n)
        net = _Net(batch.x.shape[1], batch.zp.shape[1], cfg, len(cards))
        _fit_net(net, x[idx], zp[idx], zc[idx], c[idx], y[idx], w[idx], g, gz, cfg,
                 torch.Generator().manual_seed(cfg.seed * 1000 + m))  # fmt: skip
        nets.append(net)
        out = np.setdiff1d(np.arange(n), idx)
        if len(out):
            with torch.no_grad():
                pred = net.delta(x[out], zp[out], zc[out], c[out]).numpy() * Y_SCALE
            for i, p in zip(out, pred, strict=True):
                oob[i].append(p)
    have = [i for i in range(n) if oob[i]]
    se2 = np.array([labels[i].stderr ** 2 for i in have])
    if have:
        r2 = np.array([(np.mean(oob[i]) - y[i]) ** 2 for i in have])
        sigma2 = float(np.sum(w[have] * (r2 - se2)) / np.sum(w[have]))
        rho = spearman([np.mean(oob[i]) for i in have], y[have])
    else:
        sigma2, rho = 0.0, math.nan
    sigma0 = max(math.sqrt(max(sigma2, 0.0)), cfg.sigma_floor)
    model = EditValueModel(cfg, nets, xs, zs, sigma0, clock, features, meta, cards)
    rep = FitReport(n, len(g.a) if g is not None else 0, sigma0, rho, len(nets))
    if log:
        log(f"value model: {n} labels, {rep.games} games, {len(nets)} members, sigma0 {sigma0:.4f}, "
            f"out-of-bag Spearman {rho:+.3f}")  # fmt: skip
    return model, rep


# ------------------------------------------------------------------ evaluation


def group_folds(labels: Sequence[PairedLabel], by: str = "deck", k: int = 5, seed: int = 0) -> list[int]:
    """Fold of each label: ``"deck"`` one fold per parent deck (leave one deck out), ``"random"`` ``k`` random
    folds of the distinct edits (every measurement of an edit in one fold)."""
    if by == "deck":
        names = sorted({lab.group for lab in labels})
        pos = {g: i for i, g in enumerate(names)}
        return [pos[lab.group] for lab in labels]
    if by == "random":  # by edit: repeated measurements of one edit share a fold
        keys = sorted({lab.edit_key for lab in labels})
        fold = dict(zip(keys, np.random.default_rng(seed).integers(0, k, len(keys)).tolist(), strict=True))
        return [int(fold[lab.edit_key]) for lab in labels]
    raise ValueError("folds by deck or random")


def card_value_predictions(train: Sequence[PairedLabel], test: Sequence[PairedLabel], clock: Clock) -> np.ndarray:
    """The card-value model warm-started from ``train`` (each label's standard error divided by √clock weight, as
    the edit-value model weighs it), predicting ``test`` (``gain``)."""
    cvm = CardValueModel()
    for lab in train:
        se = lab.stderr / math.sqrt(max(clock.weight(lab.checkpoint), 1e-12))
        cvm.add(Observation(lab.deck_type, tuple(e.into for e in lab.edits), tuple(e.out for e in lab.edits),
                            lab.diff, se))  # fmt: skip
    return np.array([cvm.gain([e.into for e in lab.edits], [e.out for e in lab.edits], lab.deck_type)[0]
                     for lab in test])  # fmt: skip


def cross_validate(features, labels: Sequence[PairedLabel], clock: Clock, config: ValueModelConfig | None = None, *,
                   folds: Sequence[int], games: GameData | None = None,
                   log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """Held-out predictions per method: ``value_model`` (mean and ``value_model_sd``), ``card_value``, and the
    masked model's scores (``add``: log P(card in); ``removal``: the combined removal position of the card out;
    ``add_removal``: the sum of the two as within-deck ranks; no fitting)."""
    n = len(labels)
    out = {k: np.full(n, np.nan) for k in ("value_model", "value_model_sd", "card_value")}
    folds = np.asarray(folds)
    for f in sorted(set(folds.tolist())):
        test = np.flatnonzero(folds == f)
        train = np.flatnonzero(folds != f)
        if not len(train):
            continue
        model, _ = fit(features, [labels[i] for i in train], clock, config, games=games)
        m, s = model.predict_batch(edit_features(features, [labels[i].parent for i in test],
                                                 [labels[i].edits for i in test]))  # fmt: skip
        out["value_model"][test], out["value_model_sd"][test] = m, s
        out["card_value"][test] = card_value_predictions([labels[i] for i in train], [labels[i] for i in test], clock)
        if log:
            log(f"fold {f}: {len(test)} held out")
    sc = np.stack([features.edit_scores(lab.parent, lab.edits) for lab in labels])
    out["add"], out["removal"] = sc[:, 0], sc[:, 3]
    groups = np.array([lab.group for lab in labels])
    both = np.zeros(n)
    for g in set(groups.tolist()):
        at = np.flatnonzero(groups == g)
        for col in (0, 3):
            r = np.argsort(np.argsort(sc[at, col]))
            both[at] += r / max(len(at) - 1, 1)
    out["add_removal"] = both
    return out


def summarize(labels: Sequence[PairedLabel], preds: Mapping[str, np.ndarray], *, precise: float = 0.015,
              top: float = 0.2) -> dict[str, dict]:  # fmt: skip
    """Per method: Spearman with the measured Δ over all held-out labels (``pooled``), its mean within parent
    decks (``within``, weighted by labels), over the labels measured to ``precise`` (``precise``), and the mean
    measured Δ of each deck's top ``top`` share by the method's score minus the deck's mean (``top_gain``)."""
    y = np.array([lab.diff for lab in labels])
    se = np.array([lab.stderr for lab in labels])
    groups = np.array([lab.group for lab in labels])
    out = {}
    for name, p in preds.items():
        if name.endswith("_sd"):
            continue
        p = np.round(p, 9)  # ties (e.g. the card-value model's prior 0) must not break on rounding noise
        ok = np.isfinite(p)
        within, wsum, gains = 0.0, 0, []
        for g in sorted(set(groups.tolist())):
            at = np.flatnonzero((groups == g) & ok)
            if len(at) >= 4:
                within += spearman(p[at], y[at]) * len(at)
                wsum += len(at)
                k = max(1, int(round(top * len(at))))
                best = at[np.argsort(-p[at], kind="stable")[:k]]
                gains.append((y[best].mean() - y[at].mean(), len(at)))
        prec = ok & (se <= precise)
        out[name] = {"pooled": spearman(p[ok], y[ok]), "within": within / wsum if wsum else math.nan,
                     "precise": spearman(p[prec], y[prec]) if prec.sum() >= 4 else math.nan,
                     "precise_n": int(prec.sum()), "n": int(ok.sum()),
                     "top_gain": float(sum(a * b for a, b in gains) / sum(b for _, b in gains)) if gains else math.nan}  # fmt: skip
    return out


__all__ = ["BLOCKS", "FORMAT", "SCORES", "columns", "DeckModelFeatures", "EditBatch", "EditValueModel", "FitReport", "ValueModelConfig",
           "card_value_predictions", "cross_validate", "edit_features", "file_sha256", "fit", "game_arrays",
           "group_folds", "label_batch", "load_value_model", "summarize"]  # fmt: skip
