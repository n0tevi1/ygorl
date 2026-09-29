"""Deck tuning around a given list (target 1「给定一套牌的强化」, T6.2; docs/tuning.md).

Local search over one-card edits of a base deck, judged by real games of a network policy piloting the candidate
against the environment's meta decks (piloted by the same or another policy), on the batched C++ path
(:mod:`ygorl.eval.batched`):

- **Edits** (:func:`neighbors`): swap one copy of a card in the deck for one copy of a candidate card in the same
  section (main / extra), keeping the deck size; every edited deck must be legal in the environment.
- **Candidate cards** (:func:`tech_pool`): cards from other lists of the base deck's type (its own engine variants)
  and the cards most often played across the environment's meta lists (hand traps, board breakers ...).
- **Common random numbers** (:class:`PairedEvaluator`): game ``k`` of every candidate and of the base deck uses the
  same seed, the same opponent (drawn by meta share) and both first players, so candidates are compared by their
  paired difference to the base deck, not by two noisy win rates.
- **Successive halving** (:func:`successive_halving`): all candidates get a few games, the better half gets more,
  and so on; the finalists' paired differences are reported with 95% intervals.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ygorl.cards.ydk import Deck
from ygorl.engine.duel import DuelConfig
from ygorl.eval.arena import derive_seed


@dataclass(frozen=True)
class Edit:
    out: int  # password of the copy taken out
    into: int  # password of the copy put in
    section: str  # "main" or "extra"

    def describe(self, names: Mapping[int, str] | None = None) -> str:
        n = names or {}
        return f"{self.section}: -{n.get(self.out, self.out)} +{n.get(self.into, self.into)}"


def apply(deck: Deck, edit: Edit) -> Deck:
    """``deck`` with the first copy of ``edit.out`` replaced by ``edit.into`` *in place*: the deck shuffle permutes
    positions by seed and size only, so every other card keeps its position and the same seed deals the base deck
    and the candidate the same hands but for this one card (common random numbers)."""
    cards = list(getattr(deck, edit.section))
    cards[cards.index(edit.out)] = edit.into
    return replace(deck, **{edit.section: tuple(cards)})


def tech_pool(base: Deck, same_type: Iterable[Deck], meta: Iterable[Deck], *, meta_top: int = 60,
              min_lists: int = 1) -> list[int]:  # fmt: skip
    """Candidate cards: every card in another list of the base deck's type, plus the ``meta_top`` cards that appear
    in the most meta lists (at least ``min_lists``); cards already at 3 copies in the base deck are still listed
    (legality decides later)."""
    out: dict[int, None] = {}
    for d in same_type:
        for pw in (*d.main, *d.extra):
            out.setdefault(pw)
    seen: Counter[int] = Counter()
    for d in meta:
        seen.update(set(d.main) | set(d.extra))
    for pw, n in seen.most_common(meta_top):
        if n >= min_lists:
            out.setdefault(pw)
    return list(out)


def neighbors(base: Deck, pool: Iterable[int], is_extra: Callable[[int], bool], legal: Callable[[Deck], bool],
              *, limit: int | None = None, rng: np.random.Generator | None = None) -> list[tuple[Edit, Deck]]:  # fmt: skip
    """Legal one-card swaps of ``base`` (a random ``limit`` of them when given)."""
    pool = list(dict.fromkeys(pool))
    edits = []
    for section in ("main", "extra"):
        outs = sorted(set(getattr(base, section)))
        ins = [pw for pw in pool if is_extra(pw) == (section == "extra")]
        edits.extend(Edit(o, i, section) for o in outs for i in ins if o != i)
    if rng is not None:
        rng.shuffle(edits)
    out = []
    for e in edits:
        deck = apply(base, e)
        if legal(deck):
            out.append((e, deck))
            if limit is not None and len(out) >= limit:
                break
    return out


@dataclass
class PairedEvaluator:
    """Plays decks (as deck a, piloted by ``policy``) against opponents drawn by weight (piloted by ``opponent``).

    Game pair ``k`` is the same for every deck: seed ``derive_seed(seed, 1, k)``, opponent drawn with
    ``derive_seed(seed, 2, k)``, both first players. :meth:`scores` returns, per deck, one score per pair (mean of
    its two games: 1 / 0.5 / 0 from the deck's side), so decks are compared pair by pair. A game that raised in
    the engine is missing (NaN), not a draw; a pair with both games missing is NaN. ``errors`` counts those games.

    ``control`` (optional, e.g. :class:`ygorl.build.control.CriticControlVariate`) maps the game specs to one
    zero-mean adjustment per game that is subtracted from the game's score (a control variate: the expectation is
    unchanged, the variance may drop).
    """

    env_factory: Callable[[int], object]  # num_envs -> EncodedVecEnv
    policy: object
    opponents: Sequence[Deck]
    weights: Sequence[float]
    opponent: object | None = None
    seed: int = 0
    device: str = "cpu"
    num_envs: int = 256
    config: DuelConfig = field(default_factory=lambda: DuelConfig(max_decisions=4000))
    control: Callable[[Sequence[object]], np.ndarray] | None = None
    games: int = 0
    seconds: float = 0.0
    errors: int = 0

    def _opponent(self, k: int) -> Deck:
        p = np.asarray(self.weights, dtype=float)
        idx = np.random.default_rng(derive_seed(self.seed, 2, k)).choice(len(self.opponents), p=p / p.sum())
        return self.opponents[int(idx)]

    def scores(self, decks: Sequence[Deck], pairs: range) -> np.ndarray:
        """``[deck, pair]`` scores of every deck on the same pairs."""
        if not decks:
            return np.zeros((0, len(pairs)))
        return np.stack(self.play([(d, pairs) for d in decks]))

    def specs(self, deck: Deck, pairs: Iterable[int]) -> list:
        """The game specs of ``deck`` on ``pairs``: two per pair, the deck going first then second."""
        from ygorl.eval.batched import paired_specs  # needs PyTorch (the train extra)

        return [s for k in pairs
                for s in paired_specs(deck, self._opponent(k), 1, derive_seed(self.seed, 1, k), self.config)]  # fmt: skip

    def opening_hands(self, deck: Deck, pairs: range) -> list[tuple[int, ...]]:
        """The opening hand ``deck`` is dealt on each pair (both games of a pair deal the same hand), without
        playing: the shuffle depends on the pair's seed only (docs/tuning.md, ``ygorl.build.diagnose``)."""
        from ygorl.build.diagnose import opening_hand
        from ygorl.engine.duel import shuffle_deck

        if not self.config.shuffle_decks:
            return [opening_hand(deck.main) for _ in pairs]
        return [opening_hand(shuffle_deck(deck.main, derive_seed(derive_seed(self.seed, 1, k), 0), 0)) for k in pairs]

    def play(self, jobs: Sequence[tuple[Deck, range]], *, per_game: bool = False) -> list[np.ndarray]:
        """One score per pair of each ``(deck, pairs)`` job, all jobs in one batch (the jobs may cover different
        pairs: an allocator that gives each candidate its own number of pairs still fills the GPU). With
        ``per_game``, each job's scores are ``[pair, 2]``: the game the deck went first, then the one it went second."""
        from ygorl.eval.batched import play_policies  # needs PyTorch (the train extra)

        specs = [s for d, pairs in jobs for s in self.specs(d, pairs)]
        if not specs:
            return [np.zeros((0, 2) if per_game else 0) for _ in jobs]
        env = self.env_factory(min(self.num_envs, len(specs)))
        records, stats = play_policies(env, specs, self.policy, self.opponent, device=self.device)
        self.games += len(specs)
        self.seconds += stats["seconds"]
        s = np.array([np.nan if r.reason == "exception" else 1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0
                      for r in records])  # fmt: skip
        self.errors += int(np.isnan(s).sum())
        if self.control is not None:
            t0 = time.perf_counter()
            s = s - np.asarray(self.control(specs), dtype=float)
            self.seconds += time.perf_counter() - t0
        out, at = [], 0
        for _, pairs in jobs:
            g = s[at : at + 2 * len(pairs)].reshape(len(pairs), 2)  # paired_specs: deck a first, then second
            at += 2 * len(pairs)
            if per_game:
                out.append(g)
                continue
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pairs stay NaN
                out.append(np.nanmean(g, -1))
        return out


@dataclass
class Candidate:
    edit: Edit | None  # None: the base deck
    deck: Deck
    scores: list[float] = field(default_factory=list)  # one per pair, pairs 0 .. n-1


def paired_difference(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Mean of ``a - b`` over the shared pairs (skipping pairs missing on either side) with a normal 95% interval."""
    n = min(len(a), len(b))
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float(d.mean()) if n else 0.0, -math.inf, math.inf
    h = 1.96 * float(d.std(ddof=1)) / math.sqrt(n)
    return float(d.mean()), float(d.mean()) - h, float(d.mean()) + h


def successive_halving(base: Deck, edits: Sequence[tuple[Edit, Deck]], evaluator: PairedEvaluator, *,
                       first_pairs: int = 25, keep: float = 0.5, finalists: int = 3,
                       log: Callable[[str], None] | None = None) -> tuple[Candidate, list[Candidate]]:  # fmt: skip
    """Rounds double the pairs per survivor (cumulative) and keep the best ``keep`` fraction by paired difference to
    the base deck, until ``finalists`` remain; returns the base candidate and the survivors, best first."""
    base_c = Candidate(None, base)
    alive = [Candidate(e, d) for e, d in edits]
    if not alive:
        return base_c, []
    target = first_pairs
    while True:
        todo = [c for c in [base_c, *alive] if len(c.scores) < target]
        # every candidate is behind by the same amount except the base, which may already have more pairs
        start = min(len(c.scores) for c in todo) if todo else target
        if todo:
            for c in todo:
                if len(c.scores) != start:
                    raise AssertionError("candidates out of step")
            s = evaluator.scores([c.deck for c in todo], range(start, target))
            for c, row in zip(todo, s, strict=True):
                c.scores.extend(row.tolist())
        alive.sort(key=lambda c: paired_difference(c.scores, base_c.scores)[0], reverse=True)
        if log:
            best = alive[0]
            log(
                f"{target} pairs: {len(alive)} candidates, base {np.mean(base_c.scores):.3f}, best "
                f"{paired_difference(best.scores, base_c.scores)[0]:+.3f} ({evaluator.games} games, {evaluator.seconds:.0f}s)"
            )
        if len(alive) <= finalists:
            return base_c, alive
        alive = alive[: max(finalists, int(math.ceil(len(alive) * keep)))]
        target *= 2


def validate(base: Deck, finalists: Sequence[Candidate], evaluator: PairedEvaluator, pairs: int,
             offset: int = 1_000_000) -> list[tuple[Candidate, list[float], list[float]]]:  # fmt: skip
    """Replay the finalists and the base deck on ``pairs`` fresh pairs (indices from ``offset``, never used by the
    search): the selection used the search's games, so only these give an unbiased paired difference. Returns
    ``(finalist, its fresh scores, the base deck's fresh scores)``."""
    decks = [base, *(c.deck for c in finalists)]
    s = evaluator.scores(decks, range(offset, offset + pairs))
    return [(c, s[i + 1].tolist(), s[0].tolist()) for i, c in enumerate(finalists)]


__all__ = [
    "Candidate",
    "Edit",
    "PairedEvaluator",
    "apply",
    "neighbors",
    "paired_difference",
    "successive_halving",
    "tech_pool",
    "validate",
]
