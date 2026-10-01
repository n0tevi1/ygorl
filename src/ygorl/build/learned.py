"""Learned children: deck-evolution candidates from the masked deck model (#150, docs/tuning.md「掩码卡组模型」).

:func:`learned_children` is the ``learned`` child generator of the evolution step (``RoundConfig.learned``). Its
additions come from the model's addition scores (how much a card belongs in the parent) and its removals from the
model's removal scores (by default "combined": atypical cards first, the cards the rest of the deck relies on most
last; ``removal`` selects "support" or the plain "typicality"), in place of the hand-written engine members, addition
pool and rule strata of #145, which stay as the control. Legality and the protected cards (win conditions, search
targets: :func:`ygorl.build.signals.protected_cards`) constrain the edits.

The scorer is anything with ``removal_scores(deck, kind)`` and ``addition_scores(deck, pool)`` (``password -> score``,
higher = more out of place / belongs more), e.g. :class:`ygorl.build.deck_model.DeckModel`; this module does not
need PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

import numpy as np

from ygorl.build.signals import CardValueModel, Child
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import Deck

GENERATOR = "learned"  # the child kind (lineage ``generator``)
MAX_COPIES = 3


class DeckScorer(Protocol):
    def removal_scores(self, deck: Deck, kind: str = "combined") -> Mapping[int, float]: ...

    def addition_scores(self, deck: Deck, pool: Iterable[int]) -> Mapping[int, float]: ...


REMOVAL_TEMPERATURE = 4.0  # rank points: the top ~4-6 removable cards share the draws


def learned_children(base: Deck, deck_type: str, scorer: DeckScorer, model: CardValueModel, pool: Iterable[int], *,
                     legal: Callable[[Deck], bool], is_extra: Callable[[int], bool], rng: np.random.Generator,
                     children: int = 4, max_bundle: int = 3, avoid: Iterable[Deck] = (), temperature: float = 1.0,
                     top: int = 200, protected: Iterable[int] = (), removal: str = "combined",
                     removal_temperature: float = REMOVAL_TEMPERATURE) -> list[Child]:  # fmt: skip
    """Up to ``children`` children of ``base``, distinct from each other, from ``base`` and from ``avoid``. The deck
    is scored once. Per child: draw a bundle size (1..``max_bundle``), an order of the ``top`` best-scored additions
    (cards of ``pool`` below 3 copies in ``base``; Gumbel keys on addition score / ``temperature``: a sample from the
    model's P(card | deck), so repeated calls spread over the plausible cards) and an order of the removals (Gumbel
    keys on the ``removal`` score / ``removal_temperature``; ``protected`` cards never go out). The combined removal
    score is a rank position (one point per place), so at temperature 1 the top-ranked card took about two thirds of the
    draws and a round tested nearly the same removal eight times; the default ``REMOVAL_TEMPERATURE`` spreads the draws
    over the top few removable cards. Each addition goes in one copy, taking out the first removal of the same
    section that leaves the deck legal (never a card the bundle put in, nor the card itself; a card the bundle took out
    is not put back); an addition no removal makes legal is skipped. Children are kind ``"learned"``, predicted gain = the card-value model's mean gain."""
    if children <= 0:
        return []
    counts = base.counts()
    adds = scorer.addition_scores(base, [c for c in dict.fromkeys(pool) if counts.get(c, 0) < MAX_COPIES])
    adds = {c: s for c, s in adds.items() if np.isfinite(s)}
    protected = set(protected)
    rems = {c: s for c, s in scorer.removal_scores(base, removal).items() if np.isfinite(s) and c not in protected}
    if not adds or not rems:
        return []
    add_cards = sorted(adds, key=lambda c: (-adds[c], c))[:top]
    add_s = np.array([adds[c] for c in add_cards]) / temperature
    rem_cards = sorted(rems)
    rem_s = np.array([rems[c] for c in rem_cards]) / removal_temperature
    seen = {_key(base), *(_key(d) for d in avoid)}
    out: list[Child] = []
    for _ in range(children * 4):
        if len(out) >= children:
            break
        size = int(rng.integers(1, max_bundle + 1))
        order = [add_cards[k] for k in np.argsort(-(add_s + rng.gumbel(size=len(add_s))), kind="stable")]
        outs = [rem_cards[k] for k in np.argsort(-(rem_s + rng.gumbel(size=len(rem_s))), kind="stable")]
        deck, edits = base, []
        for c in order:
            if len(edits) >= size:
                break
            if c in {e.out for e in edits}:
                continue  # never put back a card the bundle took out
            section = "extra" if is_extra(c) else "main"
            put = {e.into for e in edits}
            for o in outs:
                if o == c or o in put or o not in getattr(deck, section):
                    continue
                cand = apply(deck, Edit(o, c, section))
                if legal(cand):
                    deck, edits = cand, [*edits, Edit(o, c, section)]
                    break
        if edits and _key(deck) not in seen:
            seen.add(_key(deck))
            gain = model.gain([e.into for e in edits], [e.out for e in edits], deck_type)[0]
            out.append(Child(tuple(edits), deck, GENERATOR, gain))
    return out


def _key(deck: Deck) -> tuple:
    return tuple(sorted(deck.main)), tuple(sorted(deck.extra))


__all__ = ["GENERATOR", "DeckScorer", "learned_children"]
