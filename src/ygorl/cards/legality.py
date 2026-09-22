"""Deck legality: size limits, copy limits (banlist), card pool, deck sections.

``validate_deck`` returns every violation with a readable message instead of
stopping at the first one; ``check_deck`` raises :class:`IllegalDeck` with all
of them. Copies are counted across main + extra + side deck, and alternate
artworks (same name, ``alias`` pointing at the original) count as the original.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from ygorl.cards.lflist import Banlist
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

EXTRA_DECK_TYPES = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK
LIMIT_NAMES = {0: "forbidden", 1: "limited", 2: "semi-limited"}


@dataclass(frozen=True)
class DeckRules:
    main_min: int = 40
    main_max: int = 60
    extra_max: int = 15
    side_max: int = 15
    max_copies: int = 3


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    password: int | None = None


class IllegalDeck(ValueError):
    def __init__(self, violations: list[Violation], name: str = "") -> None:
        self.violations = violations
        head = f"deck {name!r} is illegal" if name else "deck is illegal"
        super().__init__(head + ":\n" + "\n".join(f"  - {v.message}" for v in violations))


def _canonical(password: int, cards: Mapping[int, object]) -> int:
    card = cards.get(password)
    if card is not None and card.alias and card.alias in cards and cards[card.alias].name == card.name:
        return card.alias
    return password


def _label(password: int, cards: Mapping[int, object]) -> str:
    card = cards.get(password)
    return f"{card.name} ({password})" if card is not None and card.name else str(password)


def validate_deck(
    deck: Deck,
    *,
    cards: Mapping[int, object],
    banlist: Banlist | None,
    pool: frozenset[int] | None = None,
    rules: DeckRules = DeckRules(),
) -> list[Violation]:
    """All rule violations of ``deck`` (empty list = legal).

    ``cards`` maps password -> card (needs ``name``, ``alias``, ``type``);
    ``pool`` of ``None`` means every known card is allowed.
    """
    out: list[Violation] = []
    sizes = (("main", deck.main, rules.main_min, rules.main_max), ("extra", deck.extra, 0, rules.extra_max),
             ("side", deck.side, 0, rules.side_max))  # fmt: skip
    for section, seq, lo, hi in sizes:
        if len(seq) < lo:
            out.append(Violation(f"{section}_too_small", f"{section.capitalize()} Deck has {len(seq)} cards (minimum {lo})"))
        if len(seq) > hi:
            out.append(Violation(f"{section}_too_large", f"{section.capitalize()} Deck has {len(seq)} cards (maximum {hi})"))

    known: set[int] = set()
    for pw in dict.fromkeys(deck.main + deck.extra + deck.side):
        if pw not in cards:
            out.append(Violation("unknown_card", f"Card {pw} does not exist in the card database", pw))
            continue
        known.add(pw)
        if pool is not None and pw not in pool and _canonical(pw, cards) not in pool:
            out.append(Violation("not_in_pool", f"{_label(pw, cards)} is not in this format's card pool", pw))
        if cards[pw].type & C.TYPE_TOKEN:
            out.append(Violation("token", f"{_label(pw, cards)} is a token and cannot be in a deck", pw))

    for pw in dict.fromkeys(deck.main):
        if pw in known and cards[pw].type & C.TYPE_MONSTER and cards[pw].type & EXTRA_DECK_TYPES:
            out.append(Violation("extra_in_main", f"{_label(pw, cards)} is an Extra Deck monster but is in the Main Deck", pw))
    for pw in dict.fromkeys(deck.extra):
        if pw in known and not (cards[pw].type & C.TYPE_MONSTER and cards[pw].type & EXTRA_DECK_TYPES):
            out.append(Violation("main_in_extra", f"{_label(pw, cards)} is not an Extra Deck monster but is in the Extra Deck", pw))

    counts = Counter(_canonical(pw, cards) for pw in deck.main + deck.extra + deck.side if pw in known)
    for pw, n in counts.items():
        allowed = rules.max_copies if banlist is None else min(rules.max_copies, banlist.limit(pw))
        if n <= allowed:
            continue
        if allowed == 0:
            out.append(Violation("forbidden", f"{_label(pw, cards)} is forbidden ({n} in deck)", pw))
        elif banlist is not None and banlist.limit(pw) < rules.max_copies:
            status = LIMIT_NAMES.get(allowed, f"limit {allowed}")
            out.append(Violation("over_limit", f"{_label(pw, cards)}: {n} copies, banlist allows {allowed} ({status})", pw))
        else:
            out.append(Violation("over_limit", f"{_label(pw, cards)}: {n} copies (maximum {allowed})", pw))
    return out


def check_deck(deck: Deck, **kwargs) -> None:
    """Raise :class:`IllegalDeck` listing every violation, if any."""
    violations = validate_deck(deck, **kwargs)
    if violations:
        raise IllegalDeck(violations, deck.name)
