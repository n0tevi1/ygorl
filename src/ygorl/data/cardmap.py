"""Map cards named by outside sources to passwords of our card database (BabelCDB).

Sources identify cards by a password (YGOPRODECK ``id``, masterduelmeta ``konamiID``, Yugipedia
``Password``) and a name. :class:`CardMapper` accepts the password when the card database knows it,
otherwise falls back to the English name (exact, then accent/punctuation-insensitive), and resolves
alternate artworks to the original password. Anything it cannot place is reported by the caller, never
dropped silently.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ygorl.engine import constants as C

_PUNCT = re.compile(r"[^0-9a-z]+")


def normalize_name(name: str) -> str:
    """Case-, accent- and punctuation-insensitive form of a card name."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return _PUNCT.sub("", text)


@dataclass(frozen=True)
class Mapped:
    password: int  # canonical password (the original artwork)
    how: str  # "password" | "alias" | "name"
    source_password: int | None = None


class CardMapper:
    """``map(password, name)`` -> :class:`Mapped` or None, against a password -> card mapping."""

    def __init__(self, cards: Mapping[int, object]) -> None:
        self.cards = cards
        self._exact: dict[str, set[int]] = defaultdict(set)
        self._loose: dict[str, set[int]] = defaultdict(set)
        self._arts: dict[int, set[int]] = defaultdict(set)
        for pw, card in cards.items():
            canon = self.canonical(pw)
            self._arts[canon].add(pw)
            if card.type & C.TYPE_TOKEN:  # never a deck card, never a mapping target by name
                continue
            if card.name:
                self._exact[card.name].add(canon)
                self._loose[normalize_name(card.name)].add(canon)

    def canonical(self, password: int) -> int:
        card = self.cards.get(password)
        if card is not None and card.alias and card.alias in self.cards and self.cards[card.alias].name == card.name:
            return card.alias
        return password

    def arts(self, password: int) -> set[int]:
        """Every password of the card: the original and its alternate artworks."""
        canon = self.canonical(password)
        return self._arts.get(canon, set()) | {canon}

    def by_name(self, name: str) -> int | None:
        for table, key in ((self._exact, name), (self._loose, normalize_name(name))):
            hits = table.get(key)
            if hits and len(hits) == 1:
                return next(iter(hits))
            if hits:
                return None  # ambiguous: leave it to a human
        return None

    def map(self, password: int | str | None, name: str | None = None) -> Mapped | None:
        pw = _as_int(password)
        if pw is not None and pw in self.cards:
            canon = self.canonical(pw)
            return Mapped(canon, "password" if canon == pw else "alias", pw)
        if name:
            found = self.by_name(name)
            if found is not None:
                return Mapped(found, "name", pw)
        return None

    def names(self, passwords: Iterable[int]) -> dict[int, str]:
        return {p: getattr(self.cards.get(p), "name", "") for p in passwords}


def _as_int(value: int | str | None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
