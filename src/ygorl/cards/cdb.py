"""Loading ``cards.cdb`` (BabelCDB, SQLite) and decoding its bit-packed fields.

Schema: ``datas(id, ot, alias, setcode, type, atk, def, level, race, attribute, category)``
and ``texts(id, name, desc, str1..str16)``. Packed fields:

* ``level``: bits 0-7 level/rank/link rating, bits 16-23 right pendulum scale,
  bits 24-31 left pendulum scale.
* ``setcode``: four 16-bit archetype codes (lowest first; 0 = empty slot).
* ``def``: for Link monsters this column stores the link-marker bitmask (DEF is 0).
* ``type`` / ``race`` / ``attribute``: bit flags, see ``ygorl.engine.constants``.
* ``str1..str16``: per-effect description strings; a script's
  ``aux.Stringid(code, n)`` / ``SetDescription`` refers to ``str{n+1}``.

Cards are keyed by their 8-digit ``password`` (the cdb ``id``), never by name.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from ygorl.engine import constants as C

N_STRINGS = 16
EXTRA_DECK_TYPES = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK

_FLAG_GROUPS = {"TYPE": {}, "ATTRIBUTE": {}, "RACE": {}, "LINK_MARKER": {}}
for _name in dir(C):
    for _prefix, _table in _FLAG_GROUPS.items():
        if _name.startswith(_prefix + "_") and _name not in ("RACE_ALL", "RACE_MAX", "ATTRIBUTE_ALL"):
            value = getattr(C, _name)
            if value and value & (value - 1) == 0:  # single-bit flags only
                _table[value] = _name[len(_prefix) + 1 :].lower()


def flag_names(value: int, group: str) -> list[str]:
    """Names of the single-bit ``group`` flags set in ``value`` (lowest bit first)."""
    table = _FLAG_GROUPS[group]
    return [table[bit] for bit in sorted(table) if value & bit]


@dataclass(frozen=True, slots=True)
class Card:
    password: int
    name: str
    desc: str
    strings: tuple[str, ...]
    alias: int
    ot: int
    setcodes: tuple[int, ...]
    type: int
    attack: int
    defense: int  # 0 for Link monsters (the cdb column holds link markers)
    level: int  # level, rank or link rating
    lscale: int
    rscale: int
    race: int
    attribute: int
    link_marker: int
    category: int

    # -- classification ---------------------------------------------------
    @property
    def is_monster(self) -> bool:
        return bool(self.type & C.TYPE_MONSTER)

    @property
    def is_spell(self) -> bool:
        return bool(self.type & C.TYPE_SPELL)

    @property
    def is_trap(self) -> bool:
        return bool(self.type & C.TYPE_TRAP)

    @property
    def is_token(self) -> bool:
        return bool(self.type & C.TYPE_TOKEN)

    @property
    def is_extra_deck(self) -> bool:
        return self.is_monster and bool(self.type & EXTRA_DECK_TYPES)

    @property
    def is_link(self) -> bool:
        return bool(self.type & C.TYPE_LINK)

    @property
    def is_xyz(self) -> bool:
        return bool(self.type & C.TYPE_XYZ)

    @property
    def is_pendulum(self) -> bool:
        return bool(self.type & C.TYPE_PENDULUM)

    @property
    def rank(self) -> int:
        return self.level if self.is_xyz else 0

    @property
    def link(self) -> int:
        return self.level if self.is_link else 0

    @property
    def type_names(self) -> list[str]:
        return flag_names(self.type, "TYPE")

    @property
    def attribute_names(self) -> list[str]:
        return flag_names(self.attribute, "ATTRIBUTE")

    @property
    def race_names(self) -> list[str]:
        return flag_names(self.race, "RACE")

    @property
    def link_marker_names(self) -> list[str]:
        return flag_names(self.link_marker, "LINK_MARKER")

    def effect_string(self, index: int) -> str:
        """``str{index+1}``, i.e. the text of ``aux.Stringid(password, index)``."""
        return self.strings[index]

    def to_core_tuple(self) -> tuple:
        """Fields in the order ``ygorl._core.CardDatabase.add`` / card sources expect."""
        return (
            self.password, self.alias, list(self.setcodes), self.type, self.level, self.attribute,
            self.race, self.attack, self.defense, self.lscale, self.rscale, self.link_marker,
        )  # fmt: skip


def decode_row(datas: tuple, texts: tuple | None) -> Card:
    """Build a :class:`Card` from a ``datas`` row and its (optional) ``texts`` row."""
    pid, ot, alias, setcode, type_, atk, def_, level, race, attribute, category = datas
    level = level or 0
    setcode = setcode or 0
    type_ = type_ or 0
    is_link = bool(type_ & C.TYPE_LINK)
    if texts is None:
        name, desc, strings = "", "", ("",) * N_STRINGS
    else:
        name, desc, *strs = texts[1:]
        strings = tuple((s or "") for s in strs[:N_STRINGS])
    return Card(
        password=pid,
        name=name or "",
        desc=desc or "",
        strings=strings,
        alias=alias or 0,
        ot=ot or 0,
        setcodes=tuple(c for c in ((setcode >> (16 * i)) & 0xFFFF for i in range(4)) if c),
        type=type_,
        attack=atk or 0,
        defense=0 if is_link else (def_ or 0),
        level=level & 0xFF,
        lscale=(level >> 24) & 0xFF,
        rscale=(level >> 16) & 0xFF,
        race=race or 0,
        attribute=attribute or 0,
        link_marker=(def_ or 0) if is_link else 0,
        category=category or 0,
    )


def read_cdb(path: str | Path) -> Iterator[Card]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"card database not found: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        texts = {row[0]: row for row in con.execute("SELECT * FROM texts")}
        for row in con.execute(
            "SELECT id, ot, alias, setcode, type, atk, def, level, race, attribute, category FROM datas"
        ):
            yield decode_row(row, texts.get(row[0]))
    finally:
        con.close()


class CardDB(Mapping[int, Card]):
    """Read-only password -> :class:`Card` table; later databases override earlier ones."""

    def __init__(self, cards: Iterable[Card] = ()) -> None:
        self._cards: dict[int, Card] = {}
        for card in cards:
            self._cards[card.password] = card
        self._core = None

    @classmethod
    def load(cls, *paths: str | Path) -> CardDB:
        if not paths:
            from ygorl.paths import cards_cdb

            paths = (cards_cdb(),)
        cards: list[Card] = []
        for p in paths:
            cards.extend(read_cdb(p))
        return cls(cards)

    def __getitem__(self, password: int) -> Card:
        return self._cards[password]

    def __iter__(self) -> Iterator[int]:
        return iter(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def canonical(self, password: int) -> int:
        """Resolve alternate artworks (``alias``) to the original password."""
        card = self._cards.get(password)
        if card is not None and card.alias and card.alias in self._cards:
            # Alternate arts have alias pointing at the original; rule variants reuse
            # alias too but are distinct cards if their type/stats differ.
            original = self._cards[card.alias]
            if original.name == card.name:
                return original.password
        return password

    def to_core(self):
        """A native ``ygorl._core.CardDatabase`` with every card (built once, cached)."""
        if self._core is None:
            from ygorl import _core

            db = _core.CardDatabase()
            for card in self._cards.values():
                db.add(*card.to_core_tuple())
            self._core = db
        return self._core

    def save_snapshot(self, path: str | Path) -> None:
        """Write every card as JSON (sorted by password) to pin card data to an environment."""
        rows = [asdict(self._cards[p]) for p in sorted(self._cards)]
        Path(path).write_text(json.dumps({"cards": rows}, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load_snapshot(cls, path: str | Path) -> CardDB:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Card(**{**row, "strings": tuple(row["strings"]), "setcodes": tuple(row["setcodes"])}) for row in data["cards"])


class CardVocab:
    """Stable password <-> index mapping for embeddings.

    Index 0 is padding, 1 is "unknown / face-down card"; real cards start at
    :data:`FIRST_INDEX`. :meth:`extend` only appends, so a vocab that is saved
    (:meth:`save`) and grown with ``from_db(db, base=saved)`` keeps every old
    index when the card database grows (important for warm-starting models,
    T6.3). ``from_db(db)`` without a base builds a *fresh* vocab in password
    order, whose indices shift when cards are added: persist the vocab next to
    anything trained on it (checkpoints, environment artifacts).
    """

    PAD = 0
    UNKNOWN = 1
    FIRST_INDEX = 2

    def __init__(self, passwords: Iterable[int] = ()) -> None:
        self._passwords: list[int] = []
        self._index: dict[int, int] = {}
        self.extend(passwords)

    def extend(self, passwords: Iterable[int]) -> None:
        for p in passwords:
            if p not in self._index:
                self._index[p] = len(self._passwords) + self.FIRST_INDEX
                self._passwords.append(p)

    def index(self, password: int) -> int:
        return self._index.get(password, self.UNKNOWN)

    def password(self, index: int) -> int:
        if index < self.FIRST_INDEX:
            raise KeyError(f"index {index} is a special token")
        return self._passwords[index - self.FIRST_INDEX]

    def __len__(self) -> int:
        """Embedding table size, including the special tokens."""
        return len(self._passwords) + self.FIRST_INDEX

    def __contains__(self, password: int) -> bool:
        return password in self._index

    @classmethod
    def from_db(cls, db: CardDB | Mapping[int, object], base: CardVocab | None = None) -> CardVocab:
        """All cards of ``db``: a copy of ``base`` with the missing passwords appended (in password
        order), or a fresh vocab in password order when ``base`` is None."""
        vocab = cls(base._passwords if base is not None else ())
        vocab.extend(sorted(db))
        return vocab

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"first_index": self.FIRST_INDEX, "passwords": self._passwords}))

    @classmethod
    def load(cls, path: str | Path) -> CardVocab:
        data = json.loads(Path(path).read_text())
        if data.get("first_index") != cls.FIRST_INDEX:
            raise ValueError(f"{path}: vocab layout mismatch")
        return cls(data["passwords"])
