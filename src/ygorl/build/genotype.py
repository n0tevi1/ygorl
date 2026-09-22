"""Deck genotypes, hard constraints and variation operators (T5.5).

A :class:`GenotypeSpace` fixes, for one environment, the cards a search may use
and how they are laid out; a :class:`Genotype` is one point of that space:

* **engine packages** -- a set of selected package ids (candidates from the
  synergy graph, :mod:`ygorl.build.packages`); only members of selected
  packages may appear, each with its own copy count;
* **generic slots** -- copy counts over a pool of generic cards (hand traps,
  board breakers, staples; the pool may tag each card with a role);
* **Extra Deck** -- copy counts over the Extra Deck monsters of the selected
  packages and of the generic pool.

All three live in one **count vector** ``counts`` over the space's fixed card
index (``passwords``): Main Deck cards first, then Extra Deck monsters, each block
sorted by password. The vector alone determines the deck (:meth:`GenotypeSpace.decode`);
the package set only says which cards are *allowed*. ``counts`` is the compact
numeric encoding for the surrogate model (T5.7) and for pyribs (T5.8):
:meth:`GenotypeSpace.vector` returns it, :meth:`GenotypeSpace.from_vector` maps
any real vector back to a legal genotype.

Hard constraints hold by construction: the index contains only cards of the pool
(alternate artworks folded into one entry, emitted as a password the pool
accepts), no tokens and no forbidden cards; ``cap[i] = min(max_copies, banlist
limit)``; Main vs. Extra Deck placement follows the card type. Every operator ends
in :meth:`GenotypeSpace.repair`, which clips counts to ``cap``, zeroes cards that
are not allowed, fills / trims the Main Deck into ``main_range`` (a sub-range of
the rules' 40-60) and fills the Extra Deck up to ``extra_size`` (<= 15), so every
genotype decodes to a deck that passes :func:`ygorl.cards.legality.validate_deck`.

Randomness only comes from an explicit ``numpy.random.Generator`` (or an int
seed), so every operator is deterministic given its seed. See docs/genotype.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ygorl.cards.legality import EXTRA_DECK_TYPES, DeckRules, canonical_password
from ygorl.cards.lflist import Banlist
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

OPERATORS = ("copies", "swap_generic", "add_package", "drop_package", "swap_extra")

RngLike = np.random.Generator | int | None


def as_rng(rng: RngLike) -> np.random.Generator:
    """``rng`` itself, or a fresh generator seeded with it."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class Genotype:
    """Selected package ids + a read-only int8 count vector over the space's card index."""

    __slots__ = ("packages", "counts")

    def __init__(self, packages: Iterable[int], counts: Any) -> None:
        self.packages: tuple[int, ...] = tuple(sorted({int(p) for p in packages}))
        c = np.array(counts, dtype=np.int8)
        c.flags.writeable = False
        self.counts: np.ndarray = c

    def key(self) -> bytes:
        """Hashable identity (packages and counts)."""
        return np.asarray(self.packages, dtype=np.int32).tobytes() + b"|" + self.counts.tobytes()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Genotype) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return f"Genotype(packages={self.packages}, cards={int(np.count_nonzero(self.counts))}, copies={int(self.counts.sum())})"


def _members(package: Any) -> Iterable[int]:
    return package.members if hasattr(package, "members") else package


class GenotypeSpace:
    """The cards, packages, caps and size ranges a deck search works in.

    ``cards`` maps password -> card (``name``, ``alias``, ``type``), e.g. a
    :class:`~ygorl.cards.cdb.CardDB`. ``packages`` are :class:`~ygorl.build.packages.Package`
    objects or password sequences; ``generic`` is an iterable of passwords or a
    mapping password -> role (``"hand_trap"``, ``"board_breaker"``, ...). Cards
    outside ``pool``, tokens and forbidden cards are dropped; a package keeps
    its remaining members if they are at least ``min_package_size`` and include
    a Main Deck card (identical restricted packages are kept once).

    Search-space knobs (not rules): ``min_packages`` / ``max_packages`` selected
    packages (``max_packages`` bounds sampling, ``add_package`` and crossover;
    repair exceeds it only when the Main Deck cannot reach its minimum otherwise),
    ``main_range`` (default ``(main_min, main_min + 5)``, inside the rules),
    ``extra_size`` (default the rules' ``extra_max``; the Extra Deck is always filled
    up to it as far as allowed cards permit) and ``max_generic`` distinct generic
    Main Deck cards when sampling.
    """

    def __init__(
        self,
        cards: Mapping[int, Any],
        packages: Iterable[Any],
        generic: Iterable[int] | Mapping[int, str] = (),
        *,
        banlist: Banlist | None = None,
        pool: frozenset[int] | None = None,
        rules: DeckRules = DeckRules(),
        min_packages: int = 1,
        max_packages: int = 3,
        main_range: tuple[int, int] | None = None,
        extra_size: int | None = None,
        max_generic: int = 12,
        min_package_size: int = 2,
        stamp: Mapping[str, str] | None = None,
    ) -> None:
        self.rules = rules
        self._cards = cards
        self.stamp = dict(stamp) if stamp is not None else None
        entries: dict[int, tuple[int, int, bool]] = {}  # canonical -> (emitted password, cap, is_extra)

        def resolve(pw: int) -> int | None:
            pw = int(pw)
            card = cards.get(pw)
            if card is None or card.type & C.TYPE_TOKEN:
                return None
            canon = canonical_password(pw, cards)
            if canon in entries:
                return canon
            emit = canon if pool is None or canon in pool else pw if pw in pool else None
            limit = rules.max_copies if banlist is None else min(rules.max_copies, banlist.limit(canon))
            if emit is None or limit <= 0:
                return None
            kind = cards[canon].type
            entries[canon] = (emit, limit, bool(kind & C.TYPE_MONSTER and kind & EXTRA_DECK_TYPES))
            return canon

        roles_in = dict(generic) if isinstance(generic, Mapping) else {pw: None for pw in generic}
        generic_roles: dict[int, str | None] = {}
        for pw, role in roles_in.items():
            canon = resolve(pw)
            if canon is not None and canon not in generic_roles:
                generic_roles[canon] = role

        kept: list[tuple[int, frozenset[int]]] = []
        seen: set[frozenset[int]] = set()
        for n, pkg in enumerate(packages):
            members = frozenset(c for c in (resolve(pw) for pw in _members(pkg)) if c is not None)
            if len(members) < min_package_size or all(entries[c][2] for c in members) or members in seen:
                continue
            seen.add(members)
            kept.append((n, members))

        used = set(generic_roles).union(*(m for _, m in kept))
        order = sorted(used, key=lambda c: (entries[c][2], entries[c][0]))
        self._index = {c: i for i, c in enumerate(order)}
        self.passwords: tuple[int, ...] = tuple(entries[c][0] for c in order)
        self._emit = np.array(self.passwords, dtype=np.int64)
        self.cap = np.array([entries[c][1] for c in order], dtype=np.int8)
        self.is_extra = np.array([entries[c][2] for c in order], dtype=bool)
        self.n_main = int(np.count_nonzero(~self.is_extra))
        self.is_generic = np.zeros(len(order), dtype=bool)
        self.is_generic[[self._index[c] for c in generic_roles]] = True
        self.roles: dict[int, str] = {self._index[c]: r for c, r in generic_roles.items() if r is not None}
        self.packages: tuple[tuple[int, ...], ...] = tuple(tuple(sorted(entries[c][0] for c in m)) for _, m in kept)
        self.package_source: tuple[int, ...] = tuple(n for n, _ in kept)
        self._pkg_idx = [np.array(sorted(self._index[c] for c in m), dtype=np.int64) for _, m in kept]
        self._pkg_mask = np.zeros((len(kept), len(order)), dtype=bool)
        for p, idx in enumerate(self._pkg_idx):
            self._pkg_mask[p, idx] = True

        lo, hi = main_range if main_range is not None else (rules.main_min, min(rules.main_max, rules.main_min + 5))
        if not rules.main_min <= lo <= hi <= rules.main_max:
            raise ValueError(f"main_range {(lo, hi)} is outside the rules' range {(rules.main_min, rules.main_max)}")
        self.main_range = (int(lo), int(hi))
        self.extra_size = rules.extra_max if extra_size is None else int(extra_size)
        if not 0 <= self.extra_size <= rules.extra_max:
            raise ValueError(f"extra_size {self.extra_size} is outside the rules' range (0, {rules.extra_max})")
        if not 0 <= min_packages <= max_packages:
            raise ValueError(f"need 0 <= min_packages <= max_packages, got {min_packages}, {max_packages}")
        if len(kept) < min_packages:
            raise ValueError(f"{len(kept)} engine package(s) left after restricting to the pool; need {min_packages}")
        self.min_packages, self.max_packages = min_packages, max_packages
        capacity = int(self.cap[: self.n_main].sum(dtype=np.int64))
        if capacity < lo:
            raise ValueError(f"Main Deck capacity of the space is {capacity} copies, below the minimum {lo}")
        self.max_generic = max_generic
        self._generic_main = np.flatnonzero(self.is_generic & ~self.is_extra)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "passwords": self.passwords, "cap": self.cap.tolist(), "generic": self.is_generic.tolist(),
                    "roles": sorted(self.roles.items()), "packages": self.packages, "main_range": self.main_range,
                    "extra_size": self.extra_size, "packages_range": (min_packages, max_packages),
                    "max_generic": max_generic, "rules": dataclasses.asdict(rules),
                }
            ).encode()
        ).hexdigest()  # fmt: skip

    @classmethod
    def from_environment(
        cls,
        env: Any,
        cards: Mapping[int, Any],
        packages: Iterable[Any],
        generic: Iterable[int] | Mapping[int, str] = (),
        **kw: Any,
    ) -> GenotypeSpace:
        """Space bound to an :class:`~ygorl.data.Environment` (pool, banlist, deck rules, stamp)."""
        env_kw = {"banlist": env.banlist, "pool": env.card_pool, "rules": env.deck_rules, "stamp": env.stamp()}
        return cls(cards, packages, generic, **env_kw, **kw)

    # -- index -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.passwords)

    def index_of(self, password: int) -> int:
        """Vector index of ``password`` (alternate artworks resolve to their original's entry)."""
        pw = int(password)
        i = self._index.get(pw)
        if i is None:
            i = self._index.get(canonical_password(pw, self._cards))
        if i is None:
            raise KeyError(f"card {pw} is not in this genotype space")
        return i

    def package_indices(self, package: int) -> np.ndarray:
        return self._pkg_idx[package]

    def allowed(self, packages: Iterable[int]) -> np.ndarray:
        """Cards that may have copies given the selected ``packages``."""
        pk = list(packages)
        return self.is_generic | self._pkg_mask[pk].any(axis=0) if pk else self.is_generic.copy()

    def role_counts(self, g: Genotype) -> dict[str, int]:
        """Copies per generic role (e.g. the hand-trap count descriptor of T5.8)."""
        out: dict[str, int] = {}
        for i, role in sorted(self.roles.items()):
            out[role] = out.get(role, 0) + int(g.counts[i])
        return out

    # -- phenotype ---------------------------------------------------------------
    def decode(self, g: Genotype, name: str = "") -> Deck:
        """The deck: Main / Extra Deck by card type, cards sorted by password, empty side deck."""
        c = g.counts.astype(np.int64)
        main = np.repeat(self._emit[: self.n_main], c[: self.n_main])
        extra = np.repeat(self._emit[self.n_main :], c[self.n_main :])
        return Deck(tuple(int(p) for p in main), tuple(int(p) for p in extra), (), name=name)

    def vector(self, g: Genotype) -> np.ndarray:
        """The count vector (int8, one entry per card of ``passwords``)."""
        return g.counts.copy()

    def from_vector(self, x: Any, rng: RngLike = None) -> Genotype:
        """A legal genotype for any real vector ``x``: round, clip, infer packages, repair.

        A package counts as selected when one of its non-generic members has copies.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(self),):
            raise ValueError(f"expected a vector of length {len(self)}, got shape {x.shape}")
        counts = np.clip(np.rint(np.nan_to_num(x)), 0, self.cap).astype(np.int16)
        present = (counts > 0) & ~self.is_generic
        packages = np.flatnonzero((self._pkg_mask & present).any(axis=1)) if len(self._pkg_idx) else []
        return self.repair(packages, counts, rng)

    def from_deck(self, deck: Deck, rng: RngLike = None) -> Genotype:
        """Encode an existing deck (Main + Extra Deck; cards outside the space are ignored), then repair."""
        counts = np.zeros(len(self), dtype=np.int16)
        for pw in deck.main + deck.extra:
            try:
                counts[self.index_of(pw)] += 1
            except KeyError:
                continue
        return self.from_vector(counts, rng)

    # -- constraints -------------------------------------------------------------
    def check(self, g: Genotype) -> list[str]:
        """Violated genotype invariants (empty = valid); a valid genotype decodes to a legal deck."""
        out: list[str] = []
        c = g.counts.astype(np.int64)
        if c.shape != (len(self),):
            return [f"count vector has shape {c.shape}, expected ({len(self)},)"]
        if any(not 0 <= p < len(self._pkg_idx) for p in g.packages):
            return [f"unknown package id in {g.packages}"]
        if len(g.packages) < self.min_packages:
            out.append(f"{len(g.packages)} packages selected, minimum {self.min_packages}")
        if (c < 0).any() or (c > self.cap).any():
            out.append("copy count outside [0, cap]")
        allowed = self.allowed(g.packages)
        if (c[~allowed] > 0).any():
            out.append("copies of a card that is neither generic nor in a selected package")
        main = int(c[: self.n_main].sum())
        if not self.main_range[0] <= main <= self.main_range[1]:
            out.append(f"Main Deck has {main} cards, outside {self.main_range}")
        extra = int(c[self.n_main :].sum())
        capacity = int(self.cap[self.n_main :][allowed[self.n_main :]].sum(dtype=np.int64))
        if extra != min(self.extra_size, capacity):
            out.append(f"Extra Deck has {extra} cards, expected {min(self.extra_size, capacity)}")
        return out

    def repair(self, packages: Iterable[int], counts: Any, rng: RngLike = None, main_target: int | None = None) -> Genotype:
        """Nearest valid genotype: clip to caps, drop disallowed cards, fit the deck sizes.

        The Main Deck is trimmed / filled (uniformly over copies / free copy slots of
        allowed cards) to ``main_target`` clamped into ``main_range`` (default: its
        current size clamped). If the allowed cards cannot reach the minimum, random
        unselected packages are added. The Extra Deck is trimmed to ``extra_size`` or
        filled towards it, one copy of each absent allowed card first. A valid genotype
        is returned unchanged without consuming randomness.
        """
        rng = as_rng(rng)
        pk = sorted({int(p) for p in packages})
        n_pkg = len(self._pkg_idx)
        if len(pk) < self.min_packages:
            free = [p for p in range(n_pkg) if p not in pk]
            pk = sorted(pk + [int(p) for p in rng.choice(free, self.min_packages - len(pk), replace=False)])
        c = np.clip(np.asarray(counts, dtype=np.int16), 0, self.cap).astype(np.int16)
        allowed = self.allowed(pk)
        c[~allowed] = 0
        main = ~self.is_extra
        lo, hi = self.main_range
        cur = int(c[main].sum())
        target = min(max(cur if main_target is None else int(main_target), lo), hi)
        room = int((self.cap - c)[allowed & main].sum())
        while cur + room < target:
            cand = [p for p in range(n_pkg) if p not in pk and (self._pkg_mask[p] & main & ~allowed).any()]
            if not cand:
                target = cur + room
                break
            p = int(rng.choice(cand))
            pk = sorted(pk + [p])
            allowed = allowed | self._pkg_mask[p]
            room = int((self.cap - c)[allowed & main].sum())
        self._fit(c, np.flatnonzero(allowed & main), target, rng)
        self._fit(c, np.flatnonzero(allowed & self.is_extra), self.extra_size, rng, distinct_first=True)
        return Genotype(pk, c)

    def _fit(self, c: np.ndarray, idx: np.ndarray, target: int, rng: np.random.Generator, distinct_first: bool = False) -> None:
        cur = int(c[idx].sum())
        if cur > target:
            copies = np.repeat(idx, c[idx])
            drop = rng.choice(len(copies), cur - target, replace=False)
            np.subtract.at(c, copies[drop], 1)
            return
        need = target - cur
        if need <= 0:
            return
        if distinct_first:
            absent = idx[c[idx] == 0]
            k = min(need, len(absent))
            if k:
                c[rng.choice(absent, k, replace=False)] += 1
                need -= k
        room = self.cap[idx].astype(np.int64) - c[idx]
        slots = np.repeat(idx, room)
        k = min(need, len(slots))
        if k:
            np.add.at(c, slots[rng.choice(len(slots), k, replace=False)], 1)

    # -- sampling ----------------------------------------------------------------
    def _seed_package(self, c: np.ndarray, p: int, rng: np.random.Generator) -> None:
        """Give the absent members of package ``p`` copies: Main Deck 1..cap, Extra Deck 1."""
        idx = self._pkg_idx[p]
        idx = idx[c[idx] == 0]
        is_x = self.is_extra[idx]
        c[idx[is_x]] = 1
        m = idx[~is_x]
        c[m] = rng.integers(1, self.cap[m].astype(np.int64) + 1)

    def sample(self, rng: RngLike = None) -> Genotype:
        """A random genotype: 1..max_packages packages, 0..max_generic generic cards, a random Main Deck size."""
        rng = as_rng(rng)
        n_pkg = len(self._pkg_idx)
        k = int(rng.integers(self.min_packages, min(self.max_packages, n_pkg) + 1))
        pk = sorted(int(p) for p in rng.choice(n_pkg, k, replace=False))
        c = np.zeros(len(self), dtype=np.int16)
        for p in pk:
            self._seed_package(c, p, rng)
        gm = self._generic_main[c[self._generic_main] == 0]
        n_gen = int(rng.integers(0, min(self.max_generic, len(gm)) + 1))
        if n_gen:
            pick = rng.choice(gm, n_gen, replace=False)
            c[pick] = rng.integers(1, self.cap[pick].astype(np.int64) + 1)
        target = int(rng.integers(self.main_range[0], self.main_range[1] + 1))
        return self.repair(pk, c, rng, main_target=target)

    def sample_many(self, n: int, rng: RngLike = None) -> list[Genotype]:
        rng = as_rng(rng)
        return [self.sample(rng) for _ in range(n)]

    # -- variation ---------------------------------------------------------------
    def mutate(self, g: Genotype, rng: RngLike = None, n_ops: int = 1, ops: Sequence[str] | None = None) -> Genotype:
        """Apply ``n_ops`` operators drawn uniformly from ``ops`` (default :data:`OPERATORS`), repairing after each.

        * ``copies``: set a random allowed card to a new copy count in ``0..cap`` (the Main Deck size may drift);
        * ``swap_generic``: replace every copy of a generic Main Deck card by an absent generic card;
        * ``add_package``: select an unselected package and seed its members (at ``max_packages``, replace one);
        * ``drop_package``: deselect a package (at ``min_packages``, replace it by an unselected one);
        * ``swap_extra``: replace an Extra Deck card by an absent allowed one.

        Package and swap operators keep the Main Deck size. An operator that cannot apply
        (nothing to swap, no unselected package) falls back to ``copies``.
        """
        rng = as_rng(rng)
        ops = tuple(OPERATORS if ops is None else ops)
        unknown = set(ops) - set(OPERATORS)
        if unknown or not ops:
            raise ValueError(f"unknown operators {sorted(unknown)}; expected a subset of {OPERATORS}")
        for _ in range(n_ops):
            op = ops[int(rng.integers(len(ops)))]
            pk, c = list(g.packages), g.counts.astype(np.int16)
            size = int(c[: self.n_main].sum())
            result = getattr(self, f"_op_{op}")(pk, c, rng)
            if result is None:
                result = self._op_copies(pk, c, rng)
            pk, c, keep_size = result
            g = self.repair(pk, c, rng, main_target=size if keep_size else None)
        return g

    def _op_copies(self, pk: list[int], c: np.ndarray, rng: np.random.Generator):
        idx = np.flatnonzero(self.allowed(pk) & (self.cap > 0))
        i = int(rng.choice(idx))
        values = [v for v in range(int(self.cap[i]) + 1) if v != c[i]]
        c[i] = values[int(rng.integers(len(values)))]
        return pk, c, False

    def _op_swap_generic(self, pk: list[int], c: np.ndarray, rng: np.random.Generator):
        gm = self._generic_main
        absent, present = gm[c[gm] == 0], gm[c[gm] > 0]
        if not len(absent):
            return None
        j = int(rng.choice(absent))
        if len(present):
            i = int(rng.choice(present))
            c[j], c[i] = min(int(c[i]), int(self.cap[j])), 0
        else:
            c[j] = int(rng.integers(1, int(self.cap[j]) + 1))
        return pk, c, True

    def _op_add_package(self, pk: list[int], c: np.ndarray, rng: np.random.Generator):
        free = [p for p in range(len(self._pkg_idx)) if p not in pk]
        if not free:
            return None
        out = list(pk)
        if len(out) >= self.max_packages and out:  # at the maximum: replace a package
            out.pop(int(rng.integers(len(out))))
        p = int(rng.choice(free))
        self._seed_package(c, p, rng)
        return out + [p], c, True

    def _op_drop_package(self, pk: list[int], c: np.ndarray, rng: np.random.Generator):
        out = list(pk)
        if not out:
            return None
        out.pop(int(rng.integers(len(out))))
        if len(out) < self.min_packages:  # at the minimum: replace the package instead
            free = [p for p in range(len(self._pkg_idx)) if p not in pk]
            if not free:
                return None
            p = int(rng.choice(free))
            self._seed_package(c, p, rng)
            out.append(p)
        return out, c, True

    def _op_swap_extra(self, pk: list[int], c: np.ndarray, rng: np.random.Generator):
        ex = np.flatnonzero(self.allowed(pk) & self.is_extra)
        absent, present = ex[c[ex] == 0], ex[c[ex] > 0]
        if not len(absent) or not len(present):
            return None
        i, j = int(rng.choice(present)), int(rng.choice(absent))
        c[j], c[i] = min(int(c[i]), int(self.cap[j])), 0
        return pk, c, True

    def crossover(self, a: Genotype, b: Genotype, rng: RngLike = None) -> Genotype:
        """Package-level crossover.

        Packages selected by both parents are kept; each package of only one parent
        joins with probability 1/2 (up to ``max_packages``). A kept package brings its
        members' copy counts from one parent (the parent that selected it; a random one
        if both did). Generic cards take their count from a random parent each
        (uniform crossover), the Main Deck size comes from a random parent.
        """
        rng = as_rng(rng)
        sa, sb = set(a.packages), set(b.packages)
        chosen = sorted(sa & sb)
        only = sorted(sa ^ sb)
        opt = [p for p, u in zip(only, rng.random(len(only))) if u < 0.5]
        room = max(self.max_packages - len(chosen), 0)
        if len(opt) > room:
            opt = sorted(int(p) for p in rng.choice(opt, room, replace=False))
        chosen += opt
        if len(chosen) < self.min_packages:
            rest = [p for p in only if p not in chosen]
            chosen += [int(p) for p in rng.choice(rest, min(self.min_packages - len(chosen), len(rest)), replace=False)]
        c = np.where(rng.random(len(self)) < 0.5, a.counts, b.counts).astype(np.int16)
        c[~self.is_generic] = 0
        for p in sorted(chosen):
            src = a if p in sa and (p not in sb or rng.random() < 0.5) else b
            idx = self._pkg_idx[p]
            c[idx] = src.counts[idx]
        sizes = (int(a.counts[: self.n_main].sum()), int(b.counts[: self.n_main].sum()))
        return self.repair(chosen, c, rng, main_target=sizes[int(rng.integers(2))])

    # -- persistence -------------------------------------------------------------
    def genotype_to_json(self, g: Genotype) -> dict[str, Any]:
        """``{"space": fingerprint, "packages": [...], "cards": {password: copies}}``."""
        nz = np.flatnonzero(g.counts)
        cards = {str(self.passwords[i]): int(g.counts[i]) for i in nz}
        return {"space": self.fingerprint, "packages": list(g.packages), "cards": cards}

    def genotype_from_json(self, data: Mapping[str, Any]) -> Genotype:
        if data.get("space") != self.fingerprint:
            got = str(data.get("space"))[:12]
            raise ValueError(f"genotype belongs to another genotype space ({got} != {self.fingerprint[:12]})")
        counts = np.zeros(len(self), dtype=np.int16)
        for pw, n in data["cards"].items():
            counts[self.index_of(int(pw))] = n
        return Genotype(data["packages"], counts)
