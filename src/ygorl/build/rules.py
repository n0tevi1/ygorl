"""Association rules over the deck corpus: a candidate source for deck evolution (#140, docs/tuning.md「关联规则候选」).

"Lists that run A (and B) also run C": support, confidence and lift of such rules, mined from the environment's deck
corpus (``artifacts/deck_corpus.json``). M3 showed that statistical signals do not predict a card's true value, so
the rules only *propose* edits; which proposals pay off is left to Thompson allocation and paired evaluation.

- :class:`DeckRules`: the corpus as a list × card presence matrix. Rules have one or two antecedent cards (no full
  Apriori: pairs are enough, and they are computed per query, only for antecedents in the queried deck). A type's
  lists are used alone when the type has at least ``min_type_lists`` of them; otherwise (md-2026-09 has at most 3
  lists per type) the lists of the queried cards' **archetype** when the rules know the cards' setcodes (#145: lists
  with at least ``archetype_min_cards`` distinct cards of a base setcode that as many of the queried cards carry;
  the global rules proposed other engines' cards), else all lists (the global fallback). Type-conditional
  frequencies ("``k`` of the type's ``n`` lists run C") are rules with an empty antecedent.
  :meth:`DeckRules.from_environment` caches per environment stamp.
- :func:`rule_children`: children of a parent deck whose additions are cards the rules predict, the parent lacks
  and the addition pool allows (``allowed``, #145), one copy each, each paired with the removal the card-value model
  draws lowest (never a protected card, never an antecedent of the rules that proposed the bundle; the parent's
  engine members after every other card), 1 to ``max_bundle`` edits a child, repaired to legality.
- :meth:`DeckRules.complete`: package completion for new-build exploration (#113): a partial package's usual
  missing members.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ygorl.build.deck_engine import archetypes
from ygorl.build.signals import CardValueModel, Child
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import Deck, load_ydk

GLOBAL = "*"  # the stratum name of rules mined over every list
GENERATOR = "rules"  # the child kind (lineage ``generator``)


@dataclass(frozen=True)
class Rule:
    """``antecedent`` (0-2 cards; empty: the stratum's type itself) → ``consequent``, over the ``lists`` lists of
    ``stratum`` (a deck type, or :data:`GLOBAL`). ``count`` lists contain antecedent and consequent;
    support = count / lists, confidence = count / lists with the antecedent, lift = confidence / P(consequent)
    (P over the stratum's lists; for a type's frequency rule P over every list, so its lift is how much more the type
    runs the card than lists at large)."""

    antecedent: tuple[int, ...]
    consequent: int
    count: int
    support: float
    confidence: float
    lift: float
    stratum: str
    lists: int


@dataclass(frozen=True)
class RuleConfig:
    min_count: int = 3  # lists containing antecedent and consequent
    min_support: float = 0.0  # share of the stratum's lists (min_count usually binds first)
    min_confidence: float = 0.5
    min_lift: float = 2.0
    min_type_lists: int = 20  # a type with fewer lists uses its archetype's lists (or the global rules)
    min_type_count: int = 2  # a type-conditional frequency needs this many of the type's lists
    archetype_min_cards: int = 2  # distinct cards of a base setcode that make it an archetype of a deck / list


@dataclass(frozen=True)
class Proposal:
    """A card proposed for a deck: its most confident rule and how many of the deck's cards back it."""

    rule: Rule
    evidence: int

    @property
    def weight(self) -> float:
        """How strongly the rules propose the card: confidence × evidence (a card that one generic card predicts
        ranks below one most of the deck's engine predicts)."""
        return self.rule.confidence * self.evidence


_CACHE: dict[tuple, DeckRules] = {}


class DeckRules:
    """Association rules over ``lists`` (``(deck type, deck)`` pairs); card presence, not copies, defines the rules.
    ``setcodes`` (``password -> setcodes``, e.g. :func:`ygorl.build.packages.setcodes_from_db`) enables the
    archetype strata."""

    def __init__(self, lists: Sequence[tuple[str, Deck]], config: RuleConfig | None = None, *,
                 setcodes: Mapping[int, Iterable[int]] | None = None) -> None:  # fmt: skip
        self.config = config or RuleConfig()
        self.setcodes = {int(c): tuple(v) for c, v in setcodes.items()} if setcodes is not None else None
        index: dict[int, int] = {}
        rows = []
        for _, deck in lists:
            rows.append({index.setdefault(c, len(index)): n for c, n in deck.counts().items()})
        self.cards = np.array(list(index), dtype=np.int64)
        self.col = index
        self.copies = np.zeros((len(lists), len(index)), dtype=np.int8)
        for i, row in enumerate(rows):
            for j, n in row.items():
                self.copies[i, j] = n
        self.present = (self.copies > 0).astype(np.float32)
        self.types = np.array([t for t, _ in lists], dtype=object)
        self.count = self.present.sum(0)  # lists per card
        self.p = self.count / max(len(lists), 1)
        self.bases = [{sc & 0xFFF for sc in (self.setcodes or {}).get(int(c), ())} for c in self.cards]

    def __len__(self) -> int:
        return len(self.types)

    @classmethod
    def from_environment(cls, env, config: RuleConfig | None = None, *,
                         setcodes: Mapping[int, Iterable[int]] | None = None) -> DeckRules:  # fmt: skip
        """Rules over ``env``'s deck corpus, cached per environment stamp (and corpus file, settings, and whether
        ``setcodes`` are given)."""
        path = Path(env.artifact_path("deck_corpus.json"))
        st = path.stat()
        key = (json.dumps(env.stamp(), sort_keys=True), str(path), st.st_mtime_ns, st.st_size, config or RuleConfig(),
               setcodes is not None)  # fmt: skip
        if key not in _CACHE:
            corpus = json.loads(path.read_text())
            _CACHE[key] = cls([(e["type"], load_ydk(path.parent / e["file"])) for e in corpus["decks"]], config,
                              setcodes=setcodes)  # fmt: skip
        return _CACHE[key]

    # -- strata
    def stratum(self, deck_type: str | None, cards: Iterable[int] | None = None) -> tuple[str, np.ndarray]:
        """(name, row mask) of the lists rules for ``deck_type`` are mined over: the type's own lists when it has
        ``min_type_lists`` of them; else, with setcodes and ``cards`` (the queried deck or package), the lists of
        their archetypes (:meth:`archetype_rows`) when there are any; else every list."""
        if deck_type is not None:
            rows = self.types == deck_type
            if rows.sum() >= max(self.config.min_type_lists, 1):
                return deck_type, rows
        if cards is not None and self.setcodes is not None:
            arch = archetypes(cards, self.setcodes, min_cards=self.config.archetype_min_cards)
            if arch:
                rows = self.archetype_rows(arch)
                if rows.any():
                    return "archetype:" + "+".join(f"{b:#x}" for b in sorted(arch)), rows
        return GLOBAL, np.ones(len(self), dtype=bool)

    def archetype_rows(self, bases: Iterable[int]) -> np.ndarray:
        """Lists with at least ``archetype_min_cards`` distinct cards carrying one of the base setcodes ``bases``."""
        want = set(bases)
        cols = [j for j, b in enumerate(self.bases) if b & want]
        if not cols:
            return np.zeros(len(self), dtype=bool)
        return self.present[:, cols].sum(1) >= self.config.archetype_min_cards

    def frequencies(self, deck_type: str) -> dict[int, float]:
        """Type-conditional card frequencies: share of ``deck_type``'s lists that run each card (empty for an unknown
        type)."""
        rows = self.types == deck_type
        n = int(rows.sum())
        if n == 0:
            return {}
        cnt = self.present[rows].sum(0)
        return {int(self.cards[j]): float(cnt[j] / n) for j in np.flatnonzero(cnt)}

    def typical_copies(self, card: int, deck_type: str | None = None) -> int:
        """Median copies of ``card`` in the lists that run it (the type's lists when some do), 1 for an unseen card."""
        j = self.col.get(card)
        if j is None:
            return 1
        c = self.copies[:, j]
        if deck_type is not None and (c[self.types == deck_type] > 0).any():
            c = c[self.types == deck_type]
        c = c[c > 0]
        return int(np.clip(round(float(np.median(c))), 1, 3))

    # -- rules
    def _keep(self, count: np.ndarray, confidence: np.ndarray, lift: np.ndarray, n: int) -> np.ndarray:
        cfg = self.config
        return ((count >= cfg.min_count) & (count / max(n, 1) >= cfg.min_support) & (confidence >= cfg.min_confidence)
                & (lift >= cfg.min_lift))  # fmt: skip

    def rules(self, cards: Iterable[int], deck_type: str | None = None, *, limit: int | None = None) -> list[Rule]:
        """Rules whose antecedent is one or two of ``cards`` and whose consequent is not among them (a pair rule only
        when it is more confident than both of its one-card rules), plus ``deck_type``'s frequency rules; sorted by
        lift, then confidence."""
        have = {int(c) for c in cards}
        name, rows = self.stratum(deck_type, have)
        m = self.present[rows]
        n = len(m)
        out = self._type_rules(deck_type, have)
        ante = [self.col[c] for c in sorted(have) if c in self.col]
        ante = [a for a in ante if m[:, a].sum() >= self.config.min_count]
        if not ante or n == 0:
            return self._sorted(out, limit)
        cons = np.flatnonzero((m.sum(0) >= self.config.min_count) & ~np.isin(self.cards, list(have)))
        if not len(cons):
            return self._sorted(out, limit)
        p = m[:, cons].sum(0) / n  # P(consequent) in the stratum
        ma, mc = m[:, ante], m[:, cons]
        n_a = ma.sum(0)
        n_ac = ma.T @ mc  # (antecedents, consequents)
        conf1 = n_ac / n_a[:, None]
        for i, j in zip(*np.nonzero(self._keep(n_ac, conf1, conf1 / p, n)), strict=True):
            out.append(self._rule((ante[i],), cons[j], n_ac[i, j], conf1[i, j], p[j], name, n))
        pairs = [(x, y) for x in range(len(ante)) for y in range(x + 1, len(ante))]
        if pairs:
            xs, ys = np.array(pairs).T
            both = ma[:, xs] * ma[:, ys]
            n_ab = both.sum(0)
            ok = n_ab >= self.config.min_count
            xs, ys, both, n_ab = xs[ok], ys[ok], both[:, ok], n_ab[ok]
            n_abc = both.T @ mc
            with np.errstate(invalid="ignore", divide="ignore"):
                conf2 = n_abc / n_ab[:, None]
            better = (conf2 > conf1[xs]) & (conf2 > conf1[ys])
            for k, j in zip(*np.nonzero(self._keep(n_abc, conf2, conf2 / p, n) & better), strict=True):
                out.append(self._rule((ante[xs[k]], ante[ys[k]]), cons[j], n_abc[k, j], conf2[k, j], p[j], name, n))
        return self._sorted(out, limit)

    def _rule(self, ante_cols: tuple[int, ...], col: int, count, confidence, p, stratum: str, n: int) -> Rule:
        conf = float(confidence)
        return Rule(tuple(sorted(int(self.cards[a]) for a in ante_cols)), int(self.cards[col]), int(round(float(count))),
                    float(count) / n, conf, conf / float(p), stratum, n)  # fmt: skip

    def _type_rules(self, deck_type: str | None, have: set[int]) -> list[Rule]:
        if deck_type is None:
            return []
        rows = self.types == deck_type
        n = int(rows.sum())
        if n == 0:
            return []
        cnt = self.present[rows].sum(0)
        conf = cnt / n
        keep = (cnt >= self.config.min_type_count) & (conf >= self.config.min_confidence)
        keep &= (conf / np.maximum(self.p, 1e-12) >= self.config.min_lift) & ~np.isin(self.cards, list(have))
        return [self._rule((), j, cnt[j], conf[j], self.p[j], deck_type, n) for j in np.flatnonzero(keep)]

    @staticmethod
    def _sorted(rules: list[Rule], limit: int | None) -> list[Rule]:
        rules.sort(key=lambda r: (-r.lift, -r.confidence, -r.count, r.antecedent, r.consequent))
        return rules if limit is None else rules[:limit]

    def additions(self, deck: Deck, deck_type: str | None = None) -> dict[int, Proposal]:
        """Per card the rules predict for ``deck`` but ``deck`` lacks: its most confident rule (ties: higher lift) and
        its evidence, the number of the deck's cards (the type counting as one) in the antecedents of its rules."""
        best: dict[int, Rule] = {}
        evidence: dict[int, set] = {}
        for r in self.rules(set(deck.main) | set(deck.extra), deck_type):
            b = best.get(r.consequent)
            if b is None or (r.confidence, r.lift) > (b.confidence, b.lift):
                best[r.consequent] = r
            evidence.setdefault(r.consequent, set()).update(r.antecedent or (None,))
        return {c: Proposal(r, len(evidence[c])) for c, r in best.items()}

    def complete(self, package: Iterable[int], deck_type: str | None = None, *,
                 limit: int | None = None) -> list[Rule]:  # fmt: skip
        """A partial package's usual missing members (#113): the cards most lists running the whole ``package`` also
        run (one rule with the package as antecedent); when fewer than ``min_count`` lists run all of it, the best
        one- or two-card rule per missing card. Sorted by confidence, then lift."""
        pkg = {int(c) for c in package}
        name, rows = self.stratum(deck_type, pkg)
        cols = [self.col.get(c) for c in pkg]
        n = int(rows.sum())
        if pkg and None not in cols and n:
            has = rows & (self.present[:, cols].min(1) > 0)
            n_x = int(has.sum())
            if n_x >= self.config.min_count:
                cnt = self.present[has].sum(0)
                conf = cnt / n_x
                lift = conf / np.maximum(self.present[rows].sum(0) / n, 1e-12)  # P(consequent) in the stratum
                keep = self._keep(cnt, conf, lift, n) & ~np.isin(self.cards, list(pkg))
                out = [Rule(tuple(sorted(pkg)), int(self.cards[j]), int(cnt[j]), float(cnt[j] / n), float(conf[j]),
                            float(lift[j]), name, n) for j in np.flatnonzero(keep)]  # fmt: skip
                out.sort(key=lambda r: (-r.confidence, -r.lift, r.consequent))
                return out if limit is None else out[:limit]
        best: dict[int, Rule] = {}
        for r in self.rules(pkg, deck_type):
            if r.antecedent and (r.consequent not in best or r.confidence > best[r.consequent].confidence):
                best[r.consequent] = r
        out = sorted(best.values(), key=lambda r: (-r.confidence, -r.lift, r.consequent))
        return out if limit is None else out[:limit]


def rule_children(base: Deck, deck_type: str, rules: DeckRules, model: CardValueModel, *,
                  legal: Callable[[Deck], bool], is_extra: Callable[[int], bool], rng: np.random.Generator,
                  children: int = 2, max_bundle: int = 3, protected: Iterable[int] = (),
                  avoid: Iterable[Deck] = (), allowed: Iterable[int] | None = None, copies: int = 1,
                  engine: Iterable[int] = ()) -> list[Child]:  # fmt: skip
    """Up to ``children`` children of ``base`` whose additions come from the association rules, distinct from each
    other, from ``base`` and from ``avoid``. Only cards in ``allowed`` (the engine-aware addition pool,
    :func:`ygorl.build.deck_engine.addition_pool`; every card when None) are proposed. Per child: draw a bundle size
    (1..``max_bundle``) and an order of the proposed cards (Gumbel keys on the log of :attr:`Proposal.weight`: the
    better-backed card tends to go first, repeated calls spread over the proposals); each proposed card goes in its
    typical number of copies, at most ``copies`` (one by default, #145: two copies of an off-engine card took out
    two copies of a core card), while the bundle has room, each copy taking out the card of the same section the
    card-value model draws lowest (one Thompson draw per child; ``engine`` cards only after every other card).
    Protected cards and the antecedents of the bundle's rules are never taken out, nor a card the bundle put in.
    Legality repair: a swap that leaves the deck illegal tries the next-lowest removal; a card no removal makes legal
    is dropped. Children are kind ``"rules"``, predicted gain = the model's mean gain of the bundle."""
    proposals = rules.additions(base, deck_type)
    if allowed is not None:
        ok = set(allowed)
        proposals = {c: p for c, p in proposals.items() if c in ok}
    if not proposals or children <= 0:
        return []
    protected = set(protected)
    engine = set(engine)
    seen = {_key(base), *(_key(d) for d in avoid)}
    order_cards = list(proposals)
    logw = np.log([proposals[c].weight for c in order_cards])
    out: list[Child] = []
    for _ in range(children * 4):
        if len(out) >= children:
            break
        size = int(rng.integers(1, max_bundle + 1))
        order = [order_cards[k] for k in np.argsort(-(logw + rng.gumbel(size=len(logw))), kind="stable")]
        draw = model.sample([*base.main, *base.extra], deck_type, rng)
        deck, edits, keep = base, [], set(protected)
        for c in order:
            if len(edits) >= size:
                break
            section = "extra" if is_extra(c) else "main"
            added = False
            for _copy in range(min(rules.typical_copies(c, deck_type), copies, size - len(edits))):
                cards = getattr(deck, section)
                stay = keep | set(proposals[c].rule.antecedent) | {e.into for e in edits}
                outs = sorted((o for o in dict.fromkeys(cards) if o not in stay),
                              key=lambda o: (o in engine, draw.get(o, 0.0)))  # fmt: skip
                for o in outs:
                    e = Edit(o, c, section)
                    cand = apply(deck, e)
                    if legal(cand):
                        deck, edits, added = cand, [*edits, e], True
                        break
                else:
                    break  # no removal makes another copy legal
            if added:
                keep |= set(proposals[c].rule.antecedent)
        if edits and _key(deck) not in seen:
            seen.add(_key(deck))
            gain = model.gain([e.into for e in edits], [e.out for e in edits], deck_type)[0]
            out.append(Child(tuple(edits), deck, GENERATOR, gain))
    return out


def _key(deck: Deck) -> tuple:
    return tuple(sorted(deck.main)), tuple(sorted(deck.extra))


def rules_table(rules: Sequence[Rule], names: Mapping[int, str] | None = None) -> list[str]:
    """One line per rule, cards by name when ``names`` has them."""
    n = names or {}
    out = []
    for r in rules:
        ante = " + ".join(n.get(c, str(c)) for c in r.antecedent) or f"type {r.stratum}"
        out.append(f"{ante} -> {n.get(r.consequent, r.consequent)}: support {r.support:.3f} ({r.count}/{r.lists}), "
                   f"confidence {r.confidence:.2f}, lift {r.lift:.1f}")  # fmt: skip
    return out


__all__ = ["GENERATOR", "GLOBAL", "DeckRules", "Proposal", "Rule", "RuleConfig", "rule_children", "rules_table"]
