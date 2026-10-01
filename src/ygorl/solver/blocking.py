"""Blocking boards: interruptions from script facts, board scores, and automatic solver targets (docs/solver.md).

The solver only reaches *target cards* (``password@zone``); it has no scored end board. To teach a first turn
that ends on a **blocking** board -- cards that can stop the opponent's turn -- the objective lives on our side:

* :func:`card_interruptions` reads a card's script (``ygorl.build.scripts``: per-effect categories, ``SetType``,
  ``SetRange``, and whether it can take the opponent's cards) and says where the card interrupts the opponent's
  turn: face-up on the field, set in the spell & trap zone, from the hand, or from the graveyard;
* :func:`board_interruptions` counts the interruptions of player 0 on a board summary (``targets.board_summary``);
* :func:`blocking_plan` turns a deck and an opening hand into solver target sets: the hand's traps set and its
  hand traps kept (``base``), plus one of the deck's field interrupters, best first (``pieces``);
  the driver (``tools/solve_blocking.py``) tries them, pairs two that each solve, and keeps the line whose end
  board has the most interruptions.

No card is named in this module: everything comes from the scripts' constants.

``strict=True`` (the survival pipeline, ``ygorl.solver.survival``) counts more carefully: effects that only draw or
lock (no negation, no card moved -- ``EffectFact.acts``) are dropped, an effect whose condition or target needs a
monster of its controller (``EffectFact.requires``: "if you control a Dragon", "when ... targets an Xyz monster you
control") counts only when the board has one, and a hard once-per-turn effect counts once per card name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from ygorl import paths
from ygorl.build.scripts import ScriptFacts, analyze_script, load_constants

WHERE = ("field", "set", "hand", "grave")  # where an interruption is used from
# categories that stop a card or an effect outright (a "negate")
NEGATE_CATEGORIES = ("CATEGORY_NEGATE", "CATEGORY_DISABLE", "CATEGORY_DISABLE_SUMMON")
# categories that remove or neutralise a card; they count when the effect can take the opponent's cards
DISRUPT_CATEGORIES = ("CATEGORY_DESTROY", "CATEGORY_REMOVE", "CATEGORY_TODECK", "CATEGORY_TOHAND", "CATEGORY_TOGRAVE",
                      "CATEGORY_CONTROL", "CATEGORY_POSITION", "CATEGORY_RELEASE")  # fmt: skip


@dataclass(frozen=True, slots=True)
class Interruption:
    where: str  # WHERE
    negate: bool  # a negation (NEGATE_CATEGORIES), else a disruption of the opponent's cards
    count_code: int = 0  # hard once-per-turn code (strict only): one use per name
    requires: tuple = ()  # filters of monsters its controller must control (strict only)


@cache
def _K() -> dict[str, int]:
    return load_constants()


def _mask(names: Iterable[str]) -> int:
    K = _K()
    out = 0
    for n in names:
        out |= K.get(n, 0)
    return out


def script_path(password: int) -> Path | None:
    """The card's script, in the engine's search order (``paths.script_directories``)."""
    for d in paths.script_directories():
        p = d / f"c{password}.lua"
        if p.is_file():
            return p
    return None


@cache
def script_facts(password: int) -> ScriptFacts | None:
    path = script_path(password)
    if path is None:
        return None
    return analyze_script(path.read_text(encoding="utf-8", errors="replace"), password, _K())


def interruptions_from_facts(facts: ScriptFacts, card_type: int, strict: bool = False) -> tuple[Interruption, ...]:
    """Where a card with these script facts and this ``TYPE_*`` mask interrupts the opponent's turn.

    An effect counts when it can be used during the opponent's turn -- a quick effect (``EFFECT_TYPE_QUICK_O`` /
    ``QUICK_F``), or the activation of a trap or a quick-play spell once set -- and it negates, or removes /
    neutralises cards and can take the opponent's (:data:`DISRUPT_CATEGORIES` with ``EffectFact.opponent``).
    At most one interruption per place (``where``); a negation wins over a disruption. With ``strict``, effects
    that do not act on cards (``EffectFact.acts``) or draw without negating are skipped, and the interruption
    carries the effect's once-per-turn code and requirements (an effect without requirements wins a tie).
    """
    K = _K()
    quick = K["EFFECT_TYPE_QUICK_O"] | K["EFFECT_TYPE_QUICK_F"]
    negate_mask, disrupt_mask = _mask(NEGATE_CATEGORIES), _mask(DISRUPT_CATEGORIES)
    settable = bool(card_type & K["TYPE_TRAP"]) or bool(card_type & K["TYPE_SPELL"] and card_type & K["TYPE_QUICKPLAY"])
    monster = bool(card_type & K["TYPE_MONSTER"])
    found: dict[str, Interruption] = {}
    for eff in facts.effect_facts:
        negate = bool(eff.categories & negate_mask)
        if not negate and not (eff.categories & disrupt_mask and eff.opponent):
            continue
        if strict and (not eff.acts or (eff.categories & K["CATEGORY_DRAW"] and not negate)):
            continue
        places: list[str] = []
        if eff.type & quick:
            rng = eff.range or (K["LOCATION_MZONE"] if monster else K["LOCATION_SZONE"])
            if rng & (K["LOCATION_MZONE"] | K["LOCATION_SZONE"] | K["LOCATION_FZONE"] | K["LOCATION_PZONE"]):
                places.append("field")
            if rng & K["LOCATION_HAND"]:
                places.append("hand")
            if rng & K["LOCATION_GRAVE"]:
                places.append("grave")
        elif eff.type & K["EFFECT_TYPE_ACTIVATE"] and settable:
            places.append("set")
        for w in places:
            new = Interruption(w, negate, eff.count_code, eff.requires) if strict else Interruption(w, negate)
            old = found.get(w)
            if old is None or (new.negate, not new.requires) > (old.negate, not old.requires):
                found[w] = new
    return tuple(found[w] for w in WHERE if w in found)


@cache
def _card_interruptions(password: int, card_type: int, strict: bool = False) -> tuple[Interruption, ...]:
    facts = script_facts(password)
    return interruptions_from_facts(facts, card_type, strict) if facts is not None else ()


def card_interruptions(password: int, cards, strict: bool = False) -> tuple[Interruption, ...]:
    """:func:`interruptions_from_facts` of a card (alternate artworks read the original's script)."""
    code = cards.canonical(password)
    card = cards.get(code)
    return _card_interruptions(code, card.type, strict) if card is not None else ()


def _at(password: int, where: str, cards, strict: bool = False) -> Interruption | None:
    for i in card_interruptions(password, cards, strict):
        if i.where == where:
            return i
    return None


def card_matches(flt, card) -> bool | None:
    """Whether ``card`` (a :class:`~ygorl.cards.cdb.Card`) passes a script filter; None when it cannot be told."""
    from ygorl.build.filters import And, Not, Or, Pred, _compare, setcode_matches

    if isinstance(flt, Pred):
        k, a = flt.kind, flt.args
        if k == "any":
            return True
        if k == "none":
            return False
        if k == "setcard":
            return any(setcode_matches(sc, q) for q in a for sc in card.setcodes if sc)
        if k == "code":
            return card.password in a or (card.alias in a if card.alias else False)
        if k == "type_any":
            return bool(card.type & a[0])
        if k == "type_all":
            return card.type & a[0] == a[0]
        if k == "type_eq":
            return card.type == a[0]
        if k == "race":
            return bool(card.race & a[0])
        if k == "attribute":
            return bool(card.attribute & a[0])
        if k in ("level", "rank", "link"):
            kind = "link" if card.is_link else "rank" if card.is_xyz else "level"
            return kind == k and _compare(a)(card.level)
        if k == "attack":
            return _compare(a)(card.attack)
        if k == "defense":
            return not card.is_link and _compare(a)(card.defense)
        return None
    if isinstance(flt, Not):
        v = card_matches(flt.item, card)
        return None if v is None else not v
    vals = [card_matches(f, card) for f in flt.items]
    if isinstance(flt, And):
        return False if False in vals else (None if None in vals else True)
    if isinstance(flt, Or):
        return True if True in vals else (None if None in vals else False)
    return None


def requirements_met(requires: tuple, monsters: Sequence[int], cards) -> bool:
    """Every filter of ``requires`` is passed by one of ``monsters`` (face-up monsters' passwords); unknown passes."""
    have = [cards[cards.canonical(c)] for c in monsters if cards.canonical(c) in cards]
    return all(any(card_matches(f, c) is not False for c in have) for f in requires)


@dataclass
class BoardScore:
    """Interruptions of one player's end board: one per card that can stop the opponent from where it sits."""

    interruptions: int = 0
    negates: int = 0
    pieces: list[tuple[int, str, bool]] = field(default_factory=list)  # (password, where, negate)

    def key(self) -> tuple[int, int]:
        return (self.interruptions, self.negates)

    def to_json(self) -> dict:
        return {"interruptions": self.interruptions, "negates": self.negates,
                "pieces": [[c, w, n] for c, w, n in self.pieces]}  # fmt: skip


def board_interruptions(board: Mapping, cards, player: int = 0, strict: bool = False) -> BoardScore:
    """Score ``board`` (a ``targets.board_summary`` dict) for ``player``.

    Face-up monsters and face-up spells / traps count their ``field`` interruptions, set spells / traps their
    ``set`` ones (a trap or quick-play spell whose activation interrupts), cards in hand their ``hand`` ones
    (hand traps kept), cards in the graveyard their ``grave`` ones. Face-down monsters and materials do not count.
    With ``strict`` (see the module docstring): requirements are checked against the face-up monsters of
    ``player`` and a hard once-per-turn effect counts once per code.
    """
    from ygorl.engine import constants as C

    side = board["players"][player]
    score = BoardScore()
    faceup = [c["code"] for c in side.get("mzone", []) if c["position"] & C.POS_FACEUP]
    used: set[int] = set()

    def add(code: int, where: str) -> None:
        hit = _at(code, where, cards, strict)
        if hit is not None and strict:
            if hit.requires and not requirements_met(hit.requires, faceup, cards):
                return
            if hit.count_code:
                if hit.count_code in used:
                    return
                used.add(hit.count_code)
        if hit is not None:
            score.interruptions += 1
            score.negates += hit.negate
            score.pieces.append((code, where, hit.negate))

    for c in side.get("mzone", []):
        if c["position"] & C.POS_FACEUP:
            add(c["code"], "field")
    for c in side.get("szone", []):
        add(c["code"], "field" if c["position"] & C.POS_FACEUP else "set")
    for code in side.get("hand", []):
        add(code, "hand")
    for code in side.get("grave", []):
        add(code, "grave")
    return score


@dataclass(frozen=True)
class Piece:
    """A card of the deck the solver can be asked to put on the field as an interrupter."""

    password: int
    negate: bool
    extra: bool  # an Extra Deck monster (the usual end point of a combo)
    engine: bool  # tied to the main deck: a shared archetype, or one names the other (see deck_pieces)
    target: str  # the solver target (a face-up monster)

    def rank(self) -> tuple:
        return (not self.engine, not self.negate, not self.extra, self.password)


@dataclass
class BlockingPlan:
    base: list[str]  # always required: the hand's interrupting traps / quick-play spells set, its hand traps kept
    pieces: list[Piece]  # field interrupters of the deck, best first


def _names(code: int) -> set[int]:
    facts = script_facts(code)
    return set() if facts is None else {*facts.listed_names, *facts.material_codes}


def deck_pieces(main: Sequence[int], extra: Sequence[int], cards, strict: bool = False) -> list[Piece]:
    """The deck's field interrupters: monsters whose quick effect interrupts from the monster zone, best first.

    Order: *engine* pieces first -- the piece shares an archetype (``setcodes``) with a main deck card, or its
    script names a main deck card (``listed_names``, materials) or is named by one -- since a generic Extra Deck
    monster is only sometimes reachable; then negations, then Extra Deck monsters.
    """
    out: dict[int, Piece] = {}
    extra_codes = {cards.canonical(c) for c in extra}
    main_codes = {cards.canonical(c) for c in main}
    main_sets = {s for c in main_codes if c in cards for s in cards[c].setcodes if s}
    main_names = set().union(*(_names(c) for c in main_codes)) if main_codes else set()
    for code in dict.fromkeys(cards.canonical(c) for c in (*extra, *main)):
        card = cards.get(code)
        if card is None or not card.is_monster:
            continue
        hit = _at(code, "field", cards, strict)
        if hit is None or _at(code, "hand", cards, strict) is not None:
            continue  # hand traps are kept in hand (the plan's base), not summoned
        engine = bool(set(card.setcodes) & main_sets) or code in main_names or bool(_names(code) & main_codes)
        out[code] = Piece(code, hit.negate, code in extra_codes, engine, str(code))
    return sorted(out.values(), key=Piece.rank)


def blocking_plan(main: Sequence[int], extra: Sequence[int], hand: Sequence[int], cards,
                  strict: bool = False) -> BlockingPlan:  # fmt: skip
    """Solver targets for a blocking first turn from ``hand`` (see the module docstring)."""
    base: list[str] = []
    sets = 0
    for code in hand:
        if _at(code, "set", cards, strict) is not None and sets < 5:
            base.append(f"{cards.canonical(code)}@szone:fd")
            sets += 1
        elif _at(code, "hand", cards, strict) is not None:
            base.append(f"{cards.canonical(code)}@hand")
    return BlockingPlan(base, deck_pieces(main, extra, cards, strict))


# ------------------------------------------------------------------ one hand: the target ladder


def _score_demo(demo, cards) -> BoardScore | None:
    """The best line of ``demo`` by interruptions (the line is moved first); None without a line."""
    if not demo.lines:
        return None
    scored = sorted(((board_interruptions(ln.board, cards), i) for i, ln in enumerate(demo.lines)),
                    key=lambda t: (t[0].key(), -t[1]), reverse=True)  # fmt: skip
    best, i = scored[0]
    demo.lines.insert(0, demo.lines.pop(i))
    return best


def solve_blocking(job, pieces: int = 3, pair: bool = True, cards=None, scripts=None) -> dict:
    """Solve one opening hand towards a blocking board; the record (JSON) of the best line found.

    ``job`` is a :class:`~ygorl.solver.batch.HandJob` whose ``targets`` are ignored: the targets come from
    :func:`blocking_plan`. Attempts, each a full :func:`~ygorl.solver.batch.solve_hand` at ``job.solve_ms``:

    1. ``base + piece`` for each of the first ``pieces`` pieces of the deck;
    2. with ``pair``, ``base + p1 + p2`` for the two best pieces that each solved;
    3. ``base`` alone when it is not empty and nothing above solved (set the traps, keep the hand traps).

    The record kept is the solved attempt whose best line ends on the most interruptions (then negations, then
    the earlier attempt); without any, the first attempt's record. ``solver["blocking"]`` lists every attempt and
    the kept board's score.
    """
    import shutil
    from dataclasses import replace

    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.solver.batch import _config, _load_env, sample_hand, solve_hand

    cards = cards if cards is not None else default_cards()
    deck = load_ydk(job.deck_path)
    config = _config(_load_env(job.env)) if job.env else DuelConfig()
    hand, _ = sample_hand(deck, job.hand_seed, config.player.starting_hand)
    plan = blocking_plan(deck.main, deck.extra, hand, cards)
    attempts: list[tuple[list[str], str]] = [
        ([*plan.base, p.target], f"piece:{p.password}") for p in plan.pieces[:pieces]
    ]
    results: list[tuple[dict, object, BoardScore | None]] = []

    def attempt(targets: list[str], label: str):
        sub = replace(job, targets=tuple(targets), scratch=Path(job.scratch) / f"a{len(results)}")
        demo = solve_hand(sub, cards=cards, scripts=scripts)
        score = _score_demo(demo, cards)
        info = {"label": label, "targets": list(targets), "status": demo.status,
                "wall_s": demo.solver.get("wall_s", 0.0), **(score.to_json() if score else {})}  # fmt: skip
        if demo.status != "solved" and "best_placed" in demo.solver:
            info["best_placed"] = demo.solver["best_placed"]
        results.append((info, demo, score))
        return demo.status == "solved"

    solved_pieces = []
    for targets, label in attempts:
        if attempt(targets, label):
            solved_pieces.append(targets[-1])
    if pair and len(solved_pieces) >= 2:
        attempt([*plan.base, *solved_pieces[:2]], "pair")
    if plan.base and not solved_pieces:
        attempt(list(plan.base), "base")
    if not results:  # no piece and nothing in hand to set or keep
        from ygorl.solver.demo import Demonstration

        empty =Demonstration.new(deck, hand, hand_index=job.hand_index, hand_seed=job.hand_seed, variant="plain",
                                  targets=(), environment=_load_env(job.env))  # fmt: skip
        empty.status, empty.error = "unsolved", "no interrupter in the deck and none in hand"
        empty.solver = {"wall_s": 0.0, "blocking": {"attempts": [], "base": [], "pieces": []}}
        return empty.to_json()
    solved = [(i, r) for i, r in enumerate(results) if r[2] is not None]
    if solved:
        k, (_, demo, score) = max(solved, key=lambda t: (t[1][2].key(), -t[0]))
    else:
        k, (_, demo, score) = 0, results[0]
    demo.solver["blocking"] = {
        "base": plan.base, "pieces": [p.target for p in plan.pieces], "chosen": k,
        "attempts": [r[0] for r in results], **(score.to_json() if score else {}),
    }  # fmt: skip
    demo.solver["wall_s"] = round(sum(r[0]["wall_s"] for r in results), 2)
    if not job.keep_files:
        shutil.rmtree(job.scratch, ignore_errors=True)
    return demo.to_json()


__all__ = ["BlockingPlan", "BoardScore", "Interruption", "Piece", "blocking_plan", "board_interruptions",
           "card_interruptions", "card_matches", "deck_pieces", "interruptions_from_facts", "requirements_met",
           "script_facts", "script_path", "solve_blocking"]  # fmt: skip
