"""Put a :class:`BoardSnapshot` into the core so the policy can be asked about it (design 10, 「重建」).

Two steps:

- :func:`complete`: fill in what the advisor cannot see, one sample per call. The opponent's hidden cards (hand,
  Set cards, face-down banished cards, deck, Extra Deck) are drawn from one meta deck, chosen by share among the
  meta decks consistent with the opponent's public cards (the HDT filter of design 04); the advisor's own deck
  and Extra Deck are its deck list minus the cards already seen.
- :func:`build`: a core duel holding the completed board, run to the advisor's first idle decision. The cards are
  placed by a generated Lua script (``Debug.AddCard``: Xyz materials, face-up Pendulum cards in the Extra Deck and
  counters included), served next to the card scripts. The advisor is engine player 0 and the duel starts in its
  turn-1 Main Phase 1 with no draws; ``DUEL_ATTACK_FIRST_TURN`` lifts the turn-1 battle ban when the snapshot is
  later in the game. The real turn number and who went first are restored in the observation
  (:func:`ygorl.advisor.advise.observe`).

Lost on the way (design 10「重建的已知失真」): the event history, once-per-turn and Normal Summon usage, effect
states such as equips and targets, and anything mid-chain. Only the advisor's own Main Phase is supported.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ygorl import _core
from ygorl.advisor.snapshot import BoardSnapshot, SnapshotCard
from ygorl.cards.ydk import Deck
from ygorl.data.environment import PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import DecisionPoint, DuelConfig, DuelTracker, default_cards, default_scripts, expand_seed

BOARD_SCRIPT = "ygorl_advisor_board.lua"
MAX_PRE_DECISIONS = 32  # decisions answered by the host before the idle point (standby-phase triggers, ...)
DECLINE = ("pass", "no", "cancel", "finish")  # how the host answers them: decline when it can, else the first option


class ReconstructError(ValueError):
    """A snapshot the core cannot be put into the asked state."""


@dataclass(frozen=True)
class OpponentPrior:
    """Meta decks with their shares, the opponent's possible decks."""

    decks: tuple[Deck, ...]
    shares: tuple[float, ...]

    @classmethod
    def from_environment(cls, env) -> OpponentPrior:
        if not env.meta_decks:
            raise ReconstructError(f"environment {env.version} has no meta decks to sample the opponent from")
        return cls(tuple(m.deck for m in env.meta_decks), tuple(m.share for m in env.meta_decks))

    def consistent(self, seen: Counter[int]) -> list[int]:
        """Indices of the decks holding every card seen (with multiplicity), the HDT filter."""
        out = []
        for i, deck in enumerate(self.decks):
            counts = deck.counts()
            if all(counts[p] >= n for p, n in seen.items()):
                out.append(i)
        return out

    def pick(self, seen: Counter[int], rng: random.Random) -> tuple[int, bool]:
        """(deck index, consistent?) drawn by share among the consistent decks, or among all when none is."""
        pool = self.consistent(seen)
        ok = bool(pool)
        if not ok:
            pool = list(range(len(self.decks)))
        return rng.choices(pool, weights=[self.shares[i] for i in pool])[0], ok


@dataclass
class Placed:
    """One card of a completed board, in core terms."""

    password: int
    location: int
    sequence: int = 0
    position: int = C.POS_FACEDOWN_DEFENSE
    owner: int | None = None  # engine player; None = the controller
    materials: tuple[int, ...] = ()
    counters: tuple[tuple[int, int], ...] = ()


@dataclass
class CompletedBoard:
    """A snapshot with every hidden card filled in (one sample)."""

    snapshot: BoardSnapshot
    cards: tuple[list[Placed], list[Placed]] = field(default_factory=lambda: ([], []))
    decks: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    extras: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    opponent_deck: int = -1  # index into the prior's decks
    consistent: bool = True  # the opponent deck was consistent with the public cards
    notes: list[str] = field(default_factory=list)


def _draw(pool: Counter[int], rng: random.Random, accept: Callable[[int], bool] = lambda p: True,
          fallback: Sequence[int] = ()) -> int:  # fmt: skip
    """One card out of ``pool`` (removed from it), preferring cards that ``accept``; else from ``fallback``."""
    items = [p for p, n in pool.items() for _ in range(n)]
    good = [p for p in items if accept(p)]
    choice = good or items
    if choice:
        p = rng.choice(choice)
        pool[p] -= 1
        if not pool[p]:
            del pool[p]
        return p
    good = [p for p in fallback if accept(p)] or list(fallback)
    if not good:
        raise ReconstructError("no card left to fill a hidden slot")
    return rng.choice(good)


def _from_deck(side: int, snapshot: BoardSnapshot, cards: Mapping) -> Counter[int]:
    """Known cards that left ``side``'s deck: owned by it and not tokens (Xyz materials included)."""
    out: Counter[int] = Counter()
    for card in snapshot.sides[side].cards:
        for p in (card.password, *card.materials):
            info = cards.get(p)
            if p and (card.owner is None or card.owner == side) and not (info is not None and info.type & C.TYPE_TOKEN):
                out[p] += 1
    return out


def _placed(card: SnapshotCard, password: int) -> Placed:
    # snapshot sides are (advisor, opponent) and the advisor is engine player 0, so owners map one to one
    return Placed(password, card.core_location, card.sequence, card.core_position, card.owner, card.materials,
                  card.counters)  # fmt: skip


def complete(snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, cards: Mapping, rng: random.Random
             ) -> CompletedBoard:  # fmt: skip
    """One sample of the hidden cards (see the module docstring)."""
    if snapshot.turn_player != 0:
        raise ReconstructError("only the advisor's own turn is supported")
    board = CompletedBoard(snapshot)

    def is_extra(p: int) -> bool:
        card = cards.get(p)
        return card is not None and card.is_extra_deck

    def is_main_monster(p: int) -> bool:
        card = cards.get(p)
        return card is not None and card.is_monster and not card.is_extra_deck

    def is_spell_trap(p: int) -> bool:
        card = cards.get(p)
        return card is not None and (card.is_spell or card.is_trap)

    # the advisor: its deck list minus what is already out of the deck
    me = snapshot.sides[0]
    seen = _from_deck(0, snapshot, cards)
    main_left = Counter(own_deck.main) - seen
    extra_left = Counter(own_deck.extra) - Counter({p: n for p, n in seen.items() if is_extra(p)})
    board.cards[0].extend(_placed(c, c.password) for c in me.cards)
    deck = [p for p, n in main_left.items() for _ in range(n)]
    rng.shuffle(deck)
    if len(deck) < me.deck_count:
        board.notes.append(f"own deck: {me.deck_count} cards in the game, {len(deck)} left in the list; padded")
        deck += [rng.choice(own_deck.main) for _ in range(me.deck_count - len(deck))]
    board.decks[0].extend(deck[: me.deck_count])
    extra = [p for p, n in extra_left.items() for _ in range(n)]
    rng.shuffle(extra)
    if len(extra) < me.extra_count:
        board.notes.append(f"own Extra Deck: {me.extra_count} cards in the game, {len(extra)} left in the list")
    board.extras[0].extend(extra[: me.extra_count])

    # the opponent: one meta deck consistent with its public cards, minus those cards
    op = snapshot.sides[1]
    op_seen = _from_deck(1, snapshot, cards)
    index, board.consistent = prior.pick(op_seen, rng)
    board.opponent_deck = index
    chosen = prior.decks[index]
    if not board.consistent:
        board.notes.append("no meta deck holds every public opponent card; the opponent deck is a share-weighted guess")
    main_left = Counter(chosen.main) - op_seen
    extra_left = Counter(chosen.extra) - op_seen
    for card in op.cards:
        if card.known:
            board.cards[1].append(_placed(card, card.password))
            continue
        if card.location == "mzone":
            accept = is_main_monster
        elif card.location == "szone":
            accept = is_spell_trap
        else:
            accept = lambda p: True  # noqa: E731
        board.cards[1].append(_placed(card, _draw(main_left, rng, accept, chosen.main)))
    deck = [p for p, n in main_left.items() for _ in range(n)]
    rng.shuffle(deck)
    deck += [rng.choice(chosen.main) for _ in range(op.deck_count - len(deck))]
    board.decks[1].extend(deck[: op.deck_count])
    extra = [p for p, n in extra_left.items() for _ in range(n)]
    rng.shuffle(extra)
    if chosen.extra:
        extra += [rng.choice(chosen.extra) for _ in range(op.extra_count - len(extra))]
    board.extras[1].extend(extra[: op.extra_count])
    return board


def board_script(board: CompletedBoard) -> str:
    """The Lua script that puts ``board`` into a fresh duel (``Debug.AddCard`` per card)."""
    lines = ["-- generated by ygorl.advisor.reconstruct"]
    for player in (0, 1):
        for card in board.cards[player]:
            owner = player if card.owner is None else card.owner
            proc = "true" if card.location == C.LOCATION_MZONE else "false"
            lines.append(f"local c=Debug.AddCard({card.password},{owner},{player},{card.location},{card.sequence},"
                         f"{card.position},{proc})")  # fmt: skip
            for t, n in card.counters:
                lines.append(f"if c then Debug.PreAddCounter(c,{t},{n}) end")
            for m in card.materials:  # AddCard on an occupied Monster Zone attaches an Xyz material
                lines.append(f"Debug.AddCard({m},{owner},{player},{C.LOCATION_MZONE},{card.sequence},"
                             f"{C.POS_FACEUP_ATTACK})")  # fmt: skip
        for p in board.decks[player]:
            lines.append(f"Debug.AddCard({p},{player},{player},{C.LOCATION_DECK},0,{C.POS_FACEDOWN_DEFENSE})")
        for p in board.extras[player]:
            lines.append(f"Debug.AddCard({p},{player},{player},{C.LOCATION_EXTRA},0,{C.POS_FACEDOWN_DEFENSE})")
    return "\n".join(lines) + "\n"


@dataclass
class Reconstruction:
    """A live core duel holding a completed board, stopped at the advisor's idle decision."""

    board: CompletedBoard
    core: _core.Duel
    tracker: DuelTracker
    point: DecisionPoint
    warnings: list[str]


def build(board: CompletedBoard, rule_flags: int, *, cards=None, scripts=None, seed: int = 1) -> Reconstruction:
    """Start a duel holding ``board`` and run it to the advisor's first Main Phase idle decision."""
    cards = cards if cards is not None else default_cards()
    base = scripts if scripts is not None else default_scripts()
    snap = board.snapshot
    if snap.turn > 1:
        rule_flags |= C.DUEL_ATTACK_FIRST_TURN
    script = board_script(board).encode()

    def read(name: str) -> bytes | None:
        return script if name == BOARD_SCRIPT else base.read(name)

    lp = (snap.sides[0].lp, snap.sides[1].lp)
    core = _core.Duel(expand_seed(seed), rule_flags, (lp[0], 0, 0), (lp[1], 0, 0), cards.to_core(), read)
    for name in ("constant.lua", "utility.lua", BOARD_SCRIPT):
        if not core.load_script(name):
            raise ReconstructError(f"failed to load {name}: {' | '.join(str(t) for _, t in core.pop_logs())}")
    core.start()
    config = DuelConfig(rule_flags=rule_flags, player=PlayerRules(max(lp), 0, 0), shuffle_decks=False)
    tracker = DuelTracker(config, 0, cards)
    tracker.lp = list(lp)
    warnings = list(board.notes)
    run = True  # False while a multi-select is half answered (the core still waits for the response)
    for _ in range(MAX_PRE_DECISIONS + 1):
        while run:
            status = core.process()
            tracker.on_buffer(core.get_message(), status, core.pop_logs())
            if tracker.done or tracker.awaiting:
                break
        if tracker.done:
            raise ReconstructError(f"the duel ended while being set up ({tracker.result.reason})")
        point = tracker.point()
        if point is None:
            raise ReconstructError(f"the duel stopped while being set up ({tracker.result.reason})")
        if point.player == 0 and isinstance(point.decision, M.SelectIdleCmd) and point.phase == C.PHASE_MAIN1:
            warnings += [f"script error: {e}" for e in tracker.result.script_errors]
            return Reconstruction(board, core, tracker, point, warnings)
        index = next((i for i, a in enumerate(point.actions) if a.kind in DECLINE), 0)
        warnings.append(f"answered {point.decision.name} of player {point.player} before Main Phase 1: "
                        f"{point.actions[index].kind}")  # fmt: skip
        response = tracker.act(index)
        run = response is not None
        if run:
            core.set_response(response)
    raise ReconstructError(f"no idle decision after {MAX_PRE_DECISIONS} setup decisions")


__all__ = ["BOARD_SCRIPT", "CompletedBoard", "OpponentPrior", "Placed", "Reconstruction", "ReconstructError",
           "board_script", "build", "complete"]  # fmt: skip
