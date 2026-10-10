"""Single-duel API: ``Duel(seed, env, deck_a, deck_b).run(agent_a, agent_b)``.

AEC semantics: the engine asks one player at a time; that player's agent sees a
:class:`DecisionPoint` (the decoded decision, its legal actions and the events
since the previous decision point) and returns an action index. Multi-selects
are asked card by card (see :mod:`ygorl.engine.actions`), so every agent call is
one step.

Engine player 0 always moves first; ``first=1`` seats ``deck_b`` / ``agent_b``
as engine player 0. Results are reported in (a, b) order.

The host-side bookkeeping (message parsing, undo masks, curriculum answers,
the result) is :class:`ygorl.engine.tracker.DuelTracker`; ``DecisionPoint``,
``DuelResult``, ``DuelTracker`` and the seat rule are re-exported from here.

Note: ``DecisionPoint.events`` is the engine's full (server) view, including
hidden information. Per-player observation filtering belongs to the M2
environment layer (T2.2 / T2.5).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import cache

from ygorl import _core, paths
from ygorl.cards.cdb import CardDB
from ygorl.cards.legality import check_deck
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.curriculum import FULL, MODES

# the host-side tracker lives in ygorl.engine.tracker; its public names are re-exported here for existing callers
from ygorl.engine.tracker import (  # noqa: F401
    MAX_CONSECUTIVE_RETRIES,
    MAX_ENGINE_STEPS,
    MAX_MENU_ACTIVATIONS,
    MAX_SELECTION_CANCELS,
    MAX_SELECTION_STEPS,
    WIN_REASON_DECK_OUT,
    WIN_REASON_LP,
    DecisionPoint,
    DuelResult,
    DuelTracker,
    deck_of_seat,
    seat_of_deck,
)

MASK64 = (1 << 64) - 1


def expand_seed(seed: int) -> list[int]:
    """Expand an integer seed into the core's four 64-bit words (splitmix64)."""
    state = seed & MASK64
    out = []
    for _ in range(4):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        out.append(z ^ (z >> 31))
    if not any(out):  # the core rejects an all-zero seed
        out[0] = 1
    return out


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def shuffle_deck(cards, seed: int, player: int) -> list[int]:
    """Deterministic Fisher-Yates shuffle of a main deck (host side, like EDOPro).

    The core does not shuffle decks when a duel starts; the host loads them
    already shuffled. The stream is splitmix64 seeded from ``seed`` and
    ``player``, with rejection sampling so every permutation is equally likely.
    Stable across Python versions so it can be mirrored in C++ (M2).
    """
    state = (seed & MASK64) ^ (((player + 1) * 0xD1B54A32D192ED03) & MASK64)
    out = list(cards)
    for i in range(len(out) - 1, 0, -1):
        n = i + 1
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            state, x = _splitmix64(state)
            if x < limit:
                break
        j = x % n
        out[i], out[j] = out[j], out[i]
    return out


@dataclass(frozen=True)
class DuelConfig:
    rule_flags: int = C.DUEL_MODE_MR5
    player: PlayerRules = PlayerRules()
    max_turns: int = 200
    max_decisions: int = 6000  # stops loops; the longest real game seen: 2,936 decisions (docs/cli.md)
    shuffle_decks: bool = True  # host-side shuffle of each main deck before loading
    curriculum: str = FULL  # "full" / "solo" / "handtrap": restricts the opponent in the learner's turn (T2.6)
    learner: int = 0  # the learning deck the curriculum protects: 0 = deck_a, 1 = deck_b
    augmented_start: bool = False  # the game starts from an augmented (mid-game) state (I7); shown in observations

    def __post_init__(self) -> None:
        if self.curriculum not in MODES:
            raise ValueError(f"unknown curriculum {self.curriculum!r} (expected one of {MODES})")
        if self.learner not in (0, 1):
            raise ValueError(f"learner must be 0 (deck_a) or 1 (deck_b), not {self.learner!r}")

    @classmethod
    def from_environment(cls, env: Environment, **overrides) -> DuelConfig:
        return cls(rule_flags=env.rule_flags, player=env.player, **overrides)


@cache
def default_cards() -> CardDB:
    return CardDB.load()


@cache
def default_scripts() -> _core.ScriptDirectory:
    from ygorl.engine.script_patches import chaos_angel_override, clown_crew_override, fusion_override, synchro_override

    directories = [str(p) for p in paths.script_directories()]
    original = _core.ScriptDirectory(directories)
    overrides = {}
    for name, patch in (
        ("proc_synchro.lua", synchro_override),
        ("proc_fusion.lua", fusion_override),
        ("c22850702.lua", chaos_angel_override),
        ("c83232904.lua", clown_crew_override),
    ):
        replacement = patch(original.read(name))
        if replacement is not None:
            overrides[name] = replacement
    return _core.ScriptDirectory(directories, overrides) if overrides else original


class Duel:
    """One duel between two decks. Create, then call :meth:`run` once."""

    def __init__(
        self,
        seed: int,
        env: Environment | None,
        deck_a: Deck,
        deck_b: Deck,
        *,
        cards: CardDB | None = None,
        scripts: _core.ScriptDirectory | None = None,
        config: DuelConfig | None = None,
        first: int = 0,
        validate: bool = False,
        record_messages: bool = False,
        record_steps: bool = False,
        snapshots: bool = False,
        core_seed: list[int] | None = None,
    ) -> None:
        if first not in (0, 1):
            raise ValueError("first must be 0 (deck_a starts) or 1 (deck_b starts)")
        self.seed = seed
        # The core's four RNG words: expand_seed(seed) unless given explicitly (replays recorded by other hosts,
        # e.g. EDOPro or the combo solver, carry their own words; T4a.1).
        if core_seed is None:
            self.core_seed = expand_seed(seed)
        else:
            words = [int(w) for w in core_seed]
            if len(words) != 4 or not all(0 <= w < 1 << 64 for w in words):
                raise ValueError(f"core_seed must be four 64-bit words, not {core_seed!r}")
            if not any(words):
                raise ValueError("core_seed must not be all-zero (the core rejects it)")
            self.core_seed = words
        self.env = env
        self.decks = (deck_a, deck_b)
        self.cards = cards if cards is not None else default_cards()
        self.scripts = scripts if scripts is not None else default_scripts()
        self.config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
        self.first = first
        self.record_messages = record_messages
        self.record_steps = record_steps
        self.snapshots = snapshots  # core arena for _core.Duel.snapshot()/restore() (T2.8)
        if validate:
            for deck in self.decks:
                if env is not None:
                    check_deck(deck, cards=self.cards, banlist=env.banlist, pool=env.card_pool, rules=env.deck_rules)
                else:
                    check_deck(deck, cards=self.cards, banlist=None)
        self._core: _core.Duel | None = None
        self._ran = False

    # -- helpers ---------------------------------------------------------
    def deck_of(self, player: int) -> int:
        """Index into (a, b) of the deck seated as engine ``player``."""
        return deck_of_seat(self.first, player)

    def field_state(self) -> M.ReloadField:
        """Current field summary from the engine (``OCG_DuelQueryField``)."""
        if self._core is None:
            raise RuntimeError("duel not started")
        msg = M.decode_message(bytes([C.MSG_RELOAD_FIELD]) + self._core.query_field())
        assert isinstance(msg, M.ReloadField), msg
        return msg

    def loaded_decks(self) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
        """(main, extra) per engine player, in the order the cards are loaded into the core."""
        out = []
        for team in (0, 1):
            deck = self.decks[self.deck_of(team)]
            main = shuffle_deck(deck.main, self.seed, team) if self.config.shuffle_decks else list(deck.main)
            out.append((main, list(deck.extra)))
        return out[0], out[1]

    def _setup(self) -> _core.Duel:
        p = self.config.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        core = _core.Duel(self.core_seed, self.config.rule_flags, player, player, self.cards.to_core(), self.scripts,
                          snapshots=self.snapshots)  # fmt: skip
        for base in ("constant.lua", "utility.lua"):
            if not core.load_script(base):
                raise RuntimeError(f"failed to load base script {base}")
        for team, (main, extra) in enumerate(self.loaded_decks()):
            for code in main:
                core.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
            for code in extra:
                core.new_card(team, 0, code, team, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
        core.start()
        return core

    # -- main loop -------------------------------------------------------
    def run(self, agent_a, agent_b) -> DuelResult:
        """Play the duel, asking ``agent_a`` / ``agent_b`` for every decision.

        An agent with an ``observe(point, core)`` method is shown every decision point of the duel,
        both seats', before the deciding agent's ``act`` (see :mod:`ygorl.agents.base`).
        """
        agents = (agent_a, agent_b)
        return self._loop(seat=(agents[self.deck_of(0)], agents[self.deck_of(1)]))

    def replay(self, responses: list[bytes], reference_log: list[bytes] | None = None, observer=None) -> DuelResult:
        """Feed recorded ``set_response`` payloads instead of asking agents.

        Stops with reason ``log_exhausted`` when the engine asks for more
        responses than were recorded. With ``reference_log`` (a previous
        ``message_log``), every engine message buffer is compared as it is
        produced and a :class:`ValueError` is raised at the first difference.
        An ``observer`` sees the live core: ``observer.on_start(core)`` once
        the cards are loaded (before the first ``process()``) and
        ``observer.on_buffer(core, buf)`` after every ``process()``, before
        the next response is set (the ``.yrpX`` export queries the core there).
        """
        return self._loop(responses=list(responses), reference_log=reference_log, observer=observer)

    def _loop(self, seat=None, responses: list[bytes] | None = None, reference_log: list[bytes] | None = None,
              observer=None) -> DuelResult:  # fmt: skip
        if self._ran:
            raise RuntimeError("a Duel can only be run once")
        self._ran = True
        tracker = self.tracker(reference_log=reference_log)
        core = self._core = self._setup()
        hooks = []  # optional agent hooks (docs/evaluation.md): on_duel_start(duel), on_decision(point, index)
        watchers = list({id(a): a.observe for a in seat or () if callable(getattr(a, "observe", None))}.values())
        try:
            for agent in {id(a): a for a in seat or ()}.values():
                if hasattr(agent, "on_duel_start"):
                    agent.on_duel_start(self)
                if hasattr(agent, "on_decision"):
                    hooks.append(agent.on_decision)
            if observer is not None:
                observer.on_start(core)
            while True:
                try:
                    status = core.process()
                except _core.ScriptBudgetExceeded as e:  # a script search that would not end (_core budget)
                    tracker.stop("error", str(e))
                    break
                buf = core.get_message()
                if observer is not None:
                    observer.on_buffer(core, buf)
                tracker.on_buffer(buf, status, core.pop_logs())
                if tracker.done:
                    break
                if not tracker.awaiting:
                    continue
                if responses is not None:
                    if len(tracker.result.responses) >= len(responses):
                        tracker.stop("log_exhausted")
                        break
                    response = responses[len(tracker.result.responses)]
                    tracker.use_response(response)
                else:
                    response = tracker.auto_response()
                    while response is None and not tracker.done:
                        point = tracker.point()
                        if point is None:
                            break
                        for observe in watchers:
                            observe(point, core)
                        agent = seat[point.player]
                        index = agent.act(point)
                        response = tracker.act(index, getattr(agent, "last_probs", None))
                        for hook in hooks:
                            hook(point, index)
                    if tracker.done:
                        break
                core.set_response(response)
        finally:
            core.close()
        return tracker.finish()

    def tracker(self, reference_log: list[bytes] | None = None) -> DuelTracker:
        """Host-side bookkeeping for this duel (used by :meth:`run` and by the vectorized env)."""
        return DuelTracker(self.config, self.first, self.cards, record_messages=self.record_messages,
                           record_steps=self.record_steps, reference_log=reference_log)  # fmt: skip


@dataclass(frozen=True)
class SessionSnapshot:
    """State of a :class:`DuelSession` at one agent step: the core's arena image plus the host tracker."""

    core: _core.DuelSnapshot
    tracker: DuelTracker
    session_id: int


class DuelSession:
    """A duel advanced one agent step at a time, instead of :meth:`Duel.run` asking agents.

    ``point`` is the pending :class:`DecisionPoint` (``None`` once ``done``);
    :meth:`act` answers it with an action index. With a ``Duel(...,
    snapshots=True)``, :meth:`snapshot` / :meth:`restore` capture and return
    to the complete state (core arena + host tracker), so branches can be
    explored without replaying from the start (T2.8 / T2.9). Host answers of
    curriculum modes happen inside, exactly as in :meth:`Duel.run`.
    """

    def __init__(self, duel: Duel) -> None:
        if duel._ran:
            raise RuntimeError("a Duel can only be run once")
        duel._ran = True
        self.duel = duel
        self.tracker = duel.tracker()
        self.core = duel._core = duel._setup()
        self._advance(None)

    @property
    def done(self) -> bool:
        return self.tracker.done

    @property
    def point(self) -> DecisionPoint | None:
        return None if self.tracker.done else self.tracker.point()

    def act(self, idx: int, probs=None) -> None:
        response = self.tracker.act(idx, probs)
        if response is not None and not self.tracker.done:
            self._advance(response)

    def _advance(self, response: bytes | None) -> None:
        """Run the core until an agent decision is pending or the duel is over."""
        tracker, core = self.tracker, self.core
        while True:
            if response is not None:
                core.set_response(response)
                response = None
            try:
                status = core.process()
            except _core.ScriptBudgetExceeded as e:
                tracker.stop("error", str(e))
                return
            tracker.on_buffer(core.get_message(), status, core.pop_logs())
            if tracker.done:
                return
            if not tracker.awaiting:
                continue
            response = tracker.auto_response()
            if response is None:
                tracker.point()  # may stop the duel (decision limit)
                return

    def result(self) -> DuelResult:
        """The finished game's result (whole game from the start); only valid once ``done``."""
        if not self.tracker.done:
            raise RuntimeError("the duel is not over")
        return self.tracker.finish()

    def _copy_tracker(self, tracker: DuelTracker) -> DuelTracker:
        return copy.deepcopy(tracker, {id(self.duel.cards): self.duel.cards})

    def snapshot(self) -> SessionSnapshot:
        if not self.core.snapshots_enabled:
            raise RuntimeError("this duel was created without snapshots (Duel(..., snapshots=True))")
        return SessionSnapshot(self.core.snapshot(), self._copy_tracker(self.tracker), id(self))

    def restore(self, snapshot: SessionSnapshot) -> None:
        if snapshot.session_id != id(self):
            raise ValueError("the snapshot was taken from another session")
        self.core.restore(snapshot.core)
        self.tracker = self._copy_tracker(snapshot.tracker)

    def close(self) -> None:
        self.core.close()


class ScriptedAgent:
    """Plays back a recorded list of action indices (shared by both seats, in order)."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = list(actions)
        self._next = 0

    def act(self, point: DecisionPoint) -> int:
        if self._next >= len(self._actions):
            raise IndexError("scripted action log exhausted")
        idx = self._actions[self._next]
        self._next += 1
        return idx


def run_duel(seed: int, deck_a: Deck, deck_b: Deck, agent_a, agent_b, env: Environment | None = None,
             **kwargs) -> DuelResult:  # fmt: skip
    """Convenience wrapper: ``Duel(seed, env, deck_a, deck_b, **kwargs).run(agent_a, agent_b)``."""
    return Duel(seed, env, deck_a, deck_b, **kwargs).run(agent_a, agent_b)


__all__ = [
    "DecisionPoint",
    "Duel",
    "DuelConfig",
    "DuelResult",
    "DuelSession",
    "DuelTracker",
    "ScriptedAgent",
    "deck_of_seat",
    "expand_seed",
    "run_duel",
    "seat_of_deck",
    "shuffle_deck",
]
