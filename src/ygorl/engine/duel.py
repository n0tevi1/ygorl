"""Single-duel API: ``Duel(seed, env, deck_a, deck_b).run(agent_a, agent_b)``.

AEC semantics: the engine asks one player at a time; that player's agent sees a
:class:`DecisionPoint` (the decoded decision, its legal actions and the events
since the previous decision point) and returns an action index. Multi-selects
are asked card by card (see :mod:`ygorl.engine.actions`), so every agent call is
one step.

Engine player 0 always moves first; ``first=1`` seats ``deck_b`` / ``agent_b``
as engine player 0. Results are reported in (a, b) order.

Note: ``DecisionPoint.events`` is the engine's full (server) view, including
hidden information. Per-player observation filtering belongs to the M2
environment layer (T2.2 / T2.5).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import cache

from ygorl import _core, paths
from ygorl.cards.cdb import CardDB
from ygorl.cards.legality import check_deck
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action, DecisionState, make_decision
from ygorl.engine.curriculum import FULL, MODES, allowed_actions, auto_action

MASK64 = (1 << 64) - 1
WIN_REASON_LP = 1  # MSG_WIN reasons written by the core (processor.cpp); 0x10+ come from card scripts
WIN_REASON_DECK_OUT = 2
MAX_CONSECUTIVE_RETRIES = 8


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


def deck_of_seat(first: int, seat: int) -> int:
    """Index into (a, b) of the deck at engine ``seat`` when ``first`` names the deck that moves first.

    Engine seat 0 always moves first, so seat ``p`` holds deck ``(first + p) % 2``; the one place this rule is written.
    """
    return (first + seat) % 2


def seat_of_deck(first: int, deck: int) -> int:
    """Engine seat of deck ``deck`` (0 = a, 1 = b): the inverse of :func:`deck_of_seat` (the rule is its own inverse)."""
    return deck_of_seat(first, deck)


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


@dataclass(frozen=True)
class DecisionPoint:
    index: int  # decision counter over the whole duel (each sub-step counts)
    player: int  # engine player asked to act (0 moves first)
    turn: int
    phase: int
    lp: tuple[int, int]  # indexed by engine player
    decision: M.Decision
    actions: list[Action]
    state: DecisionState
    events: tuple[M.Message, ...]  # messages since the previous decision point
    turn_player: int = 0  # engine player whose turn it is
    augmented_start: bool = False  # DuelConfig.augmented_start
    response_index: int = 0  # position this decision's response takes in DuelResult.responses
    undo: tuple[int, ...] = ()  # actions that only undo the previous step (docs/encoding.md 「撤销类空操作」)


@dataclass
class DuelResult:
    winner: int | None  # 0 = deck_a / agent_a, 1 = deck_b / agent_b, None = draw
    reason: str  # "win", "turn_limit", "decision_limit", "end", "error", "log_exhausted" (replay)
    win_reason: int | None = None  # MSG_WIN reason: WIN_REASON_LP, WIN_REASON_DECK_OUT, or 0x10+ (card effects)
    turns: int = 0
    decisions: int = 0
    lp: tuple[int, int] = (0, 0)  # in (a, b) order
    first: int = 0
    retries: int = 0
    unknown_messages: int = 0
    undecodable_messages: int = 0
    script_errors: list[str] = field(default_factory=list)
    responses: list[bytes] = field(default_factory=list)  # every set_response, in order
    actions: list[int] = field(default_factory=list)  # every agent action index, in order
    message_log: list[bytes] = field(default_factory=list)  # raw engine buffers (record_messages=True)
    steps: list[dict] = field(default_factory=list)  # per-action candidates/choice/probs (record_steps=True)
    error: str = ""
    auto_decisions: int = 0  # decisions the host answered for the opponent (curriculum); their bytes are in responses

    def summary(self) -> str:
        who = {0: "a", 1: "b", None: "draw"}[self.winner]
        return (f"winner={who} reason={self.reason} win_reason={self.win_reason} turns={self.turns} "
                f"decisions={self.decisions} lp={self.lp} retries={self.retries} unknown={self.unknown_messages} "
                f"undecodable={self.undecodable_messages} script_errors={len(self.script_errors)} {self.error}")  # fmt: skip


@cache
def default_cards() -> CardDB:
    return CardDB.load()


@cache
def default_scripts() -> _core.ScriptDirectory:
    return _core.ScriptDirectory([str(p) for p in paths.script_directories()])


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


_NOT_EVENTS = (M.Decision, M.Retry, M.Hint, M.Waiting, M.CardHint, M.PlayerHint, M.ShowHint)  # no game change
_MENUS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
# an effect activated this many times from a menu in one turn is masked there (docs/encoding.md 「撤销类空操作」 rule 3):
# a negated monster can activate an unlimited ignition effect for free forever (Expurrely Happiness); real play
# rarely activates one effect from the menu more than 3 times a turn
MAX_MENU_ACTIVATIONS = 8
# engine steps (process() calls) allowed between two decisions: a core that keeps processing without ever asking a
# player would hang the duel (the decision limit never fires); real chains take a few thousand at most (mirror of
# the C++ g_max_engine_steps)
MAX_ENGINE_STEPS = 100_000
# after this many steps in one SELECT_UNSELECT_CARD selection, unselecting is masked, so the selection can only move
# forward (docs/encoding.md 「撤销类空操作」 rule 5): a policy that never picks the card a finish needs (Swordsoul
# Blackout needs your Wyrm) would otherwise toggle the others until the decision limit
MAX_SELECTION_STEPS = 32
_INVERSE = {"select": "unselect", "unselect": "select"}


def _card_key(action: Action) -> tuple | None:
    c = action.card
    return None if c is None else (c.code, c.loc.controller, c.loc.location, c.loc.sequence)


class DuelTracker:
    """Host-side state of one duel, independent of who drives the core.

    Feed every engine buffer to :meth:`on_buffer`. When :attr:`awaiting`,
    either answer with recorded bytes (:meth:`use_response`) or first let the
    host answer for a restricted opponent (:meth:`auto_response`, curriculum
    modes) and otherwise ask agents: :meth:`point` gives the current decision
    point and :meth:`act` applies an action index, returning the response bytes
    once the decision is complete.
    Engine player indices are used internally; :meth:`finish` reports in
    (a, b) order.
    """

    def __init__(self, config: DuelConfig, first: int, cards, *, record_messages: bool = False,
                 record_steps: bool = False, reference_log: list[bytes] | None = None) -> None:  # fmt: skip
        self.config = config
        self.first = first
        self.cards = cards
        self.record_messages = record_messages
        self.record_steps = record_steps
        self.reference_log = reference_log
        self.result = DuelResult(winner=None, reason="", first=first)
        self.lp = [config.player.starting_lp, config.player.starting_lp]
        self.turn = 0
        self.turn_player = 0
        self.phase = 0
        self.events: list[M.Message] = []
        self.state: DecisionState | None = None
        self.decision: M.Decision | None = None
        self.done = False
        self._engine_winner: int | None = None
        self._last_decision: M.Decision | None = None
        self._consecutive_retries = 0
        self._engine_steps = 0  # process() calls since the last decision
        self._buffers = 0
        self._point: DecisionPoint | None = None
        self._allowed: list[int] | None = None  # point.actions -> state.actions() indices when filtered
        # no-op undo tracking (docs/encoding.md 「撤销类空操作」): the player inside a command just started from a
        # menu, and the last select / unselect of a SELECT_UNSELECT_CARD; any game event clears both
        self._inside: int | None = None
        self._toggle: tuple | None = None
        # rule 5: steps of the current SELECT_UNSELECT_CARD selection (same player, no game event in between)
        self._selection_steps = 0
        self._selection_player = -1
        self._activations: dict[tuple, int] = {}  # (player, card key, description) -> menu activations this turn
        self._activations_turn = -1
        self._stepped = False
        self._learner = seat_of_deck(first, config.learner)  # engine player of the learning deck

    @property
    def awaiting(self) -> bool:
        return not self.done and self.state is not None

    def stop(self, reason: str, error: str = "") -> None:
        self.result.reason = reason
        if error:
            self.result.error = error
        self.done = True
        self.state = None

    def on_buffer(self, buf: bytes, status: int, logs=()) -> None:
        """Consume one engine message buffer and the status of the ``process()`` that produced it."""
        res = self.result
        for t, text in logs:
            if t == _core.LOG_TYPE_ERROR:
                res.script_errors.append(text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text))
        if self.record_messages:
            res.message_log.append(buf)
        if self.reference_log is not None:
            i = self._buffers
            if i >= len(self.reference_log) or self.reference_log[i] != buf:
                raise ValueError(f"replay differs from the reference at message buffer {i}")
        self._buffers += 1
        self.state = None
        decision: M.Decision | None = None
        retried = False
        lp = self.lp
        for msg in M.decode_buffer(buf):
            self.events.append(msg)
            if not isinstance(msg, _NOT_EVENTS):
                self._inside = self._toggle = None
                self._selection_steps = 0
            if isinstance(msg, M.Decision):
                decision = msg
            elif isinstance(msg, M.NewTurn):
                self.turn += 1
                self.turn_player = msg.player
            elif isinstance(msg, M.NewPhase):
                self.phase = msg.phase
            elif isinstance(msg, (M.Damage, M.PayLpCost)):
                lp[msg.player] -= msg.amount  # the core does not clamp at 0
            elif isinstance(msg, M.Recover):
                lp[msg.player] += msg.amount
            elif isinstance(msg, M.LpUpdate):
                lp[msg.player] = msg.amount
            elif isinstance(msg, M.Win):
                self._engine_winner = msg.player if msg.player in (0, 1) else None
                res.win_reason = msg.reason
                res.reason = "win"
            elif isinstance(msg, M.Retry):
                retried = True
                res.retries += 1
            elif isinstance(msg, M.UnknownMessage):
                res.unknown_messages += 1
            elif isinstance(msg, M.UndecodableMessage):
                res.undecodable_messages += 1

        # The core keeps processing after MSG_WIN; like EDOPro's host we stop at the first one.
        if res.reason == "win":
            self.done = True
            return
        if status == _core.DUEL_STATUS_END:
            self.stop("end")
            return
        if status != _core.DUEL_STATUS_AWAITING:
            self._engine_steps += 1
            if self._engine_steps >= MAX_ENGINE_STEPS:
                self.stop("error", f"engine loop: no decision after {MAX_ENGINE_STEPS} engine steps")
            return
        self._engine_steps = 0
        if decision is None and retried:
            self._consecutive_retries += 1
            if self._consecutive_retries > MAX_CONSECUTIVE_RETRIES:
                self.stop("error", "response rejected repeatedly (MSG_RETRY)")
                return
            decision = self._last_decision
        else:
            self._consecutive_retries = 0
        if decision is None:
            self.stop("error", "engine awaits a response but sent no decodable decision")
            return
        self._last_decision = decision
        decision = M.hide_private(decision)  # what the decider may see (the message log keeps the raw bytes)
        if self.turn > self.config.max_turns:
            self.stop("turn_limit")
            return
        self.decision = decision
        if decision.player != self._inside or decision.TYPE in _MENUS:
            self._inside = None  # another player's decision, or back at a menu: not inside a command any more
        if self._toggle is not None and self._toggle[0] != decision.player:
            self._toggle = None
        if decision.TYPE != C.MSG_SELECT_UNSELECT_CARD or decision.player != self._selection_player:
            self._selection_steps = 0  # not the same selection any more
            self._selection_player = decision.player
        self.state = make_decision(decision, self.cards)
        self._point = None
        self._stepped = False  # host answers only fresh decisions, never a half-built multi-select

    def point(self) -> DecisionPoint | None:
        """The current sub-step to ask the deciding agent about (None once the duel had to stop)."""
        if not self.awaiting:
            return None
        if self._point is not None:
            return self._point
        res, state, decision = self.result, self.state, self.decision
        if res.decisions >= self.config.max_decisions:
            self.stop("decision_limit")
            return None
        actions = state.actions()
        if not actions:
            self.stop("error", f"no legal action for {decision.name}")
            return None
        self._allowed = None
        if self._restricted():
            allowed = allowed_actions(self.config.curriculum, decision, actions)
            if len(allowed) < len(actions):
                self._allowed, actions = allowed, [actions[i] for i in allowed]
        self._point = DecisionPoint(res.decisions, decision.player, self.turn, self.phase, (self.lp[0], self.lp[1]),
                                    decision, actions, state, tuple(self.events), self.turn_player,
                                    self.config.augmented_start, len(res.responses), self._undo(decision, actions))  # fmt: skip
        self.events = []
        return self._point

    def act(self, idx, probs=None) -> bytes | None:
        """Apply action ``idx`` to the current point; return the response once the decision is complete."""
        point = self.point()
        if point is None:
            raise RuntimeError("no decision is pending")
        n = len(point.actions)
        if not isinstance(idx, int) or not 0 <= idx < n:
            raise ValueError(f"agent returned action {idx!r}, out of range 0..{n - 1} for {point.decision.name}")
        res = self.result
        if self.record_steps:
            res.steps.append(_step_record(point, idx, probs))
        res.actions.append(idx)
        res.decisions += 1
        self._note_undo(point.decision, point.actions[idx])
        self._point = None
        self._stepped = True
        response = self.state.step(idx if self._allowed is None else self._allowed[idx])
        if response is not None:
            res.responses.append(response)
            self.state = None
        return response

    def _undo(self, decision: M.Decision, actions: list[Action]) -> tuple[int, ...]:
        """Indices of ``actions`` that only undo the previous step (never all of them)."""
        undo = []
        for i, a in enumerate(actions):
            if a.kind == "cancel" and self._inside is not None:
                undo.append(i)  # backs out of the command to the unchanged menu
            elif (self._toggle is not None and decision.TYPE == C.MSG_SELECT_UNSELECT_CARD
                  and (_INVERSE.get(a.kind), _card_key(a)) == self._toggle[1:]):  # fmt: skip
                undo.append(i)  # reverses the previous select / unselect
            elif (decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and a.kind == "unselect"
                  and self._selection_steps >= MAX_SELECTION_STEPS):  # fmt: skip
                undo.append(i)  # a long selection only moves forward from here (rule 5)
            elif decision.TYPE in _MENUS and a.kind == "shuffle":
                undo.append(i)  # reorders the hand, changes nothing else
            elif (decision.TYPE in _MENUS and a.kind == "activate" and self._activations_turn == self.turn
                  and self._activations.get((decision.player, _card_key(a), a.description), 0) >= MAX_MENU_ACTIVATIONS):  # fmt: skip
                undo.append(i)  # the same effect again: repeated activation limit
        return tuple(undo) if len(undo) < len(actions) else ()

    def _note_undo(self, decision: M.Decision, action: Action) -> None:
        if decision.TYPE in _MENUS:
            self._inside = decision.player
            if action.kind == "activate":
                if self._activations_turn != self.turn:
                    self._activations, self._activations_turn = {}, self.turn
                key = (decision.player, _card_key(action), action.description)
                self._activations[key] = self._activations.get(key, 0) + 1
        self._toggle = None
        if decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and action.kind in _INVERSE:
            self._toggle = (decision.player, action.kind, _card_key(action))
        if decision.TYPE == C.MSG_SELECT_UNSELECT_CARD:
            self._selection_steps += 1

    def _restricted(self) -> bool:
        """The pending decision is the opponent's, in the learner's turn, under a restricting curriculum."""
        return (self.config.curriculum != FULL and self.decision is not None and self.decision.player != self._learner
                and self.turn_player == self._learner)  # fmt: skip

    def auto_response(self) -> bytes | None:
        """Answer the pending decision on the opponent's behalf if the curriculum says so; else None.

        The answer is always a passive one (pass / no / cancel, see
        :mod:`ygorl.engine.curriculum`); it is logged in ``responses`` like any
        other, so replays need no knowledge of the curriculum.
        """
        if not self.awaiting or not self._restricted() or self._point is not None or self._stepped:
            return None
        idx = auto_action(self.config.curriculum, self.decision, self.state.actions())
        if idx is None:
            return None
        response = self.state.step(idx)
        assert response is not None, "passive answers complete a decision in one step"
        self.result.responses.append(response)
        self.result.auto_decisions += 1
        self.state = None
        return response

    def use_response(self, response: bytes) -> None:
        """Answer the pending decision with recorded bytes (replay)."""
        self.result.responses.append(response)
        self.state = None
        self.events = []

    def finish(self) -> DuelResult:
        res, lp = self.result, self.lp
        engine_winner = self._engine_winner
        if res.reason in ("turn_limit", "decision_limit", "error"):
            # the turn limit is a rule (the higher LP wins); the decision limit only stops a loop: a draw, so that
            # looping while ahead never counts as a win (mirror of Tracker::winner)
            engine_winner = (None if lp[0] == lp[1] or res.reason != "turn_limit"
                             else (0 if lp[0] > lp[1] else 1))  # fmt: skip
        res.winner = None if engine_winner is None else deck_of_seat(self.first, engine_winner)
        res.turns = self.turn
        res.lp = (lp[seat_of_deck(self.first, 0)], lp[seat_of_deck(self.first, 1)])
        return res


def _step_record(point: DecisionPoint, chosen: int, probs) -> dict:
    def action(a: Action) -> dict:
        d = {
            "kind": a.kind,
            "index": a.index,
            "code": a.card.code if a.card else 0,
            "description": a.description,
            "value": a.value,
        }
        if a.card is not None:
            d["location"] = [a.card.loc.controller, a.card.loc.location, a.card.loc.sequence]
        return d

    rec = {"index": point.index, "player": point.player, "turn": point.turn, "decision": point.decision.name,
           "actions": [action(a) for a in point.actions], "chosen": chosen}  # fmt: skip
    if probs is not None:
        rec["probs"] = [float(p) for p in probs]
    return rec


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
    "DuelTracker",
    "ScriptedAgent",
    "expand_seed",
    "run_duel",
    "shuffle_deck",
]
