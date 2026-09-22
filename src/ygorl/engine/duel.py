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
    max_decisions: int = 20000
    shuffle_decks: bool = True  # host-side shuffle of each main deck before loading

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
    ) -> None:
        if first not in (0, 1):
            raise ValueError("first must be 0 (deck_a starts) or 1 (deck_b starts)")
        self.seed = seed
        self.env = env
        self.decks = (deck_a, deck_b)
        self.cards = cards if cards is not None else default_cards()
        self.scripts = scripts if scripts is not None else default_scripts()
        self.config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
        self.first = first
        self.record_messages = record_messages
        self.record_steps = record_steps
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
        return (self.first + player) % 2

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
        core = _core.Duel(expand_seed(self.seed), self.config.rule_flags, player, player, self.cards.to_core(), self.scripts)
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
        """Play the duel, asking ``agent_a`` / ``agent_b`` for every decision."""
        agents = (agent_a, agent_b)
        return self._loop(seat=(agents[self.deck_of(0)], agents[self.deck_of(1)]))

    def replay(self, responses: list[bytes], reference_log: list[bytes] | None = None) -> DuelResult:
        """Feed recorded ``set_response`` payloads instead of asking agents.

        Stops with reason ``log_exhausted`` when the engine asks for more
        responses than were recorded. With ``reference_log`` (a previous
        ``message_log``), every engine message buffer is compared as it is
        produced and a :class:`ValueError` is raised at the first difference.
        """
        return self._loop(responses=list(responses), reference_log=reference_log)

    def _loop(self, seat=None, responses: list[bytes] | None = None, reference_log: list[bytes] | None = None) -> DuelResult:
        if self._ran:
            raise RuntimeError("a Duel can only be run once")
        self._ran = True
        tracker = self.tracker(reference_log=reference_log)
        core = self._core = self._setup()
        try:
            while True:
                status = core.process()
                tracker.on_buffer(core.get_message(), status, core.pop_logs())
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
                    response = None
                    while response is None and not tracker.done:
                        point = tracker.point()
                        if point is None:
                            break
                        agent = seat[point.player]
                        response = tracker.act(agent.act(point), getattr(agent, "last_probs", None))
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


class DuelTracker:
    """Host-side state of one duel, independent of who drives the core.

    Feed every engine buffer to :meth:`on_buffer`. When :attr:`awaiting`,
    either answer with recorded bytes (:meth:`use_response`) or ask agents:
    :meth:`point` gives the current decision point and :meth:`act` applies an
    action index, returning the response bytes once the decision is complete.
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
        self.phase = 0
        self.events: list[M.Message] = []
        self.state: DecisionState | None = None
        self.decision: M.Decision | None = None
        self.done = False
        self._engine_winner: int | None = None
        self._last_decision: M.Decision | None = None
        self._consecutive_retries = 0
        self._buffers = 0
        self._point: DecisionPoint | None = None

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
            if isinstance(msg, M.Decision):
                decision = msg
            elif isinstance(msg, M.NewTurn):
                self.turn += 1
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
            return
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
        if self.turn > self.config.max_turns:
            self.stop("turn_limit")
            return
        self.decision = decision
        self.state = make_decision(decision, self.cards)
        self._point = None

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
        self._point = DecisionPoint(res.decisions, decision.player, self.turn, self.phase, (self.lp[0], self.lp[1]),
                                    decision, actions, state, tuple(self.events))  # fmt: skip
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
        self._point = None
        response = self.state.step(idx)
        if response is not None:
            res.responses.append(response)
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
            engine_winner = None if lp[0] == lp[1] or res.reason == "error" else (0 if lp[0] > lp[1] else 1)
        res.winner = None if engine_winner is None else (self.first + engine_winner) % 2
        res.turns = self.turn
        res.lp = (lp[self.first], lp[1 - self.first])  # engine player `first` holds deck a
        return res


def _step_record(point: DecisionPoint, chosen: int, probs) -> dict:
    def action(a: Action) -> dict:
        d = {"kind": a.kind, "index": a.index, "code": a.card.code if a.card else 0, "description": a.description, "value": a.value}
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


__all__ = ["DecisionPoint", "Duel", "DuelConfig", "DuelResult", "DuelTracker", "ScriptedAgent", "expand_seed", "run_duel",
           "shuffle_deck"]
