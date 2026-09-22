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


@dataclass(frozen=True)
class DuelConfig:
    rule_flags: int = C.DUEL_MODE_MR5
    player: PlayerRules = PlayerRules()
    max_turns: int = 200
    max_decisions: int = 20000

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
    reason: str  # "win", "turn_limit", "decision_limit", "end", "error"
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

    def _setup(self) -> _core.Duel:
        p = self.config.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        core = _core.Duel(expand_seed(self.seed), self.config.rule_flags, player, player, self.cards.to_core(), self.scripts)
        for base in ("constant.lua", "utility.lua"):
            if not core.load_script(base):
                raise RuntimeError(f"failed to load base script {base}")
        for team in (0, 1):
            deck = self.decks[self.deck_of(team)]
            for code in deck.main:
                core.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
            for code in deck.extra:
                core.new_card(team, 0, code, team, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
        core.start()
        return core

    # -- main loop -------------------------------------------------------
    def run(self, agent_a, agent_b) -> DuelResult:
        if self._ran:
            raise RuntimeError("a Duel can only be run once")
        self._ran = True
        agents = (agent_a, agent_b)
        seat = [agents[self.deck_of(0)], agents[self.deck_of(1)]]
        cfg = self.config
        lp = [cfg.player.starting_lp, cfg.player.starting_lp]
        turn, phase = 0, 0
        res = DuelResult(winner=None, reason="", first=self.first)
        engine_winner: int | None = None
        last_decision: M.Decision | None = None
        events: list[M.Message] = []
        consecutive_retries = 0

        core = self._core = self._setup()
        try:
            while True:
                status = core.process()
                for t, text in core.pop_logs():
                    if t == _core.LOG_TYPE_ERROR:
                        res.script_errors.append(text.decode("utf-8", "replace"))
                decision: M.Decision | None = None
                retried = False
                for msg in M.decode_buffer(core.get_message()):
                    events.append(msg)
                    if isinstance(msg, M.Decision):
                        decision = msg
                    elif isinstance(msg, M.NewTurn):
                        turn += 1
                    elif isinstance(msg, M.NewPhase):
                        phase = msg.phase
                    elif isinstance(msg, (M.Damage, M.PayLpCost)):
                        lp[msg.player] -= msg.amount  # the core does not clamp at 0
                    elif isinstance(msg, M.Recover):
                        lp[msg.player] += msg.amount
                    elif isinstance(msg, M.LpUpdate):
                        lp[msg.player] = msg.amount
                    elif isinstance(msg, M.Win):
                        engine_winner = msg.player if msg.player in (0, 1) else None
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
                    break
                if status == _core.DUEL_STATUS_END:
                    res.reason = "end"
                    break
                if status != _core.DUEL_STATUS_AWAITING:
                    continue

                if decision is None and retried:
                    consecutive_retries += 1
                    if consecutive_retries > MAX_CONSECUTIVE_RETRIES:
                        res.reason, res.error = "error", "response rejected repeatedly (MSG_RETRY)"
                        break
                    decision = last_decision
                else:
                    consecutive_retries = 0
                if decision is None:
                    res.reason, res.error = "error", "engine awaits a response but sent no decodable decision"
                    break
                last_decision = decision

                if turn > cfg.max_turns:
                    res.reason = "turn_limit"
                    break

                state = make_decision(decision, self.cards)
                player = decision.player
                while not state.done:
                    if res.decisions >= cfg.max_decisions:
                        break
                    actions = state.actions()
                    if not actions:
                        res.reason, res.error = "error", f"no legal action for {decision.name}"
                        break
                    point = DecisionPoint(res.decisions, player, turn, phase, (lp[0], lp[1]), decision, actions, state, tuple(events))
                    events = []
                    idx = seat[player].act(point)
                    if not isinstance(idx, int) or not 0 <= idx < len(actions):
                        raise ValueError(f"agent returned action {idx!r}, out of range 0..{len(actions) - 1} for {decision.name}")
                    res.actions.append(idx)
                    res.decisions += 1
                    state.step(idx)
                if res.reason == "error":
                    break
                if not state.done:
                    res.reason = "decision_limit"
                    break
                res.responses.append(state.response)
                core.set_response(state.response)
        finally:
            core.close()

        if res.reason in ("turn_limit", "decision_limit", "error"):
            engine_winner = None if lp[0] == lp[1] or res.reason == "error" else (0 if lp[0] > lp[1] else 1)
        res.winner = None if engine_winner is None else self.deck_of(engine_winner)
        res.turns = turn
        res.lp = (lp[self.first], lp[1 - self.first])  # engine player `first` holds deck a
        return res


def run_duel(seed: int, deck_a: Deck, deck_b: Deck, agent_a, agent_b, env: Environment | None = None,
             **kwargs) -> DuelResult:  # fmt: skip
    """Convenience wrapper: ``Duel(seed, env, deck_a, deck_b, **kwargs).run(agent_a, agent_b)``."""
    return Duel(seed, env, deck_a, deck_b, **kwargs).run(agent_a, agent_b)


__all__ = ["DecisionPoint", "Duel", "DuelConfig", "DuelResult", "expand_seed", "run_duel"]
