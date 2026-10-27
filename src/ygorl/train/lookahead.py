"""Look-ahead play: best-of-K at a seat's decisions by snapshot branching (first-seat plan §13).

At a qualifying decision of a seat (:class:`LookaheadConfig`: a turn range, own / opponent turns) the seat's policy
draws K candidate actions (candidate 0 is the action the policy plays without look-ahead, the others are further
draws; duplicates are dropped). With more than one distinct candidate the game is snapshotted
(:meth:`EncodedVecEnv.snapshot <ygorl.env.encoded.EncodedVecEnv.snapshot>`) and each candidate is rolled out by both
seats' own players (the policies, a scripted line where there is one) to the horizon -- the first decision of a later
turn, or the end of the game -- and scored by :func:`lookahead_value`; the game is restored and plays the candidate of
highest score (ties: the earlier candidate, so the policy's own draw when nothing is better). Decisions inside the
rollouts never look ahead and are never seen by the caller.

The rollouts see the whole engine state (the opponent's hand, both deck orders): look-ahead is clairvoyant, an upper
bound on what one-decision selection can buy, and the look-ahead *opponent* it builds is a cheating opponent. That is
acceptable for a training opponent whose job is to react to the learner's actual board.

Random numbers (common random numbers, :class:`Uniforms`): every draw is a hash of the game key and the draw's place
-- a main decision by (seat, the seat's decision count), a candidate by (seat, decision count, candidate), a rollout
decision by (seat, decision count, rollout, rolling seat, its count in the rollout) -- so the same games with and
without look-ahead stay paired until the first changed choice, all candidates of one decision share their rollout
draws, and K = 1 is the plain policy bit for bit.

The decision procedure is a generator (:func:`look_ahead`) that yields environment requests, so any driver that
batches forwards over many slots can run it: :func:`play` here (evaluation) and the rollout collector (a look-ahead
opponent in self-play, ``Assignment.lookahead``). Requests: ``("forward", seat, event)`` -> the seat's action
probabilities (numpy); ``("step", action)`` / ``("restore", sid)`` -> the next event; ``("snapshot",)`` -> the
snapshot id (once its event arrived); ``("release", sid)`` -> None. The generator returns ``(action, info)``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ygorl.engine.outcome import TRUNCATED

_MASK64 = (1 << 64) - 1
ROLLOUT_CAP = 600  # decisions in one rollout before it is scored where it stands (a turn is ~10-150)


def _mix(*keys: int) -> int:
    z = 0
    for k in keys:
        z = ((z ^ (k & _MASK64)) + 0x9E3779B97F4A7C15) & _MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
    return z


def uniform(*keys: int) -> float:
    """A uniform number in [0, 1) from integer keys (splitmix64 chain; 53 bits)."""
    return (_mix(*keys) >> 11) / float(1 << 53)


def sample(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF draw of an action index from ``probs`` (unnormalised is fine) with the uniform ``u``."""
    cdf = np.cumsum(probs, dtype=np.float64)
    return min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(probs) - 1)


_MAIN, _CAND, _ROLL = 1, 2, 3
CANDIDATES = ("sample", "top")


@dataclass(frozen=True)
class Uniforms:
    """The draws of one game (``key``: its seed); see the module docstring."""

    key: int

    def main(self, seat: int, n: int) -> float:
        return uniform(self.key, _MAIN, seat, n)

    def candidate(self, seat: int, n: int, j: int) -> float:
        return self.main(seat, n) if j == 0 else uniform(self.key, _CAND, seat, n, j)

    def rollout(self, seat: int, n: int, m: int, mover: int, t: int) -> float:
        return uniform(self.key, _ROLL, seat, n, m, mover, t)


@dataclass(frozen=True)
class LookaheadConfig:
    """Where a seat looks ahead and how a rollout is scored.

    ``k``: candidates drawn per decision (1 = the plain policy); ``rollouts`` (M): rollouts per candidate, averaged
    (common random numbers across candidates). The decision qualifies when its turn is in
    ``[first_turn, last_turn]`` (``last_turn`` None: no end) and, with ``own_turn`` True / False, only on the seat's own
    / the opponent's turns. Score (:func:`lookahead_value`) at the horizon, from the seat's view:
    ``own_weight`` x its board - the other seat's board + ``lp_scale`` x (LP difference / 1000), ``win_value`` for a
    won game (minus for a lost one, 0 for a draw). ``own_weight`` None: 1 for a decision on the seat's own turn, 0 on
    the opponent's turn (the phase-0 window score, plan §10.2: only the opponent's board counts there, so spending a
    hand trap is not charged as a lost interruption; with ``extra_turns`` above 0 always 1). ``candidates``:
    ``"sample"`` -- K draws from the policy (duplicates dropped: a confident policy has few); ``"top"`` -- the policy's own draw, then the likeliest other
    choosable actions up to K (every choosable action when there are at most K). ``extra_turns``: the horizon is the
    first decision of turn ``turn + 1 + extra_turns`` (0: the end of the decision's turn; 2 on turn 1: the end of the
    opponent's turn 2, i.e. what of the board survived)."""

    k: int = 8
    rollouts: int = 1
    first_turn: int = 1
    last_turn: int | None = None
    own_turn: bool | None = None
    own_weight: float | None = None
    lp_scale: float = 1.0
    win_value: float = 100.0
    candidates: str = "sample"
    extra_turns: int = 0

    def __post_init__(self) -> None:
        if self.candidates not in CANDIDATES:
            raise ValueError(f"candidates must be one of {CANDIDATES}")
        if self.k < 1 or self.rollouts < 1 or self.extra_turns < 0:
            raise ValueError("look-ahead needs k >= 1, rollouts >= 1 and extra_turns >= 0")
        if self.last_turn is not None and self.last_turn < self.first_turn:
            raise ValueError("last_turn must not be before first_turn")

    def applies(self, obs) -> bool:
        g = obs["globals"]
        turn, own = int(g[3]), bool(int(g[2]))
        if turn < self.first_turn or (self.last_turn is not None and turn > self.last_turn):
            return False
        if self.own_turn is not None and own != self.own_turn:
            return False
        return int((obs["action_mask"] != 0).sum()) > 1

    @classmethod
    def segment(cls, name: str, **kw) -> LookaheadConfig:
        """The decomposition's segments: ``t1`` (turn 1), ``t2`` (turn 2), ``t3`` (turns 3 on), ``all``."""
        spans = {"t1": (1, 1), "t2": (2, 2), "t3": (3, None), "all": (1, None)}
        if name not in spans:
            raise ValueError(f"segment must be one of {sorted(spans)}")
        first, last = spans[name]
        return cls(first_turn=first, last_turn=last, **kw)


def candidates(cfg: LookaheadConfig, probs: np.ndarray, mask, draw: Callable[[int], float]) -> list[int]:
    """The distinct candidate actions of a decision; the first is always the policy's own draw (``draw(0)``)."""
    out = [sample(probs, draw(0))]
    if cfg.candidates == "sample":
        for j in range(1, cfg.k):
            a = sample(probs, draw(j))
            if a not in out:
                out.append(a)
        return out
    legal = [int(i) for i in np.flatnonzero(np.asarray(mask))]
    for a in sorted(legal, key=lambda i: (-float(probs[i]), i)):
        if len(out) >= cfg.k:
            break
        if a not in out:
            out.append(a)
    return out


def lookahead_value(ev, seat: int, own_weight: float, cfg: LookaheadConfig, cards, vocab) -> float | None:
    """The horizon score of ``ev`` (a decision or a finished game) for engine ``seat``; None for a game cut by a
    limit or an error (the candidate is then ranked last)."""
    from ygorl.train.rollout import board_score

    if ev.result is not None:
        if str(ev.result.get("reason", "")) in TRUNCATED or ev.result.get("error"):
            return None
        w = ev.result.get("winner")
        return 0.0 if w is None else (cfg.win_value if int(w) == seat else -cfg.win_value)
    obs, priv = ev.obs, ev.privileged
    mine = int(ev.player) == seat  # the event is the seat's own decision: its side is controller 0
    own = board_score(obs, priv, cards, vocab, side=0 if mine else 1)
    other = board_score(obs, priv, cards, vocab, side=1 if mine else 0)
    g = obs["globals"]
    lp = (int(g[5]) - int(g[6])) * (1 if mine else -1)
    return own_weight * own - other + cfg.lp_scale * lp / 1000.0


class GameHooks:
    """Per-game bookkeeping a rollout must carry and roll back: the default has none (two policies)."""

    def save(self) -> Any:
        return None

    def load(self, state: Any) -> None:
        pass

    def scripted(self, ev) -> int | None:
        """The scripted answer to ``ev`` (a line), or None: the seat's policy answers."""
        return None

    def note(self, ev, action: int) -> None:
        """``action`` was chosen at ``ev`` (counts windows, switches a line to a recovery line ...)."""


def look_ahead(cfg: LookaheadConfig, seat: int, n: int, ev, probs: np.ndarray, draws: Uniforms, hooks: GameHooks,
               score: Callable[[Any, int, float], float | None],
               record: bool = False) -> Generator[tuple, Any, tuple[int, dict]]:  # fmt: skip
    """Best-of-K at decision ``ev`` of engine ``seat`` (its ``n``-th decision; ``probs`` its policy there).

    Yields environment requests (module docstring) and returns ``(action, info)``: ``info`` has ``candidates``
    (distinct), ``branched`` (rolled out), ``changed`` (the action is not the policy's own draw), ``steps``
    (decisions stepped in rollouts) and, with ``record``, ``values`` per candidate. On return the environment is
    back at ``ev`` (byte for byte) and ``hooks`` as they were before it: the caller plays the action."""
    cands = candidates(cfg, probs, ev.obs["action_mask"], lambda j: draws.candidate(seat, n, j))
    info = {"candidates": len(cands), "branched": len(cands) > 1, "changed": False, "steps": 0}
    if len(cands) == 1:
        return cands[0], info
    g = ev.obs["globals"]
    turn = int(g[3])
    horizon = turn + cfg.extra_turns
    own_weight = cfg.own_weight if cfg.own_weight is not None else float(bool(int(g[2])) or cfg.extra_turns > 0)
    state = hooks.save()
    sid = yield ("snapshot",)
    values: list[float | None] = []
    fresh = True  # the environment is at the window (just snapshotted / restored)
    for a in cands:
        total, ok = 0.0, True
        for m in range(cfg.rollouts):
            if not fresh:
                yield ("restore", sid)
            fresh = False
            hooks.load(state)
            hooks.note(ev, a)
            cur = yield ("step", a)
            counts: dict[int, int] = defaultdict(int)
            steps = 0
            while True:
                if cur.result is not None or int(cur.obs["globals"][3]) > horizon or steps >= ROLLOUT_CAP:
                    v = score(cur, seat, own_weight)
                    break
                row = hooks.scripted(cur)
                if row is None:
                    p = yield ("forward", int(cur.player), cur)
                    mover = int(cur.player)
                    row = sample(p, draws.rollout(seat, n, m, mover, counts[mover]))
                    counts[mover] += 1
                    hooks.note(cur, row)
                cur = yield ("step", row)
                steps += 1
            info["steps"] += steps + 1
            if v is None:
                ok = False
            else:
                total += v
        values.append(total / cfg.rollouts if ok else None)
    yield ("restore", sid)
    yield ("release", sid)
    hooks.load(state)
    best = 0
    for i, v in enumerate(values):
        if v is not None and (values[best] is None or v > values[best]):
            best = i
    info["changed"] = best != 0
    if record:
        info["values"] = values
        info["actions"] = list(cands)
    return cands[best], info


# ------------------------------------------------------------------------------------------ evaluation driver


@dataclass
class SeatPlayer:
    """Who plays an engine seat: a policy module, optionally looking ahead (``lookahead``)."""

    module: nn.Module
    lookahead: LookaheadConfig | None = None


@dataclass
class GameSetup:
    """One game for :func:`play`: the spec, the two seats' players, an optional scripted line (``line`` a
    :class:`~ygorl.train.lines.LineScript` played by ``line_seat``; the other seat is the learner whose chain
    windows can switch it to a recovery line) and the key of its random numbers."""

    spec: Any
    seats: tuple[SeatPlayer, SeatPlayer]
    key: int
    line: Any = None
    line_seat: int = -1
    info: dict = field(default_factory=dict)


@dataclass
class LookaheadStats:
    qualifying: int = 0  # decisions where a seat's look-ahead applied
    branched: int = 0  # ... with more than one distinct candidate (rolled out)
    changed: int = 0  # ... where the chosen action is not the policy's own draw
    rollout_steps: int = 0  # decisions stepped inside rollouts
    by_turn: dict = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )  # turn -> [qualifying, branched, changed]

    def add(self, turn: int, info: dict) -> None:
        self.qualifying += 1
        self.branched += info["branched"]
        self.changed += info["changed"]
        self.rollout_steps += info["steps"]
        t = self.by_turn[min(turn, 9)]
        t[0] += 1
        t[1] += info["branched"]
        t[2] += info["changed"]

    def as_dict(self) -> dict:
        return {"qualifying": self.qualifying, "branched": self.branched, "changed": self.changed,
                "rollout_steps": self.rollout_steps, "by_turn": {str(k): v for k, v in sorted(self.by_turn.items())}}  # fmt: skip


@dataclass
class Outcome:
    index: int
    winner: int | None
    reason: str
    turns: int
    actions: list[int]  # the main game's actions in order (both seats)
    stats: LookaheadStats
    line: dict | None = None
    info: dict = field(default_factory=dict)

    def score(self, seat: int) -> float:
        """1 / 0.5 / 0 for engine ``seat`` (0.5: no winner, a limit or an error)."""
        if self.reason != "win" or self.winner not in (0, 1):
            return 0.5
        return float(self.winner == seat)


class LineHooks(GameHooks):
    """A scripted line on ``line_seat`` (:class:`~ygorl.train.lines.LinePlayer`); the learner's chain windows are
    counted as the rollout collector counts them and an activation switches the line to its recovery line."""

    def __init__(self, player, line_seat: int) -> None:
        self.line = player
        self.line_seat = line_seat
        self.windows = 0
        self.occurrences: dict[int, int] = {}

    def save(self):
        return (self.line.copy(), self.windows, dict(self.occurrences))

    def load(self, state) -> None:
        line, self.windows, occ = state
        self.line.restore(line)
        self.occurrences = dict(occ)

    def scripted(self, ev) -> int | None:
        if int(ev.player) != self.line_seat:
            return None
        return self.line.act(ev.obs)

    def note(self, ev, action: int) -> None:
        from ygorl.train.rollout import RolloutCollector

        if int(ev.player) == self.line_seat:
            return
        _, effect, occurrence, counted = RolloutCollector._count_window(self, ev)
        if counted:
            RolloutCollector._line_interrupt(self, ev, action, effect, occurrence)


class _Game:
    __slots__ = ("setup", "index", "env_id", "hooks", "decisions", "actions", "stats", "co", "co_seat", "co_ev",
                 "co_turn", "wait_sid", "t0")  # fmt: skip

    def __init__(self, setup: GameSetup, index: int, env_id: int, hooks: GameHooks) -> None:
        self.setup, self.index, self.env_id, self.hooks = setup, index, env_id, hooks
        self.decisions = [0, 0]
        self.actions: list[int] = []
        self.stats = LookaheadStats()
        self.co = None  # the running look-ahead (generator)
        self.co_seat = -1
        self.co_ev = None  # the main decision it answers
        self.co_turn = 0
        self.wait_sid: int | None = None  # a snapshot whose event is awaited


@torch.no_grad()
def play(env, games: Sequence[GameSetup], *, cards=None, vocab=None, min_batch: int | None = None,
         device: str = "cpu", progress: Callable[[int, int], None] | None = None) -> tuple[list[Outcome], dict]:  # fmt: skip
    """Play ``games`` on ``env`` (an :class:`~ygorl.env.encoded.EncodedVecEnv` with ``snapshots=True`` and
    ``privileged=True`` when any seat looks ahead; ``skip_forced`` as the policies were trained). Returns one
    :class:`Outcome` per game (in order) and timing (``seconds``, ``forwards`` rows, ``main_decisions``)."""
    from ygorl.nets.batch import collate, policy_logits
    from ygorl.train.lines import LinePlayer

    looks = any(s.lookahead is not None for g in games for s in g.seats)
    if looks and not (getattr(env, "snapshots", False) and getattr(env, "privileged", False)):
        raise ValueError("look-ahead needs EncodedVecEnv(snapshots=True, privileged=True)")
    cards = cards if cards is not None else env.cards
    vocab = vocab if vocab is not None else env.vocab
    min_batch = max(1, min_batch or env.num_envs // 2)
    outcomes: list[Outcome | None] = [None] * len(games)
    queue = iter(range(len(games)))
    running: dict[int, _Game] = {}
    timing = {"forward_rows": 0, "main_decisions": 0, "forward_seconds": 0.0}
    t0 = time.perf_counter()

    def launch(env_id: int) -> None:
        for i in queue:
            setup = games[i]
            if setup.line is not None:
                player = LinePlayer(setup.line, bool(env.skip_forced), first_seat=setup.line_seat == 1)
                hooks: GameHooks = LineHooks(player, setup.line_seat)
            else:
                hooks = GameHooks()
            running[env_id] = _Game(setup, i, env_id, hooks)
            env.reset(env_id, setup.spec)
            return
        running.pop(env_id, None)

    def finish(game: _Game, res: dict) -> None:
        w = res.get("winner")
        reason = str(res.get("reason", ""))
        line = game.hooks.line.finish() if isinstance(game.hooks, LineHooks) else None
        outcomes[game.index] = Outcome(game.index, None if w is None or reason == "error" else int(w), reason,
                                       int(res.get("turns", 0)), game.actions, game.stats, line,
                                       dict(game.setup.info))  # fmt: skip
        if progress is not None:
            progress(sum(o is not None for o in outcomes), len(games))

    def step_main(game: _Game, ev, a: int) -> None:
        seat = int(ev.player)
        game.hooks.note(ev, a)
        game.decisions[seat] += 1
        game.actions.append(int(a))
        timing["main_decisions"] += 1
        env.step(game.env_id, int(a))

    forwards: list[tuple[_Game, int, Any, str]] = []  # (game, seat, event, "main" / "start" / "co")

    def advance(game: _Game, value) -> None:
        """Run the game's look-ahead until it waits for an event or a forward, or returns."""
        while True:
            try:
                req = game.co.send(value)
            except StopIteration as stop:
                a, info = stop.value
                game.stats.add(game.co_turn, info)
                ev = game.co_ev
                game.co = game.co_ev = None
                step_main(game, ev, a)
                return
            kind = req[0]
            if kind == "forward":
                forwards.append((game, req[1], req[2], "co"))
                return
            if kind == "step":
                env.step(game.env_id, int(req[1]))
                return
            if kind == "restore":
                env.restore(game.env_id, req[1])
                return
            if kind == "snapshot":
                game.wait_sid = env.snapshot(game.env_id)
                return
            if kind == "release":
                env.release(game.env_id, req[1])
                value = None
                continue
            raise RuntimeError(f"unknown look-ahead request {kind!r}")

    def on_event(ev) -> None:
        game = running[ev.env_id]
        if game.co is not None:
            if game.wait_sid is not None:
                sid, game.wait_sid = game.wait_sid, None
                advance(game, sid)
            else:
                advance(game, ev)
            return
        if ev.result is not None:
            finish(game, ev.result)
            launch(ev.env_id)
            return
        row = game.hooks.scripted(ev)
        if row is not None:
            step_main(game, ev, row)
            return
        seat = int(ev.player)
        cfg = game.setup.seats[seat].lookahead
        forwards.append((game, seat, ev, "start" if cfg is not None and cfg.applies(ev.obs) else "main"))

    def run_forwards() -> None:
        batch, forwards[:] = list(forwards), []
        groups: dict[int, list[int]] = defaultdict(list)
        for k, (game, seat, ev, _) in enumerate(batch):
            groups[id(game.setup.seats[seat].module)].append(k)
        tf = time.perf_counter()
        probs: list[np.ndarray | None] = [None] * len(batch)
        for ks in groups.values():
            game, seat, _, _ = batch[ks[0]]
            module = game.setup.seats[seat].module
            logits = policy_logits(module, collate([batch[k][2].obs for k in ks], device)).float()
            p = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
            for r, k in enumerate(ks):
                probs[k] = p[r]
        timing["forward_seconds"] += time.perf_counter() - tf
        timing["forward_rows"] += len(batch)
        for (game, seat, ev, kind), p in zip(batch, probs, strict=True):
            draws = Uniforms(game.setup.key)
            n = game.decisions[seat]
            if kind == "main":
                step_main(game, ev, sample(p, draws.main(seat, n)))
            elif kind == "start":
                cfg = game.setup.seats[seat].lookahead

                def score(cur, s, w, cfg=cfg):
                    return lookahead_value(cur, s, w, cfg, cards, vocab)

                game.co = look_ahead(cfg, seat, n, ev, p, draws, game.hooks, score)
                game.co_seat, game.co_ev, game.co_turn = seat, ev, int(ev.obs["globals"][3])
                advance(game, None)
            else:
                advance(game, p)

    for m in {id(s.module): s.module for g in games for s in g.seats}.values():
        m.eval()
    for env_id in range(env.num_envs):
        launch(env_id)
    while running:
        if forwards:
            events = env.recv(0) if env.pending() else []
        else:
            waiting = len(running)
            events = env.recv(min(min_batch, waiting))
        for ev in events:
            on_event(ev)
        if forwards:
            run_forwards()
    timing["seconds"] = time.perf_counter() - t0
    return outcomes, timing  # type: ignore[return-value]


def paired(a: Sequence[Outcome], b: Sequence[Outcome], seats: Sequence[int]) -> dict:
    """B minus A, game by game, for the player of ``seats[i]`` in game ``i``: mean score difference, its standard
    error, the games each side won where the other did not."""
    d = np.array([y.score(s) - x.score(s) for x, y, s in zip(a, b, seats, strict=True)])
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"games": n, "diff": float(d.mean()) if n else float("nan"), "se": se, "b_better": int((d > 0).sum()),
            "a_better": int((d < 0).sum())}  # fmt: skip


__all__ = ["GameHooks", "GameSetup", "LineHooks", "LookaheadConfig", "LookaheadStats", "Outcome", "SeatPlayer",
           "Uniforms", "look_ahead", "lookahead_value", "paired", "play", "sample", "uniform"]  # fmt: skip
