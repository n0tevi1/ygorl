"""Rollout collection for PPO self-play (T4b.4) in the layout of docs/training.md §1.

:class:`RolloutCollector` drives an :class:`ygorl.env.encoded.EncodedVecEnv` (or anything with its
``reset`` / ``step`` / ``recv`` protocol, e.g. :class:`ygorl.train.toy.NimEnv`) and returns a
:class:`Rollout`: time-major ``[T, B]`` tensors, one column per environment slot, ``T`` learner decisions
per column ("fixed-length segments with auto-reset").

- **Self-play** (``Assignment.opponent is None``): the current policy plays both seats and every decision
  is a row; the rows of both seats interleave in time order in one column (the per-seat sign σ handles it).
- **Snapshot opponent** (``opponent = <snapshot id>``): only the learner seat's decisions are rows; the
  snapshot's decisions are part of the environment (docs/training.md: "只收学习方的行"). They are played
  with ``opponents(id).policy_logits`` and counted in ``Rollout.opponent_decisions``.
- **Scripted line** (``Assignment.line``, a :class:`ygorl.train.lines.LineScript`): a pool game whose opponent seat
  opens with a solver demonstration (:class:`~ygorl.train.lines.LinePlayer` picks the candidate matching each step)
  and is played by the snapshot once the line ends or breaks. The learner's rows carry the line's id
  (``Rollout.line_id``, -1 elsewhere) and ``Rollout.line_window`` (its chain-window decisions of the game, 1-based;
  0 on other decisions); a chain window with a real choice (two or more choosable rows) also carries the identity
  of the effect it answers, ``Rollout.line_effect`` (:func:`ygorl.train.lines.chain_effect`, -1 elsewhere) and
  ``Rollout.line_occurrence`` (how many times the seat has answered that effect in the game, 1-based; 0 elsewhere),
  which is what the solver-window labels are keyed by (train.line_labels: a plain window count misaligns once the
  learner's own deck gives it prompts the solver's passive responder never saw). ``FinishedGame.line`` records how
  the line went. In a first-seat scripted game (``learner_seat`` 0: the learner moves first, the line is played on the
  opponent's turn 2) only the learner's chain windows of turn 2 carry an effect and an occurrence, so the count
  starts where the line does, as the solver's responder's did; ``Rollout.line_seat`` is the learner's seat of the
  scripted game (1: the line moved first; -1 outside scripted games). When the learner activates a card at such a
  window and the script has a recovery line for it (:class:`~ygorl.train.lines.RecoveryLine`), the line player
  switches to that line (:meth:`~ygorl.train.lines.LinePlayer.interrupt`); ``FinishedGame.line`` then counts the
  switch (``recoveries``).
- **Terminal rows**: the result is attached to the column's last row of that game: ``done``; reward +1 / -1
  / 0 from that row's seat. Games cut by a limit or an engine error (:data:`TRUNCATION_REASONS`) are
  **truncations**, not outcomes: ``done`` and ``truncated``, reward 0; the estimator bootstraps them from the
  critic (:func:`ygorl.train.advantages.estimate`). An LP-decided limit never becomes a win or a loss.
- **Segment end**: a column is complete when it has ``T`` rows and its environment waits on a learner
  decision; that pending decision is the bootstrap state (``bootstrap_*``), and the next ``collect`` starts
  the column from it. Complete columns pause while the others fill up.

Behaviour log-probabilities, the full candidate distribution and the critic's Q / V are recorded at acting
time (one forward pass per batch of ready decisions).

**Window branching** (``branch=``:class:`BranchConfig`, docs/training.md §8.1 「窗口分支」, first-seat plan §5.1 A):
at a learner chain window of a scripted game on the line's turn (the opponent's turn: turn 2 when the learner moved
first, turn 1 otherwise) with a pass row and at least one activation, the slot is snapshotted
(``env.snapshot``), each candidate of the branch set (the pass and the ``k - 1`` activations the policy likes most) is
stepped and played on ``rollouts`` times to the horizon (the learner's first decision of its next turn, or the end of
the game), scored, and the slot restored (``env.restore``); the row's sampled action is then stepped as usual and the
snapshot released. Inside a branch the learner samples from a branch generator seeded by the window and the rollout
(common random numbers: every candidate of rollout m sees the same seed), the snapshot opponent too, and the line
player's state (:meth:`ygorl.train.lines.LinePlayer.copy` / ``restore``, recovery switches included) and the slot's
window counters go back with the snapshot. Branch decisions are not rows. The row keeps ``Rollout.branch_value``
(each candidate's mean value, NaN elsewhere), ``branch_valid`` and ``branch_pass``. With ``min_batch >= num_envs``
the other slots wait while a branch runs, so the main game's rows are bit for bit those of a collection without
branching; with smaller batches they keep playing (the branches only add forward rows).
"""

from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ygorl.engine.constants import MSG_SELECT_CHAIN
from ygorl.engine.outcome import TRUNCATED
from ygorl.env.driver import EnvStalled, StallWatch
from ygorl.nets.batch import policy_logits
from ygorl.env.encoding import ACTION_KINDS, CARD_INDEX, CONTROLLER, LOCATION, POSITION, SEQUENCE
from ygorl.train.lines import CHAIN_KIND, LinePlayer, chain_effect

TRUNCATION_REASONS = TRUNCATED  # games the host stopped: bootstrapped, not terminal (ygorl.engine.outcome)
PASS_KIND = ACTION_KINDS.index("pass") + 1  # the encoded kind of a chain window's pass row
HORIZONS = ("board", "critic", "both")
_MASK64 = (1 << 64) - 1


def _mix(*keys: int) -> int:
    """A 63-bit seed from ``keys`` (a splitmix64 chain; the branch generators' seeds)."""
    z = 0
    for k in keys:
        z = ((z ^ (k & _MASK64)) + 0x9E3779B97F4A7C15) & _MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
    return z >> 1


@dataclass(frozen=True)
class BranchConfig:
    """Window branching in the collector (first-seat plan §5.1 A, docs/training.md §8.1 「窗口分支」).

    ``fraction``: share of the qualifying windows that are branched (an independent draw; 0 = none, the rollout is
    unchanged); ``k``: candidates per window (the pass and the ``k - 1`` activations of highest policy probability;
    2 = fire / hold); ``rollouts`` (M): rollouts per candidate, averaged, with common random numbers across the
    candidates; ``horizon``: how a rollout that reaches the learner's next turn is scored -- ``"board"`` (minus
    ``board_scale`` times the line seat's board score, :func:`board_score`, as the phase-0 probe scores it; the plan's
    §10.5 item 2: the critic cannot tell there whether the opponent's turn worked), ``"critic"`` (the learner's
    critic V) or ``"both"`` (V minus the board term); a decided game is +1 / -1 / 0 for the learner whatever the
    horizon, a game cut by a limit or an error invalidates the window. ``require_recovery``: only branch windows at
    which the line is still being played and has a recovery line for every activation of the branch set (§5.1 risk 1:
    a snapshot that takes over a broken line favours firing). ``cards``: card passwords with solver coverage (every
    recoverable occurrence of the card has its recovery lines, so a window without one is a window where firing really
    breaks the line); non-empty: only branch windows at which every activation of the branch set (the pass not
    counted) is one of these cards -- the activations with a recovery line take it, the others leave the rest of the
    game to the snapshot (plan §11.8: a recovery-only branching never teaches the windows to fire). With
    ``require_recovery`` too, a window must meet both; () = no card condition.
    """

    fraction: float = 1.0
    k: int = 2
    rollouts: int = 1
    horizon: str = "board"
    require_recovery: bool = True
    board_scale: float = 0.1
    cards: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1:
            raise ValueError("branch fraction must be in [0, 1]")
        if self.k < 2 or self.rollouts < 1:
            raise ValueError("branching needs k >= 2 candidates and rollouts >= 1")
        if self.horizon not in HORIZONS:
            raise ValueError(f"branch horizon must be one of {HORIZONS}, got {self.horizon!r}")
        if not self.board_scale > 0:
            raise ValueError("branch board_scale must be positive")


def board_score(obs, privileged, cards, vocab, side: int = 1) -> float:
    """The board of the decider's opponent (the line seat, at the learner's horizon) in the phase-0 probe's units
    (tools/window_counterfactual_probe.py ``board_score``): 3 per interruption (:func:`ygorl.solver.blocking.
    board_interruptions`: face-up monsters and spells / traps, set spells / traps, cards in hand, cards in the
    graveyard) + 1 per monster + 0.5 per spell / trap zone card. Read from the encoded card table; the hand and the
    set spells / traps from the privileged tensors (training mode), without them those are unknown (no
    interruption). ``side`` 0 scores the decider's own board instead (its hand and set cards are in the observation;
    look-ahead, :mod:`ygorl.train.lookahead`)."""
    from ygorl.engine import constants as C
    from ygorl.solver.blocking import board_interruptions

    def code(index: int) -> int:
        return vocab.password(index) if index > 1 else 0

    monsters, spells, grave, own_hand = [], [], [], []
    for r in obs["cards"]:
        loc = int(r[LOCATION])
        if not loc or int(r[CONTROLLER]) != side:
            continue
        c, pos = code(int(r[CARD_INDEX])), int(r[POSITION])
        if loc == 3:
            monsters.append({"code": c, "position": C.POS_FACEUP_ATTACK if c and pos in (1, 3) else C.POS_FACEDOWN})
        elif loc == 4:
            spells.append({"code": c, "position": C.POS_FACEDOWN if pos == 2 or not c else C.POS_FACEUP_ATTACK,
                           "sequence": int(r[SEQUENCE])})  # fmt: skip
        elif loc == 5 and c:
            grave.append(c)
        elif loc == 2 and c and side == 0:
            own_hand.append(c)
    hand = own_hand
    if privileged is not None and side == 1:
        hand = [c for c in (code(int(row[0])) for row in privileged["op_hand"]) if c]
        op_set = privileged["op_set"]
        for z in spells:  # a set spell / trap: its identity is ground truth (rows 7-14 = spell / trap zones 0-7)
            if not z["code"] and 7 + z["sequence"] < len(op_set):
                z["code"] = code(int(op_set[7 + z["sequence"]][0]))
                z["position"] = C.POS_FACEDOWN
    zones = {"mzone": [m for m in monsters if m["code"]], "szone": [z for z in spells if z["code"]], "hand": hand,
             "grave": grave}  # fmt: skip
    bi = board_interruptions({"players": [zones]}, cards, 0)
    return 3 * bi.interruptions + len(monsters) + 0.5 * len(spells)


class EncodedBranchHooks:
    """The game-specific parts of window branching, on encoded observations (EncodedVecEnv)."""

    def __init__(self, vocab, cards) -> None:
        self.vocab = vocab
        self.cards = cards

    def candidates(self, slot, ev, probs: Tensor, k: int, require_recovery: bool, effect: int, occurrence: int,
                   cards: frozenset[int] | tuple[int, ...] = ()) -> tuple[int, list[int]] | None:  # fmt: skip
        """``(pass row, branch set)`` at a qualifying window, None elsewhere: a chain window of a scripted game on the
        line's turn (not the learner's) with a pass row and an activation; the set is the pass and the ``k - 1``
        activations of highest probability. ``require_recovery``: the line must still be played and have a recovery
        line for each of those activations at this window (``effect`` / ``occurrence``, :meth:`recovered`).
        ``cards`` (non-empty): each of those activations must be of a card with one of these passwords."""
        obs = ev.obs
        g = obs.get("globals")
        line = slot.line
        if g is None or line is None or int(g[18]) != MSG_SELECT_CHAIN or int(g[2]):
            return None
        if int(g[3]) != (2 if line.first_seat else 1):
            return None
        legal = np.flatnonzero(obs["action_mask"])
        if len(legal) < 2:
            return None
        kinds = obs["actions"][:, 0]
        passes = [int(i) for i in legal if int(kinds[i]) == PASS_KIND]
        fires = [int(i) for i in legal if int(kinds[i]) == CHAIN_KIND]
        if not passes or not fires:
            return None
        fires = sorted(fires, key=lambda i: (-float(probs[i]), i))[: k - 1]
        if cards and any(self._password(int(obs["actions"][i][2])) not in cards for i in fires):
            return None
        if require_recovery and not self.recovered(slot, ev, fires, effect, occurrence):
            return None
        return passes[0], [passes[0], *fires]

    def _password(self, index: int) -> int:
        return int(self.vocab.password(index)) if index > 1 else 0  # 0 / 1: padding / unknown card

    def recovered(self, slot, ev, fires: list[int], effect: int, occurrence: int) -> bool:
        """Whether the line is still played and has a recovery line for each activation row of ``fires`` at this
        window."""
        line = slot.line
        if line is None or line.status != "playing" or effect < 0:
            return False
        return all((effect, occurrence, int(ev.obs["actions"][i][2])) in line.recovery for i in fires)

    def turn(self, ev) -> int:
        return int(ev.obs["globals"][3])

    def horizon(self, state, ev) -> bool:
        """The learner's first decision of a later turn of its own."""
        g = ev.obs["globals"]
        return int(g[3]) > state.turn and int(g[2]) == 1

    def board(self, ev) -> float:
        return board_score(ev.obs, ev.privileged, self.cards, self.vocab)


class RolloutStalled(EnvStalled):
    """No environment produced an event for ``stall_timeout`` during a rollout (:class:`ygorl.env.driver.EnvStalled`:
    save, then end the process with ``os._exit``)."""


@dataclass(frozen=True)
class Assignment:
    """One game to play in an environment slot: the env spec and who plays the other seat."""

    spec: Any  # ``GameSpec`` for EncodedVecEnv (a pile size for NimEnv)
    opponent: int | None = None  # None: the current policy on both seats; else a snapshot id (pool opponent)
    learner_seat: int = 0  # engine seat of the learner in a pool game (ignored in self-play)
    info: Mapping[str, Any] = field(default_factory=dict)  # free-form labels (deck names ...) for the logs
    line: Any = None  # a LineScript the opponent seat opens with (needs a snapshot opponent); None: none
    # a look-ahead opponent (ygorl.train.lookahead.LookaheadConfig; needs a snapshot opponent, no line, an environment
    # with snapshots): at its qualifying decisions the snapshot plays best-of-K by snapshot branching; None: none
    lookahead: Any = None

    def is_learner(self, seat: int) -> bool:
        return self.opponent is None or seat == self.learner_seat


@dataclass
class FinishedGame:
    assignment: Assignment
    winner: int | None  # engine seat, None = draw (as reported by the environment)
    reason: str
    truncated: bool  # cut by a limit / error: not a win or a loss for training
    learner_rows: int  # training rows the game contributed (across rollouts)
    result: dict
    line: dict | None = None  # scripted game: LinePlayer.finish() (id, status completed / broken / playing /
    # recovering, ...; with recovery lines also recoveries, recovered_at, recovery_window)

    @property
    def line_seat(self) -> int:
        """The learner's seat of a scripted game (0: it moved first, the line was played on turn 2); -1 otherwise."""
        return -1 if self.line is None else self.assignment.learner_seat

    @property
    def learner_score(self) -> float | None:
        """Pool games: 1 / 0.5 / 0 for the learner (by the reported winner); None in self-play."""
        a = self.assignment
        if a.opponent is None:
            return None
        return 0.5 if self.winner is None else float(self.winner == a.learner_seat)


@dataclass
class Rollout:
    """One segment in the layout of docs/training.md §1 (row ``t * B + b`` of ``obs`` is ``[t, b]``)."""

    obs: dict[str, Tensor]  # collated observations, [T * B, ...]
    privileged: Any  # collated opponent ground truth for the critic, or None
    players: Tensor  # [T, B] engine seat of each row
    actions: Tensor  # [T, B]
    action_mask: Tensor  # [T, B, A]
    probs: Tensor  # [T, B, A] behaviour policy
    log_probs: Tensor  # [T, B] behaviour log-probability of the taken action
    q: Tensor  # [T, B, A] critic Q at acting time
    values: Tensor  # [T, B] critic V at acting time
    rewards: Tensor  # [T, B] terminal reward from the row's seat (0 elsewhere and on truncations)
    dones: Tensor  # [T, B]
    truncated: Tensor  # [T, B] subset of dones: cut by a limit, bootstrap from the critic
    bootstrap_player: Tensor  # [B]
    bootstrap_mask: Tensor  # [B, A]
    bootstrap_probs: Tensor  # [B, A]
    bootstrap_q: Tensor  # [B, A]
    bootstrap_value: Tensor  # [B]
    games: list[FinishedGame]
    opponent_decisions: int  # snapshot-opponent decisions stepped (not rows)
    seconds: float
    complete_games: bool = False
    discarded_rows: int = 0  # complete-game mode: rows of truncated games (still charged as decisions)
    line_id: Tensor | None = None  # [T, B] id of the scripted line of the row's game, -1 without one
    line_window: Tensor | None = None  # [T, B] the seat's chain-window decisions so far in its game (1-based); else 0
    line_effect: Tensor | None = None  # [T, B] effect (card vocab index) a chain window answers; -1 elsewhere
    line_occurrence: Tensor | None = None  # [T, B] answers to that effect so far in the game (1-based); 0 elsewhere
    line_seat: Tensor | None = None  # [T, B] the learner's seat of the row's scripted game (0 / 1), -1 without one
    # window branching (BranchConfig; None without it): each branch-set candidate's mean value from the learner's view
    # (NaN off the set and on rows without a branch), whether the row was branched with every rollout valid, and the
    # pass row of the set (-1 without one); branch_windows / branch_decisions count the branched windows and the engine
    # decisions stepped inside branches (not rows, not in ``decisions``)
    branch_value: Tensor | None = None  # [T, B, A]
    branch_valid: Tensor | None = None  # [T, B]
    branch_pass: Tensor | None = None  # [T, B]
    branch_windows: int = 0
    branch_decisions: int = 0
    branch_no_recovery: int = 0  # branched windows with an activation of the set without a recovery line
    # look-ahead opponents (Assignment.lookahead): their qualifying decisions, those rolled out (more than one distinct
    # candidate), those where the chosen action is not the snapshot's own draw, and decisions stepped in their rollouts
    lookahead_decisions: int = 0
    lookahead_branched: int = 0
    lookahead_changed: int = 0
    lookahead_steps: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.players.shape)  # type: ignore[return-value]

    @property
    def decisions(self) -> int:
        """All decisions stepped in the environment for this segment (rows + opponent decisions)."""
        return self.players.numel() + self.opponent_decisions + self.discarded_rows


@dataclass
class _Row:
    obs: dict
    privileged: Any
    player: int
    action: int
    log_prob: float
    probs: Tensor
    q: Tensor
    value: float
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    line_id: int = -1
    line_window: int = 0
    line_effect: int = -1
    line_occurrence: int = 0
    line_seat: int = -1
    branch_value: Tensor | None = None
    branch_valid: bool = False
    branch_pass: int = -1


@dataclass
class _BranchState:
    """A window being branched (``_Slot.branch``): the row, its sampled action (stepped once the branches are done),
    the branch set, and where the branching stands (candidate ``k`` of rollout ``m``)."""

    row: _Row | None
    window: Any  # the window's event (what every restore returns to)
    action: int
    candidates: list[int]
    pass_row: int
    turn: int
    seed: int
    counted: bool  # a counted window of the line (effect / occurrence): an activation may switch it to recovery
    effect: int
    occurrence: int
    windows: int  # the slot's window counters at the window (after counting it)
    occurrences: dict[int, int]
    line_pre: LinePlayer | None  # the line player before the row's action told it anything
    line_post: LinePlayer | None = None  # ... and after (what the main game goes on with)
    sid: int = -1
    k: int = 0
    m: int = 0
    phase: str = "snapshot"  # "snapshot" / "restore": waiting for the window's observation; "run": rolling out
    sums: list[float] = field(default_factory=list)
    ok: bool = True
    recovered: bool = True  # every activation of the set has a recovery line here (else the snapshot takes over)
    gen: torch.Generator = field(default_factory=torch.Generator)


@dataclass
class _LookState:
    """A look-ahead opponent's decision in progress: the generator (:func:`ygorl.train.lookahead.look_ahead`), the
    decision's event, and a snapshot whose event is awaited."""

    gen: Any
    ev: Any
    turn: int
    wait_sid: int | None = None


@dataclass
class _Slot:
    assignment: Assignment
    opponent: nn.Module | None = None  # the snapshot, resolved when the game starts (it may leave the pool)
    rows: int = 0  # rows of the current game, across rollouts
    rows_here: int = 0  # rows of the current game in the column being filled
    line: LinePlayer | None = None  # scripted game: the opponent seat's line
    windows: int = 0  # the learner's chain-window decisions so far in this game
    occurrences: dict[int, int] = field(default_factory=dict)  # effect (vocab index) -> chain windows answered on it
    serial: int = 0  # the game's number in the collector (branch seeds)
    branch: _BranchState | None = None  # a window being branched
    look: _LookState | None = None  # a look-ahead opponent's decision being rolled out (Assignment.lookahead)
    looks: int = 0  # look-ahead decisions of the opponent so far in this game (their random-number keys)


class RolloutCollector:
    """Collect ``num_steps`` learner rows per environment slot from ``env`` with ``model``.

    ``next_game() -> Assignment`` chooses every new game (deck pairing, first player, opponent);
    ``opponents(snapshot_id) -> module`` returns the frozen snapshot for pool games. ``min_batch``: wait for
    at least this many ready decisions before a forward pass (default: half the environments).
    """

    def __init__(self, env, model: nn.Module, next_game: Callable[[], Assignment], num_steps: int, *,
                 opponents: Callable[[int], nn.Module] | None = None, seed: int = 0, min_batch: int | None = None,
                 device: torch.device | str = "cpu", stall_timeout: float | None = 900.0,
                 turn_discount: float = 1.0, complete_games: bool = False, branch: BranchConfig | None = None,
                 branch_hooks=None) -> None:  # fmt: skip
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        # window branching (BranchConfig): the environment must take snapshots; branch_hooks are the game-specific
        # parts (EncodedBranchHooks by default, train.toy.NimBranchHooks for the toy game)
        if branch is not None:
            if complete_games:
                raise ValueError("window branching runs with fixed-length segments, not complete_games")
            if not getattr(env, "snapshots", False):
                raise ValueError("window branching needs an environment with snapshots (EncodedVecEnv(snapshots=True))")
            if branch_hooks is None:
                branch_hooks = EncodedBranchHooks(env.vocab, env.cards)
        self.branch = branch
        self.branch_hooks = branch_hooks
        self.seed = seed
        self._branch_rng = random.Random(_mix(seed, 0xB4A9C4))  # the fraction draws (never the main generator)
        self._branching = 0  # slots with a window being branched
        self._branch_cards = frozenset(branch.cards) if branch is not None else frozenset()
        # windows branched, decisions stepped in branches, branched windows without a recovery line (this collection)
        self._branch_counts = [0, 0, 0]
        self._look_counts = [0, 0, 0, 0]  # look-ahead decisions, branched, changed, rollout steps (this collection)
        self.env = env
        self.model = model
        self.next_game = next_game
        self.num_steps = num_steps
        self.complete_games = complete_games
        self.opponents = opponents
        # speed pressure (design T6, docs/spikes/reward-signal.md): a won / lost game's terminal reward is
        # +/- turn_discount ** turns, so an earlier win is worth more and a later loss costs less; 1.0 = off
        if not 0 < turn_discount <= 1:
            raise ValueError("turn_discount must be in (0, 1]")
        self.turn_discount = turn_discount
        self.min_batch = max(1, min_batch if min_batch is not None else env.num_envs // 2)
        self.device = torch.device(device)
        # no event from any environment for this long while some are still owed rows: an engine stuck inside one
        # duel (its worker thread never returns) -- raise instead of waiting forever; None = wait forever
        self.stall_timeout = stall_timeout
        self._watch = StallWatch(env, stall_timeout, describe=self._describe, error=RolloutStalled)
        self.generator = torch.Generator().manual_seed(seed)
        self._slots: list[_Slot | None] = [None] * env.num_envs
        self._held: dict[int, Any] = {}  # env -> pending learner decision (the bootstrap state)
        self.games_started = 0
        # snapshot module id -> (the module, its copy with Linear parameters in the autocast dtype, that dtype); the
        # module itself is kept so its id cannot be reused by another object while the entry lives (_opponent_module)
        self._cast: dict[int, tuple[nn.Module, nn.Module, torch.dtype]] = {}

    # -- public -------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> Rollout:
        # act with the network as evaluation sees it (dropout off); the learner switches back to train mode
        training = self.model.training
        self.model.eval()
        try:
            return self._collect_complete() if self.complete_games else self._collect()
        finally:
            self.model.train(training)

    def _collect(self) -> Rollout:
        t0 = time.perf_counter()
        B, T = self.env.num_envs, self.num_steps
        for env_id, slot in enumerate(self._slots):
            if slot is None:
                self._new_game(env_id)
            else:
                slot.rows_here = 0
        cols: list[list[_Row]] = [[] for _ in range(B)]
        games: list[FinishedGame] = []
        opponent_decisions = 0
        ready = list(self._held.values())
        self._held.clear()
        self._branch_counts = [0, 0, 0]
        self._look_counts = [0, 0, 0, 0]
        lockstep = self.min_batch >= B
        deferred: list = []  # lockstep: the other slots' decisions, held back while a window is branched
        while True:
            learner, others, branching, looking, look_starts = [], defaultdict(list), [], [], []
            # Full batches provide stable membership; worker order must not reassign RNG draws or deals.
            # Partial asynchronous collection retains its existing scheduling/sampling behavior.
            if lockstep:
                ready.sort(key=lambda ev: ev.env_id)
            for ev in ready:
                slot = self._slots[ev.env_id]
                if slot.branch is not None:
                    branching.append(ev)
                elif slot.look is not None:
                    looking.append(ev)
                elif lockstep and self._branching:
                    deferred.append(ev)
                elif ev.result is not None:
                    games.append(self._finish(ev, cols[ev.env_id]))
                    self._new_game(ev.env_id)
                elif slot.assignment.is_learner(ev.player):
                    if len(cols[ev.env_id]) >= T:
                        self._held[ev.env_id] = ev
                    else:
                        learner.append(ev)
                elif slot.line is not None and self._act_line(slot, ev):
                    opponent_decisions += 1
                elif slot.assignment.lookahead is not None and slot.assignment.lookahead.applies(ev.obs):
                    look_starts.append(ev)
                else:
                    others[id(slot.opponent)].append(ev)
            if looking or look_starts:
                opponent_decisions += self._look_round(looking, look_starts)
            b_learner, b_horizon, b_groups = self._sort_branch(branching) if branching else ([], [], {})
            pending, b_pending = self._forward_opponents(others, b_groups)  # queued before the learner's wait
            if learner or b_learner or b_horizon:
                self._act_learner(learner, cols, b_learner, b_horizon)
            opponent_decisions += self._step_opponents(pending)
            self._step_branch_opponents(b_pending)
            if len(self._held) == B:
                break
            if lockstep and self._branching:
                ready = self._recv(1)
            elif deferred:  # the branches are done: the held-back round, completed
                owed = B - len(self._held) - len(deferred)
                ready, deferred = deferred + (self._recv(owed) if owed > 0 else []), []
            else:
                ready = self._recv(min(self.min_batch, B - len(self._held)))
        return self._assemble(cols, games, opponent_decisions, time.perf_counter() - t0)

    def _collect_complete(self) -> Rollout:
        """Exactly one new game per slot, packed in slot order as [rows, 1]. No padding or open tails.

        Truncated games consume the budget but contribute no training rows in either target arm.
        Finished slots pause until the next update; no preferential refilling of short games.
        """
        if self._held or any(slot is not None for slot in self._slots):
            raise ValueError("complete-game collection cannot continue fixed-length segments")
        t0 = time.perf_counter()
        B = self.env.num_envs
        cols: list[list[_Row]] = [[] for _ in range(B)]
        games: list[FinishedGame] = []
        opponent_decisions = discarded = 0
        for e in range(B):
            self._new_game(e)
        while len(self._held) < B:
            ready = sorted(self._recv(min(self.min_batch, B - len(self._held))), key=lambda ev: ev.env_id)
            learner, others = [], defaultdict(list)
            for ev in ready:
                slot = self._slots[ev.env_id]
                if ev.result is not None:
                    game = self._finish(ev, cols[ev.env_id])
                    games.append(game)
                    if game.truncated:
                        discarded += len(cols[ev.env_id])
                        cols[ev.env_id].clear()
                    self._held[ev.env_id] = ev  # also exclude finished slots from stall reports
                    self._slots[ev.env_id] = None
                elif slot.assignment.is_learner(ev.player):
                    learner.append(ev)
                elif slot.line is not None and self._act_line(slot, ev):
                    opponent_decisions += 1
                else:
                    others[id(slot.opponent)].append(ev)
            pending = self._forward_opponents(others)
            if learner:
                self._act_learner(learner, cols)
            opponent_decisions += self._step_opponents(pending)
        self._held.clear()
        rows = [row for col in cols for row in col]
        if not rows:
            raise ValueError(
                f"complete-game batch has no valid learner rows ({len(games)} games, "
                f"{discarded} discarded rows); reduce truncation, do not label it a draw"
            )
        ro = self._assemble([rows], games, opponent_decisions, time.perf_counter() - t0, complete=True)
        ro.discarded_rows = discarded
        return ro

    def _recv(self, n: int) -> list:
        return self._watch.recv(n, owing=[e for e in range(self.env.num_envs) if e not in self._held])

    def _describe(self, env_id: int) -> str:
        """A slot's game, for a stall report (the stuck duel is among the silent ones)."""
        slot = self._slots[env_id]
        a = slot.assignment if slot is not None else None
        spec = getattr(a, "spec", None)
        return (f"game rows {slot.rows if slot else 0}, seed {getattr(spec, 'seed', None)}, "
                f"first {getattr(spec, 'first', None)}, {dict(a.info) if a is not None else {}}")  # fmt: skip

    # -- game bookkeeping ---------------------------------------------------------------------------
    def _new_game(self, env_id: int) -> None:
        assignment = self.next_game()
        opponent = None
        if assignment.opponent is not None:
            if self.opponents is None:
                raise RuntimeError("a pool game needs opponents(snapshot_id)")
            opponent = self.opponents(assignment.opponent)
        line = None
        if assignment.lookahead is not None:
            if opponent is None or assignment.line is not None:
                raise RuntimeError("a look-ahead opponent needs a snapshot opponent and no scripted line")
            if self.complete_games or not getattr(self.env, "snapshots", False):
                raise RuntimeError(
                    "a look-ahead opponent needs fixed-length segments and EncodedVecEnv(snapshots=True)"
                )
        if assignment.line is not None:
            if opponent is None:
                raise RuntimeError("a scripted-line game needs a snapshot opponent to continue the line")
            line = LinePlayer(assignment.line, bool(getattr(self.env, "skip_forced", False)),
                              first_seat=assignment.learner_seat == 0)  # fmt: skip
        self._slots[env_id] = _Slot(assignment, opponent, line=line, serial=self.games_started)
        self._watch.started(env_id)
        self.games_started += 1
        self.env.reset(env_id, assignment.spec)

    def _finish(self, ev, col: list[_Row]) -> FinishedGame:
        slot = self._slots[ev.env_id]
        res = ev.result
        reason = str(res.get("reason", ""))
        truncated = reason in TRUNCATION_REASONS
        winner = res.get("winner")
        if slot.rows_here:
            row = col[-1]
            row.done, row.truncated = True, truncated
            if not truncated and winner is not None:
                scale = self.turn_discount ** int(res.get("turns", 0)) if self.turn_discount < 1 else 1.0
                row.reward = scale if winner == row.player else -scale
        return FinishedGame(slot.assignment, winner, reason, truncated, slot.rows, dict(res),
                            slot.line.finish() if slot.line is not None else None)  # fmt: skip

    # -- acting ---------------------------------------------------------------------------------------
    def _act_learner(self, events: list, cols: list[list[_Row]], branch_learner: list = (),
                     branch_horizon: list = ()) -> None:  # fmt: skip
        """One forward over the learner's ready decisions: ``events`` become rows (sampled with the collector's
        generator), the branch decisions after them (:meth:`_sort_branch`) only share the forward."""
        model = self.model
        extra = [*branch_learner, *branch_horizon]
        both = [*events, *extra] if extra else events
        batch = model.collate([ev.obs for ev in both], self.device)
        priv = model.collate_privileged([ev.privileged for ev in both], self.device)
        out = model(batch, priv)
        logp = torch.log_softmax(out.logits.float(), -1)
        width = logp.shape[1]
        # everything the rows keep (probabilities, log-probabilities, Q, V) in one device-to-host copy
        packed = torch.cat([logp.exp(), logp, out.q.float(), out.v.float().unsqueeze(1)], 1).cpu()
        if extra:
            self._act_branch_learner(list(branch_learner), list(branch_horizon), packed[len(events) :], width)
            packed = packed[: len(events)]
            if not events:
                return
        probs, logp = packed[:, :width].contiguous(), packed[:, width : 2 * width]
        q, v = packed[:, 2 * width : 3 * width], packed[:, 3 * width]
        actions = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        chosen = logp.gather(1, actions.unsqueeze(1)).squeeze(1)
        for i, ev in enumerate(events):
            a = int(actions[i])
            slot = self._slots[ev.env_id]
            window, effect, occurrence, counted = self._count_window(slot, ev)
            branch = self._branch_window(slot, ev, a, probs[i], counted, effect, occurrence) if self.branch else None
            if counted:
                self._line_interrupt(slot, ev, a, effect, occurrence)
            scripted = slot.assignment.line is not None
            line_id = slot.assignment.line.id if scripted else -1
            row = _Row(ev.obs, ev.privileged, int(ev.player), a, float(chosen[i]), probs[i], q[i], float(v[i]),
                       line_id=line_id, line_window=window, line_effect=effect, line_occurrence=occurrence,
                       line_seat=slot.assignment.learner_seat if scripted else -1)  # fmt: skip
            cols[ev.env_id].append(row)
            slot.rows += 1
            slot.rows_here += 1
            if branch is None:
                self.env.step(ev.env_id, a)
            else:  # branch first; the row's action is stepped once the branches are done (_branch_next)
                branch.row = row
                branch.line_post = None if slot.line is None else slot.line.copy()
                slot.branch = branch
                self._branching += 1
                branch.sid = self.env.snapshot(ev.env_id)

    @staticmethod
    def _count_window(slot: _Slot, ev) -> tuple[int, int, int, bool]:
        """Count a learner decision that is a chain window: ``(window, effect, occurrence, counted)`` -- the seat's
        chain-window number in the game (0 elsewhere) and, for a window of the line's turn with a real choice, the
        effect it answers, its occurrence and True (-1, 0, False otherwise)."""
        g = ev.obs.get("globals")  # absent on toy environments (train.toy)
        if g is None or int(g[18]) != MSG_SELECT_CHAIN:
            return 0, -1, 0, False
        slot.windows += 1
        # a forced prompt (nothing to activate: the pass is the only choosable row) is not a window the
        # labels know; the solver's passive responder is only counted where it had a choice too
        # first seat: only the windows of the line's turn (the opponent's turn 2) are counted
        first = slot.line is not None and slot.line.first_seat
        if int((ev.obs["action_mask"] != 0).sum()) > 1 and (not first or (int(g[3]) == 2 and not int(g[2]))):
            effect = chain_effect(ev.obs)
            occurrence = slot.occurrences[effect] = slot.occurrences.get(effect, 0) + 1
            return slot.windows, effect, occurrence, True
        return slot.windows, -1, 0, False

    @staticmethod
    def _line_interrupt(slot: _Slot, ev, a: int, effect: int, occurrence: int) -> None:
        """The learner's activation at a window of the line: a recovery line takes over, if there is one."""
        if slot.line is not None and int(ev.obs["actions"][a][0]) == CHAIN_KIND:
            slot.line.interrupt(effect, occurrence, int(ev.obs["actions"][a][2]))

    # -- window branching -------------------------------------------------------------------------------
    def _branch_window(self, slot: _Slot, ev, a: int, probs: Tensor, counted: bool, effect: int,
                       occurrence: int) -> _BranchState | None:  # fmt: skip
        """A branching of this learner decision (taken before its action tells the line anything), or None: not a
        qualifying window, or not drawn (``BranchConfig.fraction``)."""
        cfg = self.branch
        if cfg.fraction <= 0:
            return None
        hooks = self.branch_hooks
        found = hooks.candidates(slot, ev, probs, cfg.k, cfg.require_recovery, effect, occurrence, self._branch_cards)
        if found is None or (cfg.fraction < 1 and self._branch_rng.random() >= cfg.fraction):
            return None
        pass_row, candidates = found
        recovered = hooks.recovered(slot, ev, [c for c in candidates if c != pass_row], effect, occurrence)
        return _BranchState(None, ev, a, candidates, pass_row, hooks.turn(ev),
                            _mix(self.seed, slot.serial, slot.rows), counted, effect, occurrence, slot.windows,
                            dict(slot.occurrences), None if slot.line is None else slot.line.copy(),
                            sums=[0.0] * len(candidates), recovered=recovered)  # fmt: skip

    def _sort_branch(self, events: list) -> tuple[list, list, dict[int, list]]:
        """Answer what needs no network among the decisions of slots being branched -- the window's observation after
        a snapshot / restore starts the next candidate (or ends the branching), a rollout ends at a board horizon or
        the game's end, the line plays its step -- and return the rest: the learner's decisions, its decisions at a
        critic horizon (V needed) and the snapshot's decisions by module. They join the main forward passes
        (:meth:`_act_learner`, :meth:`_forward_opponents`) and are sampled with the branch's generator."""
        cfg, hooks = self.branch, self.branch_hooks
        learner, horizon, groups = [], [], defaultdict(list)
        for ev in events:
            slot = self._slots[ev.env_id]
            st = slot.branch
            if st.phase != "run":
                self._branch_next(ev.env_id, slot)
            elif ev.result is not None:
                winner = ev.result.get("winner")
                if str(ev.result.get("reason", "")) in TRUNCATION_REASONS:
                    self._branch_value(ev.env_id, slot, None)
                else:
                    self._branch_value(ev.env_id, slot, 0.0 if winner is None else
                                       (1.0 if winner == slot.assignment.learner_seat else -1.0))  # fmt: skip
            elif slot.assignment.is_learner(ev.player):
                if not hooks.horizon(st, ev):
                    learner.append(ev)
                elif cfg.horizon == "board":
                    self._branch_value(ev.env_id, slot, -cfg.board_scale * hooks.board(ev))
                else:
                    horizon.append(ev)
            elif slot.line is not None and self._act_line(slot, ev):
                self._branch_counts[1] += 1
            else:
                groups[id(slot.opponent)].append(ev)
        return learner, horizon, groups

    def _act_branch_learner(self, learner: list, horizon: list, packed: Tensor, width: int) -> None:
        """The learner's branch decisions (``packed``: their rows of the learner forward, probabilities first and V at
        column ``3 * width``): sample with the branch generator and step; at a critic horizon end the rollout."""
        cfg = self.branch
        for i, ev in enumerate(learner):
            slot = self._slots[ev.env_id]
            a = int(torch.multinomial(packed[i, :width], 1, generator=slot.branch.gen))
            _, effect, occurrence, counted = self._count_window(slot, ev)
            if counted:
                self._line_interrupt(slot, ev, a, effect, occurrence)
            self._branch_counts[1] += 1
            self.env.step(ev.env_id, a)
        for i, ev in enumerate(horizon, len(learner)):
            value = float(packed[i, 3 * width])
            if cfg.horizon == "both":
                value -= cfg.board_scale * self.branch_hooks.board(ev)
            self._branch_value(ev.env_id, self._slots[ev.env_id], value)

    def _step_branch_opponents(self, pending: list[tuple[list, Tensor]]) -> None:
        for evs, probs in pending:
            probs = probs.cpu()
            for i, ev in enumerate(evs):
                a = int(torch.multinomial(probs[i], 1, generator=self._slots[ev.env_id].branch.gen))
                self._branch_counts[1] += 1
                self.env.step(ev.env_id, a)

    def _branch_value(self, env_id: int, slot: _Slot, value: float | None) -> None:
        """A rollout ended (None: cut by a limit or an error, the window is invalid); go back to the window."""
        st = slot.branch
        if value is None:
            st.ok = False
        else:
            st.sums[st.k] += value
        st.k += 1
        if st.k == len(st.candidates):
            st.k, st.m = 0, st.m + 1
        st.phase = "restore"
        self.env.restore(env_id, st.sid)

    def _branch_next(self, env_id: int, slot: _Slot) -> None:
        """At the window again (snapshot taken / restored): step the next candidate with the window's line state and
        counters and the rollout's seed, or, after the last one, release the snapshot, record the values on the row
        and step its action with the line state the main game had."""
        st, cfg = slot.branch, self.branch
        slot.windows, slot.occurrences = st.windows, dict(st.occurrences)
        if st.m < cfg.rollouts:
            if slot.line is not None:
                slot.line.restore(st.line_pre)
            st.gen.manual_seed(_mix(st.seed, st.m))
            a = st.candidates[st.k]
            if st.counted:
                self._line_interrupt(slot, st.window, a, st.effect, st.occurrence)
            st.phase = "run"
            self._branch_counts[1] += 1
            self.env.step(env_id, a)
            return
        self.env.release(env_id, st.sid)
        if slot.line is not None:
            slot.line.restore(st.line_post)
        row = st.row
        row.branch_pass = st.pass_row
        if st.ok:
            value = torch.full_like(row.probs, float("nan"))
            for c, total in zip(st.candidates, st.sums):
                value[c] = total / cfg.rollouts
            row.branch_value, row.branch_valid = value, True
        slot.branch = None
        self._branching -= 1
        self._branch_counts[0] += 1
        self._branch_counts[2] += not st.recovered
        self.env.step(env_id, st.action)

    # -- look-ahead opponents ------------------------------------------------------------------------
    def _look_round(self, events: list, starts: list) -> int:
        """Advance the look-ahead opponents (``Assignment.lookahead``; :func:`ygorl.train.lookahead.look_ahead`):
        ``starts`` are qualifying snapshot decisions (the look-ahead begins), ``events`` the environment's answers to
        running ones. Their forwards are batched by module (the snapshot for its seat, the learner's model for the
        learner's seat, no rows); a finished look-ahead steps its action in the main game. Returns the snapshot's
        main-game decisions stepped. The slots count as branching (``_branching``), so lockstep collection holds the
        other slots back as it does for window branches."""
        from ygorl.train.lookahead import GameHooks, Uniforms, look_ahead, lookahead_value

        stepped = 0
        env_of: dict[int, int] = {}
        forwards: list[tuple[_Slot, int, Any, str]] = []

        def send(slot: _Slot, value) -> None:
            nonlocal stepped
            st, env_id = slot.look, env_of[id(slot)]
            while True:
                try:
                    req = st.gen.send(value)
                except StopIteration as stop:
                    action, info = stop.value
                    counts = self._look_counts
                    counts[0] += 1
                    counts[1] += info["branched"]
                    counts[2] += info["changed"]
                    counts[3] += info["steps"]
                    slot.look = None
                    slot.looks += 1
                    self._branching -= 1
                    stepped += 1
                    self.env.step(env_id, int(action))
                    return
                kind = req[0]
                if kind == "forward":
                    forwards.append((slot, req[1], req[2], "co"))
                    return
                if kind == "release":
                    self.env.release(env_id, req[1])
                    value = None
                    continue
                if kind == "step":
                    self.env.step(env_id, int(req[1]))
                elif kind == "restore":
                    self.env.restore(env_id, req[1])
                elif kind == "snapshot":
                    st.wait_sid = self.env.snapshot(env_id)
                else:
                    raise RuntimeError(f"unknown look-ahead request {kind!r}")
                return

        for ev in [*events, *starts]:
            env_of[id(self._slots[ev.env_id])] = ev.env_id
        forwards.extend((self._slots[ev.env_id], int(ev.player), ev, "start") for ev in starts)
        for ev in events:
            slot = self._slots[ev.env_id]
            if slot.look.wait_sid is not None:
                sid, slot.look.wait_sid = slot.look.wait_sid, None
                send(slot, sid)
            else:
                send(slot, ev)
        while forwards:
            batch = list(forwards)
            forwards.clear()
            groups: dict[int, list[int]] = defaultdict(list)
            modules: dict[int, nn.Module] = {}
            for k, (slot, seat, _, _) in enumerate(batch):
                module = self.model if seat == slot.assignment.learner_seat else self._opponent_module(slot.opponent)
                modules[id(module)] = module
                groups[id(module)].append(k)
            probs: list[Any] = [None] * len(batch)
            for key, ks in groups.items():
                obs = self.model.collate([batch[k][2].obs for k in ks], self.device)
                p = torch.softmax(policy_logits(modules[key], obs).float(), -1).cpu().numpy().astype(np.float64)
                for r, k in enumerate(ks):
                    probs[k] = p[r]
            for (slot, seat, ev, kind), p in zip(batch, probs, strict=True):
                if kind == "co":
                    send(slot, p)
                    continue
                cfg = slot.assignment.lookahead

                def score(cur, s, w, cfg=cfg):
                    return lookahead_value(cur, s, w, cfg, self.env.cards, self.env.vocab)

                gen = look_ahead(cfg, seat, slot.looks, ev, p, Uniforms(_mix(self.seed, 0x100CA, slot.serial)),
                                 GameHooks(), score)  # fmt: skip
                slot.look = _LookState(gen, ev, int(ev.obs["globals"][3]))
                self._branching += 1
                send(slot, None)
        return stepped

    def _act_line(self, slot: _Slot, ev) -> bool:
        """Play the opponent seat's decision from its scripted line; False once the line is over (the snapshot plays)."""
        row = slot.line.act(ev.obs)
        if row is None:
            return False
        self.env.step(ev.env_id, row)
        return True

    def _forward_opponents(self, groups: Mapping[int, list], branch_groups: Mapping[int, list] | None = None):
        """Queue one forward per snapshot module (``groups``: the ready decisions of each) and return the candidate
        probabilities on the device, without waiting for them (:meth:`_step_opponents` samples). With
        ``branch_groups`` (window branching: the snapshot's decisions inside branches, by module) they share the
        module's forward, after the main ones, and ``(pending, branch_pending)`` is returned."""
        pending, branch_pending = [], []
        for key in [*groups, *(k for k in branch_groups or () if k not in groups)]:
            main, extra = groups.get(key, []), (branch_groups or {}).get(key, [])
            evs = [*main, *extra] if extra else main
            module = self._opponent_module(self._slots[evs[0].env_id].opponent)
            batch = self.model.collate([ev.obs for ev in evs], self.device)
            probs = torch.softmax(policy_logits(module, batch).float(), -1)
            if main:
                pending.append((main, probs[: len(main)] if extra else probs))
            if extra:
                branch_pending.append((extra, probs[len(main) :]))
        return pending if branch_groups is None else (pending, branch_pending)

    def _opponent_module(self, module: nn.Module) -> nn.Module:
        """The module a snapshot's forward runs through. Under autocast on the device it is a frozen copy whose
        ``nn.Linear`` weights and biases are already in the autocast dtype: autocast casts them on every call
        otherwise (its cast cache only keeps parameters that require a gradient, which snapshots do not), one
        kernel per tensor per forward. Casting once ahead gives the same values (the same rounding). The snapshot
        itself is untouched (it is what the pool saves and evaluates)."""
        kind = self.device.type
        if kind == "cpu" or not torch.is_autocast_enabled(kind):
            return module
        dtype = torch.get_autocast_dtype(kind)
        entry = self._cast.get(id(module))
        if entry is not None and entry[0] is module and entry[2] == dtype:
            return entry[1]
        if len(self._cast) >= 64:  # keep the opponents still in a slot (the pool evicts, snapshots come and go)
            live = {id(s.opponent) for s in self._slots if s is not None and s.opponent is not None}
            self._cast = {k: v for k, v in self._cast.items() if k in live}
        cast = copy.deepcopy(module)
        for m in cast.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.to(dtype)
                if m.bias is not None:
                    m.bias.data = m.bias.data.to(dtype)
        self._cast[id(module)] = (module, cast, dtype)
        return cast

    def _step_opponents(self, pending: list[tuple[list, Tensor]]) -> int:
        """Sample and step the opponents' decisions of :meth:`_forward_opponents`: one device-to-host copy and one
        sampling call over all groups (``torch.multinomial`` draws row by row, so the draws are those of one call per
        group, in group order); the number of decisions stepped."""
        if not pending:
            return 0
        probs = torch.cat([p for _, p in pending]).cpu() if len(pending) > 1 else pending[0][1].cpu()
        actions = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1).tolist()
        events = [ev for evs, _ in pending for ev in evs]
        for ev, a in zip(events, actions):
            self.env.step(ev.env_id, int(a))
        return len(events)

    def _act_opponent(self, module: nn.Module, events: list) -> None:
        """One snapshot's decisions, forwarded, sampled and stepped at once (the two-phase methods above are what the
        collection loop uses)."""
        self._step_opponents(self._forward_opponents({id(module): events}))

    # -- assembly ---------------------------------------------------------------------------------
    def _assemble(
        self, cols: list[list[_Row]], games, opponent_decisions: int, seconds: float, *, complete: bool = False
    ) -> Rollout:
        B, T = len(cols), len(cols[0]) if complete else self.num_steps
        rows = [cols[b][t] for t in range(T) for b in range(B)]  # time-major
        model = self.model
        obs = model.collate([r.obs for r in rows], self.device)
        priv = model.collate_privileged([r.privileged for r in rows], self.device)

        def grid(values, dtype) -> Tensor:
            return torch.tensor(values, dtype=dtype).reshape(T, B)

        def grid_a(tensors) -> Tensor:
            return torch.stack(tensors).reshape(T, B, -1)

        if complete:
            # All games end in done=True, so these shape-compatible sentinels are never bootstrapped.
            boot_player = torch.tensor([rows[-1].player])
            boot_mask = obs["action_mask"][-1:].cpu()
            boot_probs, boot_q = rows[-1].probs[None], torch.zeros_like(rows[-1].q[None])
            boot_value = torch.zeros(1)
        else:
            held = [self._held[b] for b in range(B)]
            boot_batch = model.collate([ev.obs for ev in held], self.device)
            boot_priv = model.collate_privileged([ev.privileged for ev in held], self.device)
            boot = model(boot_batch, boot_priv)
            boot_player = torch.tensor([int(ev.player) for ev in held])
            boot_mask = boot_batch["action_mask"].cpu()
            width = boot.logits.shape[1]
            packed = torch.cat(
                [torch.softmax(boot.logits.float(), -1), boot.q.float(), boot.v.float()[:, None]], 1
            ).cpu()
            boot_probs, boot_q = packed[:, :width].contiguous(), packed[:, width : 2 * width].contiguous()
            boot_value = packed[:, 2 * width].contiguous()
        return Rollout(
            obs=obs, privileged=priv,
            players=grid([r.player for r in rows], torch.long), actions=grid([r.action for r in rows], torch.long),
            action_mask=obs["action_mask"].reshape(T, B, -1).cpu(), probs=grid_a([r.probs for r in rows]),
            log_probs=grid([r.log_prob for r in rows], torch.float32), q=grid_a([r.q for r in rows]),
            values=grid([r.value for r in rows], torch.float32), rewards=grid([r.reward for r in rows], torch.float32),
            dones=grid([r.done for r in rows], torch.bool), truncated=grid([r.truncated for r in rows], torch.bool),
            bootstrap_player=boot_player, bootstrap_mask=boot_mask, bootstrap_probs=boot_probs, bootstrap_q=boot_q,
            bootstrap_value=boot_value, games=games, opponent_decisions=opponent_decisions, seconds=seconds,
            complete_games=complete, line_id=grid([r.line_id for r in rows], torch.long),
            line_window=grid([r.line_window for r in rows], torch.long),
            line_effect=grid([r.line_effect for r in rows], torch.long),
            line_occurrence=grid([r.line_occurrence for r in rows], torch.long),
            line_seat=grid([r.line_seat for r in rows], torch.long),
            lookahead_decisions=self._look_counts[0], lookahead_branched=self._look_counts[1],
            lookahead_changed=self._look_counts[2], lookahead_steps=self._look_counts[3],
            **(self._branch_fields(rows, T, B) if self.branch is not None else {}),
        )  # fmt: skip

    def _branch_fields(self, rows: list[_Row], T: int, B: int) -> dict:
        width = rows[0].probs.shape[0]
        value = torch.full((T * B, width), float("nan"))
        for i, r in enumerate(rows):
            if r.branch_value is not None:
                value[i] = r.branch_value
        return {"branch_value": value.reshape(T, B, width),
                "branch_valid": torch.tensor([r.branch_valid for r in rows], dtype=torch.bool).reshape(T, B),
                "branch_pass": torch.tensor([r.branch_pass for r in rows], dtype=torch.long).reshape(T, B),
                "branch_windows": self._branch_counts[0], "branch_decisions": self._branch_counts[1],
                "branch_no_recovery": self._branch_counts[2]}  # fmt: skip


__all__ = [
    "Assignment",
    "BranchConfig",
    "EncodedBranchHooks",
    "FinishedGame",
    "HORIZONS",
    "Rollout",
    "RolloutCollector",
    "RolloutStalled",
    "TRUNCATION_REASONS",
    "board_score",
]
