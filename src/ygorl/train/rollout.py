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
- **Terminal rows**: the result is attached to the column's last row of that game: ``done``; reward +1 / -1
  / 0 from that row's seat. Games cut by a limit or an engine error (:data:`TRUNCATION_REASONS`) are
  **truncations**, not outcomes: ``done`` and ``truncated``, reward 0; the estimator bootstraps them from the
  critic (:func:`ygorl.train.advantages.estimate`). An LP-decided limit never becomes a win or a loss.
- **Segment end**: a column is complete when it has ``T`` rows and its environment waits on a learner
  decision; that pending decision is the bootstrap state (``bootstrap_*``), and the next ``collect`` starts
  the column from it. Complete columns pause while the others fill up.

Behaviour log-probabilities, the full candidate distribution and the critic's Q / V are recorded at acting
time (one forward pass per batch of ready decisions).
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

TRUNCATION_REASONS = frozenset({"turn_limit", "decision_limit", "error"})


class RolloutStalled(RuntimeError):
    """No environment produced an event for ``stall_timeout``: an engine is stuck inside a duel. Its worker thread
    never returns, so destroying the environment pool (joining its threads) would hang too: the caller should save
    what it needs and end the process with ``os._exit`` (tools/train_ppo.py does)."""


@dataclass(frozen=True)
class Assignment:
    """One game to play in an environment slot: the env spec and who plays the other seat."""

    spec: Any  # ``GameSpec`` for EncodedVecEnv (a pile size for NimEnv)
    opponent: int | None = None  # None: the current policy on both seats; else a snapshot id (pool opponent)
    learner_seat: int = 0  # engine seat of the learner in a pool game (ignored in self-play)
    info: Mapping[str, Any] = field(default_factory=dict)  # free-form labels (deck names ...) for the logs

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

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.players.shape)  # type: ignore[return-value]

    @property
    def decisions(self) -> int:
        """All decisions stepped in the environment for this segment (rows + opponent decisions)."""
        return self.players.numel() + self.opponent_decisions


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


@dataclass
class _Slot:
    assignment: Assignment
    opponent: nn.Module | None = None  # the snapshot, resolved when the game starts (it may leave the pool)
    rows: int = 0  # rows of the current game, across rollouts
    rows_here: int = 0  # rows of the current game in the column being filled


def _logits_of(module: nn.Module, batch) -> Tensor:
    fn = getattr(module, "policy_logits", None)
    return fn(batch) if fn is not None else module(batch).logits


class RolloutCollector:
    """Collect ``num_steps`` learner rows per environment slot from ``env`` with ``model``.

    ``next_game() -> Assignment`` chooses every new game (deck pairing, first player, opponent);
    ``opponents(snapshot_id) -> module`` returns the frozen snapshot for pool games. ``min_batch``: wait for
    at least this many ready decisions before a forward pass (default: half the environments).
    """

    def __init__(self, env, model: nn.Module, next_game: Callable[[], Assignment], num_steps: int, *,
                 opponents: Callable[[int], nn.Module] | None = None, seed: int = 0, min_batch: int | None = None,
                 device: torch.device | str = "cpu", stall_timeout: float | None = 900.0) -> None:  # fmt: skip
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        self.env = env
        self.model = model
        self.next_game = next_game
        self.num_steps = num_steps
        self.opponents = opponents
        self.min_batch = max(1, min_batch if min_batch is not None else env.num_envs // 2)
        self.device = torch.device(device)
        # no event from any environment for this long while some are still owed rows: an engine stuck inside one
        # duel (its worker thread never returns) -- raise instead of waiting forever; None = wait forever
        self.stall_timeout = stall_timeout
        self._last_event: dict[int, float] = {}
        self.generator = torch.Generator().manual_seed(seed)
        self._slots: list[_Slot | None] = [None] * env.num_envs
        self._held: dict[int, Any] = {}  # env -> pending learner decision (the bootstrap state)
        self.games_started = 0

    # -- public -------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> Rollout:
        # act with the network as evaluation sees it (dropout off); the learner switches back to train mode
        training = self.model.training
        self.model.eval()
        try:
            return self._collect()
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
        while True:
            learner, others = [], defaultdict(list)
            for ev in ready:
                slot = self._slots[ev.env_id]
                if ev.result is not None:
                    games.append(self._finish(ev, cols[ev.env_id]))
                    self._new_game(ev.env_id)
                elif slot.assignment.is_learner(ev.player):
                    if len(cols[ev.env_id]) >= T:
                        self._held[ev.env_id] = ev
                    else:
                        learner.append(ev)
                else:
                    others[id(slot.opponent)].append(ev)
            if learner:
                self._act_learner(learner, cols)
            for evs in others.values():
                self._act_opponent(self._slots[evs[0].env_id].opponent, evs)
                opponent_decisions += len(evs)
            if len(self._held) == B:
                break
            ready = self._recv(min(self.min_batch, B - len(self._held)))
        return self._assemble(cols, games, opponent_decisions, time.perf_counter() - t0)

    def _recv(self, n: int) -> list:
        if self.stall_timeout is None:
            return self.env.recv(n)
        start = time.monotonic()
        while True:
            got = self.env.recv(n, timeout=min(60.0, self.stall_timeout))
            now = time.monotonic()
            for ev in got:
                self._last_event[ev.env_id] = now
            if got:
                return got
            if now - start >= self.stall_timeout:
                raise RolloutStalled(self._stall_report(now))

    def _stall_report(self, now: float) -> str:
        """Which environments owe events and how long they have been silent (the stuck duel is among them)."""
        waiting = [e for e in range(self.env.num_envs) if e not in self._held]
        waiting.sort(key=lambda e: self._last_event.get(e, 0.0))
        lines = []
        for e in waiting[:8]:
            slot = self._slots[e]
            a = slot.assignment if slot is not None else None
            spec = getattr(a, "spec", None)
            lines.append(f"env {e}: silent {now - self._last_event.get(e, now):.0f}s, game rows {slot.rows if slot else 0}, "
                         f"seed {getattr(spec, 'seed', None)}, first {getattr(spec, 'first', None)}, "
                         f"{dict(a.info) if a is not None else {}}")  # fmt: skip
        return (
            f"no environment event for {self.stall_timeout:.0f}s while {len(waiting)} environment(s) still owe "
            "rows (an engine stuck inside a duel?); longest silent:\n  " + "\n  ".join(lines)
        )

    # -- game bookkeeping ---------------------------------------------------------------------------
    def _new_game(self, env_id: int) -> None:
        assignment = self.next_game()
        opponent = None
        if assignment.opponent is not None:
            if self.opponents is None:
                raise RuntimeError("a pool game needs opponents(snapshot_id)")
            opponent = self.opponents(assignment.opponent)
        self._slots[env_id] = _Slot(assignment, opponent)
        self._last_event[env_id] = time.monotonic()
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
                row.reward = 1.0 if winner == row.player else -1.0
        return FinishedGame(slot.assignment, winner, reason, truncated, slot.rows, dict(res))

    # -- acting ---------------------------------------------------------------------------------------
    def _act_learner(self, events: list, cols: list[list[_Row]]) -> None:
        model = self.model
        batch = model.collate([ev.obs for ev in events], self.device)
        priv = model.collate_privileged([ev.privileged for ev in events], self.device)
        out = model(batch, priv)
        logp = torch.log_softmax(out.logits.float(), -1)
        probs = logp.exp()
        actions = torch.multinomial(probs.cpu(), 1, generator=self.generator).squeeze(-1)
        chosen = logp.cpu().gather(1, actions.unsqueeze(1)).squeeze(1)
        q, v, probs = out.q.float().cpu(), out.v.float().cpu(), probs.cpu()
        for i, ev in enumerate(events):
            a = int(actions[i])
            slot = self._slots[ev.env_id]
            cols[ev.env_id].append(_Row(ev.obs, ev.privileged, int(ev.player), a, float(chosen[i]), probs[i], q[i],
                                        float(v[i])))  # fmt: skip
            slot.rows += 1
            slot.rows_here += 1
            self.env.step(ev.env_id, a)

    def _act_opponent(self, module: nn.Module, events: list) -> None:
        batch = self.model.collate([ev.obs for ev in events], self.device)
        probs = torch.softmax(_logits_of(module, batch).float(), -1).cpu()
        actions = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        for ev, a in zip(events, actions.tolist()):
            self.env.step(ev.env_id, int(a))

    # -- assembly ---------------------------------------------------------------------------------
    def _assemble(self, cols: list[list[_Row]], games, opponent_decisions: int, seconds: float) -> Rollout:
        B, T = len(cols), self.num_steps
        rows = [cols[b][t] for t in range(T) for b in range(B)]  # time-major
        model = self.model
        obs = model.collate([r.obs for r in rows], self.device)
        priv = model.collate_privileged([r.privileged for r in rows], self.device)

        def grid(values, dtype) -> Tensor:
            return torch.tensor(values, dtype=dtype).reshape(T, B)

        def grid_a(tensors) -> Tensor:
            return torch.stack(tensors).reshape(T, B, -1)

        held = [self._held[b] for b in range(B)]
        boot_batch = model.collate([ev.obs for ev in held], self.device)
        boot_priv = model.collate_privileged([ev.privileged for ev in held], self.device)
        boot = model(boot_batch, boot_priv)
        return Rollout(
            obs=obs, privileged=priv,
            players=grid([r.player for r in rows], torch.long), actions=grid([r.action for r in rows], torch.long),
            action_mask=obs["action_mask"].reshape(T, B, -1).cpu(), probs=grid_a([r.probs for r in rows]),
            log_probs=grid([r.log_prob for r in rows], torch.float32), q=grid_a([r.q for r in rows]),
            values=grid([r.value for r in rows], torch.float32), rewards=grid([r.reward for r in rows], torch.float32),
            dones=grid([r.done for r in rows], torch.bool), truncated=grid([r.truncated for r in rows], torch.bool),
            bootstrap_player=torch.tensor([int(ev.player) for ev in held]), bootstrap_mask=boot_batch["action_mask"].cpu(),
            bootstrap_probs=torch.softmax(boot.logits.float(), -1).cpu(), bootstrap_q=boot.q.float().cpu(),
            bootstrap_value=boot.v.float().cpu(), games=games, opponent_decisions=opponent_decisions, seconds=seconds,
        )  # fmt: skip


__all__ = ["Assignment", "FinishedGame", "Rollout", "RolloutCollector", "RolloutStalled", "TRUNCATION_REASONS"]
