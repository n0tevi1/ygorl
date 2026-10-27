"""A toy two-player game with the :class:`ygorl.env.encoded.EncodedVecEnv` interface, for tests and debugging.

Nim (subtraction game): a pile of ``n`` stones, the seats alternate taking 1-3 stones, whoever takes the last
stone wins. The optimal move from ``n`` is to take ``n % 4``; a multiple of 4 is lost against perfect play.
``NimEnv`` speaks the same protocol as ``EncodedVecEnv`` (``reset`` / ``step`` / ``recv`` events with
``obs`` / ``result`` / ``privileged``), so the rollout collector, the PPO learner and the league run on it
unchanged; ``NimModel`` has the model interface of :class:`ygorl.nets.actor_critic.ActorCritic`.

``NimEnv(snapshots=True)`` adds the snapshot interface of ``EncodedVecEnv`` (``snapshot`` / ``restore`` / ``release``
/ ``snapshot_count``: jobs whose event is the current observation), and :class:`NimBranchHooks` the game-specific
parts of the collector's window branching (``ygorl.train.rollout.BranchConfig``), so branching is testable on a game
whose best move is known.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor, nn

from ygorl.engine.outcome import TURN_LIMIT
from ygorl.env.encoded import EncodedEvent
from ygorl.nets.actor_critic import ActorCriticOutput
from ygorl.nets.heads import MASKED_LOGIT
from ygorl.train.critic import Critic

N_ACTIONS = 3  # take 1, 2 or 3 stones (candidate row i = take i + 1)


def optimal_action(pile: int) -> int:
    """Stones to take from a winning pile (``pile % 4 != 0``)."""
    return pile % 4


class NimEnv:
    """``num_envs`` Nim games stepped synchronously. ``reset(env_id, pile)``: engine seat 0 moves first.

    ``max_moves`` cuts a game (result reason ``"turn_limit"``, winner None), for truncation tests.
    Observations: ``pile`` (int), ``player`` (the seat to move), ``action_mask`` ``[3]``.
    """

    def __init__(
        self, num_envs: int, max_pile: int = 15, max_moves: int | None = None, snapshots: bool = False
    ) -> None:
        self.num_envs = num_envs
        self.snapshots = snapshots
        self._snapshots: dict[int, tuple[int, int, int, int, int]] = {}  # id -> (env, reset count, pile, player, moves)
        self._next_sid = 0
        self._resets = [0] * num_envs
        self.max_pile = max_pile
        self.max_moves = max_moves
        self._pile = [0] * num_envs
        self._player = [0] * num_envs
        self._moves = [0] * num_envs
        self._ready: deque[EncodedEvent] = deque()

    @staticmethod
    def observe(pile: int, player: int, max_pile: int) -> dict[str, np.ndarray]:
        mask = np.array([pile >= k for k in range(1, N_ACTIONS + 1)], dtype=np.int32)
        return {"pile": np.array(pile, dtype=np.int64), "player": np.array(player, dtype=np.int64),
                "action_mask": mask}  # fmt: skip

    def reset(self, env_id: int, spec: int) -> None:
        if not 1 <= spec <= self.max_pile:
            raise ValueError(f"pile must be in 1..{self.max_pile}")
        self._pile[env_id], self._player[env_id], self._moves[env_id] = int(spec), 0, 0
        self._resets[env_id] += 1
        self._emit(env_id)

    def step(self, env_id: int, action: int) -> None:
        take = int(action) + 1
        if not 1 <= take <= min(N_ACTIONS, self._pile[env_id]):
            raise ValueError(f"illegal action {action} with {self._pile[env_id]} stones")
        self._pile[env_id] -= take
        self._moves[env_id] += 1
        if self._pile[env_id] == 0:
            result = {"winner": self._player[env_id], "reason": "win", "decisions": self._moves[env_id]}
            self._ready.append(EncodedEvent(env_id, self._player[env_id], None, result))
            return
        self._player[env_id] ^= 1
        if self.max_moves is not None and self._moves[env_id] >= self.max_moves:
            result = {"winner": None, "reason": TURN_LIMIT, "decisions": self._moves[env_id]}
            self._ready.append(EncodedEvent(env_id, self._player[env_id], None, result))
            return
        self._emit(env_id)

    def snapshot(self, env_id: int) -> int:
        """A snapshot of ``env_id``'s current decision; its event is the unchanged observation (EncodedVecEnv)."""
        if not self.snapshots:
            raise RuntimeError("snapshots are disabled; construct with snapshots=True")
        if self._pile[env_id] == 0:
            raise RuntimeError("snapshot needs a game at a decision")
        sid = self._next_sid
        self._next_sid += 1
        self._snapshots[sid] = (env_id, self._resets[env_id], self._pile[env_id], self._player[env_id],
                                self._moves[env_id])  # fmt: skip
        self._emit(env_id)
        return sid

    def restore(self, env_id: int, sid: int) -> None:
        env, resets, pile, player, moves = self._snapshots[sid]
        if env != env_id or resets != self._resets[env_id]:
            raise ValueError("a snapshot only restores in its env's game")
        self._pile[env_id], self._player[env_id], self._moves[env_id] = pile, player, moves
        self._emit(env_id)

    def release(self, env_id: int, sid: int) -> None:
        if self._snapshots.get(sid, (env_id,))[0] != env_id:
            raise ValueError("not a snapshot of this env")
        self._snapshots.pop(sid, None)

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EncodedEvent]:
        out = list(self._ready)
        self._ready.clear()
        return out

    def pending(self) -> int:
        return len(self._ready)

    def _emit(self, env_id: int) -> None:
        obs = self.observe(self._pile[env_id], self._player[env_id], self.max_pile)
        self._ready.append(EncodedEvent(env_id, self._player[env_id], obs, None))


class NimBranchHooks:
    """Window branching on Nim (``RolloutCollector(branch=..., branch_hooks=NimBranchHooks())``): every learner decision
    of a pool game with two or more legal takes is a window; the branch set is the ``k`` takes of highest probability
    (Nim has no pass); the horizon is the learner's next decision; the opponent's "board" is 1 when the learner then
    faces a multiple of 4 (a lost pile against perfect play), else 0. ``require_recovery`` (scripted lines only) and
    ``cards`` (Nim has no cards) leave no window."""

    def candidates(self, slot, ev, probs: Tensor, k: int, require_recovery: bool, effect: int, occurrence: int,
                   cards=()) -> tuple[int, list[int]] | None:  # fmt: skip
        legal = np.flatnonzero(ev.obs["action_mask"])
        if require_recovery or cards or slot.assignment.opponent is None or len(legal) < 2:
            return None
        return -1, sorted((int(i) for i in legal), key=lambda i: (-float(probs[i]), i))[:k]

    def recovered(self, slot, ev, fires: list[int], effect: int, occurrence: int) -> bool:
        return False

    def turn(self, ev) -> int:
        return 0

    def horizon(self, state, ev) -> bool:
        return True

    def board(self, ev) -> float:
        return float(int(ev.obs["pile"]) % 4 == 0)


class NimModel(nn.Module):
    """Pile one-hot -> MLP context; learned candidate embeddings; dot-product actor + :class:`Critic`."""

    def __init__(self, max_pile: int = 15, hidden: int = 32) -> None:
        super().__init__()
        self.max_pile = max_pile
        self.trunk = nn.Sequential(nn.Linear(max_pile + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.candidates = nn.Parameter(torch.randn(N_ACTIONS, hidden) * 0.1)
        self.query = nn.Linear(hidden, hidden)
        self.critic = Critic(history_dim=hidden, action_dim=hidden, hidden=hidden)

    # -- the model interface used by the collector and the learner ------------------------------------
    @staticmethod
    def collate(observations, device=None) -> dict[str, Tensor]:
        return {"pile": torch.as_tensor(np.stack([o["pile"] for o in observations])),
                "action_mask": torch.as_tensor(np.stack([o["action_mask"] for o in observations])) != 0}  # fmt: skip

    @staticmethod
    def collate_privileged(privileged, device=None) -> None:
        return None

    def _context(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        h = self.trunk(nn.functional.one_hot(obs["pile"].long(), self.max_pile + 1).float())
        cand = self.candidates.expand(h.shape[0], -1, -1)
        return h, cand

    def policy_logits(self, obs: dict[str, Tensor]) -> Tensor:
        h, cand = self._context(obs)
        logits = torch.einsum("bad,bd->ba", cand, self.query(h))
        return logits.masked_fill(~obs["action_mask"], MASKED_LOGIT)

    def forward(self, obs: dict[str, Tensor], privileged=None) -> ActorCriticOutput:
        h, cand = self._context(obs)
        logits = torch.einsum("bad,bd->ba", cand, self.query(h)).masked_fill(~obs["action_mask"], MASKED_LOGIT)
        crit = self.critic(h, cand, obs["action_mask"])
        return ActorCriticOutput(logits, crit.q, crit.v)
