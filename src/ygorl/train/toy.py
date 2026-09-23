"""A toy two-player game with the :class:`ygorl.env.encoded.EncodedVecEnv` interface, for tests and debugging.

Nim (subtraction game): a pile of ``n`` stones, the seats alternate taking 1-3 stones, whoever takes the last
stone wins. The optimal move from ``n`` is to take ``n % 4``; a multiple of 4 is lost against perfect play.
``NimEnv`` speaks the same protocol as ``EncodedVecEnv`` (``reset`` / ``step`` / ``recv`` events with
``obs`` / ``result`` / ``privileged``), so the rollout collector, the PPO learner and the league run on it
unchanged; ``NimModel`` has the model interface of :class:`ygorl.nets.actor_critic.ActorCritic`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor, nn

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

    def __init__(self, num_envs: int, max_pile: int = 15, max_moves: int | None = None) -> None:
        self.num_envs = num_envs
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
            result = {"winner": None, "reason": "turn_limit", "decisions": self._moves[env_id]}
            self._ready.append(EncodedEvent(env_id, self._player[env_id], None, result))
            return
        self._emit(env_id)

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EncodedEvent]:
        out = list(self._ready)
        self._ready.clear()
        return out

    def pending(self) -> int:
        return len(self._ready)

    def _emit(self, env_id: int) -> None:
        obs = self.observe(self._pile[env_id], self._player[env_id], self.max_pile)
        self._ready.append(EncodedEvent(env_id, self._player[env_id], obs, None))


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
