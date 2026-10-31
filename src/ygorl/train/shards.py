"""Sharded segment collection: N actor subprocesses collect the segment in parallel under identical weights.

The parent trainer splits ``[T, num_envs]`` into N shards; each shard process runs its own
:class:`~ygorl.env.encoded.EncodedVecEnv` and :class:`~ygorl.train.rollout.RolloutCollector` with an acting
copy of the current policy (refreshed before every segment) and a local mirror of the snapshot pool. The
merged rollout is layout-identical to a single collector's (rows ``t * B + b``), so the learner, the league
bookkeeping and every estimator see exactly the synchronous-PPO data they would see otherwise: every row is
on-policy with respect to the pre-update weights — no staleness of any kind.

Wire protocol (spawn context; pickles over duplex pipes): the parent sends an ``init`` message (config dict,
shard id, vocab path) and then, per segment, a ``collect`` message (actor weights, changed pool nets); the
shard answers with a pickled :class:`~ygorl.train.rollout.Rollout` (CPU tensors).
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import torch
from torch import Tensor, nn

from ygorl.train.rollout import Rollout

_T_B_FIELDS = (
    "players",
    "actions",
    "log_probs",
    "values",
    "rewards",
    "dones",
    "truncated",
    "line_id",
    "line_window",
    "line_effect",
    "line_occurrence",
    "line_seat",
    "branch_valid",
    "branch_pass",
)
_T_B_A_FIELDS = ("probs", "q", "action_mask")
_T_B_A_OPTIONAL = ("branch_value",)
_B_FIELDS = ("bootstrap_player", "bootstrap_mask", "bootstrap_probs", "bootstrap_q", "bootstrap_value")
_SCALAR_FIELDS = (
    "opponent_decisions",
    "branch_windows",
    "branch_decisions",
    "branch_no_recovery",
    "lookahead_decisions",
    "lookahead_branched",
    "lookahead_changed",
    "lookahead_steps",
    "discarded_rows",
)


def merge_rollouts(rollouts: list[Rollout]) -> Rollout:
    """Concatenate shard rollouts along the env dimension ``B``, preserving the ``t * B + b`` row layout.

    ``obs`` rows are shard-major inside each shard; reshaping every shard to ``[T, B_s, ...]`` first and
    concatenating along ``B`` reproduces exactly the single-collector layout. Tensors may be any mix of
    devices (the learner moves the result once, as with a single CPU-side collector).
    """
    if len(rollouts) == 1:
        return rollouts[0]
    T, B_s = rollouts[0].players.shape
    T_all = {r.players.shape[0] for r in rollouts}
    assert len(T_all) == 1 and T_all == {T}, f"shards disagree on segment length: {T_all}"

    kwargs: dict[str, object] = {}
    for name in _T_B_FIELDS:
        vals = [getattr(r, name) for r in rollouts]
        if all(v is not None for v in vals):
            kwargs[name] = torch.cat(vals, dim=1)
        elif any(v is not None for v in vals):
            raise ValueError(f"rollouts disagree on optional field {name}")
    for name in _T_B_A_FIELDS:
        kwargs[name] = torch.cat([getattr(r, name) for r in rollouts], dim=1)
    for name in _T_B_A_OPTIONAL:
        vals = [getattr(r, name) for r in rollouts]
        if all(v is not None for v in vals):
            kwargs[name] = torch.cat(vals, dim=1)
        elif any(v is not None for v in vals):
            raise ValueError(f"rollouts disagree on optional field {name}")
    for name in _B_FIELDS:
        kwargs[name] = torch.cat([getattr(r, name) for r in rollouts], dim=0)
    for name in _SCALAR_FIELDS:
        kwargs[name] = sum(getattr(r, name) for r in rollouts)

    obs = {}
    for key, first in rollouts[0].obs.items():
        parts = [r.obs[key] for r in rollouts]
        if parts[0].dim() == 1:  # [T * B_s] per-row scalars: rows only, no B axis beyond the row
            obs[key] = torch.cat([p.reshape(T, -1) for p in parts], dim=1).reshape(-1)
        else:
            obs[key] = torch.cat([p.reshape(T, B_s, *p.shape[1:]) for p in parts], dim=1).reshape(
                -1, *parts[0].shape[1:]
            )
    kwargs["obs"] = obs
    privs = [r.privileged for r in rollouts]
    if all(p is not None for p in privs):
        merged_priv = {}
        for key in privs[0]:
            parts = [p[key] for p in privs]
            if parts[0].dim() == 1:
                merged_priv[key] = torch.cat([q.reshape(T, -1) for q in parts], dim=1).reshape(-1)
            else:
                merged_priv[key] = torch.cat([q.reshape(T, B_s, *q.shape[1:]) for q in parts], dim=1).reshape(
                    -1, *parts[0].shape[1:]
                )
        kwargs["privileged"] = merged_priv
    else:
        kwargs["privileged"] = None
    games = []
    for r in rollouts:
        games.extend(r.games)
    return Rollout(
        games=games,
        seconds=max(r.seconds for r in rollouts),
        complete_games=any(r.complete_games for r in rollouts),
        **kwargs,
    )


def rollout_to_cpu(ro: Rollout) -> Rollout:
    """Move every tensor of a rollout to host memory (MPS tensors cannot cross a multiprocessing pipe)."""
    for name in (*_T_B_FIELDS, *_T_B_A_FIELDS, *_T_B_A_OPTIONAL, *_B_FIELDS):
        t = getattr(ro, name, None)
        if isinstance(t, Tensor) and t.device.type != "cpu":
            setattr(ro, name, t.cpu())
    ro.obs = {k: (v.cpu() if isinstance(v, Tensor) and v.device.type != "cpu" else v) for k, v in ro.obs.items()}
    if ro.privileged is not None:
        ro.privileged = {
            k: (v.cpu() if isinstance(v, Tensor) and v.device.type != "cpu" else v) for k, v in ro.privileged.items()
        }
    return ro


def _weights_to_bytes(model: nn.Module) -> bytes:
    return pickle.dumps({k: v.detach().cpu() for k, v in model.state_dict().items()})


def _load_weights_bytes(model: nn.Module, data: bytes) -> None:
    model.load_state_dict(pickle.loads(data))


@dataclass
class ShardState:
    process: Process
    pipe: Connection
    sent_pool: int = -1  # pool version the shard last received (skip resends)


class ShardSupervisor:
    """Parent-side driver: keeps N collector subprocesses, feeds weights, merges their rollouts.

    The trainer calls :meth:`collect` where it used to call ``RolloutCollector.collect``; the returned
    rollout is layout-identical, so nothing downstream changes. The snapshot pool lives in the parent (the
    league keeps its bookkeeping there); shards receive the pool's frozen nets whenever the pool changes.
    """

    def __init__(self, cfg_dict: dict, num_shards: int, model: nn.Module, pool, run_dir,
                 log=print) -> None:  # fmt: skip
        from ygorl.train.shard_worker import shard_main

        self.model = model
        self.pool = pool
        self.num_shards = num_shards
        self.log = log
        self.shards: list[ShardState] = []
        for shard in range(num_shards):
            parent, child = Pipe(True)
            p = Process(target=shard_main, args=(child, cfg_dict, shard, num_shards, str(run_dir)), daemon=True)
            p.start()
            child.close()
            self.shards.append(ShardState(process=p, pipe=parent))
        self._push(weights=True, pool=True)

    def _pool_nets(self) -> tuple[int, dict[int, bytes]]:
        snaps = {**self.pool._pinned, **self.pool._snaps} if hasattr(self.pool, "_pinned") else dict(self.pool._snaps)
        # The signature (not the count) versions the mirror: an eviction can keep the size while changing the set
        signature = hash(tuple(sorted(snaps)))
        return signature, {sid: _weights_to_bytes(s.model) for sid, s in snaps.items()}

    def _push(self, *, weights: bool, pool: bool) -> None:
        version, nets = self._pool_nets()
        for s in self.shards:
            need_pool = pool and s.sent_pool != version
            s.pipe.send(
                {
                    "cmd": "sync",
                    "weights": _weights_to_bytes(self.model) if weights else None,
                    "pool": nets if need_pool else None,
                    "pool_version": version,
                }
            )
            if need_pool:
                s.sent_pool = version

    def collect(self) -> Rollout:
        t0 = time.perf_counter()
        self._push(weights=True, pool=True)  # the acting copy already holds the post-update weights
        for s in self.shards:
            s.pipe.send({"cmd": "collect"})
        rollouts = []
        for s in self.shards:
            if not s.process.is_alive():
                raise RuntimeError(f"collection shard died (exitcode {s.process.exitcode})")
            rollouts.append(s.pipe.recv())
        merged = merge_rollouts(rollouts)
        merged.seconds = time.perf_counter() - t0
        return merged

    def close(self) -> None:
        for s in self.shards:
            try:
                s.pipe.send({"cmd": "stop"})
                s.pipe.close()
            except (OSError, ValueError):
                pass
        for s in self.shards:
            s.process.join(timeout=10)
            if s.process.is_alive():
                s.process.terminate()
