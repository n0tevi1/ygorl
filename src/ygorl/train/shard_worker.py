"""Shard worker main loop (spawned): build the env shard through the trainer's own construction, serve collect commands."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch


def shard_main(pipe, cfg_dict: dict, shard: int, num_shards: int, run_dir: str) -> None:
    from ygorl.cards.cdb import CardVocab
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.eval.arena import derive_seed
    from ygorl.train.checkpoint import CriticConfig, build_actor_critic
    from ygorl.train.critic import load_allowing_new_heads
    from ygorl.train.lines import LineLibrary
    from ygorl.train.selfplay import SnapshotPool
    from ygorl.train.shards import _load_weights_bytes, rollout_to_cpu
    from ygorl.train.trainer import TrainConfig, build_collection

    cfg = TrainConfig.from_dict(cfg_dict)
    device = torch.device(cfg.collect_device) if cfg.collect_device else torch.device("cpu")
    cards = default_cards()
    vocab = CardVocab.load(Path(run_dir) / "vocab.json")
    environment = None
    if cfg.env is not None:
        from ygorl.data import load_environment

        environment = load_environment(cfg.env, cards=cards)
    overrides = {k: v for k, v in (("max_turns", cfg.max_turns), ("max_decisions", cfg.max_decisions)) if v}
    duel_config = (DuelConfig.from_environment(environment, **overrides) if environment is not None
                   else DuelConfig(**overrides))  # fmt: skip

    net_config = cfg.actor_config(len(vocab), None)
    critic_cfg = CriticConfig.from_train_config(cfg_dict)
    acting = build_actor_critic(net_config, None, critic_cfg).to(device)
    acting.eval()
    acting.requires_grad_(False)

    pool = SnapshotPool(cfg.pool_size, cfg.pinned_share, cfg.pool_sampling, cfg.pfsp_power, device=device)
    lines = None
    if cfg.line_demos and cfg.line_share > 0:
        lines = LineLibrary.load_cached(
            cfg.line_demos,
            vocab,
            cards=cards,
            env=environment,
            recovery=cfg.line_recovery,
            event_length=cfg.event_length,
            cache=Path("out") / "linecache" / "lib.pkl",
        )

    per_shard = max(1, cfg.num_envs // num_shards)
    _, _, collector = build_collection(
        cfg, model=acting, cards=cards, vocab=vocab, net_config=net_config, pool=pool, lines=lines,
        evolved=None, duel_config=duel_config, environment=environment,
        num_envs=per_shard, env_threads=max(1, cfg.env_threads),
        seed=derive_seed(cfg.seed, 2000 + shard), device=device)  # fmt: skip

    while True:
        try:
            msg = pipe.recv()
        except EOFError:  # the parent is gone (normal exit or crash): die quietly
            return
        if msg["cmd"] == "stop":
            return
        if msg["cmd"] == "sync":
            if msg["weights"] is not None:
                _load_weights_bytes(acting, msg["weights"])
            if msg["pool"] is not None:  # a full mirror refresh: replace, so evictions propagate
                from ygorl.train.selfplay import Snapshot

                pool._snaps.clear()
                for sid, blob in msg["pool"].items():
                    fresh = build_actor_critic(net_config, None, critic_cfg)
                    load_allowing_new_heads(fresh, pickle.loads(blob))
                    pool._snaps[sid] = Snapshot(sid, 0, pool._freeze(fresh), "mirror", None)
                pool._next_id = max(pool._snaps, default=-1) + 1
        elif msg["cmd"] == "collect":
            with torch.no_grad():
                pipe.send(rollout_to_cpu(collector.collect()))
