"""The PPO self-play training loop on real duels (T4b.4); see docs/training.md §8.

One iteration: collect a rollout on :class:`ygorl.env.encoded.EncodedVecEnv` (current policy on both seats,
or against a snapshot from the pool), one :class:`ygorl.train.ppo.PPOLearner` update, then the league
bookkeeping: a snapshot every ``snapshot_every`` updates, a checkpoint every ``checkpoint_every``, and every
``eval_every`` updates an evaluation of the saved checkpoint against baseline agents on paired seeds (the
arena path, ``policy:<checkpoint>`` vs ``greedy`` / ``random``) with keep-best: the checkpoint with the best
win rate against ``keep_best_by`` is copied to ``best.pt`` and pinned in the pool.

Run directory: ``config.json``, ``vocab.json`` (``CardVocab.save``), ``metrics.jsonl`` (one line per update),
``eval.jsonl`` (one line per evaluation and opponent), ``checkpoints/latest.pt`` + ``checkpoints/update_N.pt``
(the evaluated ones), ``best.pt``.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.nets.actor_critic import ActorCritic
from ygorl.nets.config import NetConfig
from ygorl.nets.gemm import use_split_k
from ygorl.nets.text import TextFeatures
from ygorl.train.checkpoint import (
    load_actor,
    load_checkpoint,
    save_checkpoint,
    vocab_from_text,
    vocab_passwords,
    vocab_to_text,
)
from ygorl.train.ppo import PPOConfig, PPOLearner
from ygorl.train.rollout import Rollout, RolloutCollector
from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool

SMALL_NET = {"d_model": 64, "n_heads": 4, "board_layers": 1, "history_layers": 1}


@dataclass(frozen=True)
class TrainConfig:
    decks: tuple[str, ...] = ()  # .ydk files of the deck pool
    pairings: str = "cross"  # DeckPool pairings: "all", "cross" or "mirror"
    env: str | None = None  # environment version / directory (rules, stamped into checkpoints)
    max_turns: int | None = None  # DuelConfig overrides (None = the environment's / default)
    max_decisions: int | None = None
    num_envs: int = 32
    env_threads: int = 2
    steps: int = 64  # rows per environment slot per rollout (T)
    min_batch: int | None = None  # ready decisions per forward pass (default num_envs // 2)
    event_length: int = 64  # event tokens per observation (window mode)
    skip_forced: bool = True  # decisions with one choosable row are played in C++ and produce no rows
    net: dict = field(default_factory=lambda: dict(SMALL_NET))  # NetConfig overrides (vocab_size is set)
    text_dir: str | None = None  # frozen text tables (T5.2); None = off
    privileged_critic: bool = True  # design I9: the critic sees the opponent ground truth
    privileged_dim: int = 64
    critic_hidden: int = 128
    shared_backbone: bool = True
    ppo: PPOConfig = field(default_factory=PPOConfig)
    selfplay_fraction: float = 0.75  # deals against the current policy; the rest against pool snapshots
    pool_size: int = 8
    # fixed opponents kept in the pool for the whole run (policy or PPO checkpoints, e.g. the BC warm start) and
    # their share of the pool games; the run's card vocab and event length must match
    pin_opponents: tuple[str, ...] = ()
    pinned_share: float = 0.5
    snapshot_every: int = 10  # updates
    pool_sampling: str = "pfsp"  # "pfsp" (design I1) or "uniform" (SnapshotPool): opponents the learner loses to come up more
    pfsp_power: float = 2.0
    # a due snapshot joins the pool only if the learner scored above this in its pool games since the last one joined
    # (at least snapshot_min_games of them; ygo-agent's OSFP uses 0.55); None = always; an empty pool always takes one
    snapshot_min_win_rate: float | None = None
    snapshot_min_games: int = 20
    checkpoint_every: int = 10
    eval_every: int = 25
    eval_pairs: int = 8  # paired seeds per deck pairing and baseline (2 games each)
    eval_pairings: int = 0  # evaluate on this many training pairings, a fixed sample (seeded); 0 = all of them
    eval_opponents: tuple[str, ...] = ("greedy", "random")
    keep_best_by: str = "greedy"
    eval_workers: int = 2
    eval_greedy_policy: bool = False  # evaluate the argmax policy (policy-greedy) instead of sampling
    seed: int = 0
    device: str = "cpu"  # PyTorch device of the learner and the acting network ("cuda" also covers ROCm)
    # Experimental, not in the design (synchronous PPO; docs/scaling.md S3): collect the next rollout on a thread with
    # a copy of the weights the update starts from, while the update runs; every rollout is then one update stale
    overlap_collect: bool = False
    bf16: bool = False  # experimental: bf16 autocast for acting and the update on a GPU
    torch_threads: int = 4  # during updates
    collect_threads: int = 2  # PyTorch threads while collecting (the env threads share the cores)
    bc_prior: str | None = None  # checkpoint of a BC policy: KL prior (ppo.kl_prior_coef)
    init_from: str | None = None  # initialize the actor from a checkpoint (e.g. BC warm start)

    def __post_init__(self) -> None:
        if self.keep_best_by and self.keep_best_by not in self.eval_opponents:
            raise ValueError("keep_best_by must be one of eval_opponents")
        if self.num_envs < 1 or self.steps < 1:
            raise ValueError("num_envs and steps must be at least 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ppo"] = self.ppo.to_dict()
        d["decks"], d["eval_opponents"] = list(self.decks), list(self.eval_opponents)
        d["pin_opponents"] = list(self.pin_opponents)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in data.items() if k in known}
        d["ppo"] = PPOConfig.from_dict(d.get("ppo", {}))
        for key in ("decks", "eval_opponents", "pin_opponents"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def _mean(xs) -> float | None:
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


class Trainer:
    """Build with a :class:`TrainConfig` and a run directory, or :meth:`resume` from a checkpoint."""

    def __init__(self, cfg: TrainConfig, run_dir: str | Path, *, state: dict | None = None,
                 log: Callable[[str], None] | None = print) -> None:  # fmt: skip
        if not cfg.decks:
            raise ValueError("the deck pool is empty (TrainConfig.decks)")
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda _msg: None)
        self.environment = None
        if cfg.env is not None:
            from ygorl.data import load_environment

            self.environment = load_environment(cfg.env, cards=default_cards())
        overrides = {k: v for k, v in (("max_turns", cfg.max_turns), ("max_decisions", cfg.max_decisions)) if v}
        self.duel_config = (DuelConfig.from_environment(self.environment, **overrides) if self.environment is not None
                            else DuelConfig(**overrides))  # fmt: skip
        self.decks = [load_ydk(p) for p in cfg.decks]
        self.cards = default_cards()
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and torch.version.hip:  # fused attention with a mask (docs/benchmarks.md)
            os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        init = load_actor(cfg.init_from, cfg.text_dir) if cfg.init_from and state is None else None
        if state is not None:
            self.vocab = vocab_from_text(state["vocab"])
        else:  # a warm start brings its own vocab: its ID embeddings are indexed by it
            self.vocab = init.vocab if init is not None else CardVocab.from_db(self.cards)
        self.vocab_text = vocab_to_text(self.vocab, self.run_dir / "vocab.json")
        text = TextFeatures.load(cfg.text_dir, self.vocab) if cfg.text_dir else None
        self.net_config = NetConfig(**{**cfg.net, "vocab_size": len(self.vocab)}).with_text(text)
        self._text = text
        torch.manual_seed(derive_seed(cfg.seed, 0))
        self.model = self._new_model()
        if init is not None:
            if init.net_config.to_dict() != self.net_config.to_dict():
                raise ValueError(f"{cfg.init_from}: its network {init.net_config.to_dict()} differs from the configured "
                                 f"{self.net_config.to_dict()} (set the same net options)")  # fmt: skip
            self.model.actor.load_state_dict(init.net.state_dict())
            self.log(f"initialized the actor from {cfg.init_from} (the critic starts fresh)")
        prior = None
        if cfg.bc_prior:
            loaded = load_actor(cfg.bc_prior, cfg.text_dir)
            if vocab_passwords(loaded.vocab) != vocab_passwords(self.vocab):
                raise ValueError(f"{cfg.bc_prior}: its card vocab differs from the run's; the prior would read "
                                 "other cards from the same indices")  # fmt: skip
            prior = loaded.net.to(self.device)
        self.learner = PPOLearner(self.model, cfg.ppo, prior)
        self.pool = SnapshotPool(cfg.pool_size, cfg.pinned_share, cfg.pool_sampling, cfg.pfsp_power)
        for path in cfg.pin_opponents:
            pinned = load_actor(path, cfg.text_dir)
            if vocab_passwords(pinned.vocab) != vocab_passwords(self.vocab):
                raise ValueError(f"{path}: its card vocab differs from the run's (pinned opponent)")
            if pinned.event_length != cfg.event_length:
                raise ValueError(f"{path}: trained with {pinned.event_length} event tokens, the run uses "
                                 f"{cfg.event_length} (pinned opponent)")  # fmt: skip
            self.pool.pin(pinned.net.to(self.device), tag=f"pinned:{Path(path).name}")
        self.schedule = SelfPlaySchedule(DeckPool(self.decks, cfg.pairings), self.pool, self.duel_config,
                                         selfplay_fraction=cfg.selfplay_fraction, seed=cfg.seed)  # fmt: skip
        self.env = EncodedVecEnv(cfg.num_envs, cfg.env_threads, cards=self.cards, vocab=self.vocab,
                                 privileged=cfg.privileged_critic, event_length=cfg.event_length,
                                 skip_forced=cfg.skip_forced)  # fmt: skip
        # the acting network: the model itself, or a copy refreshed before each overlapped collection
        self.acting = copy.deepcopy(self.model) if cfg.overlap_collect else self.model
        self._pending: Rollout | None = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="collect") if cfg.overlap_collect else None
        self._stream = torch.cuda.Stream(self.device) if cfg.overlap_collect and self.device.type == "cuda" else None
        self.collector = RolloutCollector(self.env, self.acting, self.schedule, cfg.steps, opponents=self.schedule.opponent,
                                          seed=derive_seed(cfg.seed, 3), min_batch=cfg.min_batch,
                                          device=self.device)  # fmt: skip
        self.counters = {"updates": 0, "rows": 0, "decisions": 0, "games": 0, "seconds": 0.0, "truncated": 0,
                         "errors": 0, "snapshots": 0, "snapshots_skipped": 0}  # fmt: skip
        self.league = {"games": 0, "points": 0.0}  # the learner's pool games since the last snapshot joined
        self.best = {"score": None, "update": None}
        if state is not None:
            self._restore(state)
        (self.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    def _new_model(self) -> ActorCritic:
        c = self.cfg
        model = ActorCritic(self.net_config, self._text, privileged=c.privileged_critic, privileged_dim=c.privileged_dim,
                           critic_hidden=c.critic_hidden, shared_backbone=c.shared_backbone)  # fmt: skip
        return use_split_k(model).to(self.device)

    # -- persistence ----------------------------------------------------------------------------------
    @classmethod
    def resume(cls, checkpoint: str | Path, run_dir: str | Path | None = None, *, log=print, **overrides) -> Trainer:
        """Continue a run from ``checkpoint``; ``overrides`` replace config fields (e.g. ``eval_every``)."""
        state = load_checkpoint(checkpoint)
        cfg = replace(TrainConfig.from_dict(state["config"]), **overrides)
        if run_dir is None:  # RUN/checkpoints/NAME.pt, or a checkpoint copied elsewhere (its own directory)
            here = Path(checkpoint).resolve().parent
            run_dir = here.parent if here.name == "checkpoints" else here
        return cls(cfg, Path(run_dir), state=state, log=log)

    def state_dict(self) -> dict:
        return {"config": self.cfg.to_dict(), "net_config": self.net_config.to_dict(), "vocab": self.vocab_text,
                "environment": self.environment.stamp() if self.environment is not None else None,
                "learner": self.learner.state_dict(), "pool": self.pool.state_dict(),
                "schedule": self.schedule.state_dict(), "counters": dict(self.counters), "best": dict(self.best), "league": dict(self.league),
                "rng": {"torch": torch.get_rng_state(), "collector": self.collector.generator.get_state()}}  # fmt: skip

    def _restore(self, state: dict) -> None:
        if state["net_config"] != self.net_config.to_dict():
            raise ValueError("the checkpoint's network does not match the configuration")
        self.learner.load_state_dict(state["learner"])
        self.pool.load_state_dict(state["pool"], self._new_model)
        self.schedule.load_state_dict(state["schedule"])
        self.counters.update(state["counters"])
        self.best.update(state["best"])
        self.league.update(state.get("league", {}))
        torch.set_rng_state(state["rng"]["torch"])
        self.collector.generator.set_state(state["rng"]["collector"])

    def save(self, name: str = "latest.pt") -> Path:
        return save_checkpoint(self.state_dict(), self.run_dir / "checkpoints" / name)

    # -- training ---------------------------------------------------------------------------------------
    def train(self, max_updates: int | None = None, max_minutes: float | None = None) -> dict:
        """Run until ``max_updates`` more updates or ``max_minutes`` of wall time; returns the last metrics."""
        cfg = self.cfg
        start, done, last = time.time(), 0, {}
        try:
            while (max_updates is None or done < max_updates) and (
                max_minutes is None or time.time() - start < max_minutes * 60
            ):
                last = self.step()
                done += 1
                u = self.counters["updates"]
                if cfg.snapshot_every and u % cfg.snapshot_every == 0:
                    self.maybe_snapshot(u)
                if cfg.eval_every and u % cfg.eval_every == 0:
                    self.evaluate()
                elif cfg.checkpoint_every and u % cfg.checkpoint_every == 0:
                    self.save()
        except KeyboardInterrupt:
            self.log("interrupted: saving checkpoints/latest.pt")
        self.save()
        return last

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16)

    def _collect(self) -> Rollout:
        if self._stream is None:
            with self._autocast():
                return self.collector.collect()
        with torch.cuda.stream(self._stream), self._autocast():
            ro = self.collector.collect()
        self._stream.synchronize()  # the rollout's tensors are complete before the learner's stream reads them
        return ro

    def maybe_snapshot(self, update: int) -> bool:
        """Add a snapshot of the model to the pool, unless the admission threshold holds it back."""
        cfg, lg = self.cfg, self.league
        regular = [i for i in self.pool.ids() if i >= 0]
        if cfg.snapshot_min_win_rate is not None and regular:
            if lg["games"] < cfg.snapshot_min_games or lg["points"] / lg["games"] <= cfg.snapshot_min_win_rate:
                self.counters["snapshots_skipped"] += 1
                return False
        self.pool.add(self.model, update=update)
        self.counters["snapshots"] += 1
        self.league.update(games=0, points=0.0)
        return True

    def step(self) -> dict:
        cfg = self.cfg
        t_step = time.perf_counter()
        torch.set_num_threads(cfg.collect_threads)
        future = None
        if cfg.overlap_collect:
            ro, self._pending = self._pending or self._collect(), None
            self.acting.load_state_dict(self.model.state_dict())
            if self._stream is not None:
                self._stream.wait_stream(torch.cuda.current_stream(self.device))
            future = self._executor.submit(self._collect)
        else:
            ro = self._collect()
        torch.set_num_threads(cfg.torch_threads)
        t0 = time.perf_counter()
        with self._autocast():
            stats = self.learner.update(ro)
        update_s = time.perf_counter() - t0
        if future is not None:  # the league bookkeeping between steps never runs next to a collection
            self._pending = future.result()
        step_s = time.perf_counter() - t_step
        c = self.counters
        c["updates"] = self.learner.updates
        c["rows"] += ro.players.numel()
        c["decisions"] += ro.decisions
        c["games"] += len(ro.games)
        c["seconds"] += step_s
        c["truncated"] += sum(g.truncated for g in ro.games)
        c["errors"] += sum(g.reason == "error" for g in ro.games)
        for g in ro.games:
            if g.learner_score is not None and not g.truncated:
                self.pool.record(g.assignment.opponent, g.learner_score)
                self.league["games"] += 1
                self.league["points"] += g.learner_score
        record = {"update": c["updates"], "time": round(time.time(), 1), **self._rollout_stats(ro, update_s),
                  "step_s": round(step_s, 2), "rows_per_s": ro.players.numel() / max(step_s, 1e-9),
                  **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
                  "total": dict(c), "pool": len(self.pool)}  # fmt: skip
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.log(f"update {c['updates']}: {record['decisions_per_s']:.0f} decisions/s collect, "
                 f"{record['rows_per_s']:.0f} rows/s overall, games {len(ro.games)} "
                 f"(truncated {record['truncated_games']}), entropy {stats['entropy']:.3f}, kl_ref {stats['kl_ref']:.4f}, "
                 f"q_loss {stats['q_loss']:.4f}, pool win {record['pool_win_rate']}")  # fmt: skip
        return record

    @staticmethod
    def _rollout_stats(ro: Rollout, update_s: float) -> dict:
        games = ro.games
        pool_scores = [g.learner_score for g in games if g.learner_score is not None]
        decided = [g for g in games if not g.truncated and g.assignment.opponent is None]
        first_wins = [0.5 if g.winner is None else float(g.winner == 0) for g in decided]
        return {
            "rows": ro.players.numel(), "decisions": ro.decisions, "collect_s": round(ro.seconds, 2),
            "update_s": round(update_s, 2), "decisions_per_s": ro.decisions / max(ro.seconds, 1e-9),
            "games": len(games),
            "selfplay_games": sum(g.assignment.opponent is None for g in games),
            "pool_games": len(pool_scores), "pool_win_rate": None if not pool_scores else round(_mean(pool_scores), 3),
            "first_player_win_rate": None if not first_wins else round(_mean(first_wins), 3),
            "truncated_games": sum(g.truncated for g in games), "errors": sum(g.reason == "error" for g in games),
            "reasons": dict(Counter(g.reason for g in games)),
            "mean_game_decisions": _mean(int(g.result.get("decisions", 0)) for g in games),
            "mean_turns": _mean(int(g.result.get("turns", 0)) for g in games),
        }  # fmt: skip

    # -- evaluation and keep-best -----------------------------------------------------------------------
    def eval_pairings(self) -> list[tuple[int, int]]:
        """The deck pairings of the periodic evaluation: all training pairings, or a fixed seeded sample of them."""
        pairs = self.schedule.decks.pairs
        k = self.cfg.eval_pairings
        if not k or k >= len(pairs):
            return list(pairs)
        idx = np.random.default_rng(derive_seed(self.cfg.seed, 5)).choice(len(pairs), k, replace=False)
        return [pairs[i] for i in sorted(idx)]

    def evaluate(self) -> dict[str, dict]:
        """Save ``checkpoints/update_N.pt`` and play it against every baseline; update keep-best."""
        cfg = self.cfg
        u = self.counters["updates"]
        path = self.save(f"update_{u:06d}.pt")
        shutil.copyfile(path, self.run_dir / "checkpoints" / "latest.pt")
        results = {}
        for opponent in cfg.eval_opponents:
            rep, seconds = evaluate_checkpoint(path, self.decks, opponent, pairings=self.eval_pairings(),
                                               pairs=cfg.eval_pairs, config=self.duel_config, env=self.environment,
                                               workers=cfg.eval_workers, seed=derive_seed(cfg.seed, 4),
                                               greedy_policy=cfg.eval_greedy_policy)  # fmt: skip
            results[opponent] = {"update": u, "vs": opponent, "games": rep.games, "win_rate": rep.win_rate,
                                 "ci": list(rep.ci), "wins": rep.wins, "losses": rep.losses, "draws": rep.draws,
                                 "reasons": rep.reasons, "errors": rep.errors, "mean_turns": rep.mean_turns,
                                 "seconds": round(seconds, 1)}  # fmt: skip
        score = results[cfg.keep_best_by]["win_rate"] if cfg.keep_best_by else None
        is_best = score is not None and (self.best["score"] is None or score > self.best["score"])
        if is_best:
            self.best = {"score": score, "update": u}
            self.pool.set_best(self.model, update=u, score=score)
            path = self.save(f"update_{u:06d}.pt")  # re-save with the new best recorded
            shutil.copyfile(path, self.run_dir / "best.pt")
            shutil.copyfile(path, self.run_dir / "checkpoints" / "latest.pt")
        with (self.run_dir / "eval.jsonl").open("a") as f:
            for r in results.values():
                f.write(json.dumps({**r, "best": is_best and r["vs"] == cfg.keep_best_by}) + "\n")
        summary = ", ".join(f"vs {k} {v['win_rate']:.3f} ({v['games']} games)" for k, v in results.items())
        self.log(f"eval at update {u}: {summary}{' -> new best' if is_best else ''}")
        return results


def evaluate_checkpoint(checkpoint: str | Path, decks, opponent: str, *, pairings, pairs: int, config: DuelConfig,
                        env=None, workers: int = 1, seed: int = 0, greedy_policy: bool = False):  # fmt: skip
    """Arena report of ``policy:<checkpoint>`` (on deck i) against ``opponent`` (on deck j) over the given
    ``(i, j)`` pairings, ``pairs`` paired seeds each; returns ``(merged report, seconds)``."""
    from ygorl.agents.registry import AgentSpec
    from ygorl.eval.arena import Arena, merge

    kind = "policy-greedy" if greedy_policy else "policy"
    arena = Arena(AgentSpec(f"{kind}:{Path(checkpoint).resolve()}"), AgentSpec(opponent), env=env, config=config,
                  workers=workers, mp_context="spawn" if workers > 1 else None)  # fmt: skip
    cells = [(decks[i], decks[j], derive_seed(seed, k)) for k, (i, j) in enumerate(pairings)]
    t0 = time.time()
    reports = arena.run_many(cells, pairs)
    return merge(reports), time.time() - t0


def summarize_metrics(path: str | Path, window: int = 10) -> dict:
    """First / last ``window`` update averages of the main metrics in a ``metrics.jsonl`` (for reports)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = ("entropy", "kl_ref", "q_loss", "v_loss", "policy_loss", "approx_kl", "clip_frac", "q_explained_var",
            "decisions_per_s", "rows_per_s", "mean_game_decisions", "pool_win_rate")  # fmt: skip

    def avg(sub, k):
        vals = [r[k] for r in sub if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        return round(statistics.fmean(vals), 4) if vals else None

    return {"updates": len(rows), "first": {k: avg(rows[:window], k) for k in keys},
            "last": {k: avg(rows[-window:], k) for k in keys}, "total": rows[-1]["total"] if rows else {}}  # fmt: skip


__all__ = ["SMALL_NET", "TrainConfig", "Trainer", "evaluate_checkpoint", "summarize_metrics"]
