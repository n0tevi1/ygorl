"""The PPO self-play training loop on real duels (T4b.4); see docs/training.md §8.

One iteration: collect a rollout on :class:`ygorl.env.encoded.EncodedVecEnv` (current policy on both seats,
or against a snapshot from the pool), one :class:`ygorl.train.ppo.PPOLearner` update, then the league
bookkeeping: a snapshot every ``snapshot_every`` updates, a checkpoint every ``checkpoint_every``, and every
``eval_every`` updates an evaluation of the saved checkpoint against baseline agents on paired seeds (the
arena path, ``policy:<checkpoint>`` vs ``greedy`` / ``random``) with keep-best: the checkpoint with the best
win rate against ``keep_best_by`` is copied to ``best.pt`` and pinned in the pool.

Run directory: ``config.json``, ``vocab.json`` (``CardVocab.save``), ``metrics.jsonl`` (one line per update),
``eval.jsonl`` (one line per evaluation and opponent), ``games.jsonl.gz`` (with ``log_games``: one line per finished
game, :meth:`Trainer._log_games`), ``checkpoints/latest.pt`` + ``checkpoints/update_N.pt``
(the evaluated ones), ``best.pt``.
"""

from __future__ import annotations

import copy
import fcntl
import gzip
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
    FORMAT,
    CriticConfig,
    Signature,
    build_actor_critic,
    checkpoint_format,
    load_actor,
    load_checkpoint,
    save_checkpoint,
    torch_device,
    vocab_from_text,
    vocab_to_text,
    warm_start,
)
from ygorl.train.ppo import PPOConfig, PPOLearner
from ygorl.train.rollout import BranchConfig, Rollout, RolloutCollector, RolloutStalled
from ygorl.train.league import League
from ygorl.train.line_labels import LineLabels
from ygorl.train.lines import LineLibrary
from ygorl.train.selfplay import DeckPool, EvolvedDecks, SelfPlaySchedule, SnapshotPool
from ygorl.train.shards import ShardSupervisor

SMALL_NET = {"d_model": 64, "n_heads": 4, "board_layers": 1, "history_layers": 1}


PPO_EVENT_LENGTH = 64  # a fresh PPO run's event window (the encoders default to env.events.DEFAULT_EVENT_LENGTH = 128)


@dataclass(frozen=True)
class TrainConfig:
    decks: tuple[str, ...] = ()  # .ydk files of the deck pool
    pairings: str = "cross"  # DeckPool pairings: "all", "cross" or "mirror"
    # an evolution process's deck-pool manifest (selfplay.EvolvedDecks, #108), re-read every deck_pool_every updates;
    # evolved_share of the deals pair one of its probation / active decks, weighted by (1 - p) ** evolved_power
    deck_pool: str | None = None
    deck_pool_every: int = 10
    evolved_share: float = 0.3
    evolved_power: float = 1.0
    env: str | None = None  # environment version / directory (rules, stamped into checkpoints)
    max_turns: int | None = None  # DuelConfig overrides (None = the environment's / default)
    max_decisions: int | None = None
    num_envs: int = 128  # x steps = 16,384 rows per update (design I1: update noise is the plateau)
    env_threads: int = 2
    complete_games: bool = False  # experimental: one whole game per env per update, steps unused
    steps: int = 128  # rows per environment slot per rollout (T)
    min_batch: int | None = None  # ready decisions per forward pass (default num_envs // 2)
    # event tokens per observation (window mode); None: the --init-from checkpoint's, else PPO_EVENT_LENGTH. A
    # Trainer always stores the number it used (checkpoints, resume)
    event_length: int | None = None
    skip_forced: bool = True  # decisions with one choosable row are played in C++ and produce no rows
    cancel_budget: int = 32  # experimental environment mask, both learner and pool-opponent seats
    net: dict = field(default_factory=lambda: dict(SMALL_NET))  # NetConfig overrides (vocab_size is set)
    text_dir: str | None = None  # frozen text tables (T5.2); None = off
    privileged_critic: bool = True  # design I9: the critic sees the opponent ground truth
    privileged_dim: int = 64
    critic_hidden: int = 128
    shared_backbone: bool = True
    # separate critic on a warm start (shared_backbone=False with init_from): "actor" copies the actor's trunk into the
    # critic trunk and, when init_from is a PPO checkpoint, loads its critic heads too (the arm-G recipe, #434: a
    # critic that starts from the shared trunk reaches 0.90 in 384 updates where a fresh one dips first); "fresh" keeps
    # the new critic's random initialization
    critic_init: str = "actor"
    critic_deck_order: bool = False  # the privileged critic also reads both players' next draws (docs/encoding.md)
    # auxiliary critic heads (#440, design R4): 0 = none, else len(train.aux.AUX_TARGETS); trained with ppo.aux_coef
    critic_aux: int = 0
    # survival-potential head Φ on the critic trunk (docs/training.md §8.2): trained with ppo.phi_coef, shapes rewards
    # with ppo.phi_shape_coef; a resumed checkpoint without it starts the head fresh
    critic_phi: bool = False
    ppo: PPOConfig = field(default_factory=PPOConfig)
    selfplay_fraction: float = 0.75  # deals against the current policy; the rest against pool snapshots
    pool_size: int = 8
    # fixed opponents kept in the pool for the whole run (policy or PPO checkpoints, e.g. the BC warm start) and
    # their share of the pool games; the run's card vocab and event length must match
    pin_opponents: tuple[str, ...] = ()
    pinned_share: float = 0.5
    # scripted-line opponents (train.lines): demonstration files (docs/solver.md) whose solved plain lines the opponent
    # seat opens with in line_share of the pool games (the snapshot plays on once the line ends or breaks)
    line_demos: tuple[str, ...] = ()
    line_share: float = 0.0
    # share of the scripted deals that are first-seat games: the learner's deck moves first and the demonstration deck
    # plays the line on its turn 2 (pressure to interrupt on turn 1 and on the opponent's turn); 0 = all second-seat
    line_first_share: float = 0.0
    # --fire demonstration files of those lines' hands (train.lines recovery lines): when the learner's hand trap hits a
    # chain window the combo recovers from, the opponent seat switches to the solver's recovery line instead of
    # breaking and handing the seat to the snapshot; () = off
    line_recovery: tuple[str, ...] = ()
    # graded hand-trap labels of those lines (train.line_labels; the window cache is written next to the file): the
    # learner's labelled chain windows get ppo.line_bonus_coef * bonus on their advantage
    line_labels: str | None = None
    # deal only the lines that carry a label (line_labels): the labelled rows are sparse otherwise (a labelled line and
    # the learner holding the hand trap are both needed); scripted games then never use unlabelled lines
    line_labelled_only: bool = False
    # window branching (train.rollout.BranchConfig, docs/training.md §8.1 「窗口分支」): at the learner's chain windows of
    # the scripted games on the line's turn the slot is snapshotted and the pass and the branch_k - 1 likeliest
    # activations are each played on branch_rollouts times to the learner's next turn (branch_horizon: board score,
    # critic V or both; branch_board_scale per board point), in branch_fraction of the windows; branch_require_recovery:
    # only where the line has a recovery line for those activations; branch_cards (non-empty): only where each of those
    # activations is of one of these passwords (cards whose recovery lines are stored for every recoverable occurrence,
    # plan §11.8; both given: both conditions). The values feed ppo.branch_mode / branch_q_coef / branch_ce_coef (whose
    # margin is in board points: times branch_board_scale)
    branch_windows: bool = False
    branch_k: int = 2
    branch_rollouts: int = 1
    branch_horizon: str = "board"
    branch_fraction: float = 1.0
    branch_require_recovery: bool = True
    branch_board_scale: float = 0.1
    branch_cards: tuple[int, ...] = ()
    # look-ahead opponents (train.lookahead, first-seat plan §13): in lookahead_share of the unscripted pool deals the
    # snapshot plays its first own turn (turn 2 when the learner moves first; with lookahead_second also turn 1 when it
    # moves second) best-of-lookahead_k by snapshot branching, each candidate rolled out lookahead_rollouts times to the
    # end of the turn and scored by its board (ygorl.train.lookahead.lookahead_value)
    lookahead_share: float = 0.0
    lookahead_k: int = 8
    lookahead_rollouts: int = 1
    lookahead_second: bool = False
    # throughput of the look-ahead (first-seat plan §13.4; the choices do not change): lookahead_forks spare environment
    # slots replay a look-ahead game so a decision's candidates roll out side by side (ygorl.train.lookahead.Forks);
    # lookahead_rounds look-ahead-only rounds between two full rounds of asynchronous collection
    lookahead_forks: int = 64
    lookahead_rounds: int = 4
    snapshot_every: int = 10  # updates
    pool_sampling: str = (
        "pfsp"  # "pfsp" (design I1) or "uniform" (SnapshotPool): opponents the learner loses to come up more
    )
    pfsp_power: float = 2.0
    # active snapshot cohort (SelfPlaySchedule active_k): pool deals draw from pool_active_k snapshots redrawn every
    # pool_active_deals deals (0 = the number of environments), so in-flight games batch into few opponent forwards
    pool_active_k: int = 0
    pool_active_deals: int = 0
    # record Rollout.opp_board (the opponent's board on the first seat's first turn-3 row; metrics opp_board_mean)
    # without rewarding it; ppo.opp_board_coef > 0 records it too
    score_opp_board: bool = False
    # a due snapshot joins the pool only if the learner scored above this in its pool games since the last one joined
    # (at least snapshot_min_games of them; ygo-agent's OSFP uses 0.55); None = always; an empty pool always takes one
    snapshot_min_win_rate: float | None = None
    snapshot_min_games: int = 20
    register_every: int = 0  # publish immutable checkpoints for an independent matrix consumer; 0 = off
    register_matrix: str | None = None  # existing matrix path, resolved at publication
    checkpoint_every: int = 10
    eval_every: int = 25
    eval_pairs: int = 8  # paired seeds per deck pairing and baseline (2 games each)
    # raise when no environment produces an event for this long (an engine stuck inside a duel); None = wait forever
    stall_timeout: float | None = 900.0
    eval_pairings: int = 0  # evaluate on this many training pairings, a fixed sample (seeded); 0 = all of them
    eval_opponents: tuple[str, ...] = ("greedy", "random")
    keep_best_by: str = "greedy"
    eval_workers: int = 2
    eval_greedy_policy: bool = False  # evaluate the argmax policy (policy-greedy) instead of sampling
    seed: int = 0
    device: str = "cpu"  # PyTorch device of the learner and the acting network ("cuda" also covers ROCm)
    # Acting-side device override: e.g. "cpu" keeps the learner alone on the GPU (MPS wedges its kernels when
    # several processes stream inference commands concurrently; docs/benchmarks.md "Apple M5 Ultra(MPS)"). None:
    # the acting side shares ``device``. No effect on the algorithm: still synchronous PPO, the acting copy is
    # refreshed after every update (as with ``overlap_collect``).
    collect_device: str | None = None
    # Sharded segment collection: N actor subprocesses each run their own env shard and collector under the
    # same weights, and the parent merges the [T, B/N] rollouts along B before the update. Algorithmically
    # identical to synchronous PPO (every row is on-policy w.r.t. the pre-update weights; no staleness).
    collect_shards: int = 1
    # Snapshot-opponent inference on N CPU threads, overlapping the learner's GPU forward (profiling: the
    # opponents' small-batch forwards were ~58% of a segment). Sampling stays on the collect thread with the
    # same generator order; the draws' distribution is unchanged, their exact stream differs from on-device
    # forwards exactly as between device choices. 0 = the previous in-loop forwards.
    opp_cpu_threads: int = 0
    # Opponent inference through the traced actor graph in ygorl._infer (C++, GIL-free): several collection
    # threads forward concurrently. Needs a libtorch build of _infer and opp_cpu_threads >= 1; the graph is
    # exported from the run's actor at startup (net changes need a fresh export, checked by fingerprint).
    opp_cpp: bool = False
    # Experimental, not in the design (synchronous PPO; docs/scaling.md S3): collect the next rollout on a thread with
    # a copy of the weights the update starts from, while the update runs; every rollout is then one update stale
    overlap_collect: bool = False
    bf16: bool = False  # experimental: bf16 autocast for acting and the update on a GPU
    learner_precision: str = "inherit"  # update only: inherit bf16, or explicitly select fp32 / bf16
    torch_threads: int = 4  # during updates
    collect_threads: int = 2  # PyTorch threads while collecting (the env threads share the cores)
    bc_prior: str | None = None  # checkpoint of a BC policy: KL prior (ppo.kl_prior_coef)
    init_from: str | None = None  # initialize the actor from a checkpoint (e.g. BC warm start)
    # critic warm-up: a critic that starts fresh (BC / actor-only warm starts) gives noise advantages that can push the
    # policy off for hundreds of updates; the first updates then train the critic only (the policy is frozen) until
    # its Q explained variance averages critic_warmup_ev over 5 updates, or for critic_warmup updates at most. 0 = off
    critic_warmup: int = 0
    critic_warmup_ev: float = 0.6
    # the per-card ID embedding (most of a small actor's parameters, dense gradients) stays at its initial or
    # warm-start values; the structured card columns and any text / facts views carry the card identity
    freeze_id_embedding: bool = False
    # speed pressure, a diagnostic arm for now (design T6 conflicts with C3): won / lost games end in
    # +/- turn_discount ** turns instead of +/- 1; 1.0 = off (docs/spikes/reward-signal.md R5)
    turn_discount: float = 1.0
    # append one record per finished game to games.jsonl.gz (decks, first player, winner, turns, reason, update,
    # environment): labels for the deck surrogate (design 05 §5.3); off by default
    log_games: bool = False
    trace_games: bool = False

    def __post_init__(self) -> None:
        if type(self.cancel_budget) is not int or self.cancel_budget not in (0, 1, 4, 32):
            raise ValueError("cancel_budget must be 0, 1, 4 or 32")
        if self.learner_precision not in ("inherit", "fp32", "bf16"):
            raise ValueError("learner_precision must be inherit, fp32, or bf16")
        if self.register_every < 0 or (self.register_every and not self.register_matrix):
            raise ValueError("register_every must be nonnegative and needs register_matrix when enabled")
        if self.ppo.critic_target == "terminal" and not self.complete_games:
            raise ValueError("terminal critic targets require complete_games")
        if self.complete_games and (self.turn_discount != 1 or self.overlap_collect):
            raise ValueError("complete_games requires turn_discount=1 and synchronous collection")
        if self.keep_best_by and self.keep_best_by not in self.eval_opponents:
            raise ValueError("keep_best_by must be one of eval_opponents")
        if self.num_envs < 1 or self.steps < 1:
            raise ValueError("num_envs and steps must be at least 1")
        if not 0 <= self.line_share <= 1 or (self.line_share > 0 and not self.line_demos):
            raise ValueError("line_share must be in [0, 1] and needs line_demos when above 0")
        if not 0 <= self.line_first_share <= 1 or (self.line_first_share > 0 and self.line_share <= 0):
            raise ValueError("line_first_share must be in [0, 1] and needs a line_share above 0 when above 0")
        if self.line_recovery and not (self.line_demos and self.line_share > 0):
            raise ValueError("line_recovery needs line_demos and a line_share above 0 (it recovers those lines)")
        if self.line_labels and not (self.line_demos and self.line_share > 0):
            raise ValueError("line_labels needs line_demos and a line_share above 0 (the labels are on those lines)")
        if self.line_labelled_only and not self.line_labels:
            raise ValueError("line_labelled_only needs line_labels")
        self.branch_config()  # validates the branch_* values
        if self.branch_cards and not self.branch_windows:
            raise ValueError("branch_cards needs branch_windows")
        if self.branch_windows:
            if not (self.line_demos and self.line_share > 0):
                raise ValueError("branch_windows needs line_demos and a line_share above 0 (it branches their windows)")
            if self.branch_require_recovery and not self.line_recovery:
                raise ValueError("branch_require_recovery needs line_recovery (or --no-branch-require-recovery)")
            if self.branch_cards and not self.line_recovery:
                raise ValueError("branch_cards needs line_recovery (the covered cards' recovery lines)")
            if self.complete_games:
                raise ValueError("branch_windows needs fixed-length segments, not complete_games")
        if not 0 <= self.lookahead_share <= 1:
            raise ValueError("lookahead_share must be in [0, 1]")
        if self.lookahead_share > 0:
            self.lookahead_config()  # validates k / rollouts
            if self.complete_games:
                raise ValueError("lookahead_share needs fixed-length segments, not complete_games")
            if not self.privileged_critic:
                raise ValueError("lookahead_share needs the privileged environment (the rollouts score both boards)")
        if self.lookahead_forks < 0 or self.lookahead_rounds < 0:
            raise ValueError("lookahead_forks and lookahead_rounds must not be negative")
        from ygorl.train.aux import AUX_TARGETS

        if not 0 <= self.critic_aux <= len(AUX_TARGETS):
            raise ValueError(f"critic_aux must be in [0, {len(AUX_TARGETS)}]")
        if (self.critic_aux > 0) != (self.ppo.aux_coef > 0):
            # heads without a loss train nothing (the fs-aux runs of 2026-10-31 were silent no-ops this way)
            raise ValueError("critic_aux and ppo.aux_coef go together: auxiliary heads need --aux-coef > 0 and back")
        if self.ppo.opp_board_coef > 0 and not self.privileged_critic:
            raise ValueError("ppo.opp_board_coef needs the privileged environment (the opponent's hand and set cards)")
        if self.ppo.phi_coef > 0 and not self.critic_phi:
            raise ValueError("ppo.phi_coef needs critic_phi")
        if self.ppo.phi_shape_coef > 0 and not (self.critic_phi and self.ppo.phi_coef > 0):
            raise ValueError("ppo.phi_shape_coef needs critic_phi and ppo.phi_coef > 0 (an untrained potential "
                             "only adds noise)")  # fmt: skip
        uses_branch = self.ppo.branch_mode != "off" or self.ppo.branch_q_coef > 0 or self.ppo.branch_ce_coef > 0
        if uses_branch and not self.branch_windows:
            raise ValueError(
                "ppo.branch_mode / branch_q_coef / branch_ce_coef need branch_windows (the rollouts carry no branch "
                "values)"
            )

    def lookahead_config(self):
        from ygorl.train.lookahead import LookaheadConfig

        return LookaheadConfig(k=self.lookahead_k, rollouts=self.lookahead_rollouts)

    def branch_config(self) -> BranchConfig:
        return BranchConfig(fraction=self.branch_fraction, k=self.branch_k, rollouts=self.branch_rollouts,
                            horizon=self.branch_horizon, require_recovery=self.branch_require_recovery,
                            board_scale=self.branch_board_scale, cards=tuple(self.branch_cards))  # fmt: skip

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ppo"] = self.ppo.to_dict()
        d["decks"], d["eval_opponents"] = list(self.decks), list(self.eval_opponents)
        d["pin_opponents"] = list(self.pin_opponents)
        d["line_demos"] = list(self.line_demos)
        d["line_recovery"] = list(self.line_recovery)
        d["branch_cards"] = list(self.branch_cards)
        return d

    @property
    def critic(self) -> CriticConfig:
        return CriticConfig.from_train_config(self.to_dict())

    def actor_config(self, vocab_size: int, text: TextFeatures | None) -> NetConfig:
        """The actor's network: :attr:`net` over a vocab of ``vocab_size``, widened for the frozen tables in
        ``text`` (loaded from :attr:`text_dir`). tools/train_bc.py builds its network here too."""
        return NetConfig(**{**self.net, "vocab_size": vocab_size}).with_text(text)

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in data.items() if k in known}
        d["ppo"] = PPOConfig.from_dict(d.get("ppo", {}))
        for key in ("decks", "eval_opponents", "pin_opponents", "line_demos", "line_recovery", "branch_cards"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def _mean(xs) -> float | None:
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def line_stats(games, recovery: bool = False) -> dict:
    """Scripted-line counts of a rollout's finished games: games, lines completed / broken (and the mean step the
    broken ones broke at), lines still playing when the game ended (``line_unfinished``); the same for the first-seat
    games alone (``line_first_*``: the learner moved first, the line was played on turn 2), with the games that
    passed over leading steps of their line (``line_first_skipped``). With ``recovery`` (TrainConfig.line_recovery)
    also the lines that switched to a recovery line (``line_recovered``) and how many of those completed it
    (``line_recovered_completed``); a line still recovering at the end counts as unfinished."""
    out = {}
    for prefix, seat in (("line", None), ("line_first", 0)):
        lines = [g.line for g in games if g.line is not None and (seat is None or g.line_seat == seat)]
        broken = [ln["broken_at"] for ln in lines if ln["status"] == "broken"]
        out.update({f"{prefix}_games": len(lines), f"{prefix}_completed": sum(ln["status"] == "completed" for ln in lines),
                    f"{prefix}_broken": len(broken),
                    f"{prefix}_broken_step": None if not broken else round(_mean(broken), 2),
                    f"{prefix}_unfinished": sum(ln["status"] == "playing" for ln in lines)})  # fmt: skip
        if seat is not None:
            out[f"{prefix}_skipped"] = sum(bool(ln.get("skipped")) for ln in lines)
        if recovery:
            recovered = [ln for ln in lines if ln.get("recoveries")]
            out[f"{prefix}_unfinished"] += sum(ln["status"] == "recovering" for ln in lines)
            out.update({f"{prefix}_recovered": len(recovered),
                        f"{prefix}_recovered_completed": sum(ln["status"] == "completed" for ln in recovered)})  # fmt: skip
    return out


def build_collection(
    cfg: "TrainConfig",
    *,
    model,
    cards,
    vocab,
    net_config,
    pool,
    lines,
    evolved,
    duel_config,
    environment,
    num_envs: int,
    env_threads: int,
    seed: int,
    device,
    record_games: bool = False,
    opp_artifact: str | None = None,
):
    """Schedule + env + RolloutCollector of one collection process — the trainer itself or a shard worker.

    One construction site for both keeps the sharded and the in-process collections bit-compatible; the
    trainer-only extras (look-ahead forks, window branching, deck-pool evolution) activate through ``cfg``
    just as before. Shard workers must run with those extras off (asserted in :mod:`ygorl.train.shards`).
    """

    decks = [load_ydk(p) for p in cfg.decks]
    schedule = SelfPlaySchedule(DeckPool(decks, cfg.pairings), pool, duel_config,
                                selfplay_fraction=cfg.selfplay_fraction, seed=cfg.seed,
                                evolved=evolved, lines=lines, line_share=cfg.line_share,
                                line_first_share=cfg.line_first_share,
                                lookahead=cfg.lookahead_config() if cfg.lookahead_share > 0 else None,
                                lookahead_share=cfg.lookahead_share,
                                lookahead_second=cfg.lookahead_second, active_k=cfg.pool_active_k,
                                active_deals=cfg.pool_active_deals or max(1, num_envs))  # fmt: skip
    forks = cfg.lookahead_forks if cfg.lookahead_share > 0 else 0
    env = EncodedVecEnv(num_envs + forks, env_threads, cards=cards, vocab=vocab,
                        privileged=cfg.privileged_critic, event_length=cfg.event_length,
                        skip_forced=cfg.skip_forced, record_actions=True, record_games=record_games,
                        selection_history=net_config.selection_history, attack_history=net_config.attack_history,
                        cancel_budget=cfg.cancel_budget,
                        snapshots=cfg.branch_windows or cfg.lookahead_share > 0)  # fmt: skip
    collector = RolloutCollector(env, model, schedule, cfg.steps, opponents=schedule.opponent,
                                 seed=seed, min_batch=cfg.min_batch,
                                 device=device, stall_timeout=cfg.stall_timeout,
                                 turn_discount=cfg.turn_discount, complete_games=cfg.complete_games,
                                 branch=cfg.branch_config() if cfg.branch_windows else None,
                                 forks=forks, look_rounds=cfg.lookahead_rounds,
                                 opp_cpu_threads=cfg.opp_cpu_threads, opp_artifact=opp_artifact,
                                 score_opp_board=cfg.score_opp_board or cfg.ppo.opp_board_coef > 0)  # fmt: skip
    return schedule, env, collector


class Trainer:
    """Build with a :class:`TrainConfig` and a run directory, or :meth:`resume` from a checkpoint."""

    def __init__(self, cfg: TrainConfig, run_dir: str | Path, *, state: dict | None = None,
                 log: Callable[[str], None] | None = print) -> None:  # fmt: skip
        if not cfg.decks:
            raise ValueError("the deck pool is empty (TrainConfig.decks)")
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive run lock: two trainers writing one run directory corrupt metrics and checkpoints (the OS
        # releases the lock when the holder dies, so a crashed run never leaves a stale lock behind).
        lock_path = self.run_dir / ".lock"
        try:
            holder = int(lock_path.read_text().strip() or 0)
        except (OSError, ValueError):
            holder = 0
        if holder == os.getpid():  # same-process re-entry (save-then-resume in place): retire the old inode
            lock_path.unlink()
        self._lock_file = lock_path.open("a+")  # never truncates: a refused process must not clobber the holder
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError(f"another trainer is already running in {self.run_dir}") from None
        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(f"{os.getpid()}\n")
        self._lock_file.flush()
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
        self.device = torch_device(cfg.device)
        # An overlapped collection must not encode Metal work next to the learner (MPS has no streams): on MPS
        # the acting side then runs on CPU, leaving the GPU to the learner alone. Sampling stays on the host
        # either way (the collector's generator is a CPU generator), so only the forward's numerics move.
        # An overlapped collection must not encode Metal work next to the learner (MPS has no streams): on MPS
        # the acting side then runs on CPU, leaving the GPU to the learner alone. An explicit ``collect_device``
        # does the same split for synchronous training (same algorithm: the acting copy is refreshed every step).
        self._collect_device = (
            torch_device(cfg.collect_device)
            if cfg.collect_device is not None
            else torch.device("cpu")
            if cfg.overlap_collect and self.device.type == "mps"
            else self.device
        )
        init = load_actor(cfg.init_from, cfg.text_dir) if cfg.init_from and state is None else None
        cfg = self.cfg = self._event_length(cfg, init)
        if state is not None:
            self.vocab = vocab_from_text(state["vocab"])
        else:  # a warm start brings its own vocab: its ID embeddings are indexed by it
            self.vocab = init.vocab if init is not None else CardVocab.from_db(self.cards)
        self.vocab_text = vocab_to_text(self.vocab, self.run_dir / "vocab.json")
        text = TextFeatures.load(cfg.text_dir, self.vocab) if cfg.text_dir else None
        self.net_config = cfg.actor_config(len(self.vocab), text)
        self._text = text
        torch.manual_seed(derive_seed(cfg.seed, 0))
        self.model = self._new_model()
        if init is not None:
            added = warm_start(self.model.actor, init)
            added = f"; new input features start fresh: {added}" if added else ""
            critic = (
                self._init_separate_critic(cfg) if self.model.critic_trunk is not None else "the critic starts fresh"
            )
            self.log(f"initialized the actor from {cfg.init_from} ({critic}{added})")
        if cfg.freeze_id_embedding:
            emb = self.model.actor.identity.id_embedding
            if emb is None:
                raise ValueError("freeze_id_embedding needs an ID embedding (net.id_embedding = True)")
            emb.weight.requires_grad_(False)
        signature = Signature.of(
            self.vocab, cfg.event_length, self.net_config.selection_history, self.net_config.attack_history
        )
        prior = None
        if cfg.bc_prior:
            loaded = load_actor(cfg.bc_prior, cfg.text_dir)
            if why := signature.mismatches(loaded.signature, event_length=False):
                raise ValueError(f"{cfg.bc_prior}: the run and the prior differ: {'; '.join(why)}; the prior would "
                                 "read other cards from the same indices")  # fmt: skip
            prior = loaded.net.to(self.device)
        self.lines = self.line_labels = None
        if cfg.line_demos and cfg.line_share > 0:
            self.lines = LineLibrary.load_cached(cfg.line_demos, self.vocab, cards=self.cards, env=self.environment,
                                                 recovery=cfg.line_recovery, event_length=cfg.event_length,
                                                 cache=Path("out") / "linecache" / "lib.pkl")  # fmt: skip
            self.log(f"scripted lines: {len(self.lines)} from {len(cfg.line_demos)} file(s) in {self.lines.seconds:.0f} s"
                     + "".join(f"; left out {p}" for p in self.lines.problems[:10])
                     + (f"; ... {len(self.lines.problems)} left out" if len(self.lines.problems) > 10 else ""))  # fmt: skip
            if not len(self.lines):
                raise ValueError("line_demos has no solved plain line that replays (see the log)")
            if cfg.line_recovery:
                rp = self.lines.recovery_problems
                self.log(f"recovery lines: {self.lines.recovery_windows} windows on "
                         f"{sum(bool(s.recovery) for s in self.lines.scripts)} lines from {len(cfg.line_recovery)} "
                         f"file(s)" + "".join(f"; left out {p}" for p in rp[:5])
                         + (f"; ... {len(rp)} left out" if len(rp) > 5 else ""))  # fmt: skip
            if cfg.line_labels:
                self.line_labels = LineLabels.load(cfg.line_labels, self.lines, self.vocab, cards=self.cards,
                                                   env=self.environment, event_length=cfg.event_length, log=self.log)  # fmt: skip
                if cfg.line_labelled_only:
                    self.lines.dealable = sorted({key[0] for key in self.line_labels.table})
                    self.log(f"scripted lines dealt from the {len(self.lines.dealable)} labelled lines only")
        self.learner = PPOLearner(self.model, cfg.ppo, prior, line_labels=self.line_labels,
                                  branch_board_scale=cfg.branch_board_scale)  # fmt: skip
        self.pool = SnapshotPool(
            cfg.pool_size, cfg.pinned_share, cfg.pool_sampling, cfg.pfsp_power, device=self._collect_device
        )
        for path in cfg.pin_opponents:
            pinned = load_actor(path, cfg.text_dir)
            if why := signature.mismatches(pinned.signature):
                raise ValueError(f"{path}: the run and the pinned opponent differ: {'; '.join(why)}"
                                 + (" (the run's event length came from --init-from; set --event-length)"
                                    if self._adopted_event_length else ""))  # fmt: skip
            self.pool.pin(pinned.net.to(self._collect_device), tag=f"pinned:{Path(path).name}")
        self.evolved = None
        if cfg.deck_pool is not None:
            env = self.environment
            self.evolved = EvolvedDecks(cfg.deck_pool, share=cfg.evolved_share, power=cfg.evolved_power,
                                        validate=None if env is None else lambda d: [str(v) for v in env.validate_deck(d, self.cards)])  # fmt: skip
            if state is None:
                self._reload_deck_pool()
        if cfg.collect_shards > 1:
            for unsupported, flag in (
                ("deck_pool", cfg.deck_pool),
                ("lookahead_share", cfg.lookahead_share),
                ("branch_windows", cfg.branch_windows),
                ("overlap_collect", cfg.overlap_collect),
                ("text_dir", cfg.text_dir),
                ("trace_games", cfg.trace_games),
                ("complete_games", cfg.complete_games),
            ):
                if flag:
                    raise ValueError(f"collect_shards does not support {unsupported} (yet)")
            self.schedule = self.env = self.collector = None
            self.acting = self.model  # the shards hold their own acting copies; the parent only sends weights
            self.shards = ShardSupervisor(
                self.cfg.to_dict(), cfg.collect_shards, self.model, self.pool, self.run_dir, log=self.log
            )
            self.log(f"sharded collection: {cfg.collect_shards} x {max(1, cfg.num_envs // cfg.collect_shards)} envs")
        else:
            opp_artifact = None
            if cfg.opp_cpp:
                from ygorl.train.export_actor import export_actor

                class _Loaded:  # export_actor's interface: the run's actor + its event window
                    pass

                import copy as _copy

                _l = _Loaded()
                # the traced graph must be strictly CPU (a C++ forward on device tensors would encode Metal
                # from collection threads — forbidden; copy first, same values)
                _l.net = _copy.deepcopy(self.model.actor).cpu().eval()
                _l.event_length = cfg.event_length
                artifact = self.run_dir / "actor_actpath.pt"
                meta = export_actor(_l, artifact)
                self.log(f"opponent inference: traced actor graph {meta['fingerprint']} -> {artifact}")
                opp_artifact = str(artifact)
            self.schedule, self.env, self.collector = build_collection(
                cfg, model=None, cards=self.cards, vocab=self.vocab, net_config=self.net_config,
                pool=self.pool, lines=self.lines, evolved=self.evolved, duel_config=self.duel_config,
                environment=self.environment, num_envs=cfg.num_envs, env_threads=cfg.env_threads,
                seed=derive_seed(cfg.seed, 3), device=self._collect_device, record_games=cfg.trace_games,
                opp_artifact=opp_artifact)  # fmt: skip
            # the acting copy: the model itself, or a device copy refreshed before each (overlapped or split-device) collection
            self.acting = (
                copy.deepcopy(self.model).to(self._collect_device)
                if cfg.overlap_collect or self._collect_device != self.device
                else self.model
            )
            self.collector.model = self.acting
            self.shards = None
            if cfg.overlap_collect and self._collect_device != self.device:
                self.log(
                    "overlapped collection acts on CPU: Metal cannot encode two threads' GPU work concurrently"
                    " (the learner keeps the GPU; bf16 acting is off)"
                )
        self._pending: Rollout | None = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="collect") if cfg.overlap_collect else None
        self._stream = torch.cuda.Stream(self.device) if cfg.overlap_collect and self.device.type == "cuda" else None
        self._warmup_ev: list[float] = []
        self.counters = {"critic_warmup_done": 0, "updates": 0, "rows": 0, "decisions": 0, "games": 0, "seconds": 0.0, "truncated": 0,
                         "errors": 0, "snapshots": 0, "snapshots_skipped": 0}  # fmt: skip
        self.league = League(self.pool, self.evolved, min_win_rate=cfg.snapshot_min_win_rate,
                             min_games=cfg.snapshot_min_games)  # fmt: skip
        if state is not None:
            self._restore(state)
        self._acting_update = self.learner.updates
        (self.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    def _init_separate_critic(self, cfg: "TrainConfig") -> str:
        """Initialize a separate critic on a warm start per ``cfg.critic_init``; returns a log phrase."""
        if cfg.critic_init == "fresh":
            return "the critic starts fresh"
        if cfg.critic_init != "actor":
            raise ValueError(f"critic_init must be 'actor' or 'fresh', not {cfg.critic_init!r}")
        self.model.critic_trunk.load_state_dict(self.model.actor.state_dict())
        if checkpoint_format(cfg.init_from) != FORMAT:
            return "the critic trunk copies the actor; its heads start fresh"
        source = load_checkpoint(cfg.init_from)["learner"]["model"]
        heads = {k: v for k, v in source.items() if not k.startswith(("actor.", "critic_trunk."))}
        missing, unexpected = self.model.load_state_dict(heads, strict=False)
        fresh = sorted(k for k in missing if not k.startswith(("actor.", "critic_trunk.")))
        note = f"; critic parameters not in the checkpoint start fresh: {fresh}" if fresh else ""
        if unexpected:
            raise ValueError(f"{cfg.init_from}: unexpected critic parameters {unexpected}")
        return f"the critic trunk copies the actor and its heads load from the checkpoint{note}"

    def _event_length(self, cfg: TrainConfig, init) -> TrainConfig:
        """``cfg`` with its event window settled: a warm start keeps the window its network was trained with (like
        its vocab) unless ``event_length`` is set; a set one that differs is used, with a warning."""
        self._adopted_event_length = cfg.event_length is None and init is not None
        if cfg.event_length is None:
            n = init.event_length if init is not None else PPO_EVENT_LENGTH
            if init is not None:
                pins = (
                    " (pinned opponents must read the same; set --event-length to choose)" if cfg.pin_opponents else ""
                )
                self.log(f"event length {n}, as {cfg.init_from} was trained with{pins}")
            return replace(cfg, event_length=n)
        if init is not None and init.event_length != cfg.event_length:
            self.log(f"warning: {cfg.init_from} was trained with {init.event_length} event tokens per observation; "
                     f"this run reads {cfg.event_length}")  # fmt: skip
        return cfg

    def _new_model(self) -> ActorCritic:
        return use_split_k(build_actor_critic(self.net_config, self._text, self.cfg.critic)).to(self.device)

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
        rng = {
            "torch": torch.get_rng_state(),
            "collector": None if self.collector is None else self.collector.generator.get_state(),
        }
        if self.device.type == "cuda":
            # PPO shuffles minibatches on the learner device, independently of the CPU acting generator.
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {"config": self.cfg.to_dict(), "net_config": self.net_config.to_dict(), "vocab": self.vocab_text,
                "environment": self.environment.stamp() if self.environment is not None else None,
                "learner": self.learner.state_dict(), **self.league.state_dict(),
                "schedule": self.schedule.state_dict(), "counters": dict(self.counters), "rng": rng}  # fmt: skip

    def _restore(self, state: dict) -> None:
        saved_net = {
            "selection_history": False,
            "attack_history": False,
            "battle_context": False,
            **state["net_config"],
        }
        if saved_net != self.net_config.to_dict():
            raise ValueError("the checkpoint's network does not match the configuration")
        self.learner.load_state_dict(state["learner"])
        if self.learner.new_heads:
            self.log(f"critic heads not in the checkpoint start fresh: {self.learner.new_heads}")
        self.league.load_state_dict(state, self._new_model)
        self.schedule.load_state_dict(state["schedule"])
        self.counters.update(state["counters"])
        torch.set_rng_state(state["rng"]["torch"])
        if self.collector is not None:
            self.collector.generator.set_state(state["rng"]["collector"])
        if self.device.type == "cuda" and "cuda" in state["rng"]:
            torch.cuda.set_rng_state(state["rng"]["cuda"], self.device)

    def save(self, name: str = "latest.pt") -> Path:
        return save_checkpoint(self.state_dict(), self.run_dir / "checkpoints" / name)

    # -- training ---------------------------------------------------------------------------------------
    def train(self, max_updates: int | None = None, max_minutes: float | None = None) -> dict:
        """Run until ``max_updates`` more updates or ``max_minutes`` of wall time; returns the last metrics."""
        cfg = self.cfg
        start, done, last = time.time(), 0, {}
        wall_start = time.monotonic()
        wall_before = self.counters.get("wall_seconds", 0.0)
        try:
            while (max_updates is None or done < max_updates) and (
                max_minutes is None or time.time() - start < max_minutes * 60
            ):
                last = self.step()
                done += 1
                u = self.counters["updates"]
                if self.evolved is not None and cfg.deck_pool_every and u % cfg.deck_pool_every == 0:
                    self._reload_deck_pool()
                if cfg.snapshot_every and u % cfg.snapshot_every == 0:
                    self.maybe_snapshot(u)
                if cfg.register_every:
                    self.counters["wall_seconds"] = wall_before + time.monotonic() - wall_start
                    if u % cfg.register_every == 0:
                        from ygorl.train.registration import register_checkpoint

                        register_checkpoint(self)
                if cfg.eval_every and u % cfg.eval_every == 0:
                    self.evaluate()
                elif cfg.checkpoint_every and u % cfg.checkpoint_every == 0:
                    self.save()
        except KeyboardInterrupt:
            self.log("interrupted: saving checkpoints/latest.pt")
        except RolloutStalled as exc:
            # save while the process still can; the stuck worker thread keeps the env pool from being destroyed,
            # so the caller must end the process (os._exit) rather than unwind normally
            self.log(f"rollout stalled, saving checkpoints/latest.pt: {exc}")
            self.save()
            raise
        finally:
            if self.shards is not None:
                self.shards.close()
        if cfg.register_every:
            self.counters["wall_seconds"] = wall_before + time.monotonic() - wall_start
        self.save()
        return last

    def _reload_deck_pool(self) -> None:
        ev = self.evolved
        if ev.reload():
            live, hist = ev.ids("probation", "active"), ev.ids("history")
            self.log(f"deck pool {ev.path}: {len(live)} evolved decks dealt, {len(hist)} history"
                     + "".join(f"; left out {p}" for p in ev.problems))  # fmt: skip

    def _autocast(self, *, learner: bool = False):
        enabled = self.cfg.bf16
        if learner and self.cfg.learner_precision != "inherit":
            enabled = self.cfg.learner_precision == "bf16"
        if not learner and enabled and self._collect_device.type == "cpu" and self.device.type != "cpu":
            enabled = False  # CPU-acting fallback of overlapped collection: bf16 CPU inference is slower
        return torch.autocast(
            self.device.type if learner else self._collect_device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
            cache_enabled=not learner,
        )

    def _collect(self) -> Rollout:
        if self.shards is not None:
            return self.shards.collect()
        model_update = self._acting_update if self.cfg.overlap_collect else self.learner.updates
        self.env.trace_model_update = model_update
        if self._stream is None:
            with self._autocast():
                ro = self.collector.collect()
        else:
            with torch.cuda.stream(self._stream), self._autocast():
                ro = self.collector.collect()
            self._stream.synchronize()
        if self.cfg.trace_games:
            self._log_game_traces(ro.games, model_update)
        return ro

    @property
    def best(self) -> dict:
        """Keep-best's score and update (``League.best``)."""
        return self.league.best

    def maybe_snapshot(self, update: int) -> bool:
        """Add a snapshot of the model to the pool, unless the admission threshold holds it back (League.admit)."""
        joined = self.league.admit(self.model, update)
        self.counters["snapshots" if joined else "snapshots_skipped"] += 1
        return joined

    def step(self) -> dict:
        cfg = self.cfg
        t_step = time.perf_counter()
        torch.set_num_threads(cfg.collect_threads)
        future = None
        if cfg.overlap_collect:
            ro, self._pending = self._pending or self._collect(), None
            self.acting.load_state_dict(self.model.state_dict())
            self._acting_update = self.learner.updates
            if self._stream is not None:
                self._stream.wait_stream(torch.cuda.current_stream(self.device))
            future = self._executor.submit(self._collect)
        else:
            ro = self._collect()
            if self.acting is not self.model:  # split-device acting: the copy tracks the post-update weights
                self.acting.load_state_dict(self.model.state_dict())
                self._acting_update = self.learner.updates
        torch.set_num_threads(cfg.torch_threads)
        t0 = time.perf_counter()
        # Adam mutates weights between minibatches: cached BF16 casts would become stale within this context.
        with self._autocast(learner=True):
            warm = self.cfg.critic_warmup > 0 and not self.counters.get("critic_warmup_done")
            stats = self.learner.update(ro, policy=not warm)
        update_s = time.perf_counter() - t0
        if self.device.type == "mps":
            # The MPS allocator deadlocks after ~100-600 s of sustained kernel encoding (pytorch/pytorch#201378);
            # the confirmed workaround is periodic cache clearing + synchronize, done once per update here.
            torch.mps.empty_cache()
            torch.mps.synchronize()
        if warm:
            self._warmup_ev.append(stats["q_explained_var"])
            recent = self._warmup_ev[-5:]
            if self.learner.updates >= self.cfg.critic_warmup or (
                    len(recent) == 5 and sum(recent) / 5 >= self.cfg.critic_warmup_ev):  # fmt: skip
                self.counters["critic_warmup_done"] = self.learner.updates
                self.log(f"critic warm-up done after {self.learner.updates} updates (Q explained variance "
                         f"{sum(recent) / len(recent):.3f}): the policy trains from now on")  # fmt: skip
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
        self._log_errors(ro.games)
        self._log_truncations(ro.games)
        if cfg.log_games:
            self._log_games(ro.games)
        evolved_games = self.league.record(ro.games)
        record = {"update": c["updates"], "time": round(time.time(), 1), **self._rollout_stats(ro, update_s),
                  "step_s": round(step_s, 2), "rows_per_s": ro.players.numel() / max(step_s, 1e-9),
                  **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
                  "total": dict(c), "pool": len(self.pool),
                  **({"evolved_games": evolved_games} if self.evolved is not None else {}),
                  **(line_stats(ro.games, bool(self.cfg.line_recovery)) if self.lines is not None else {})}  # fmt: skip
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.log(f"update {c['updates']}: {record['decisions_per_s']:.0f} decisions/s collect, "
                 f"{record['rows_per_s']:.0f} rows/s overall, games {len(ro.games)} "
                 f"(truncated {record['truncated_games']}), entropy {stats['entropy']:.3f}, kl_ref {stats['kl_ref']:.4f}, "
                 f"q_loss {stats['q_loss']:.4f}, pool win {record['pool_win_rate']}"
                 + (f", lines {record['line_completed']}/{record['line_games']} completed, {record['line_broken']} "
                    f"broken (mean step {record['line_broken_step']})" if self.lines is not None else "")
                 + (f", {record['line_recovered']} recovered ({record['line_recovered_completed']} completed)"
                    if "line_recovered" in record else "")
                 + (f", first-seat {record['line_first_completed']}/{record['line_first_games']} completed, "
                    f"{record['line_first_broken']} broken (mean step {record['line_first_broken_step']}, "
                    f"{record['line_first_skipped']} skipped leading steps)"
                    if self.lines is not None and self.cfg.line_first_share > 0 else "")
                 + (f", labelled windows {record['line_labelled_rows']} ({record['line_fire_rows']} fired, mean bonus "
                    f"{record['line_bonus_mean']:.3f})" if "line_labelled_rows" in record else "")
                 + (f", line CE rows {record['line_ce_rows']} ({record['line_ce_fire_targets']} fire targets, loss "
                    f"{record.get('line_ce_loss', 0.0):.4f})" if "line_ce_rows" in record else "")
                 + (f", branched windows {record['branch_windows']} ({record['branch_decisions']} branch decisions"
                    + (f", no-recovery share {record['branch_no_recovery_share']:.2f}"
                       if record.get("branch_no_recovery_share") is not None else "")
                    + (f", {record['branch_rows']} rows used, mean adv {record['branch_adv_mean']:.3f}, fire share "
                       f"{record['branch_fire_share']:.2f}, tie share {record['branch_tie_share']:.2f}, hold share "
                       f"{record['branch_hold_share']:.2f}" if "branch_rows" in record else "")
                    + (f", CE rows {record['branch_ce_rows']} (loss {record['branch_ce_loss']:.4f})"
                       if "branch_ce_loss" in record else "") + ")"
                    if "branch_windows" in record else "")
                 + (f", look-ahead decisions {record['lookahead_decisions']} ({record['lookahead_branched']} rolled out, "
                    f"{record['lookahead_changed']} changed, {record['lookahead_steps']} rollout decisions)"
                    if "lookahead_decisions" in record else "")
                 + (f", SKIPPED {stats['nonfinite_minibatches']} non-finite-loss / {stats['nonfinite_grad_minibatches']} "
                    "non-finite-gradient minibatches" if stats.get("nonfinite_minibatches")
                    or stats.get("nonfinite_grad_minibatches") else ""))  # fmt: skip
        return record

    def _diagnostic_record(self, game) -> dict:
        """Actual decks/rules and cross-update actions are needed to replay a changing training policy."""
        a, result = game.assignment, game.result
        return {"format": "ygorl.training-diagnostic.v1", "update": self.learner.updates,
                "seed": a.spec.seed, "first": a.spec.first, "info": dict(a.info), "spec": asdict(a.spec),
                "environment": self.environment.stamp() if self.environment is not None else None,
                "skip_forced": self.cfg.skip_forced, "opponent": a.opponent, "learner_seat": a.learner_seat,
                "reason": game.reason, "truncated": game.truncated, "winner": game.winner,
                "turns": result.get("turns"), "decisions": result.get("decisions"),
                "error": str(result.get("error", "")), "script_errors": result.get("script_errors"),
                "retries": result.get("retries"), "unknown_messages": result.get("unknown_messages"),
                "undecodable_messages": result.get("undecodable_messages", 0),
                "action_indices": result.get("action_indices"),
                "responses": [bytes(r).hex() for r in result.get("responses", [])]}  # fmt: skip

    def _log_game_traces(self, games, model_update: int) -> None:
        """Record collection evidence before optimization, also for successful games."""
        if not games:
            return
        with gzip.open(self.run_dir / "game-traces.jsonl.gz", "at") as f:
            for game in games:
                record = self._diagnostic_record(game)
                record.pop("update")  # finishing collection does not identify every action's policy
                record.update(
                    format="ygorl.training-game-trace.v1",
                    collection_model_update=model_update,
                    action_model_updates=game.result["action_model_updates"],
                    encoding={
                        "event_length": self.cfg.event_length,
                        "selection_history": self.net_config.selection_history,
                        "attack_history": self.net_config.attack_history,
                        "cancel_budget": self.cfg.cancel_budget,
                    },
                )
                f.write(json.dumps(record) + "\n")

    def _log_errors(self, games) -> None:
        """Preserve unhealthy games even when the engine subsequently reported a winner."""
        errors = [
            g
            for g in games
            if g.reason == "error"
            or any(
                g.result.get(k)
                for k in ("error", "retries", "unknown_messages", "script_errors", "undecodable_messages")
            )
        ]
        if not errors:
            return
        with (self.run_dir / "errors.jsonl").open("a") as f:
            for g in errors:
                f.write(json.dumps(self._diagnostic_record(g)) + "\n")

    def _log_truncations(self, games) -> None:
        limits = [g for g in games if g.truncated and g.reason != "error"]
        if not limits:
            return
        with (self.run_dir / "truncations.jsonl").open("a") as f:
            for g in limits:
                f.write(json.dumps(self._diagnostic_record(g)) + "\n")

    def _log_games(self, games) -> None:
        """Append one line per finished game to ``games.jsonl.gz`` (a gzip member per update)::

            {"update": 12, "decks": ["branded", "evo-0003"], "evolved": 1, "first": 0, "winner": 1, "turns": 7,
             "reason": "win", "truncated": false, "opponent": null, "learner": null, "seed": 123,
             "environment": "md-2026-09", "fingerprint": "..."}

        ``decks`` are deck a and b: a corpus deck's name is its .ydk stem (the path is in config.json ``decks``), an
        evolved deck's is its manifest id (``evolved``: which of the two it is; the manifest is config.json
        ``deck_pool``; a manifest's history decks, dealt as opponents only, also carry their ids). ``first`` and
        ``winner`` are deck indices (0 = a, 1 = b; ``winner`` as the engine reported it, null = none; ``truncated``:
        cut by a limit or an error, not a result for training). ``update`` is the update the games' rollout fed: the
        policy that played them is the one after ``update - 1`` updates (``update - 2`` with ``overlap_collect``).
        ``opponent`` is null in self-play, else the pool snapshot id, and ``learner`` the learner's deck index."""
        stamp = self.environment.stamp() if self.environment is not None else {}
        env = {"environment": stamp.get("environment"), "fingerprint": stamp.get("fingerprint")}
        lines = []
        for g in games:
            a, spec = g.assignment, g.assignment.spec
            info = a.info
            evolved = info.get("evolved")
            lines.append(json.dumps({
                "update": self.learner.updates, "decks": [info.get("deck_a"), info.get("deck_b")],
                "evolved": None if evolved is None else spec.deck_of_seat(info["evolved_seat"]),
                "first": spec.first, "winner": None if g.winner is None else spec.deck_of_seat(g.winner),
                "turns": int(g.result.get("turns", 0)), "reason": g.reason, "truncated": g.truncated,
                "opponent": a.opponent, "learner": None if a.opponent is None else spec.deck_of_seat(a.learner_seat),
                "seed": getattr(spec, "seed", None),
                "retries": g.result.get("retries"), "unknown_messages": g.result.get("unknown_messages"),
                "script_errors": g.result.get("script_errors"), "error": g.result.get("error"), **env}))  # fmt: skip
        if lines:
            with gzip.open(self.run_dir / "games.jsonl.gz", "at", compresslevel=6) as f:
                f.write("\n".join(lines) + "\n")

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
            **({"discarded_rows": ro.discarded_rows} if ro.complete_games else {}),
            **({"branch_windows": ro.branch_windows, "branch_decisions": ro.branch_decisions,
                "branch_no_recovery_share": round(ro.branch_no_recovery / ro.branch_windows, 3)
                if ro.branch_windows else None} if ro.branch_value is not None else {}),
            **({"lookahead_decisions": ro.lookahead_decisions, "lookahead_branched": ro.lookahead_branched,
                "lookahead_changed": ro.lookahead_changed, "lookahead_steps": ro.lookahead_steps}
               if ro.lookahead_decisions else {}),
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
        # Spawned eval workers initializing Metal next to a live trainer deadlocked the run twice on MPS
        # (2026-10-29, both arms at their first eval): single-process evaluation there.
        workers = 1 if self.device.type == "mps" else cfg.eval_workers
        for opponent in cfg.eval_opponents:
            rep, seconds = evaluate_checkpoint(path, self.decks, opponent, pairings=self.eval_pairings(),
                                               pairs=cfg.eval_pairs, config=self.duel_config, env=self.environment,
                                               workers=workers, seed=derive_seed(cfg.seed, 4),
                                               greedy_policy=cfg.eval_greedy_policy)  # fmt: skip
            results[opponent] = {"update": u, "vs": opponent, "games": rep.games, "attempted_games": rep.attempted_games,
                                 "valid": rep.errors == 0 and rep.games > 0, "win_rate": rep.win_rate,
                                 "ci": list(rep.ci), "wins": rep.wins, "losses": rep.losses, "draws": rep.draws,
                                 "reasons": rep.reasons, "errors": rep.errors, "mean_turns": rep.mean_turns,
                                 "retries": rep.retries, "script_errors": rep.script_errors,
                                 "unknown_messages": rep.unknown_messages, "undecodable_messages": rep.undecodable_messages,
                                 "seconds": round(seconds, 1)}  # fmt: skip
            if not results[opponent]["valid"]:
                with (self.run_dir / "eval-errors.jsonl").open("a") as f:
                    f.write(json.dumps({"update": u, "vs": opponent, "report": rep.to_dict()}) + "\n")
        valid = all(r["valid"] for r in results.values())
        score = results[cfg.keep_best_by]["win_rate"] if cfg.keep_best_by and valid else None
        is_best = self.league.offer_best(self.model, u, score)
        if is_best:
            path = self.save(f"update_{u:06d}.pt")  # re-save with the new best recorded
            shutil.copyfile(path, self.run_dir / "best.pt")
            shutil.copyfile(path, self.run_dir / "checkpoints" / "latest.pt")
        with (self.run_dir / "eval.jsonl").open("a") as f:
            for r in results.values():
                f.write(json.dumps({**r, "best": is_best and r["vs"] == cfg.keep_best_by}) + "\n")
        summary = ", ".join(f"vs {k} {v['win_rate']:.3f} ({v['games']} games)" for k, v in results.items())
        status = " -> new best" if is_best else " (invalid evaluation; best unchanged)" if not valid else ""
        self.log(f"eval at update {u}: {summary}{status}")
        return results


def evaluate_checkpoint(checkpoint: str | Path, decks, opponent: str, *, pairings, pairs: int, config: DuelConfig,
                        env=None, workers: int = 1, seed: int = 0, greedy_policy: bool = False):  # fmt: skip
    """Arena report of ``policy:<checkpoint>`` (on deck i) against ``opponent`` (on deck j) over the given
    ``(i, j)`` pairings, ``pairs`` paired seeds each; returns ``(merged report, seconds)``."""
    from ygorl.agents.registry import AgentSpec, policy_spec
    from ygorl.eval.arena import Arena, merge

    agent = AgentSpec(policy_spec(Path(checkpoint).resolve(), greedy=greedy_policy))
    arena = Arena(agent, AgentSpec(opponent), env=env, config=config,
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


__all__ = ["SMALL_NET", "TrainConfig", "Trainer", "evaluate_checkpoint", "line_stats", "summarize_metrics"]
