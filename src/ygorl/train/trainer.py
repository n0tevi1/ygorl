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
import gzip
import json
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
from ygorl.train.rollout import Rollout, RolloutCollector, RolloutStalled
from ygorl.train.league import League
from ygorl.train.line_labels import LineLabels
from ygorl.train.lines import LineLibrary
from ygorl.train.selfplay import DeckPool, EvolvedDecks, SelfPlaySchedule, SnapshotPool

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
    # graded hand-trap labels of those lines (train.line_labels; the window cache is written next to the file): the
    # learner's labelled chain windows get ppo.line_bonus_coef * bonus on their advantage
    line_labels: str | None = None
    # deal only the lines that carry a label (line_labels): the labelled rows are sparse otherwise (a labelled line and
    # the learner holding the hand trap are both needed); scripted games then never use unlabelled lines
    line_labelled_only: bool = False
    snapshot_every: int = 10  # updates
    pool_sampling: str = (
        "pfsp"  # "pfsp" (design I1) or "uniform" (SnapshotPool): opponents the learner loses to come up more
    )
    pfsp_power: float = 2.0
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
        if self.line_labels and not (self.line_demos and self.line_share > 0):
            raise ValueError("line_labels needs line_demos and a line_share above 0 (the labels are on those lines)")
        if self.line_labelled_only and not self.line_labels:
            raise ValueError("line_labelled_only needs line_labels")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ppo"] = self.ppo.to_dict()
        d["decks"], d["eval_opponents"] = list(self.decks), list(self.eval_opponents)
        d["pin_opponents"] = list(self.pin_opponents)
        d["line_demos"] = list(self.line_demos)
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
        for key in ("decks", "eval_opponents", "pin_opponents", "line_demos"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def _mean(xs) -> float | None:
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def line_stats(games) -> dict:
    """Scripted-line counts of a rollout's finished games: games, lines completed / broken (and the mean step the
    broken ones broke at), lines still playing when the game ended (``line_unfinished``)."""
    lines = [g.line for g in games if g.line is not None]
    broken = [ln["broken_at"] for ln in lines if ln["status"] == "broken"]
    return {"line_games": len(lines), "line_completed": sum(ln["status"] == "completed" for ln in lines),
            "line_broken": len(broken), "line_broken_step": None if not broken else round(_mean(broken), 2),
            "line_unfinished": sum(ln["status"] == "playing" for ln in lines)}  # fmt: skip


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
        self.device = torch_device(cfg.device)
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
            self.lines = LineLibrary.load(cfg.line_demos, self.vocab, cards=self.cards, env=self.environment)
            self.log(f"scripted lines: {len(self.lines)} from {len(cfg.line_demos)} file(s) in {self.lines.seconds:.0f} s"
                     + "".join(f"; left out {p}" for p in self.lines.problems[:10])
                     + (f"; ... {len(self.lines.problems)} left out" if len(self.lines.problems) > 10 else ""))  # fmt: skip
            if not len(self.lines):
                raise ValueError("line_demos has no solved plain line that replays (see the log)")
            if cfg.line_labels:
                self.line_labels = LineLabels.load(cfg.line_labels, self.lines, self.vocab, cards=self.cards,
                                                   env=self.environment, event_length=cfg.event_length, log=self.log)  # fmt: skip
                if cfg.line_labelled_only:
                    self.lines.dealable = sorted({key[0] for key in self.line_labels.table})
                    self.log(f"scripted lines dealt from the {len(self.lines.dealable)} labelled lines only")
        self.learner = PPOLearner(self.model, cfg.ppo, prior, line_labels=self.line_labels)
        self.pool = SnapshotPool(cfg.pool_size, cfg.pinned_share, cfg.pool_sampling, cfg.pfsp_power)
        for path in cfg.pin_opponents:
            pinned = load_actor(path, cfg.text_dir)
            if why := signature.mismatches(pinned.signature):
                raise ValueError(f"{path}: the run and the pinned opponent differ: {'; '.join(why)}"
                                 + (" (the run's event length came from --init-from; set --event-length)"
                                    if self._adopted_event_length else ""))  # fmt: skip
            self.pool.pin(pinned.net.to(self.device), tag=f"pinned:{Path(path).name}")
        self.evolved = None
        if cfg.deck_pool is not None:
            env = self.environment
            self.evolved = EvolvedDecks(cfg.deck_pool, share=cfg.evolved_share, power=cfg.evolved_power,
                                        validate=None if env is None else lambda d: [str(v) for v in env.validate_deck(d, self.cards)])  # fmt: skip
            if state is None:
                self._reload_deck_pool()
        self.schedule = SelfPlaySchedule(DeckPool(self.decks, cfg.pairings), self.pool, self.duel_config,
                                         selfplay_fraction=cfg.selfplay_fraction, seed=cfg.seed,
                                         evolved=self.evolved, lines=self.lines, line_share=cfg.line_share)  # fmt: skip
        self.env = EncodedVecEnv(cfg.num_envs, cfg.env_threads, cards=self.cards, vocab=self.vocab,
                                 privileged=cfg.privileged_critic, event_length=cfg.event_length,
                                 skip_forced=cfg.skip_forced, record_actions=True, record_games=cfg.trace_games,
                                 selection_history=self.net_config.selection_history, attack_history=self.net_config.attack_history, cancel_budget=cfg.cancel_budget)  # fmt: skip
        # the acting network: the model itself, or a copy refreshed before each overlapped collection
        self.acting = copy.deepcopy(self.model) if cfg.overlap_collect else self.model
        self._pending: Rollout | None = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="collect") if cfg.overlap_collect else None
        self._stream = torch.cuda.Stream(self.device) if cfg.overlap_collect and self.device.type == "cuda" else None
        self.collector = RolloutCollector(self.env, self.acting, self.schedule, cfg.steps, opponents=self.schedule.opponent,
                                          seed=derive_seed(cfg.seed, 3), min_batch=cfg.min_batch,
                                          device=self.device, stall_timeout=cfg.stall_timeout,
                                          turn_discount=cfg.turn_discount, complete_games=cfg.complete_games)  # fmt: skip
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
        rng = {"torch": torch.get_rng_state(), "collector": self.collector.generator.get_state()}
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
        self.league.load_state_dict(state, self._new_model)
        self.schedule.load_state_dict(state["schedule"])
        self.counters.update(state["counters"])
        torch.set_rng_state(state["rng"]["torch"])
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
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=not learner)

    def _collect(self) -> Rollout:
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
        torch.set_num_threads(cfg.torch_threads)
        t0 = time.perf_counter()
        # Adam mutates weights between minibatches: cached BF16 casts would become stale within this context.
        with self._autocast(learner=True):
            warm = self.cfg.critic_warmup > 0 and not self.counters.get("critic_warmup_done")
            stats = self.learner.update(ro, policy=not warm)
        update_s = time.perf_counter() - t0
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
                  **(line_stats(ro.games) if self.lines is not None else {})}  # fmt: skip
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.log(f"update {c['updates']}: {record['decisions_per_s']:.0f} decisions/s collect, "
                 f"{record['rows_per_s']:.0f} rows/s overall, games {len(ro.games)} "
                 f"(truncated {record['truncated_games']}), entropy {stats['entropy']:.3f}, kl_ref {stats['kl_ref']:.4f}, "
                 f"q_loss {stats['q_loss']:.4f}, pool win {record['pool_win_rate']}"
                 + (f", lines {record['line_completed']}/{record['line_games']} completed, {record['line_broken']} "
                    f"broken (mean step {record['line_broken_step']})" if self.lines is not None else "")
                 + (f", labelled windows {record['line_labelled_rows']} ({record['line_fire_rows']} fired, mean bonus "
                    f"{record['line_bonus_mean']:.3f})" if "line_labelled_rows" in record else ""))  # fmt: skip
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
