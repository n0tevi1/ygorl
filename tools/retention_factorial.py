"""Bounded 2x2 native-deal coverage x frozen replay pilot. No automatic resume/promotion.

prepare freezes inputs and the evaluation deals; run executes each training arm
in a fresh process, then evaluates only the shared initial and fixed endpoints.
"""

import argparse
from dataclasses import asdict
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

from behavior_replay import result_fields, sha
from ygorl import _core, paths
from ygorl.agents.registry import agent_factory
from ygorl.data import load_environment
from ygorl.engine.duel import Duel, DuelConfig, ScriptedAgent, default_cards
from ygorl.engine.outcome import HOST_LIMITS
from ygorl.engine.replay import Replay
from ygorl.env.encoding import ACTION_KINDS
from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings
from ygorl.eval.arena import derive_seed
from ygorl.solver.resume import tree_digest
from ygorl.train.checkpoint import load_checkpoint
from ygorl.train.registration import atomic_json
from ygorl.train.retention import FrozenReplay, MechanismDeckPool, prompt_groups
from ygorl.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]
ARMS = ("uniform", "uniform_replay", "mechanism", "mechanism_replay")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def runtime():
    env = load_environment("md-2026-09")
    return dict(
        native=str(_core.__file__),
        native_sha256=sha(_core.__file__),
        cards=sha(paths.cards_cdb()),
        scripts=tree_digest(paths.card_scripts(), "*.lua"),
        environment=env.stamp(),
        torch=str(torch.__version__),
    )


def panel(reg):
    env = load_environment("md-2026-09")
    decks = [d.deck for d in sorted(env.meta_decks, key=lambda x: x.deck.name)]
    cfg = DuelConfig.from_environment(env)
    pairs = sample_pairings(len(decks), reg["clusters"], reg["panel_seed"])
    return env, cfg, pairing_slots(decks, pairs, reg["panel_seed"], cfg)


def panel_record(reg):
    _, cfg, slots = panel(reg)
    return json.loads(
        json.dumps(
            dict(
                config=asdict(cfg),
                clusters=[dict(seed=s, decks=[asdict(d) for d in ds], agent_seeds=seeds) for s, ds, seeds in slots],
            )
        )
    )


def prepare(root, source, opponent, smoke):
    require(not root.exists(), "output already exists; never overwrite research evidence")
    source, opponent = source.resolve(), opponent.resolve()
    state = load_checkpoint(source)
    env = load_environment("md-2026-09")
    require(state["environment"] == env.stamp(), "source environment mismatch")
    require(load_checkpoint(opponent)["environment"] == env.stamp(), "opponent environment mismatch")
    decks = sorted(env.meta_decks, key=lambda x: x.deck.name)
    require(len(decks) == 20 and all(not env.validate_deck(d.deck, default_cards()) for d in decks), "invalid decks")
    reg = dict(
        version=1,
        smoke=smoke,
        source=str(source),
        source_sha256=sha(source),
        opponent=str(opponent),
        opponent_sha256=sha(opponent),
        training_seeds=[2026101901] if smoke else [2026101911, 2026101912, 2026101913],
        bank_seed=2026101902 if smoke else 2026101914,
        panel_seed=2026101903 if smoke else 2026101915,
        bootstrap_seed=2026101916,
        clusters=1 if smoke else 32,
        updates=1 if smoke else 16,
        steps=128,
        num_envs=4 if smoke else 8,
        bank_batches=1 if smoke else 8,
        bank_per_group=16 if smoke else 64,
        retention_coef=0.5,
        retention_batch=32 if smoke else 128,
        mechanism_share=0.5,
        max_hours=0.5 if smoke else 4,
        max_output_gib=4,
        reserve_gib=2,
        workers=2,
        auto_promote=False,
        primary="coverage, replay and interaction contrasts of final policy whole-game scores",
        uncertainty="paired deal-cluster and training-seed bootstrap; 3 seeds and 32 clusters are a pilot, not definitive power",
        censoring="host limits contribute [0,1] outcome bounds, never LP-adjudicated wins",
        restart="fresh matched actor/critic, optimizer, reference and frozen source pool; no exact historical resume",
        data_scope="native full games on existing 20-deck environment; fresh deals, not unseen-card transfer",
        coverage="card-type opportunity groups, not semantic mechanism mastery; report actual prompt/phase/offered actions",
        replay="fixed teacher probabilities on independent native states; equal prompt/phase groups; no PPO/critic targets",
        selection="fixed final update only; no tuning or checkpoint selection on locked evaluation",
        budgets="equal fresh rollout rows, not equal FLOPs; report accepted optimizer steps and replay cost",
        artifacts="finished training traces, metrics and final checkpoints; no exact mid-arm resume; preserve interrupted evidence",
    )
    root.mkdir(parents=True)
    atomic_json(root / "registration.json", reg)
    atomic_json(root / "panel.json", panel_record(reg))
    pool = MechanismDeckPool([d.deck for d in decks], default_cards(), share=reg["mechanism_share"])
    atomic_json(
        root / "sampling.json",
        dict(
            decks=[d.deck.name for d in decks],
            pairs=pool.pairs,
            probabilities=pool.probabilities.tolist(),
            groups=pool.groups,
        ),
    )
    files = {str(p): sha(p) for base in ("src/ygorl", "tools") for p in sorted((REPO / base).rglob("*.py"))}
    files.update({str(d.path): sha(d.path) for d in decks})
    files.update(
        {
            str(p): sha(p)
            for p in (source, opponent, root / "registration.json", root / "panel.json", root / "sampling.json")
        }
    )
    identity = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        runtime=runtime(),
        files=files,
    )
    atomic_json(root / "identity.json", identity)
    print(json.dumps(dict(root=str(root), identity=sha(root / "identity.json"), registration=reg), indent=2))


def verify(root):
    require(not (root / "STOP.json").exists(), "study STOP exists")
    ident = read(root / "identity.json")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == ident["commit"],
        "commit drift",
    )
    for filename, digest in ident["files"].items():
        require(sha(filename) == digest, f"identity drift: {filename}")
    require(runtime() == ident["runtime"], "native/data/environment drift")
    require(panel_record(read(root / "registration.json")) == read(root / "panel.json"), "evaluation panel drift")


def guard(root, reg):
    require(not (root / "STOP.json").exists(), "study STOP exists")
    require(shutil.disk_usage(root).free >= reg["reserve_gib"] * 1024**3, "disk reserve exhausted")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    require(size < reg["max_output_gib"] * 1024**3, "output budget exhausted")
    if (root / "run.json").exists():
        require(time.time() < read(root / "run.json")["deadline_unix"], "wall budget exhausted")


def healthy(result):
    return result["reason"] in ("win", "end", *HOST_LIMITS) and not any(
        result.get(k) for k in ("error", "script_errors", "retries", "unknown_messages", "undecodable_messages")
    )


def trainer(root, name, seed, reg, *, mechanism=False, replay=False):
    path = root / name
    require(not path.exists(), "arm already exists; no silent retry")
    state = load_checkpoint(reg["source"])
    cfg = dict(state["config"])
    cfg.update(
        decks=[str(d.path) for d in sorted(load_environment("md-2026-09").meta_decks, key=lambda x: x.deck.name)],
        init_from=reg["source"],
        env="md-2026-09",
        seed=seed,
        device="cpu",
        num_envs=reg["num_envs"],
        env_threads=2,
        min_batch=reg["num_envs"],
        steps=reg["steps"],
        torch_threads=2,
        collect_threads=2,
        trace_games=True,
        log_games=True,
        bf16=False,
        learner_precision="fp32",
        overlap_collect=False,
        checkpoint_every=0,
        eval_every=0,
        snapshot_every=0,
        register_every=0,
        critic_warmup=0,
        deck_pool=None,
        pin_opponents=[],
        complete_games=False,
        cancel_budget=32,
        stall_timeout=120,
    )
    cfg["ppo"] = {
        **cfg["ppo"],
        "retention_coef": reg["retention_coef"] if replay else 0,
        "retention_batch": reg["retention_batch"],
    }
    t = Trainer(TrainConfig.from_dict(cfg), path)
    t.model.load_state_dict(state["learner"]["model"], strict=True)
    t.learner.reference.load_state_dict(state["learner"]["model"], strict=True)
    require(
        all(torch.equal(v, state["learner"]["model"][k]) for k, v in t.model.state_dict().items()),
        "initial actor/critic restoration mismatch",
    )
    print("restored complete actor AND critic; matched fresh optimizer/reference/pool", flush=True)
    t.pool.add(t.model, 0, tag="fixed-initial")
    require(not t.learner.optimizer.state, "optimizer must start empty")
    if mechanism:
        t.schedule.decks = MechanismDeckPool(t.decks, t.cards, cfg["pairings"], share=reg["mechanism_share"])
    # Check every game, including unfinished ones, against reserved eval and other data streams.
    eval_seeds = {x["seed"] for x in read(root / "panel.json")["clusters"]}
    excluded = eval_seeds | {
        derive_seed(s, 2, i) for s in [reg["bank_seed"], *reg["training_seeds"]] if s != seed for i in range(20000)
    }
    schedule = t.collector.next_game

    def next_game():
        a = schedule()
        require(a.spec.seed not in excluded, "train/bank/evaluation seed collision")
        require(t.schedule.deals <= 20000, "registered deal bound exceeded")
        return a

    t.collector.next_game = next_game
    torch.manual_seed(seed)
    return t


def coverage(ro):
    groups, counts = torch.unique(prompt_groups(ro.obs), return_counts=True)
    acts, mask = ro.obs["actions"], ro.obs["action_mask"].bool()
    return dict(
        rows=ro.players.numel(),
        games=len(ro.games),
        censored_games=sum(g.truncated for g in ro.games),
        prompt_phase={str(int(k)): int(v) for k, v in zip(groups, counts, strict=True)},
        offered={k: int(((acts[..., 0] == i + 1) & mask).any(-1).sum()) for i, k in enumerate(ACTION_KINDS)},
    )


def replay_training(t):
    """Replay recorded actions, including games spanning updates, without a policy."""
    from ygorl.cards.ydk import Deck
    from ygorl.data.environment import PlayerRules
    from ygorl.env import GameSpec
    from ygorl.env.encoded import EncodedVecEnv

    path = t.run_dir / "game-traces.jsonl.gz"
    if not path.exists():
        require(t.counters["games"] == 0, "missing finished training traces")
        return 0
    env = EncodedVecEnv(
        1,
        1,
        vocab=t.vocab,
        event_length=t.cfg.event_length,
        skip_forced=t.cfg.skip_forced,
        cancel_budget=t.cfg.cancel_budget,
        selection_history=t.net_config.selection_history,
        attack_history=t.net_config.attack_history,
    )
    count = 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            spec = dict(record["spec"])
            for key in ("deck_a", "deck_b"):
                d = spec[key]
                spec[key] = Deck(name=d["name"], main=tuple(d["main"]), extra=tuple(d["extra"]), side=tuple(d["side"]))
            cfg = dict(spec["config"])
            cfg["player"] = PlayerRules(**cfg["player"])
            spec["config"] = DuelConfig(**cfg)
            env.reset(0, GameSpec(**spec))
            i = 0
            while True:
                ev = env.recv(timeout=120)[0]
                if ev.result is not None:
                    break
                action = record["action_indices"][i]
                require(ev.obs["action_mask"][action], "illegal training replay action")
                env.step(0, action)
                i += 1
            require(
                i == len(record["action_indices"]) == len(record["action_model_updates"]),
                "training trace length mismatch",
            )
            require(
                all(ev.result[k] == record[k] for k in ("reason", "winner", "turns", "decisions")),
                "training replay result mismatch",
            )
            require(
                [bytes(r).hex() for r in ev.result["responses"]] == record["responses"],
                "training replay response mismatch",
            )
            count += 1
    require(count == t.counters["games"], "finished training trace count mismatch")
    return count


def bank(root):
    verify(root)
    reg = read(root / "registration.json")
    t = trainer(root, "bank", reg["bank_seed"], reg)
    observations, probabilities, audits = [], [], []
    for _ in range(reg["bank_batches"]):
        guard(root, reg)
        ro = t._collect()
        require(all(healthy(g.result) for g in ro.games), "unhealthy bank game")
        observations.append(ro.obs)
        probabilities.append(ro.probs.reshape(-1, ro.probs.shape[-1]))
        audits.append(coverage(ro))
    t.counters["games"] = sum(a["games"] for a in audits)
    exact_replays = replay_training(t)
    obs = {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    probs = torch.cat(probabilities)
    groups = prompt_groups(obs)
    gen = torch.Generator().manual_seed(reg["bank_seed"])
    selected = []
    for group in torch.unique(groups):
        ix = torch.where(groups == group)[0]
        selected.append(ix[torch.randperm(len(ix), generator=gen)[: reg["bank_per_group"]]])
    ix = torch.cat(selected)
    data = dict(obs={k: v[ix] for k, v in obs.items()}, probabilities=probs[ix], groups=groups[ix])
    FrozenReplay(**data)  # validate before saving
    torch.save(data, root / "bank.pt")
    atomic_json(
        root / "bank-report.json",
        dict(
            sha256=sha(root / "bank.pt"),
            rows=len(ix),
            exact_training_replays=exact_replays,
            coverage=audits,
            identity_sha256=sha(root / "identity.json"),
            seed=reg["bank_seed"],
            unfinished_games=len(t.env._action_traces),
        ),
    )
    verify(root)


def arm(root, name):
    verify(root)
    reg = read(root / "registration.json")
    si, label = name.split("-", 1)
    require(label in ARMS, "unknown arm")
    seed = reg["training_seeds"][int(si)]
    t = trainer(root, name, seed, reg, mechanism=label.startswith("mechanism"), replay=label.endswith("replay"))
    bank_report = read(root / "bank-report.json")
    require(sha(root / "bank.pt") == bank_report["sha256"], "bank drift")
    require(bank_report["identity_sha256"] == sha(root / "identity.json"), "bank belongs to another study")
    replay = FrozenReplay(**torch.load(root / "bank.pt", weights_only=True), seed=seed + 100)
    original = t.learner.update
    audits = []

    def update(ro, policy=True):
        audit = coverage(ro)
        atomic_json(t.run_dir / "latest-rollout.json", audit)
        require(all(healthy(g.result) for g in ro.games), "unhealthy game before optimizer")
        require(
            all(torch.isfinite(getattr(ro, k)).all() for k in ("q", "values", "probs", "rewards")), "nonfinite rollout"
        )
        require(
            torch.allclose(ro.log_probs.exp(), ro.probs.gather(-1, ro.actions[..., None]).squeeze(-1), atol=1e-6),
            "behavior probability mismatch",
        )
        audits.append(audit)
        return original(ro, policy=policy, replay=replay)

    t.learner.update = update
    for u in range(reg["updates"]):
        guard(root, reg)
        record = t.step()
        require(all(torch.isfinite(p).all() for p in t.model.parameters()), "nonfinite parameters")
        atomic_json(
            root / "progress.json",
            dict(
                stage="training",
                arm=name,
                update=u + 1,
                updates=reg["updates"],
                rows=t.counters["rows"],
                optimizer_steps=record["minibatches"],
                updated_unix=time.time(),
            ),
        )
    exact_replays = replay_training(t)
    final = t.save("final.pt")
    torch.save(replay.state_dict(), t.run_dir / "replay-rng.pt")
    atomic_json(
        t.run_dir / "report.json",
        dict(
            checkpoint=str(final),
            sha256=sha(final),
            counters=t.counters,
            coverage=audits,
            replay_rows=replay.sampled,
            exact_training_replays=exact_replays,
            unfinished_games=len(t.env._action_traces),
            pair_draws=dict(getattr(t.schedule.decks, "draws", {})),
            auto_promote=False,
        ),
    )
    verify(root)


def evaluation_game(job):
    directory, number, spec = job
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    duel = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first, record_steps=True)
    result = duel.run(spec.agent_a(spec.agent_seeds[0]), spec.agent_b(spec.agent_seeds[1]))
    replay = directory / f"{number}.replay.json.gz"
    Replay.from_duel(duel, result).save(replay)
    script = ScriptedAgent(result.actions)
    cold = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first).run(script, script)
    row = dict(
        game_id=number,
        cluster=spec.pair,
        first=spec.first,
        seed=spec.seed,
        agent_seeds=spec.agent_seeds,
        result=result_fields(result),
        replay=str(replay),
        sha256=sha(replay),
        cold_equal=result_fields(cold) == result_fields(result) and cold.responses == result.responses,
    )
    atomic_json(directory / f"{number}.json", row)
    require(healthy(row["result"]) and row["cold_equal"], "unhealthy evaluation or cold replay mismatch")
    return row


def outcome_bounds(result):
    if result["reason"] in HOST_LIMITS:
        return (0.0, 1.0)
    require(healthy(result), "cannot score unhealthy game")
    value = 0.5 if result["winner"] is None else float(result["winner"] == 0)
    return (value, value)


def factorial_contrasts(values):
    """values [seed, arm, ...], same paired units; fixed arm order ARMS."""
    return dict(
        coverage=((values[:, 2] - values[:, 0]) + (values[:, 3] - values[:, 1])) / 2,
        replay=((values[:, 1] - values[:, 0]) + (values[:, 3] - values[:, 2])) / 2,
        interaction=values[:, 3] - values[:, 2] - values[:, 1] + values[:, 0],
        combined_minus_baseline=values[:, 3] - values[:, 0],
    )


def summarize(root, reg, cells):
    n = len(reg["training_seeds"])
    # Keep training seeds, opponents, deal clusters and their 4 paired games distinct.
    values = np.array(
        [[[cells[f"{s}-{a}__{o}"]["bounds"] for o in ("greedy", "source", "strong")] for a in ARMS] for s in range(n)]
    )
    values = values.reshape(n, 4, 3, reg["clusters"], 4, 2).mean(axis=(2, 4))  # [seed, arm, cluster, bound]
    mid = values.mean(-1)
    rng = np.random.default_rng(reg["bootstrap_seed"])
    si = rng.integers(n, size=(10000, n))
    ci = rng.integers(reg["clusters"], size=(10000, reg["clusters"]))
    contrasts = {}
    # Linear contrasts: conservative censoring bounds use coefficient signs.
    weights = dict(
        coverage=[-0.5, -0.5, 0.5, 0.5],
        replay=[-0.5, 0.5, -0.5, 0.5],
        interaction=[1, -1, -1, 1],
        combined_minus_baseline=[-1, 0, 0, 1],
    )
    for name, x in factorial_contrasts(mid).items():
        bs = x[si[:, :, None], ci[:, None, :]].mean((1, 2))
        w = np.asarray(weights[name])[None, :, None]
        lo = (np.maximum(w, 0) * values[..., 0] + np.minimum(w, 0) * values[..., 1]).sum(1)
        hi = (np.maximum(w, 0) * values[..., 1] + np.minimum(w, 0) * values[..., 0]).sum(1)
        contrasts[name] = dict(
            midpoint=float(x.mean()),
            per_seed=x.mean(1).tolist(),
            midpoint_ci95=np.quantile(bs, [0.025, 0.975]).tolist(),
            censoring_bounds=[float(lo.mean()), float(hi.mean())],
            note="midpoint imputes 0.5 to censored games; not an observed terminal score",
        )
    initial = np.asarray([cells[f"initial__{o}"]["bounds"] for o in ("greedy", "source", "strong")])
    initial = initial.reshape(3, reg["clusters"], 4, 2).mean(axis=(0, 2))
    retention = {}
    for i, name in enumerate(ARMS):
        delta = mid[:, i] - initial.mean(-1)[None, :]
        bs = delta[si[:, :, None], ci[:, None, :]].mean((1, 2))
        retention[name] = dict(
            midpoint_delta_from_initial=float(delta.mean()),
            per_seed=delta.mean(1).tolist(),
            midpoint_ci95=np.quantile(bs, [0.025, 0.975]).tolist(),
            censoring_bounds=[
                float((values[:, i, :, 0] - initial[:, 1]).mean()),
                float((values[:, i, :, 1] - initial[:, 0]).mean()),
            ],
        )
    atomic_json(
        root / "report.json",
        dict(
            complete=True,
            healthy=True,
            study_sha256=sha(root / "identity.json"),
            identity_sha256=sha(root / "identity.json"),
            cells=cells,
            contrasts=contrasts,
            retention=retention,
            auto_promote=False,
            uncertainty=reg["uncertainty"],
            decision="Review fixed-budget results and retention before extending; no automatic selection.",
        ),
    )


def evaluate(root):
    verify(root)
    reg = read(root / "registration.json")
    env, cfg, slots = panel(reg)
    candidates = {"initial": reg["source"]}
    for si in range(len(reg["training_seeds"])):
        for a in ARMS:
            r = read(root / f"{si}-{a}" / "report.json")
            require(sha(r["checkpoint"]) == r["sha256"], "final checkpoint drift")
            candidates[f"{si}-{a}"] = r["checkpoint"]
    opponents = dict(greedy="greedy", source="policy:" + reg["source"], strong="policy:" + reg["opponent"])
    jobs = []
    for name, path in candidates.items():
        for opp, policy in opponents.items():
            specs = cell_specs(agent_factory("policy:" + path), agent_factory(policy), slots, env, cfg)
            jobs.extend((str(root / "games" / f"{name}__{opp}"), i, s) for i, s in enumerate(specs))
    require(not (root / "games").exists(), "evaluation already exists; no overwrite")
    jobs.sort(key=lambda j: (j[1], j[0]))
    with mp.get_context("spawn").Pool(reg["workers"]) as pool:
        for count, _ in enumerate(pool.imap_unordered(evaluation_game, jobs), 1):
            guard(root, reg)
            atomic_json(
                root / "progress.json", dict(stage="evaluation", done=count, total=len(jobs), updated_unix=time.time())
            )
    cells = {}
    for path in sorted((root / "games").iterdir()):
        rows = [read(path / f"{i}.json") for i in range(4 * reg["clusters"])]
        bounds = [outcome_bounds(r["result"]) for r in rows]
        cells[path.name] = dict(
            games=len(rows),
            censored=sum(r["result"]["reason"] in HOST_LIMITS for r in rows),
            bounds=bounds,
            score_bounds=np.mean(bounds, axis=0).tolist(),
            mean_turns=float(np.mean([r["result"]["turns"] for r in rows])),
        )
    verify(root)
    summarize(root, reg, cells)


def run(root):
    from colab_checkpoint import run_lock

    with run_lock(root / "controller.lock"):
        verify(root)
        reg = read(root / "registration.json")
        require(not (root / "run.json").exists(), "run already attempted; inspect evidence before a new registration")
        atomic_json(
            root / "run.json", dict(started_unix=time.time(), deadline_unix=time.time() + reg["max_hours"] * 3600)
        )
        done = threading.Event()

        def beat():
            while not done.is_set():
                atomic_json(root / "pipeline-status.json", dict(stage="running", updated_unix=time.time()))
                done.wait(15)

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            # Rotate arm order across seeds to reduce time/order confounding.
            stages = [("bank", None)]
            for si in range(len(reg["training_seeds"])):
                stages += [("arm", f"{si}-{ARMS[(i + si) % 4]}") for i in range(4)]
            stages += [("evaluate", None)]
            for command, name in stages:
                guard(root, reg)
                args = [sys.executable, __file__, command, "--root", str(root)]
                if name:
                    args += ["--name", name]
                with (root / f"{name or command}.log").open("x") as log:
                    subprocess.run(
                        args,
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=max(1, read(root / "run.json")["deadline_unix"] - time.time()),
                    )
        except BaseException as exc:
            if not (root / "STOP.json").exists():
                atomic_json(
                    root / "STOP.json", dict(reason=repr(exc), traceback=traceback.format_exc(), time=time.time())
                )
            raise
        finally:
            done.set()
            thread.join()
            atomic_json(
                root / "pipeline-status.json",
                dict(stage="stopped" if (root / "STOP.json").exists() else "complete", updated_unix=time.time()),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "bank", "arm", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    if args.command == "prepare":
        prepare(root, args.source, args.opponent, args.smoke)
    elif args.command == "arm":
        arm(root, args.name)
    else:
        globals()[args.command](root)


if __name__ == "__main__":
    main()
