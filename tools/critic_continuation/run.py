"""Preregistered critic-weight transfer study: isolated training phases and fixed opponent cells."""

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ygorl.agents.registry import agent_factory
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_cards
from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings, spec_of
from ygorl.eval.arena import GameRecord
from ygorl.solver.resume import implementation_identity, output_lock
from ygorl.train.checkpoint import load_checkpoint
from ygorl.train.registration import atomic_json
from ygorl.train.rollout import RolloutStalled
from ygorl.train.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "terminal-critic-policy-long-2026-10-06"
EVAL_SEED = 2026100904
BASELINES = ("greedy", "old-256x2", "initial-128x2", "historical-rl")
NODES = (128, 256, 512)
DRIVERS = ("run.py", "analyze.py", "launch.py", "preflight.py", "audit.py", "watch.py", "retention.py")
BC = ROOT.parent / "teacher-cancel-2026-10-05/models/128x2/policy.pt"
OLD = ROOT.parent / "bounded-capacity-2026-10-05/models/256x2/policy.pt"
SOLVER = ROOT.parent / "declaration-list-2026-10-05/build/bin/combosolver"
INITIAL = PREVIOUS / "seed-0/cold/run/checkpoints/update_00000000.pt"
HISTORY = PREVIOUS / "seed-0/warm/run/checkpoints/update_00000128.pt"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def actor_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.actor.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def source_record(seed_id, arm):
    path = PREVIOUS / f"seed-{seed_id}/{arm}/nodes/update_00000128.json"
    record = json.loads(path.read_text())
    assert record["update"] == 128
    assert record["study_sha256"] == sha(PREVIOUS / "identity.json")
    assert record["sha256"] == sha(record["checkpoint"])
    return record


def assert_equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype, path
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu()), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_equal(a, b, f"{path}[{i}]")
    else:
        assert actual == expected, path


def verify_restored(trainer, source):
    actual = trainer.state_dict()
    source = {**source, "config": TrainConfig.from_dict(source["config"]).to_dict()}
    for key in actual:
        assert_equal(actual[key], source[key], key)
    assert trainer.counters["updates"] == 128 and trainer.counters["rows"] == 2097152
    assert trainer.learner.optimizer.state and actual["pool"]["snapshots"]
    assert trainer.environment.stamp() == source["environment"]
    return {
        "verified_keys": sorted(actual),
        "counters": dict(trainer.counters),
        "actor_sha256": actor_hash(trainer.model),
        "optimizer_and_pool_restored": True,
        "active_duels_restored": False,
    }


def inputs():
    env = load_environment("md-2026-09")
    meta = sorted(env.meta_decks, key=lambda m: m.deck.name)
    assert len(meta) == 20 and all(not env.validate_deck(m.deck, default_cards()) for m in meta)
    return env, meta


def expected_identity():
    env, meta = inputs()
    return {
        "implementation": implementation_identity(SOLVER, Path(__file__)),
        "drivers": {name: sha(ROOT / name) for name in DRIVERS},
        "environment": env.stamp(),
        "decks": {m.deck.name: sha(m.path) for m in meta},
        "initial_ppo_sha256": sha(INITIAL),
        "aborted_study_sha256": sha(ROOT.parent / "terminal-critic-continuation-restart-2026-10-08/identity.json"),
        "aborted_stop_sha256": sha(ROOT.parent / "terminal-critic-continuation-restart-2026-10-08/STOP.json"),
        "bc_sha256": sha(BC),
        "old_bc_sha256": sha(OLD),
        "historical_rl_sha256": sha(HISTORY),
        "source_models": {f"{i}-{arm}": source_record(i, arm) for i in range(3) for arm in ("cold", "warm")},
        "previous_analysis_sha256": sha(PREVIOUS / "analysis.json"),
        "previous_audit_sha256": sha(
            ROOT.parent / "terminal-critic-policy-long-monitor-2026-10-06/completion-audit.json"
        ),
        "preflight_sha256": sha(ROOT / "preflight.json"),
        "recovery_audit_sha256": sha(ROOT.parent / "activation-health-canary-2026-10-09/independent-audit.json"),
        "preregistration": json.loads((ROOT / "registration.json").read_text()),
        "eval_seed": EVAL_SEED,
        "nodes": NODES,
        "additional_updates": 2304,
        "retention_helper_sha256": sha(ROOT.parent / "terminal-critic-retention-2026-10-06/run.py"),
        "fit_helper_sha256": sha(ROOT.parent / "terminal-critic-fit-2026-10-06/run.py"),
        "retention_corpus_sha256": sha(ROOT.parent / "terminal-critic-confirm-2026-10-06/confirm/report.json"),
    }


def verify_identity():
    old = json.loads((ROOT / "identity.json").read_text())
    assert old == json.loads(json.dumps(expected_identity())), "study inputs or source changed"
    return old


def stop(reason, **extra):
    with output_lock(ROOT / "STOP.json"):
        if not (ROOT / "STOP.json").exists():
            atomic_json(ROOT / "STOP.json", {"reason": reason, "time": time.time(), **extra})


def check_stop():
    assert not (ROOT / "STOP.json").exists(), "study stopped; retain partial evidence"


def prepare():
    recovery = json.loads((ROOT.parent / "activation-health-canary-2026-10-09/independent-audit.json").read_text())
    assert recovery["healthy"] and recovery["checkpoint_update"] == 310
    assert recovery["delta"]["updates"] == 16 and recovery["delta"]["errors"] == 0
    pf = json.loads((ROOT / "preflight.json").read_text())
    assert len(pf["checks"]) == 6 and pf["training_rows_generated"] == 0
    assert pf["analysis_and_auditor_fixture_passed"]
    assert all(sha(ROOT / name) == digest for name, digest in pf["driver_sha256"].items())
    previous = json.loads((PREVIOUS / "analysis.json").read_text())
    audit = json.loads(
        (ROOT.parent / "terminal-critic-policy-long-monitor-2026-10-06/completion-audit.json").read_text()
    )
    assert previous["criterion_to_test_longer_training"]
    assert audit["analysis_sha256"] == sha(PREVIOUS / "analysis.json")
    with output_lock(ROOT / "identity.json"):
        assert not (ROOT / "identity.json").exists()
        atomic_json(ROOT / "identity.json", expected_identity())
    print("PREPARED", sha(ROOT / "identity.json"), flush=True)


def phase_root(seed_id, arm):
    return ROOT / f"seed-{seed_id}" / arm


def node_record(seed_id, arm, update):
    return phase_root(seed_id, arm) / "nodes" / f"update_{update:08d}.json"


def checked_node(seed_id, arm, update):
    path = node_record(seed_id, arm, update)
    data = json.loads(path.read_text())
    assert data["study_sha256"] == sha(ROOT / "identity.json")
    assert sha(data["checkpoint"]) == data["sha256"]
    return data


def publish(trainer, root, extra):
    u = trainer.counters["updates"]
    name = f"update_{u:08d}.pt"
    target = trainer.run_dir / "checkpoints" / name
    record = root / "nodes" / f"update_{u:08d}.json"
    assert not target.exists() and not record.exists()
    checkpoint = trainer.save(name)
    record.parent.mkdir(exist_ok=True)
    atomic_json(
        record,
        {
            "study_sha256": sha(ROOT / "identity.json"),
            "update": u,
            "checkpoint": str(checkpoint.resolve()),
            "sha256": sha(checkpoint),
            "actor_sha256": actor_hash(trainer.model),
            "counters": dict(trainer.counters),
            "elapsed_seconds": time.monotonic() - trainer.phase_started,
            **extra,
        },
    )


def unhealthy(game):
    return game.reason == "error" or any(
        game.result.get(k) for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
    )


def behavior_selection(game):
    reasons = []
    if game.result.get("turns", 0) > 20:
        reasons.append("turns_over_20")
    if game.result.get("decisions", 0) > 1000:
        reasons.append("decisions_over_1000")
    if game.assignment.spec.seed % 64 == 0:
        reasons.append("seed_mod_64")
    return reasons


class AuditedTrainer(Trainer):
    def _collect(self):
        check_stop()
        ro = super()._collect()
        bad = [g for g in ro.games if unhealthy(g)]
        if bad:
            self._log_errors(bad)
            self._log_games(bad)
            self.save("health-stop.pt")
            stop("unhealthy training rollout before optimizer", run=str(self.run_dir))
            raise RuntimeError("unhealthy training rollout")
        selected = []
        for game in ro.games:
            reasons = behavior_selection(game)
            if reasons:
                record = self._diagnostic_record(game)
                assert record["responses"], "selected game has no native replay responses"
                record.update(
                    behavior_selection=reasons,
                    training_update=self.counters["updates"] + 1,
                    replay_basis="native_responses",
                )
                selected.append(json.dumps(record))
        if selected:
            with gzip.open(self.run_dir / "behavior-traces.jsonl.gz", "at", compresslevel=6) as f:
                f.write("\n".join(selected) + "\n")
        check_stop()
        if self.counters["updates"] == 128:
            path = self.run_dir.parent / "first-rollout.pt"
            assert not path.exists()
            torch.save(ro, path)
            self.first_rollout_sha = sha(path)
        return ro

    def step(self):
        r = super().step()
        assert all(not isinstance(v, float) or math.isfinite(v) for v in r.values()), "nonfinite metrics"
        assert self.cfg.critic_warmup == 0 and r["critic_warmup"] == 0
        games = self.counters["games"] - self.source_counters["games"]
        truncated = self.counters["truncated"] - self.source_counters["truncated"]
        if games >= 100 and truncated / games > 0.01:
            self.save("truncation-stop.pt")
            stop("training truncation rate exceeds 1%", run=str(self.run_dir))
            raise RuntimeError("training truncation stop")
        return r


def train(seed_id, arm):
    verify_identity()
    check_stop()
    root = phase_root(seed_id, arm)
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / "report.json"):
        assert not (root / "identity.json").exists(), "partial/completed phase retained; do not overwrite"
        source = source_record(seed_id, arm)
        state = load_checkpoint(source["checkpoint"])
        trainer = AuditedTrainer.resume(source["checkpoint"], root / "run", log=lambda msg: print(msg, flush=True))
        restored = verify_restored(trainer, state)
        trainer.source_counters = dict(trainer.counters)
        trainer.phase_started = time.monotonic()
        atomic_json(
            root / "identity.json",
            {
                "study_sha256": sha(ROOT / "identity.json"),
                "source": source,
                "config": trainer.cfg.to_dict(),
                "torch": str(torch.__version__),
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0),
            },
        )
        atomic_json(root / "start-audit.json", restored)
        publish(trainer, root, {"resumed_from": source})
        for endpoint in NODES[1:]:
            trainer.train(max_updates=endpoint - trainer.counters["updates"])
            publish(trainer, root, {})
        check_stop()
        assert trainer.counters["updates"] == 512 and trainer.counters["rows"] == 8388608
        delta = {
            k: trainer.counters[k] - trainer.source_counters[k]
            for k in ("updates", "rows", "games", "truncated", "errors")
        }
        games = [json.loads(line) for line in gzip.open(root / "run/games.jsonl.gz", "rt")]
        assert len(games) == delta["games"]
        records = [json.loads(p.read_text()) for p in sorted((root / "nodes").glob("*.json"))]
        assert [r["update"] for r in records] == list(NODES)
        atomic_json(
            root / "report.json",
            {
                "study_sha256": sha(ROOT / "identity.json"),
                "identity_sha256": sha(root / "identity.json"),
                "counters": dict(trainer.counters),
                "source_counters": trainer.source_counters,
                "added": delta,
                "elapsed_seconds": time.monotonic() - trainer.phase_started,
                "nodes": records,
                "first_rollout_sha256": sha(root / "first-rollout.pt"),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        print("PHASE COMPLETE", seed_id, arm, delta, flush=True)


class Watch:
    def __init__(self, inner):
        assert hasattr(inner, "host"), "candidate must use the PPO HostDuel adapter"
        self.inner = inner
        self.actions = []
        self.counts = {"material_cancel_available": 0, "material_cancel_masked": 0, "material_cancel_chosen": 0}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def on_decision(self, point, index):
        self.actions.append(index)  # both seats; retained even if an agent later raises without a DuelResult
        hook = getattr(self.inner, "on_decision", None)
        if hook:
            hook(point, index)

    def act(self, point):
        index = self.inner.act(point)
        if point.decision.TYPE == C.MSG_SELECT_UNSELECT_CARD:
            cancels = [i for i, action in enumerate(point.actions) if action.kind == "cancel"]
            if cancels:
                assert len(cancels) == 1
                cancel = cancels[0]
                mask = self.inner.host.observe()["action_mask"]
                masked = cancel >= len(mask) or not mask[cancel]
                self.counts["material_cancel_available"] += 1
                self.counts["material_cancel_masked"] += int(masked)
                self.counts["material_cancel_chosen"] += int(index == cancel)
                assert not (masked and index == cancel)
        return index


def play(job):
    game_id, spec, trace_root = job
    torch.set_num_threads(1)
    base = {"pair": spec.pair, "first": spec.first, "seed": spec.seed}
    watched = Watch(spec.agent_a(spec.agent_seeds[0]))
    result = None
    try:
        duel = Duel(
            spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first, record_steps=True
        )
        result = duel.run(watched, spec.agent_b(spec.agent_seeds[1]))
        fields = {
            k: getattr(result, k)
            for k in (
                "winner",
                "reason",
                "turns",
                "decisions",
                "win_reason",
                "lp",
                "retries",
                "unknown_messages",
                "undecodable_messages",
                "error",
            )
        }
        record = GameRecord(**base, **fields, script_errors=len(result.script_errors))
    except Exception as exc:
        record = GameRecord(**base, winner=None, reason="exception", error=repr(exc))
    row = {
        "game_id": game_id,
        "agent_a": spec_of(spec.agent_a),
        "agent_b": spec_of(spec.agent_b),
        "agent_seeds": spec.agent_seeds,
        "deck_a": asdict(spec.deck_a),
        "deck_b": asdict(spec.deck_b),
        "config": asdict(spec.config),
        "result": asdict(record),
        "candidate_counts": watched.counts,
    }
    if not (record.healthy and record.reason == "win" and record.winner in (0, 1)):
        path = Path(trace_root) / f"failed-{game_id}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(
                {
                    "spec": row,
                    "explicit_action_prefix": watched.actions,
                    "trace": None if result is None else asdict(result),
                },
                f,
                default=lambda x: x.hex() if isinstance(x, bytes) else str(x),
            )
        row["failure_trace"] = {"path": str(path), "sha256": sha(path)}
    return row


def wait_for(path):
    deadline = time.monotonic() + 48 * 3600
    while not path.exists():
        check_stop()
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(5)
    check_stop()


def evaluate_node(pool, candidate, label, slots, env, cfg, initial):
    root = ROOT / "evaluation" / label
    root.mkdir(parents=True, exist_ok=True)
    opponents = {
        "greedy": "greedy",
        "old-256x2": f"policy:{OLD}",
        "initial-128x2": f"policy:{initial}",
        "historical-rl": f"policy:{HISTORY}",
    }
    reports = []
    for name in BASELINES:
        check_stop()
        target = root / f"{name}.json"
        raw = root / f"{name}.jsonl"
        ident = {
            "study_sha256": sha(ROOT / "identity.json"),
            "candidate": candidate,
            "opponent": opponents[name],
            "opponent_sha256": "" if name == "greedy" else sha(opponents[name].split(":", 1)[1]),
        }
        if target.exists():
            previous = json.loads(target.read_text())
            assert previous["identity"] == ident and previous["raw_sha256"] == sha(raw)
            assert previous["games"] == 256 and previous["all_healthy"]
            reports.append(previous)
            continue
        assert not raw.exists(), "partial cell retained; investigate, do not replace or append"
        specs = cell_specs(
            agent_factory(f"policy:{candidate['checkpoint']}"), agent_factory(opponents[name]), slots, env, cfg
        )
        assert len(specs) == 256
        counts = {k: 0 for k in ("material_cancel_available", "material_cancel_masked", "material_cancel_chosen")}
        wins = 0
        with raw.open("x") as f:
            jobs = [(i, spec, str(root)) for i, spec in enumerate(specs)]
            for row in pool.imap(play, jobs, chunksize=1):
                f.write(json.dumps(row) + "\n")
                f.flush()
                result = GameRecord(**row["result"])
                if not (result.healthy and result.reason == "win" and result.winner in (0, 1)):
                    stop(
                        "evaluation health/limit stop",
                        label=label,
                        opponent=name,
                        game_id=row["game_id"],
                        result=row["result"],
                        raw=str(raw),
                    )
                    raise RuntimeError("evaluation health/limit stop; trace retained")
                wins += int(result.winner == 0)
                for key in counts:
                    counts[key] += row["candidate_counts"][key]
        report = {
            "identity": ident,
            "raw_sha256": sha(raw),
            "games": 256,
            "wins": wins,
            "win_rate": wins / 256,
            "all_healthy": True,
            "candidate_counts": counts,
        }
        atomic_json(target, report)
        reports.append(report)
        print("CELL COMPLETE", label, name, wins, "/256", counts, flush=True)
    return reports


def consume():
    verify_identity()
    env, meta = inputs()
    cfg = DuelConfig.from_environment(env)
    pairs = sample_pairings(len(meta), 64, EVAL_SEED)
    slots = pairing_slots([m.deck for m in meta], pairs, EVAL_SEED, cfg)
    first = {"checkpoint": str(INITIAL), "sha256": sha(INITIAL)}
    manifest = {
        "study_sha256": sha(ROOT / "identity.json"),
        "pairings": pairs,
        "seed": EVAL_SEED,
        "config": asdict(cfg),
        "baseline_initial": first,
        "workers": 4,
        "torch_threads": 1,
    }
    with output_lock(ROOT / "evaluation-report.json"):
        path = ROOT / "evaluation-identity.json"
        if path.exists():
            assert json.loads(path.read_text()) == json.loads(json.dumps(manifest))
        else:
            atomic_json(path, manifest)
        done = {}
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            done["initial"] = evaluate_node(pool, first, "initial", slots, env, cfg, first["checkpoint"])
            for seed_id in range(3):
                for arm in ("cold", "warm"):
                    for update in NODES:
                        wait_for(node_record(seed_id, arm, update))
                        candidate = checked_node(seed_id, arm, update)
                        label = f"seed-{seed_id}-{arm}-u{update}"
                        done[label] = evaluate_node(pool, candidate, label, slots, env, cfg, first["checkpoint"])
        assert sum(r["games"] for reports in done.values() for r in reports) == 19456
        atomic_json(
            ROOT / "evaluation-report.json",
            {"study_sha256": sha(ROOT / "identity.json"), "all_healthy": True, "games": 19456, "cells": done},
        )
        print("EVALUATION COMPLETE", 19456, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "consume"))
    parser.add_argument("--seed-id", type=int, choices=range(3))
    parser.add_argument("--arm", choices=("cold", "warm"))
    args = parser.parse_args()
    torch.set_num_threads(2 if args.mode == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if args.mode == "prepare":
            prepare()
        elif args.mode == "train":
            train(args.seed_id, args.arm)
        else:
            consume()
    except BaseException as exc:
        stop(repr(exc), mode=args.mode, seed_id=args.seed_id, arm=args.arm)
        if isinstance(exc, RolloutStalled):
            import traceback

            traceback.print_exc()
            os._exit(1)
        raise
