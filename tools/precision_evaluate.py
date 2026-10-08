"""Independent CPU evaluation of the registered learner-precision pilot; no model promotion."""

import argparse
import gzip
import json
import multiprocessing as mp
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from colab_checkpoint import atomic, run_lock, sha
from precision_colab import validate_registration
from precision_worker import validate_precision
from ygorl import _core, paths
from ygorl.agents.registry import agent_factory, parse_spec
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_cards
from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings, spec_of
from ygorl.eval.arena import GameRecord
from ygorl.eval.behavior import length_stats
from ygorl.solver.resume import output_lock, tree_digest
from ygorl.train.checkpoint import load_checkpoint

REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def stop(root, reason, **extra):
    with output_lock(root / "STOP.json"):
        if not (root / "STOP.json").exists():
            atomic(root / "STOP.json", {"reason": reason, "owner": "precision_evaluate", "time": time.time(), **extra})


def check_stop(root):
    require(not (root / "STOP.json").exists(), "pilot stopped; retain partial evidence")


def validate_panel(panel):
    require(panel["seed"] == 2026100822 and panel["clusters"] == 64, "evaluation panel differs")
    require(
        panel["bootstrap_seed"] == 2026100823 and panel["bootstrap_replicates"] == 20000,
        "bootstrap registration differs",
    )
    require(sorted(s["seed"] for s in panel["starts"]) == [0, 1, 2], "three paired starts required")
    names = [o["name"] for o in panel["opponents"]]
    require(len(names) == len(set(names)) == 3, "three unique opponents required")
    require(all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names), "unsafe opponent name")
    for opponent in panel["opponents"]:
        parsed = parse_spec(opponent["spec"])
        require(opponent["spec"] == "greedy" or parsed.is_policy, "unsupported opponent")
        digest = sha(parsed.checkpoint) if parsed.is_policy else ""
        require(digest == opponent["checkpoint_sha256"], "opponent checkpoint changed")


def bind_inputs(root):
    """Bind local evaluator independently; remote native binaries need not share the local build hash."""
    registration = read_json(root / "registration.json")
    validate_registration(registration)
    controller = read_json(root / "identity.json")
    package = read_json(root / "package.json")
    require(controller["registration_sha256"] == sha(root / "registration.json"), "registration changed")
    require(controller["package_sha256"] == sha(root / "package.json"), "package changed")
    require(controller["run_id"] == registration["run_id"], "controller run differs")
    panel = registration["evaluation"]
    validate_panel(panel)
    for name in ("archive", "manifest", "bootstrap"):
        require(sha(package[name]) == package[name + "_sha256"], f"package {name} changed")
    source = read_json(package["manifest"])["files"]
    bindings = {}
    for name, digest in source.items():
        if name.startswith(("src/", "tools/")) and name.endswith(".py"):
            path = REPO / name
            require(sha(path) == digest, f"registered source differs: {name}")
            bindings[str(path)] = digest
    require(str(Path(__file__).resolve()) in bindings, "evaluator absent from registered source")
    # Also bind local Python files absent from a package allowlist, which imports could otherwise pick up.
    env = load_environment("md-2026-09")
    meta = sorted(env.meta_decks, key=lambda m: m.deck.name)
    require(len(meta) == 20 and all(not env.validate_deck(m.deck, default_cards()) for m in meta), "deck pool differs")
    deck_hashes = {m.path.name: sha(m.path) for m in meta}
    for start in panel["starts"]:
        require(sha(start["checkpoint"]) == start["sha256"], "starting checkpoint changed")
        state = load_checkpoint(start["checkpoint"])
        require(state["environment"] == env.stamp(), "starting environment differs")
        require(state["counters"]["updates"] == 128, "starting checkpoint is not u128")
        require({Path(p).name: sha(p) for p in state["config"]["decks"]} == deck_hashes, "starting decks differ")
        for job in registration["jobs"]:
            if job["seed"] == start["seed"]:
                require(job["checkpoint_sha256"] == start["sha256"], "evaluation start differs from training")
                bound = controller["jobs"][job["job_id"]]
                expected = {
                    "job_id": job["job_id"],
                    "seed": job["seed"],
                    "learner_precision": job["learner_precision"],
                    "source_checkpoint_sha256": start["sha256"],
                    "origin_update": 128,
                    "target_update": 160,
                    "registration_sha256": controller["registration_sha256"],
                    "package_sha256": controller["package_sha256"],
                    "environment": env.stamp(),
                }
                require(all(bound.get(k) == v for k, v in expected.items()), "controller job binding differs")
    cfg = DuelConfig.from_environment(env)
    pairs = sample_pairings(len(meta), panel["clusters"], panel["seed"])
    ident = {
        "controller_sha256": sha(root / "identity.json"),
        "registration_sha256": sha(root / "registration.json"),
        "package_sha256": sha(root / "package.json"),
        "bindings": bindings,
        "python_sha256": tree_digest(REPO / "src/ygorl", "*.py"),
        "native_sha256": sha(_core.__file__),
        "cards_sha256": sha(paths.cards_cdb()),
        "scripts_sha256": tree_digest(paths.card_scripts(), "*.lua"),
        "environment": env.stamp(),
        "decks": deck_hashes,
        "pairings": pairs,
        "config": asdict(cfg),
        "panel": panel,
        "candidate_precision": "cpu-fp32",
        "torch": str(torch.__version__),
    }
    ident = json.loads(json.dumps(ident))
    target = root / "evaluator/identity.json"
    if target.exists():
        require(read_json(target) == ident, "evaluator identity changed; refuse resume")
    else:
        require(not list((root / "evaluation").glob("*/*")), "unbound existing evaluation artifacts")
        atomic(target, ident)
    return registration, controller, env, cfg, pairing_slots([m.deck for m in meta], pairs, panel["seed"], cfg)


class Watch:
    def __init__(self, inner):
        require(hasattr(inner, "host"), "candidate must use PPO HostDuel")
        self.inner, self.actions = inner, []
        self.counts = dict.fromkeys(
            ("material_cancel_available", "material_cancel_masked", "material_cancel_chosen"), 0
        )

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def on_decision(self, point, index):
        self.actions.append(index)
        hook = getattr(self.inner, "on_decision", None)
        if hook:
            hook(point, index)

    def act(self, point):
        index = self.inner.act(point)
        if point.decision.TYPE in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD):
            cancels = [i for i, action in enumerate(point.actions) if action.kind == "cancel"]
            if cancels:
                require(len(cancels) == 1, "ambiguous cancel")
                cancel = cancels[0]
                mask = self.inner.host.observe()["action_mask"]
                masked = cancel >= len(mask) or not mask[cancel]
                self.counts["material_cancel_available"] += 1
                self.counts["material_cancel_masked"] += int(masked)
                self.counts["material_cancel_chosen"] += int(index == cancel)
                require(not (masked and index == cancel), "masked cancel chosen")
        return index


def play(job):
    game_id, spec, trace_root = job
    torch.set_num_threads(1)
    watched, result = None, None
    base = {"pair": spec.pair, "first": spec.first, "seed": spec.seed}
    try:
        watched = Watch(spec.agent_a(spec.agent_seeds[0]))
        duel = Duel(
            spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first, record_steps=True
        )
        # Policy agents are loaded on CPU, without autocast regardless of the checkpoint's training config.
        with torch.autocast("cpu", enabled=False):
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
        "candidate_counts": {} if watched is None else watched.counts,
    }
    if not healthy(row):
        path = Path(trace_root) / f"failed-{game_id}.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump(
                {
                    "spec": row,
                    "explicit_action_prefix": [] if watched is None else watched.actions,
                    "trace": None if result is None else asdict(result),
                },
                stream,
                default=lambda x: x.hex() if isinstance(x, bytes) else str(x),
            )
        row["failure_trace"] = {"path": str(path), "sha256": sha(path)}
    return row


def healthy(row):
    result = GameRecord(**row["result"])
    return result.healthy and result.reason == "win" and result.winner in (0, 1)


def read_cell(target, ident=None, games=256):
    report = read_json(target)
    raw = target.with_suffix(".jsonl")
    require(ident is None or report["identity"] == ident, "cell identity changed")
    require(report["raw_sha256"] == sha(raw), "completed raw changed")
    text = raw.read_text()
    require(text.endswith("\n"), "partial raw tail")
    rows = [json.loads(line) for line in text.splitlines()]
    require(len(rows) == games and [r["game_id"] for r in rows] == list(range(games)), "missing/duplicate game")
    require(all(healthy(r) for r in rows) and report["all_healthy"], "unhealthy completed cell")
    wins = sum(r["result"]["winner"] == 0 for r in rows)
    require(
        report["games"] == games and report["wins"] == wins and report["win_rate"] == wins / games, "cell score differs"
    )
    return report, rows


def receive_rows(root, iterator, deadline):
    """Check STOP and the wall bound even if the next ordered worker result has not arrived."""
    while True:
        check_stop(root)
        try:
            if deadline is None:
                row = next(iterator)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("evaluation wall budget expired; partial evidence retained")
                try:
                    row = iterator.next(timeout=min(remaining, 30))
                except mp.TimeoutError:
                    continue
            yield row
        except StopIteration:
            return


def evaluate_cell(root, pool, candidate, label, opponent, slots, env, cfg, *, deadline=None):
    check_stop(root)
    require(sha(candidate["checkpoint"]) == candidate["sha256"], "candidate checkpoint changed")
    target = root / "evaluation" / label / (opponent["name"] + ".json")
    raw = target.with_suffix(".jsonl")
    ident = {
        "study_sha256": sha(root / "identity.json"),
        "evaluator_sha256": sha(root / "evaluator/identity.json"),
        "candidate": candidate,
        "opponent": opponent["spec"],
        "opponent_sha256": opponent["checkpoint_sha256"],
    }
    if target.exists():
        return read_cell(target, ident)[0]
    require(not raw.exists(), "partial cell retained; investigate, never replace or reroll")
    target.parent.mkdir(parents=True, exist_ok=True)
    trace_root = target.parent / opponent["name"]
    trace_root.mkdir(exist_ok=True)
    specs = cell_specs(
        agent_factory(f"policy:{candidate['checkpoint']}"), agent_factory(opponent["spec"]), slots, env, cfg
    )
    require(len(specs) == 256, "cell size differs")
    rows = []
    with raw.open("x") as stream:
        iterator = pool.imap(play, [(i, spec, str(trace_root)) for i, spec in enumerate(specs)], chunksize=1)
        for row in receive_rows(root, iterator, deadline):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            if not healthy(row):
                stop(
                    root, "evaluation health/limit stop", label=label, opponent=opponent["name"], raw=str(raw), row=row
                )
                raise RuntimeError("evaluation health/limit stop; partial artifacts retained")
            check_stop(root)
    wins = sum(r["result"]["winner"] == 0 for r in rows)
    require(len(rows) == 256, "worker returned incomplete cell")
    report = {
        "identity": ident,
        "raw_sha256": sha(raw),
        "games": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows),
        "all_healthy": True,
        "candidate_counts": dict(sum((Counter(r["candidate_counts"]) for r in rows), Counter())),
        "turns": length_stats([r["result"]["turns"] for r in rows]),
        "decisions": length_stats([r["result"]["decisions"] for r in rows]),
        "deck_out": sum(r["result"]["win_reason"] == 2 for r in rows),
    }
    atomic(target, report)
    print("CELL COMPLETE", label, opponent["name"], wins, "/256", flush=True)
    return report


def completed_candidate(root, job, controller):
    path = root / "completed" / (job["job_id"] + ".json")
    if not path.exists():
        return None
    record = read_json(path)
    for key in ("job_id", "seed", "learner_precision"):
        require(record[key] == job[key], "completed job differs")
    identity = controller["jobs"][job["job_id"]]
    require(record["identity"] == identity, "completed identity differs")
    directory = Path(record["snapshot"])
    validate_precision(directory, identity)
    require(
        Path(record["checkpoint"]).resolve() == (directory / "checkpoint.pt").resolve(), "checkpoint outside snapshot"
    )
    require(sha(record["checkpoint"]) == record["checkpoint_sha256"], "completed checkpoint changed")
    state = load_checkpoint(record["checkpoint"])
    require(state["counters"]["updates"] == 160, "endpoint is not u160")
    return {
        "checkpoint": record["checkpoint"],
        "sha256": record["checkpoint_sha256"],
        "completion_sha256": sha(path),
        "seed": job["seed"],
        "learner_precision": job["learner_precision"],
        "update": 160,
    }


def crossed_bootstrap(differences, *, seed, reps):
    """Resample seeds and the shared deal clusters independently; retain all seats/opponents within a cell."""
    values = np.asarray(differences, dtype=float)
    require(values.ndim == 2 and min(values.shape) > 0 and np.isfinite(values).all(), "invalid bootstrap values")
    rng = np.random.default_rng(seed)
    si = rng.integers(values.shape[0], size=(reps, values.shape[0]))
    ci = rng.integers(values.shape[1], size=(reps, values.shape[1]))
    samples = values[si[:, :, None], ci[:, None, :]].mean(axis=(1, 2))
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "per_seed": values.mean(axis=1).tolist(),
        "seed": seed,
        "replicates": reps,
        "method": "crossed seed x shared deal-cluster percentile bootstrap; equal opponent weights",
    }


def game_key(row):
    return [row[k] for k in ("game_id", "agent_seeds", "deck_a", "deck_b", "config")] + [
        row["result"][k] for k in ("pair", "first", "seed")
    ]


def analyze(root, registration):
    panel = registration["evaluation"]
    values = np.empty((3, 3, len(panel["opponents"]), 64, 4))  # seed, start/fp32/bf16, opponent, cluster, variant
    reference = None
    cells = {}
    for seed in range(3):
        for a, arm in enumerate(("start", "fp32", "bf16")):
            label = f"seed-{seed}-{arm}"
            for o, opponent in enumerate(panel["opponents"]):
                path = root / "evaluation" / label / (opponent["name"] + ".json")
                if not path.exists():
                    return None
                report, rows = read_cell(path)
                keys = [game_key(r) for r in rows]
                require(reference is None or keys == reference, "paired evaluation panel differs")
                reference = keys
                require([r["result"]["pair"] for r in rows] == [i // 4 for i in range(256)], "cluster order differs")
                values[seed, a, o] = np.array([r["result"]["winner"] == 0 for r in rows]).reshape(64, 4)
                cells[label + "/" + opponent["name"]] = {
                    "sha256": sha(path),
                    "raw_sha256": report["raw_sha256"],
                    "win_rate": report["win_rate"],
                }
    cluster = values.mean(axis=(2, 4))
    kwargs = {"seed": panel["bootstrap_seed"], "reps": panel["bootstrap_replicates"]}
    primary = crossed_bootstrap(cluster[:, 2] - cluster[:, 1], **kwargs)
    report = {
        "evaluator_sha256": sha(root / "evaluator/identity.json"),
        "games": int(values.size),
        "all_healthy": True,
        "primary_bf16_minus_fp32": primary,
        "improvement_from_start": {
            arm: crossed_bootstrap(cluster[:, a] - cluster[:, 0], **kwargs) for a, arm in ((1, "fp32"), (2, "bf16"))
        },
        "per_seed_opponent": {
            f"seed-{s}/{opponent['name']}": {
                arm: float(values[s, a, o].mean()) for a, arm in enumerate(("start", "fp32", "bf16"))
            }
            for s in range(3)
            for o, opponent in enumerate(panel["opponents"])
        },
        "quality_gate": {
            "nominal_ci_lower_above_minus_2pp": primary["ci95"][0] > -0.02,
            "no_seed_loses_more_than_5pp": min(primary["per_seed"]) >= -0.05,
        },
        "behavior_review": "required; distribution/cancellation signals are not automatic error labels",
        "training_health_and_efficiency_review": "required separately",
        "auto_promote": False,
        "interpretation": "small three-seed pilot; unresolved interval is inconclusive, not equivalence",
        "cells": cells,
    }
    atomic(root / "evaluation-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=7)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    require(1 <= args.workers <= 4 and 0 < args.poll <= 60 and 0 < args.max_hours <= 7, "invalid evaluator bounds")
    root = args.root.resolve()
    deadline = time.monotonic() + args.max_hours * 3600
    torch.set_num_threads(1)
    with run_lock(root / "evaluator"):
        while not (root / "identity.json").exists():
            check_stop(root)
            if args.once or time.monotonic() >= deadline:
                atomic(
                    root / "evaluator/status.json", {"status": "waiting-for-controller", "checked_unix": time.time()}
                )
                return 0
            time.sleep(args.poll)
        try:
            with mp.get_context("spawn").Pool(args.workers) as pool:
                while time.monotonic() < deadline:
                    check_stop(root)
                    registration, controller, env, cfg, slots = bind_inputs(root)
                    candidates = [
                        (f"seed-{s['seed']}-start", {**s, "update": 128}) for s in registration["evaluation"]["starts"]
                    ]
                    for job in registration["jobs"]:
                        candidate = completed_candidate(root, job, controller)
                        if candidate:
                            candidates.append((job["job_id"], candidate))
                    for label, candidate in candidates:
                        for opponent in registration["evaluation"]["opponents"]:
                            (root / "evaluation" / label / opponent["name"]).mkdir(parents=True, exist_ok=True)
                            evaluate_cell(root, pool, candidate, label, opponent, slots, env, cfg, deadline=deadline)
                    report = analyze(root, registration) if len(candidates) == 9 else None
                    atomic(
                        root / "evaluator/status.json",
                        {
                            "status": "complete" if report else "waiting-for-endpoints",
                            "checked_unix": time.time(),
                            "candidates_evaluated": [label for label, _ in candidates],
                        },
                    )
                    if report or args.once:
                        return 0
                    time.sleep(args.poll)
            atomic(root / "evaluator/status.json", {"status": "deadline-incomplete", "checked_unix": time.time()})
        except BaseException as exc:
            stop(root, repr(exc))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
