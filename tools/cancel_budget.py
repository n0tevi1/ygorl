"""Candidate-only inference ablation; keep the native engine and checkpoints frozen."""

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from behavior_replay import result_fields, sha
from colab_checkpoint import run_lock
from precision_evaluate import crossed_bootstrap, healthy, require
from ygorl import _core, paths
from ygorl.agents.registry import agent_factory
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, ScriptedAgent, default_cards
from ygorl.engine.replay import Replay
from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings
from ygorl.eval.behavior import length_stats
from ygorl.solver.resume import tree_digest
from ygorl.train.registration import atomic_json

REPO = Path(__file__).resolve().parents[1]
BUDGETS = (32, 4, 1, 0)


def restrict_cancel(point, observation, count, budget):
    """Intersect the existing encoded mask; never remove its sole remaining exit."""
    if budget not in BUDGETS or count < 0:
        raise ValueError("invalid cancellation budget/count")
    mask = observation["action_mask"]
    require(bool(np.any(mask)), "empty baseline action mask")
    info = {"intervened": 0, "sole_exit_preserved": 0}
    if point.decision.TYPE not in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD) or count < budget:
        return observation, info
    indices = [i for i, a in enumerate(point.actions) if i < len(mask) and mask[i] and a.kind == "cancel"]
    if not indices:
        return observation, info
    changed = mask.copy()
    changed[indices] = False
    if not np.any(changed):
        info["sole_exit_preserved"] = 1
        return observation, info
    info["intervened"] = 1
    return {**observation, "action_mask": changed}, info


class MaskView:
    """Expose an already audited observation to the unmodified policy sampler."""

    def __init__(self, host):
        self.host, self.observation = host, None

    def __getattr__(self, name):
        return getattr(self.host, name)

    def observe(self):
        require(self.observation is not None, "missing decision observation")
        return self.observation


def play(job):
    destination, game_id, spec, budget, control = job
    torch.set_num_threads(1)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    duel = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first, record_steps=True)
    agents = [spec.agent_a(spec.agent_seeds[0]), spec.agent_b(spec.agent_seeds[1])]
    session = DuelSession(duel)
    counts, interventions, actions = Counter(), [], []
    try:
        for a in agents:
            if hasattr(a, "on_duel_start"):
                a.on_duel_start(duel)
        candidate = agents[0]
        original_host = candidate.host
        view = MaskView(original_host)
        candidate.host = view
        while (point := session.point) is not None:
            for a in agents:
                if hasattr(a, "observe"):
                    a.observe(point, session.core)
            actor = agents[duel.deck_of(point.player)]
            if actor is candidate:
                obs = original_host.observe()
                count = session.tracker._selection_cancels
                view.observation, info = restrict_cancel(point, obs, count, budget)
                counts.update(info)
                if info["intervened"]:
                    interventions.append(
                        {
                            "decision": point.index,
                            "prior_cancels": count,
                            "actions": [asdict(a) for a in point.actions],
                            "baseline_mask": obs["action_mask"].tolist(),
                            "restricted_mask": view.observation["action_mask"].tolist(),
                        }
                    )
            idx = actor.act(point)
            if actor is candidate:
                require(bool(view.observation["action_mask"][idx]), "sampled masked action")
                if point.decision.TYPE in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD):
                    counts["cancel_chosen"] += point.actions[idx].kind == "cancel"
                if interventions and interventions[-1]["decision"] == point.index:
                    interventions[-1]["chosen"] = idx
            actions.append(idx)
            session.act(idx)
            for a in agents:
                if hasattr(a, "on_decision"):
                    a.on_decision(point, idx)
        result = session.result()
    except Exception as exc:
        atomic_json(
            destination / f"{game_id}.failure.json",
            {
                "error": repr(exc),
                "actions": actions,
                "interventions": interventions,
                "budget": budget,
                "game_id": game_id,
            },
        )
        raise
    finally:
        session.close()
    replay = destination / f"{game_id}.replay.json.gz"
    Replay.from_duel(duel, result).save(replay)
    script = ScriptedAgent(actions)
    cold = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first).run(script, script)
    equal = (
        result_fields(result) == result_fields(cold) and result.responses == cold.responses and actions == cold.actions
    )
    row = {
        "game_id": game_id,
        "result": {"pair": spec.pair, "first": spec.first, "seed": spec.seed, **result_fields(result)},
        "agent_seeds": spec.agent_seeds,
        "budget": budget,
        "counts": dict(counts),
        "interventions": interventions,
        "cold_replay_equal": equal,
        "replay": str(replay),
        "replay_sha256": sha(replay),
    }
    if control:
        require(budget == 32, "control must use baseline budget")
        plain = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first).run(
            spec.agent_a(spec.agent_seeds[0]), spec.agent_b(spec.agent_seeds[1])
        )
        row["unwrapped_equal"] = (
            result_fields(result) == result_fields(plain)
            and actions == plain.actions
            and result.responses == plain.responses
        )
    target = destination / f"{game_id}.json"
    atomic_json(target, row)
    atomic_json(target.with_suffix(".sha.json"), {"sha256": sha(target)})
    require(equal and healthy(row), "unhealthy game or replay mismatch; retain artifacts")
    require(not control or row["unwrapped_equal"], "budget 32 differs from original policy")
    return row


def bindings(registration):
    env = load_environment("md-2026-09")
    decks = sorted(env.meta_decks, key=lambda d: d.deck.name)
    require(len(decks) == 20 and all(not env.validate_deck(d.deck, default_cards()) for d in decks), "decks differ")
    for path, digest in registration["checkpoints"].items():
        require(sha(path) == digest, "checkpoint changed")
    ident = {
        "registration": registration,
        "python": tree_digest(REPO / "src/ygorl", "*.py"),
        "tools": {p.name: sha(p) for p in sorted((REPO / "tools").glob("*.py"))},
        "native": sha(_core.__file__),
        "cards": sha(paths.cards_cdb()),
        "scripts": tree_digest(paths.card_scripts(), "*.lua"),
        "environment": env.stamp(),
        "decks": {d.path.name: sha(d.path) for d in decks},
        "torch": str(torch.__version__),
    }
    cfg = DuelConfig.from_environment(env)
    pairs = sample_pairings(len(decks), registration["clusters"], registration["panel_seed"])
    slots = pairing_slots([d.deck for d in decks], pairs, registration["panel_seed"], cfg)
    return json.loads(json.dumps(ident)), env, cfg, slots


def read_game(path):
    require(sha(path) == json.loads(path.with_suffix(".sha.json").read_text())["sha256"], "game changed")
    row = json.loads(path.read_text())
    require(healthy(row) and row["cold_replay_equal"], "existing failed game; refuse replacement")
    require(sha(row["replay"]) == row["replay_sha256"], "replay changed")
    require(row.get("unwrapped_equal", True), "existing baseline mismatch")
    return row


def summary(root, reg, rows):
    cells = {}
    for label, cell in rows.items():
        cells[label] = {
            "games": len(cell),
            "wins": sum(r["result"]["winner"] == 0 for r in cell),
            "turns": length_stats([r["result"]["turns"] for r in cell]),
            "decisions": length_stats([r["result"]["decisions"] for r in cell]),
            "counts": dict(sum((Counter(r["counts"]) for r in cell), Counter())),
        }
    report = {"cells": cells, "auto_promote": False, "exploratory": True}
    if len(cells) == 36 and all(c["games"] == 4 * reg["clusters"] for c in cells.values()):
        values = np.array(
            [
                [
                    [[r["result"]["winner"] == 0 for r in rows[f"seed-{s}-b{b}-{o['name']}"]] for o in reg["opponents"]]
                    for b in BUDGETS
                ]
                for s in range(3)
            ]
        )
        values = values.reshape(3, 4, 3, reg["clusters"], 4).mean(axis=(2, 4))
        report["win_difference_from_32"] = {
            str(b): crossed_bootstrap(values[:, i] - values[:, 0], seed=2026100825, reps=20000)
            for i, b in enumerate(BUDGETS)
            if i
        }
    atomic_json(root / "summary.json", report)


def run(root, workers):
    reg = json.loads((root / "registration.json").read_text())
    ident, env, cfg, slots = bindings(reg)
    path = root / "identity.json"
    if path.exists():
        require(json.loads(path.read_text()) == ident, "identity changed")
    else:
        require(not (root / "games").exists(), "unbound existing games")
        atomic_json(path, ident)
    runtime = root / "runtime.json"
    if not runtime.exists():
        atomic_json(runtime, {"deadline_unix": time.time() + reg["max_hours"] * 3600})
    deadline = json.loads(runtime.read_text())["deadline_unix"]
    rows = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for s, candidate in enumerate(reg["candidates"]):
            for opponent in reg["opponents"]:
                specs = cell_specs(agent_factory(candidate), agent_factory(opponent["spec"]), slots, env, cfg)
                for budget in BUDGETS:
                    label = f"seed-{s}-b{budget}-{opponent['name']}"
                    dest = root / "games" / label
                    cell = []
                    jobs = []
                    for i, spec in enumerate(specs):
                        existing = dest / f"{i}.json"
                        if existing.exists():
                            row = read_game(existing)
                            require(row["game_id"] == i and row["budget"] == budget, "game identity differs")
                            require(row["agent_seeds"] == list(spec.agent_seeds), "agent seeds differ")
                            require(
                                row["result"]["seed"] == spec.seed
                                and row["result"]["first"] == spec.first
                                and row["result"]["pair"] == spec.pair,
                                "panel differs",
                            )
                            cell.append(row)
                        else:
                            require(not (dest / f"{i}.failure.json").exists(), "failed game retained")
                            require(not (dest / f"{i}.replay.json.gz").exists(), "partial game retained")
                            jobs.append((str(dest), i, spec, budget, budget == 32 and i == 0))
                    iterator = pool.imap_unordered(play, jobs, chunksize=1)
                    pending = len(jobs)
                    while pending:
                        require(not (root / "STOP.json").exists(), "study stopped")
                        require(time.time() < deadline, "registered wall deadline expired")
                        atomic_json(
                            root / "status.json",
                            {
                                "stage": "running",
                                "cell": label,
                                "cell_games": len(cell),
                                "completed_games": sum(map(len, rows.values())) + len(cell),
                                "checked_unix": time.time(),
                                "deadline_unix": deadline,
                            },
                        )
                        try:
                            cell.append(iterator.next(timeout=min(20, max(0.01, deadline - time.time()))))
                            pending -= 1
                        except mp.TimeoutError:
                            pass
                    cell.sort(key=lambda r: r["game_id"])
                    require(len(cell) == len(specs), "incomplete cell")
                    rows[label] = cell
                    summary(root, reg, rows)
                    print(
                        json.dumps(
                            {"cell": label, "games": len(cell), "wins": sum(r["result"]["winner"] == 0 for r in cell)}
                        ),
                        flush=True,
                    )
    atomic_json(
        root / "status.json",
        {"stage": "complete", "completed_games": sum(map(len, rows.values())), "checked_unix": time.time()},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    require(1 <= args.workers <= 4, "worker limit")
    with run_lock(args.root / "run.lock"):
        require(not (args.root / "STOP.json").exists(), "study stopped; inspect retained evidence")
        try:
            run(args.root, args.workers)
        except Exception as exc:
            atomic_json(args.root / "STOP.json", {"reason": repr(exc), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
