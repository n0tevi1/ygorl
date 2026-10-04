"""Collect candidate-owned response windows against a fixed panel and fit public response teachers.

All continuations are training labels. Independent full-game evaluation uses a separate seed and directory.
The observation-only window predicate and policy sampling match evaluate_response_reranker.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from compare_checkpoints import digest, git
from response_probe import sample, signature, write
from ygorl import _core
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.driver import ABANDON, drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.agent_matrix import pairing_slots, sample_pairings
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import _uniform
from ygorl.eval.response_probe import ResponseRidge, common_valid, public_features, response_window, simple_response
from ygorl.nets.batch import collate, policy_logits
from ygorl.paths import cards_cdb, third_party
from ygorl.train.checkpoint import load_actor

BIASES = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


def collect(a, name, opponent_index, candidate, opponent, slots, config):
    out = a.out / name
    out.mkdir(exist_ok=True)
    specs, streams = [], []
    for seed, decks, seeds in slots:
        for m in (0, 1):
            for g in (0, 1):
                specs.append(
                    GameSpec(
                        seed=seed,
                        deck_a=decks[m],
                        deck_b=decks[1 - m],
                        first=int(m != g),
                        config=replace(config, shuffle_decks=False),
                    )
                )
                streams.append((seeds[m], seeds[1 - m]))
    env = EncodedVecEnv(
        128, 4, vocab=candidate.vocab, event_length=candidate.event_length, skip_forced=True, privileged=False
    )
    policies = (candidate, opponent)
    roots, outcomes = {}, {}
    saved = out / "states.json"
    if saved.exists():
        data = json.loads(saved.read_text())
        roots, outcomes = {int(k): v for k, v in data["roots"].items()}, data["outcomes"]
    else:

        def result(game, res):
            outcomes[str(game.index)] = {"status": "error" if res["reason"] == "error" else "no_window", "result": res}

        @torch.no_grad()
        def decide(ready):
            actions = [None] * len(ready)
            for side, policy in enumerate(policies):
                ids = [j for j, (game, ev) in enumerate(ready) if game.spec.deck_of_seat(ev.player) == side]
                if not ids:
                    continue
                observations = [ready[j][1].obs for j in ids]
                probs = (
                    torch.softmax(policy_logits(policy.net, collate(observations, a.device)).float(), -1).cpu().numpy()
                )
                for row, j in enumerate(ids):
                    game, ev = ready[j]
                    if int(ev.obs["globals"][3]) > 4:
                        outcomes[str(game.index)] = {"status": "no_window"}
                        actions[j] = ABANDON
                        continue
                    window = response_window(ev.obs) if side == 0 else None
                    if window is not None:
                        legal, passed = window
                        f = public_features(candidate.net, [ev.obs], a.device)[0, legal]
                        roots[game.index] = {
                            "game": game.index,
                            "pairing": game.index // 4,
                            "player": ev.player,
                            "turn": int(ev.obs["globals"][3]),
                            "chain_length": int(ev.obs["globals"][17]),
                            "prefix": game.state["prefix"],
                            "signature": signature(ev.obs, ev.player),
                            "actions": legal.tolist(),
                            "pass": int(np.flatnonzero(legal == passed)[0]),
                            "probs": probs[row, legal].tolist(),
                            "features": f.tolist(),
                            "action_rows": ev.obs["actions"][legal].tolist(),
                        }
                        np.savez_compressed(out / f"obs-{game.index}.npz", **ev.obs)
                        outcomes[str(game.index)] = {"status": "root"}
                        actions[j] = ABANDON
                        continue
                    n = game.state["counts"][ev.player]
                    chosen = sample(probs[row], _uniform(streams[game.index][side], 0, 0, n))
                    game.state["counts"][ev.player] += 1
                    game.state["prefix"].append([chosen, signature(ev.obs, ev.player)])
                    actions[j] = chosen
            return actions

        drive(env, specs, decide, result, min_batch=32, start=lambda i, sp: {"counts": [0, 0], "prefix": []})
        write(saved, {"roots": roots, "outcomes": outcomes})
    print(name, "collected", len(roots), "roots from", len(specs), "games", flush=True)
    for index, root in sorted(roots.items()):
        path = out / f"rollouts-{index}.json"
        if path.exists():
            continue
        jobs = [
            {"choice": c, "rep": r, "pos": 0, "counts": [0, 0]}
            for c in range(len(root["actions"]))
            for r in range(a.continuations)
        ]
        records = [None] * len(jobs)

        def result(game, res):
            records[game.index] = res

        @torch.no_grad()
        def branch(ready):
            actions = [None] * len(ready)
            play = []
            for j, (game, ev) in enumerate(ready):
                state = game.state
                pos = state["pos"]
                prefix = root["prefix"]
                if pos < len(prefix):
                    actions[j], wanted = prefix[pos]
                    if signature(ev.obs, ev.player) != wanted:
                        raise RuntimeError(f"{name}/{index}: prefix mismatch {pos}")
                elif pos == len(prefix):
                    if signature(ev.obs, ev.player) != root["signature"]:
                        raise RuntimeError(f"{name}/{index}: root mismatch")
                    actions[j] = root["actions"][state["choice"]]
                else:
                    play.append(j)
                state["pos"] += 1
            for side, policy in enumerate(policies):
                ids = [j for j in play if ready[j][0].spec.deck_of_seat(ready[j][1].player) == side]
                if not ids:
                    continue
                probs = (
                    torch.softmax(
                        policy_logits(policy.net, collate([ready[j][1].obs for j in ids], a.device)).float(), -1
                    )
                    .cpu()
                    .numpy()
                )
                for row, j in enumerate(ids):
                    game, ev = ready[j]
                    state = game.state
                    seed = derive_seed(a.seed, opponent_index, index, state["rep"], 29)
                    actions[j] = sample(probs[row], _uniform(seed, 0, ev.player, state["counts"][ev.player]))
                    state["counts"][ev.player] += 1
            return actions

        drive(env, [specs[index]] * len(jobs), branch, result, min_batch=32, start=lambda i, sp: jobs[i])
        write(path, {"game": index, "actions": root["actions"], "records": records})
        print(
            f"{time.monotonic() - a.started:.0f}s: {name} game {index}, {len(records)} continuations saved", flush=True
        )


def fit(a, opponents):
    data = []
    excluded = set()
    errors = 0
    total = 0
    coverage = {}
    for name in opponents:
        states = json.loads((a.out / name / "states.json").read_text())
        coverage[name] = {"roots": len(states["roots"]), "base_games": a.pairings * 4}
        excluded |= {int(i) // 4 for i, s in states["outcomes"].items() if s["status"] == "error"}
        for i, r in states["roots"].items():
            records = json.loads((a.out / name / f"rollouts-{i}.json").read_text())["records"]
            total += len(records)
            errors += sum(z["reason"] == "error" for z in records)
            scores = np.array(
                [
                    np.nan
                    if z["reason"] == "error"
                    else 0.5
                    if z["winner"] is None
                    else float(z["winner"] == r["player"])
                    for z in records
                ]
            ).reshape(-1, a.continuations)
            valid = common_valid(scores)
            if valid.sum() < a.continuations // 2:
                excluded.add(r["pairing"])
                continue
            data.append(
                {"opponent": name, "root": r, "scores": scores[:, valid].mean(axis=1), "valid": int(valid.sum())}
            )
    data = [d for d in data if d["root"]["pairing"] not in excluded]
    if a.pairings - len(excluded) < 2:
        raise ValueError("too few common training pairings")
    x = []
    y = []
    gains = {b: 0.0 for b in BIASES}
    for d in data:
        r = d["root"]
        features = np.asarray(r["features"])
        features -= features[r["pass"]].copy()
        values = d["scores"]
        p = np.asarray(r["probs"])
        p /= p.sum()
        baseline = float(p @ values)
        for k in range(len(features)):
            if k != r["pass"]:
                x.append(features[k])
                y.append(float(values[k] - values[r["pass"]]))
        for b in BIASES:
            chosen = simple_response(p, np.arange(len(p)), r["pass"], pass_bias=b)
            gains[b] += float(values[chosen] - baseline)
    denominator = (a.pairings - len(excluded)) * 4 * len(opponents)
    gains = {b: v / denominator for b, v in gains.items()}
    selected = max(BIASES, key=lambda b: gains[b])
    teacher = ResponseRidge.fit(x, y, regularization=10.0)
    np.savez(a.out / "public-reranker.npz", scale=teacher.scale, weight=teacher.weight)
    report = {
        "coverage": coverage,
        "excluded_pairings": sorted(excluded),
        "training_roots": len(data),
        "training_examples": len(x),
        "continuations": total,
        "error_continuations": errors,
        "selected_pass_bias": selected,
        "training_bias_gains": gains,
        "regularization": 10.0,
        "scope": "All continuation means are training estimates, not independent efficacy evidence.",
        "used_roots": [{"opponent": d["opponent"], "game": d["root"]["game"], "valid_seeds": d["valid"]} for d in data],
    }
    write(a.out / "training-summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "used_roots"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--environment", default="environments/md-2026-09")
    p.add_argument("--pairings", type=int, default=32)
    p.add_argument("--continuations", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026100401)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.started = time.monotonic()
    if a.pairings < 2 or a.continuations < 4:
        p.error("need >=2 pairings and >=4 continuations")
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    environment = load_environment(a.environment)
    environment.check_meta_decks(default_cards())
    panel = json.loads(a.panel.read_text())
    environment.check_stamp(panel["environment"])
    config = DuelConfig.from_environment(environment, max_decisions=4000)
    pairs = sample_pairings(len(environment.meta_decks), a.pairings, a.seed)
    slots = pairing_slots([m.deck for m in environment.meta_decks], pairs, a.seed, config)
    manifest = {
        "environment": environment.stamp(),
        "config": asdict(config),
        "pairings": pairs,
        "seed": a.seed,
        "continuations": a.continuations,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": digest(a.checkpoint),
        "opponents": panel["opponents"],
        "device": a.device,
        "core_sha256": digest(_core.__file__),
        "cards_sha256": digest(cards_cdb()),
        "tool_sha256": digest(__file__),
        "shared_tool_sha256": {
            n: digest(Path(__file__).with_name(n)) for n in ("response_probe.py", "compare_checkpoints.py")
        },
        "source_hashes": {str(f.relative_to(Path("src"))): digest(f) for f in sorted(Path("src/ygorl").rglob("*.py"))},
        "submodules": {
            n: {
                "commit": git(third_party() / n, "rev-parse", "HEAD"),
                "dirty": git(third_party() / n, "status", "--porcelain"),
            }
            for n in ("CardScripts", "BabelCDB", "ygopro-core")
        },
        "training_only": True,
        "regularization": 10.0,
        "pass_bias_grid": BIASES,
        "protocol": "candidate first eligible public response, same window/sampling as deployed evaluation; whole-game continuations with original policy/opponent",
    }
    manifest = json.loads(json.dumps(manifest))
    path = a.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest mismatch")
    write(path, manifest)
    candidate = load_actor(a.checkpoint)
    candidate.net.to(a.device).eval()
    for oi, (name, entry) in enumerate(panel["opponents"].items()):
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("opponent changed")
        opponent = load_actor(entry["path"])
        opponent.net.to(a.device).eval()
        if candidate.signature.mismatches(opponent.signature):
            raise ValueError("incompatible policy signatures")
        collect(a, name, oi, candidate, opponent, slots, config)
    if digest(a.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("candidate changed during collection")
    for entry in panel["opponents"].values():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("opponent changed during collection")
    fit(a, panel["opponents"])


if __name__ == "__main__":
    main()
