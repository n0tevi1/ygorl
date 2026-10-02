"""Fixed-state response-value pilot; exact prefix replay, paired continuations and held-out public reranking.

The oracle and critic use privileged state only as diagnostics. The public reranker uses frozen actor features
and training-pairing labels. Candidate selection and scoring use separate rollout seeds. Costs, targets and all
subsequent decisions follow the unchanged sampled policy; this is not a search over optimal multi-step responses.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ygorl import _core
from ygorl.data.environment import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.driver import ABANDON, drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.eval.agent_matrix import pairing_slots, sample_pairings
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import _uniform
from ygorl.eval.response_probe import ResponseRidge, bootstrap_clusters, common_valid, public_features
from ygorl.nets.actor_critic import collate_privileged
from ygorl.nets.batch import collate
from ygorl.train.checkpoint import load_actor_critic

PASS = ACTION_KINDS.index("pass") + 1


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def signature(obs, player):
    h = hashlib.sha256(bytes([player]))
    for key in sorted(obs):
        h.update(key.encode())
        h.update(np.asarray(obs[key]).tobytes())
    return h.hexdigest()


def json_default(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False, default=json_default) + "\n")
    tmp.replace(path)


def sample(prob, u):
    cdf = np.cumsum(prob, dtype=float)
    return min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(prob) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--environment", default="environments/md-2026-09")
    p.add_argument("--pairings", default=32, type=int)
    p.add_argument("--continuations", default=32, type=int)
    p.add_argument("--seed", default=20261003, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.pairings < 4 or a.pairings % 2 or a.continuations < 4 or a.continuations % 2:
        p.error("need even pairings >=4 and even continuations >=4")
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    environment = load_environment(a.environment)
    environment.check_meta_decks(default_cards())
    config = DuelConfig.from_environment(environment, max_decisions=4000)
    decks = [m.deck for m in environment.meta_decks]
    pairs = sample_pairings(len(decks), a.pairings, a.seed)
    slots = pairing_slots(decks, pairs, a.seed, config)
    specs = []
    for s, ds, seeds in slots:
        for m in (0, 1):
            for g in (0, 1):
                specs.append(
                    GameSpec(
                        seed=s,
                        deck_a=ds[m],
                        deck_b=ds[1 - m],
                        first=int(m != g),
                        config=replace(config, shuffle_decks=False),
                    )
                )
    manifest = {
        "environment": environment.stamp(),
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": digest(a.checkpoint),
        "core_sha256": digest(_core.__file__),
        "tool_sha256": digest(__file__),
        "stats_sha256": digest(Path(__file__).parents[1] / "src/ygorl/eval/response_probe.py"),
        "config": asdict(config),
        "seed": a.seed,
        "pairings": pairs,
        "continuations": a.continuations,
        "device": a.device,
        "train_pairings": "even indices",
        "test_pairings": "odd indices",
        "selection_seeds": "first half",
        "evaluation_seeds": "second half",
        "max_actions": 8,
        "max_turn": 4,
    }
    manifest = json.loads(json.dumps(manifest))
    path = a.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest mismatch")
    write(path, manifest)
    loaded = load_actor_critic(a.checkpoint)
    model = loaded.model.to(a.device).eval()
    env = EncodedVecEnv(128, 4, vocab=loaded.vocab, event_length=loaded.event_length, skip_forced=True, privileged=True)
    t0 = time.time()
    roots = {}
    collection = {}
    states_path = a.out / "states.json"
    if states_path.exists():
        saved = json.loads(states_path.read_text())
        roots = {int(k): v for k, v in saved["roots"].items()}
        collection = saved["collection"]
    else:

        def start(i, sp):
            return {"prefix": [], "counts": [0, 0]}

        def result(game, res):
            collection[str(game.index)] = {
                "status": "error" if res["reason"] == "error" else "no_window",
                "result": res,
            }

        @torch.no_grad()
        def collect(ready):
            batch = collate([ev.obs for _, ev in ready], a.device)
            output = model(batch, collate_privileged([ev.privileged for _, ev in ready], a.device))
            probs = torch.softmax(output.logits.float(), -1).cpu().numpy()
            features = public_features(model.actor, [ev.obs for _, ev in ready], a.device)
            qs = output.q.float().cpu().numpy()
            actions = []
            for row, (game, ev) in enumerate(ready):
                obs = ev.obs
                glob = obs["globals"]
                legal = np.flatnonzero(obs["action_mask"]).tolist()
                if int(glob[3]) > 4:
                    collection[str(game.index)] = {"status": "no_window"}
                    actions.append(ABANDON)
                    continue
                passes = [k for k in legal if int(obs["actions"][k, 0]) == PASS]
                if int(glob[18]) == C.MSG_SELECT_CHAIN and int(glob[2]) == 0 and passes and len(legal) > 1:
                    if len(legal) > 8:
                        collection[str(game.index)] = {"status": "too_many_actions"}
                        actions.append(ABANDON)
                        continue
                    root = {
                        "game": game.index,
                        "pairing": game.index // 4,
                        "player": ev.player,
                        "turn": int(glob[3]),
                        "prefix": game.state["prefix"],
                        "signature": signature(obs, ev.player),
                        "actions": legal,
                        "pass": legal.index(passes[0]),
                        "probs": probs[row, legal].tolist(),
                        "q": qs[row, legal].tolist(),
                        "features": features[row, legal].tolist(),
                        "action_rows": obs["actions"][legal].tolist(),
                    }
                    roots[game.index] = root
                    np.savez_compressed(a.out / f"obs-{game.index}.npz", **obs)
                    collection[str(game.index)] = {"status": "root"}
                    actions.append(ABANDON)
                    continue
                n = game.state["counts"][ev.player]
                chosen = sample(probs[row], _uniform(derive_seed(a.seed, game.index, 11), 0, ev.player, n))
                game.state["counts"][ev.player] += 1
                game.state["prefix"].append([chosen, signature(obs, ev.player)])
                actions.append(chosen)
            return actions

        drive(env, specs, collect, result, start=start)
        write(states_path, {"roots": roots, "collection": collection})
    print(f"collected {len(roots)} roots from {len(specs)} games in {time.time() - t0:.1f}s", flush=True)
    # Play one state's full candidate x continuation rectangle at a time, atomically retaining completed states.
    # Prefix replay is exact (every observation tensor hashed), not just matching candidate count / turn.
    for index, root in sorted(roots.items()):
        path = a.out / f"rollouts-{index}.json"
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
        def decide(ready):
            actions = [None] * len(ready)
            play = []
            for j, (game, ev) in enumerate(ready):
                job = game.state
                pos = job["pos"]
                prefix = root["prefix"]
                if pos < len(prefix):
                    chosen, want = prefix[pos]
                    if signature(ev.obs, ev.player) != want:
                        raise RuntimeError(f"prefix mismatch game {index} at {pos}")
                    actions[j] = chosen
                elif pos == len(prefix):
                    if signature(ev.obs, ev.player) != root["signature"]:
                        raise RuntimeError(f"root mismatch game {index}")
                    actions[j] = root["actions"][job["choice"]]
                else:
                    play.append(j)
                job["pos"] += 1
            if play:
                probs = (
                    torch.softmax(model.policy_logits(collate([ready[j][1].obs for j in play], a.device)).float(), -1)
                    .cpu()
                    .numpy()
                )
                for row, j in enumerate(play):
                    game, ev = ready[j]
                    job = game.state
                    seed = derive_seed(a.seed, index, job["rep"], 29)
                    actions[j] = sample(probs[row], _uniform(seed, 0, ev.player, job["counts"][ev.player]))
                    job["counts"][ev.player] += 1
            return actions

        drive(env, [specs[index]] * len(jobs), decide, result, start=lambda i, sp: jobs[i])
        write(path, {"game": index, "actions": root["actions"], "records": records})
        print(f"{time.time() - t0:.0f}s: game {index}, {len(jobs)} continuations saved", flush=True)
    summarize(a, roots, collection)


def summarize(a, roots, collection):
    half = a.continuations // 2
    data = {}
    trainx = []
    trainy = []
    errors = 0
    invalid = []
    for index, root in sorted(roots.items()):
        saved = json.loads((a.out / f"rollouts-{index}.json").read_text())
        records = saved["records"]
        scores = np.array(
            [
                np.nan
                if r["reason"] == "error"
                else 0.5
                if r.get("winner") is None
                else float(r["winner"] == root["player"])
                for r in records
            ]
        ).reshape(-1, a.continuations)
        errors += sum(r["reason"] == "error" for r in records)
        valid = common_valid(scores)
        sel = valid[:half]
        ev = valid[half:]
        if sel.sum() < half // 2 or ev.sum() < half // 2:
            invalid.append(index)
            continue
        selection = scores[:, :half][:, sel].mean(axis=1)
        evaluation = scores[:, half:][:, ev].mean(axis=1)
        x = np.asarray(root["features"])
        x = x - x[root["pass"]]
        data[index] = {
            "root": root,
            "x": x,
            "selection": selection,
            "evaluation": evaluation,
            "scores": scores,
            "valid": valid,
            "selection_n": int(sel.sum()),
            "evaluation_n": int(ev.sum()),
        }
        if root["pairing"] % 2 == 0:
            for k in range(len(x)):
                if k != root["pass"]:
                    trainx.append(x[k])
                    trainy.append(selection[k] - selection[root["pass"]])
    teacher = ResponseRidge.fit(trainx, trainy)
    np.savez(a.out / "public-reranker.npz", scale=teacher.scale, weight=teacher.weight)
    choices = []
    paired = {k: {} for k in ("oracle", "critic", "public", "argmax", "pass")}
    confident = []
    excluded_pairs = {i // 4 for i in invalid}
    excluded_pairs |= {int(i) // 4 for i, v in collection.items() if v["status"] == "error"}
    for index, d in data.items():
        root = d["root"]
        if root["pairing"] % 2 == 0 or root["pairing"] in excluded_pairs:
            continue
        prob = np.asarray(root["probs"])
        prob /= prob.sum()
        q = np.asarray(root["q"])
        ev = d["evaluation"]
        baseline = float(prob @ ev)
        picks = {
            "oracle": int(np.argmax(d["selection"])),
            "critic": int(np.argmax(q)),
            "public": int(np.argmax(teacher.scores(d["x"]))),
            "argmax": int(np.argmax(prob)),
            "pass": root["pass"],
        }
        for name, k in picks.items():
            paired[name][index] = float(ev[k] - baseline)
        # Independent eval half: record paired action-pass error bars before interpreting critic signs.
        for k in range(len(q)):
            if k == root["pass"]:
                continue
            ds = d["scores"][k, half:] - d["scores"][root["pass"], half:]
            ds = ds[d["valid"][half:]]
            se = float(ds.std(ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else float("inf")
            diff = float(ds.mean())
            if abs(diff) > 1.96 * se:
                confident.append(
                    {
                        "game": index,
                        "action": root["actions"][k],
                        "difference": diff,
                        "se": se,
                        "critic_difference": float((q[k] - q[root["pass"]]) / 2),
                        "critic_sign_correct": bool((q[k] - q[root["pass"]]) * diff > 0),
                    }
                )
        choices.append(
            {
                "game": index,
                "pairing": root["pairing"],
                "turn": root["turn"],
                "player": root["player"],
                "actions": root["actions"],
                "picks": picks,
                "baseline": baseline,
                "evaluation": ev.tolist(),
                "selection_n": d["selection_n"],
                "evaluation_n": d["evaluation_n"],
            }
        )
    testgames = [i for i in range(a.pairings * 4) if (i // 4) % 2 and i // 4 not in excluded_pairs]
    metrics = {
        name: bootstrap_clusters([diffs.get(i, 0.0) for i in testgames], [i // 4 for i in testgames], seed=a.seed)
        for name, diffs in paired.items()
    }
    report = {
        "environment": json.loads((a.out / "manifest.json").read_text())["environment"],
        "roots": len(roots),
        "train_examples": len(trainx),
        "test_games": len(testgames),
        "test_roots": len(choices),
        "excluded_pairs": sorted(excluded_pairs),
        "error_rollouts": errors,
        "metrics": metrics,
        "choices": choices,
        "confident_action_pairs": confident,
        "critic_sign_accuracy": float(np.mean([x["critic_sign_correct"] for x in confident])) if confident else None,
        "scope": "pilot; fixed checkpoint; privileged-state continuations; public teacher trained only on even pairings; pointwise cluster intervals",
    }
    write(a.out / "summary.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("choices", "confident_action_pairs")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
