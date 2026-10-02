"""Frozen public response reranker versus fixed external opponents, with paired full-game controls.

One intervention at the candidate's first eligible response; all other decisions use unchanged sampled actors.
No privileged tensors, critic, or decision-time engine search enter the reranker. Resume complete cells only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from compare_checkpoints import digest, git, write_json
from response_probe import PASS, sample
from ygorl import _core
from ygorl.data.environment import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.driver import drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.agent_matrix import pairing_slots, sample_pairings
from ygorl.eval.batched import _uniform
from ygorl.eval.paired_panel import compare_panel
from ygorl.eval.response_probe import ResponseRidge, public_features
from ygorl.nets.batch import collate, policy_logits
from ygorl.paths import cards_cdb, third_party
from ygorl.train.checkpoint import load_actor


def play(candidate, opponent, teacher, slots, config, device, enabled):
    specs, streams = [], []
    for seed, ds, ss in slots:
        for m in (0, 1):
            for g in (0, 1):
                specs.append(
                    GameSpec(
                        seed=seed,
                        deck_a=ds[m],
                        deck_b=ds[1 - m],
                        first=int(m != g),
                        config=replace(config, shuffle_decks=False),
                    )
                )
                streams.append((ss[m], ss[1 - m]))
    env = EncodedVecEnv(
        128, 4, vocab=candidate.vocab, event_length=candidate.event_length, skip_forced=True, privileged=False
    )
    records = [None] * len(specs)

    def result(game, res):
        first = game.spec.first
        winner = res["winner"]
        records[game.index] = {
            "pair": game.index // 4,
            "seed": game.spec.seed,
            "first": first,
            "winner": None if winner is None else game.spec.deck_of_seat(winner),
            "reason": res["reason"],
            "error": res.get("error", ""),
            "turns": res["turns"],
            "decisions": res["decisions"],
            "lp": [res["lp"][first], res["lp"][1 - first]],
            "intervention": game.state["intervention"],
        }

    @torch.no_grad()
    def decide(ready):
        actions = [0] * len(ready)
        for side, policy in enumerate((candidate, opponent)):
            ids = [j for j, (game, ev) in enumerate(ready) if game.spec.deck_of_seat(ev.player) == side]
            if not ids:
                continue
            observations = [ready[j][1].obs for j in ids]
            probs = torch.softmax(policy_logits(policy.net, collate(observations, device)).float(), -1).cpu().numpy()
            eligible = []
            for row, j in enumerate(ids):
                game, ev = ready[j]
                state = game.state
                n = state["counts"][ev.player]
                actions[j] = sample(probs[row], _uniform(streams[game.index][side], 0, 0, n))
                state["counts"][ev.player] += 1
                glob = ev.obs["globals"]
                legal = np.flatnonzero(ev.obs["action_mask"])
                passes = [k for k in legal if ev.obs["actions"][k, 0] == PASS]
                if (
                    side == 0
                    and state["intervention"] is None
                    and int(glob[3]) <= 4
                    and int(glob[18]) == C.MSG_SELECT_CHAIN
                    and int(glob[2]) == 0
                    and passes
                    and 1 < len(legal) <= 8
                ):
                    eligible.append((j, legal, passes[0]))
            if eligible:
                features = public_features(policy.net, [ready[j][1].obs for j, _, _ in eligible], device)
                for f, (j, legal, passed) in zip(features, eligible, strict=True):
                    game, ev = ready[j]
                    picked = int(legal[np.argmax(teacher.scores(f[legal] - f[passed]))])
                    game.state["intervention"] = {
                        "turn": int(ev.obs["globals"][3]),
                        "chain_length": int(ev.obs["globals"][17]),
                        "sampled": actions[j],
                        "reranked": picked,
                        "changed": bool(picked != actions[j]),
                        "applied": enabled,
                    }
                    if enabled:
                        actions[j] = picked
        return actions

    drive(env, specs, decide, result, min_batch=32, start=lambda i, sp: {"counts": [0, 0], "intervention": None})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--environment", default="environments/md-2026-09")
    p.add_argument("--pairings", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((a.pilot / "manifest.json").read_text())
    panel = json.loads(a.panel.read_text())
    environment = load_environment(a.environment)
    environment.check_stamp(pilot["environment"])
    environment.check_stamp(panel["environment"])
    environment.check_meta_decks(default_cards())
    config = DuelConfig.from_environment(environment, max_decisions=4000)
    checkpoint = Path(pilot["checkpoint"])
    if digest(checkpoint) != pilot["checkpoint_sha256"]:
        raise ValueError("pilot checkpoint changed")
    candidate = load_actor(checkpoint)
    candidate.net.to(a.device).eval()
    frozen = a.pilot / "public-reranker.npz"
    with np.load(frozen) as z:
        teacher = ResponseRidge(z["scale"], z["weight"])
    pairs = sample_pairings(len(environment.meta_decks), a.pairings, a.seed)
    slots = pairing_slots([m.deck for m in environment.meta_decks], pairs, a.seed, config)
    manifest = {
        "environment": environment.stamp(),
        "config": asdict(config),
        "pairings": pairs,
        "seed": a.seed,
        "checkpoint": pilot["checkpoint"],
        "checkpoint_sha256": digest(checkpoint),
        "reranker": str(frozen.resolve()),
        "reranker_sha256": digest(frozen),
        "opponents": panel["opponents"],
        "pilot_manifest_sha256": digest(a.pilot / "manifest.json"),
        "device": a.device,
        "core_sha256": digest(_core.__file__),
        "cards_sha256": digest(cards_cdb()),
        "tool_sha256": digest(__file__),
        "probe_tool_sha256": digest(Path(__file__).with_name("response_probe.py")),
        "source_hashes": {str(f.relative_to(Path("src"))): digest(f) for f in sorted(Path("src/ygorl").rglob("*.py"))},
        "submodules": {
            n: {
                "commit": git(third_party() / n, "rev-parse", "HEAD"),
                "dirty": git(third_party() / n, "status", "--porcelain"),
            }
            for n in ("CardScripts", "BabelCDB", "ygopro-core")
        },
        "protocol": "first eligible response of candidate, <=4 turns, <=8 legal actions; one intervention; public only",
        "gate": "difference >=0.02 and pointwise paired CI95 lower >0; fixed sample; no optional stopping",
    }
    manifest = json.loads(json.dumps(manifest))
    path = a.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest mismatch")
    write_json(path, manifest)
    arrays = {name: np.empty((a.pairings, len(panel["opponents"]), 4)) for name in ("control", "reranked")}
    coverage = {}
    for oi, (name, entry) in enumerate(panel["opponents"].items()):
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("opponent changed")
        opponent = load_actor(entry["path"])
        opponent.net.to(a.device).eval()
        if candidate.signature.mismatches(opponent.signature):
            raise ValueError("incompatible policy signatures")
        for label in arrays:
            path = a.out / f"{label}-{name}.json"
            if not path.exists():
                print("playing", label, name, flush=True)
                write_json(
                    path, {"records": play(candidate, opponent, teacher, slots, config, a.device, label == "reranked")}
                )
            records = json.loads(path.read_text())["records"]
            arrays[label][:, oi, :] = np.array(
                [
                    np.nan if r["reason"] == "error" else 0.5 if r["winner"] is None else float(r["winner"] == 0)
                    for r in records
                ]
            ).reshape(a.pairings, 4)
            coverage[f"{label}-{name}"] = {
                "eligible": sum(r["intervention"] is not None for r in records),
                "changed": sum(bool(r["intervention"] and r["intervention"]["changed"]) for r in records),
            }
            print("saved", label, name, flush=True)
    if digest(frozen) != manifest["reranker_sha256"] or digest(checkpoint) != pilot["checkpoint_sha256"]:
        raise ValueError("frozen candidate changed during evaluation")
    for entry in panel["opponents"].values():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("opponent changed during evaluation")
    report = compare_panel(arrays, "control", seed=a.seed)
    report["coverage"] = coverage
    report["opponents"] = list(panel["opponents"])
    write_json(a.out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
