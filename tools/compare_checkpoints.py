"""Compare frozen checkpoints on a fixed external panel; retain games and resume completed cells.

Uses the environment's meta decks with uniform ordered pairing sampling (not meta-share weighting).
Every candidate faces every opponent on the same four-way pairings. The control must be one candidate.
Example:
  PYTHONPATH=src python tools/compare_checkpoints.py --environment environments/md-2026-09 \
    --candidate control=/abs/control.pt --candidate teacher=/abs/teacher.pt --control control \
    --opponent anchor=/abs/anchor.pt --pairings 128 --seed 20261002 --out out/compare
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import ygorl
from ygorl import _core
from ygorl.agents.registry import agent_factory
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.eval.agent_matrix import ERROR_REASONS, _play_batched, pairing_slots, sample_pairings
from ygorl.eval.matchup import _deck_hash
from ygorl.eval.paired_panel import compare_panel
from ygorl.paths import cards_cdb, third_party


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def checkpoints(items):
    out = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in out:
            raise ValueError("checkpoints must be distinct NAME=PATH entries; names use letters, numbers, _ or -")
        path = Path(path).resolve(strict=True)
        out[name] = {"path": str(path), "sha256": digest(path)}
    return out


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_cell(path, candidate, opponent, slots):
    data = json.loads(path.read_text())
    if data["candidate"] != candidate or data["opponent"] != opponent or len(data["records"]) != 4 * len(slots):
        raise ValueError(f"mismatched cell: {path}")
    values = []
    for i, r in enumerate(data["records"]):
        m, g = (i % 4) // 2, i % 2
        if (r["pair"], r["seed"], r["first"]) != (i // 4, slots[i // 4][0], int(g != m)):
            raise ValueError(f"mismatched game order: {path} record {i}")
        if r["winner"] not in (0, 1, None):
            raise ValueError(f"invalid winner: {path} record {i}")
        values.append(np.nan if r["reason"] in ERROR_REASONS else
                      (0.5 if r["winner"] is None else float(r["winner"] == 0)))  # fmt: skip
    return np.asarray(values).reshape(len(slots), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--environment", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--opponent", action="append", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--pairings", type=int, default=128)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-decisions", type=int, default=4000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import torch

    torch.set_num_threads(1)
    candidates, opponents = checkpoints(args.candidate), checkpoints(args.opponent)
    if args.control not in candidates or len(candidates) < 2 or args.pairings < 2:
        p.error("need a control, at least two candidates and at least two pairings")
    if set(candidates) & set(opponents) or (
        {v["sha256"] for v in candidates.values()} & {v["sha256"] for v in opponents.values()}
    ):
        p.error("external opponents cannot include a compared checkpoint (including renamed copies)")
    env = load_environment(args.environment)
    env.check_meta_decks(default_cards())
    decks = [md.deck for md in env.meta_decks]
    config = DuelConfig.from_environment(env, max_decisions=args.max_decisions)
    pairs = sample_pairings(len(decks), args.pairings, args.seed)
    slots = pairing_slots(decks, pairs, args.seed, config)
    package = Path(ygorl.__file__).parent
    root = Path(__file__).resolve().parents[1]
    sources = {str(f.relative_to(package)): digest(f) for f in sorted(package.rglob("*.py"))}
    manifest = {
        "format": "ygorl-paired-panel-v1",
        "environment": env.stamp(),
        "config": asdict(config),
        "candidates": candidates,
        "opponents": opponents,
        "control": args.control,
        "deck_sampling": "uniform ordered distinct meta-deck pairs, with replacement",
        "decks": [{"name": md.name, "hash": _deck_hash(md.deck)} for md in env.meta_decks],
        "pairings": pairs,
        "seed": args.seed,
        "device": args.device,
        "sampling": {"greedy": False, "temperature": 1.0},
        "git_commit": git(root, "rev-parse", "HEAD"),
        "package_path": str(package),
        "source_hashes": sources,
        "tool_sha256": digest(__file__),
        "core_sha256": digest(_core.__file__),
        "cards_sha256": digest(cards_cdb()),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "submodules": {
            n: {
                "commit": git(third_party() / n, "rev-parse", "HEAD"),
                "dirty": git(third_party() / n, "status", "--porcelain"),
            }
            for n in ("CardScripts", "BabelCDB", "ygopro-core")
        },
    }
    # Normalize tuples before comparing a manifest loaded from JSON on resume.
    manifest = json.loads(json.dumps(manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("run inputs / implementation changed; use a new output directory")
    else:
        if list(args.out.iterdir()):
            raise ValueError("output directory has files but no manifest; use a new directory")
        write_json(path, manifest)
    scores = {c: np.empty((args.pairings, len(opponents), 4)) for c in candidates}
    for ci, (candidate, ca) in enumerate(candidates.items()):
        for oi, (opponent, op) in enumerate(opponents.items()):
            path = args.out / f"cell-{ci}-{oi}.json"
            if not path.exists():
                print(f"playing {candidate} vs {opponent}: {4 * args.pairings} games", flush=True)
                rows = [(name, f"policy:{entry['path']}", entry["sha256"],
                         agent_factory(f"policy:{entry['path']}"), True)
                        for name, entry in ((candidate, ca), (opponent, op))]  # fmt: skip
                played = _play_batched([(0, 1)], rows, slots, config, args.device)
                if (0, 1) not in played:
                    raise ValueError("this tool requires compatible plain policy checkpoints")
                write_json(path, {"candidate": candidate, "opponent": opponent,
                                  "records": [asdict(r) for r in played[0, 1]]})  # fmt: skip
            scores[candidate][:, oi, :] = read_cell(path, candidate, opponent, slots)
            print(f"saved {candidate} vs {opponent}", flush=True)
    # Catch a checkpoint being overwritten during evaluation rather than labeling mixed weights as frozen.
    for entry in (*candidates.values(), *opponents.values()):
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("checkpoint changed during evaluation; discard this output directory")
    summary = compare_panel(scores, args.control, seed=args.seed)
    summary["environment"] = env.stamp()
    summary["opponents"] = list(opponents)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
