"""Which demonstrated turn-1 steps lose probability across checkpoints? (strength diagnosis, #83)

Replays solver demonstration lines (``ygorl.train.bc.build_dataset``: the deck under study's non-forced
decisions, encoded with each checkpoint's own vocab) and scores the demonstrated action under every checkpoint.
Steps are grouped by type: idle-command plays (summon / special summon / activate / set), idle-command passes
(end / battle phase), card selection (select / unselect / finish in SELECT_CARD, SELECT_UNSELECT_CARD, tribute,
sum), chain responses (chain / pass in SELECT_CHAIN), yes/no and option prompts, place / position, other; and by
position in the line (thirds). Reports per group: mean p(demo action), geometric mean, top-1 accuracy, entropy
and max probability.

Usage: tools/demo_step_probs.py DEMOS.jsonl CKPT [CKPT ...] [--out rows.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ygorl.engine import constants as C
from ygorl.env.encoding import ACTION_KINDS

PLAY = {"summon", "spsummon", "activate", "mset", "sset", "reposition"}
SELECT_TYPES = {C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM}


def step_type(dtype: int, kind: str) -> str:
    if dtype == C.MSG_SELECT_IDLECMD:
        return (
            "idle:play"
            if kind in PLAY
            else "idle:pass(end/bp)"
            if kind in ("end_phase", "battle_phase")
            else "idle:other"
        )
    if dtype in SELECT_TYPES:
        return "select:finish" if kind == "finish" else "select:card"
    if dtype == C.MSG_SELECT_CHAIN:
        return "chain:activate" if kind == "chain" else "chain:pass"
    if dtype in (C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION):
        return "yesno/option"
    if dtype in (C.MSG_SELECT_PLACE, C.MSG_SELECT_POSITION, C.MSG_SELECT_DISFIELD):
        return "place/position"
    return "other"


def score(ckpt: str, demos, threads: int) -> list[dict]:
    from ygorl.train.bc import build_dataset
    from ygorl.train.checkpoint import load_actor

    pol = load_actor(ckpt)
    data = build_dataset(demos, pol.vocab, event_length=pol.event_length)
    net = pol.net.eval()
    rows = []
    with torch.no_grad():
        for b in range(0, len(data), 256):
            idx = np.arange(b, min(len(data), b + 256))
            obs, target = data.batch(idx)
            logp = net(obs).log_probs()
            p = logp.exp()
            mask = obs["action_mask"].bool()
            ent = -(torch.where(mask, p * logp, torch.zeros(()))).sum(-1)
            for j, i in enumerate(idx):
                a = int(target[j])
                kind = ACTION_KINDS[int(obs["actions"][j, a, 0]) - 1]
                dtype = int(obs["globals"][j, 18])
                m = data.meta[i]
                rows.append({"key": (m["deck"], m["hand_index"], m["variant"], m["line"], m["step"]),
                             "type": step_type(dtype, kind), "kind": kind, "n": int(mask[j].sum()),
                             "turn": int(obs["globals"][j, 3]), "p": float(p[j, a]), "top1": int(logp[j].argmax() == a),
                             "ent": float(ent[j]), "pmax": float(p[j].max()), "solver": m["solver"],
                             "argmax_kind": ACTION_KINDS[int(obs["actions"][j, int(logp[j].argmax()), 0]) - 1]})  # fmt: skip
    # position in the line, thirds of the kept steps of each line
    by_line = defaultdict(list)
    for r in rows:
        by_line[r["key"][:4]].append(r)
    for rs in by_line.values():
        rs.sort(key=lambda r: r["key"][4])
        for k, r in enumerate(rs):
            r["third"] = ("early", "mid", "late")[min(2, 3 * k // len(rs))]
    return rows


def table(name: str, results: dict[str, list[dict]], group) -> None:
    names = list(results)
    groups = sorted({group(r) for r in next(iter(results.values()))})
    print(f"\n== by {name}: mean p(demo) / geo-mean p / top-1 / max-p  [n rows]")
    print(f"{'group':22s} {'n':>5s} " + " ".join(f"{Path(c).parent.name[:14]:>30s}" for c in names))
    for g in groups:
        cells = []
        n = 0
        for c in names:
            rs = [r for r in results[c] if group(r) == g]
            n = len(rs)
            p = np.array([r["p"] for r in rs])
            cells.append(f"{p.mean():.3f}/{np.exp(np.log(np.maximum(p, 1e-9)).mean()):.3f}/"
                         f"{np.mean([r['top1'] for r in rs]):.2f}/{np.mean([r['pmax'] for r in rs]):.2f}")  # fmt: skip
        print(f"{str(g):22s} {n:5d} " + " ".join(f"{c:>30s}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demos")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from ygorl.solver import read_jsonl

    demos = list(read_jsonl(Path(a.demos)))
    results = {c: score(c, demos, a.threads) for c in a.checkpoints}
    lens = {c: len(r) for c, r in results.items()}
    if len(set(lens.values())) != 1:
        raise SystemExit(f"the checkpoints kept different steps: {lens}")
    table("all", results, lambda r: "all")
    table("step type", results, lambda r: r["type"])
    table("position in line", results, lambda r: r["third"])
    table(
        "legal choices",
        results,
        lambda r: "2" if r["n"] == 2 else "3-5" if r["n"] <= 5 else "6-15" if r["n"] <= 15 else "16+",
    )
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for c, rs in results.items():
                for r in rs:
                    f.write(json.dumps({"ckpt": c, **r}) + "\n")


if __name__ == "__main__":
    main()
