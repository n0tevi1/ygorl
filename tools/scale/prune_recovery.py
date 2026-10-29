"""Prune a recovery-line file (fire records of handtrap_labels_graded.py --recovery-out) to the lines LineLibrary.load
would keep: per (deck, hand, card), in file order, the first line per window identity that replays as a recovery
line (lines.recovery_line against the hand's plain record); the other lines are dropped, records left without lines
are dropped. LineLibrary.load(recovery=[pruned]) then joins the same lines but replays a fraction of them (the
Mulcharmy Fuwalos / Maxx "C" records have ~27 lines per record, one window often holding several).
  uv run --no-sync python out/claude-scale/prune_recovery.py PLAIN.jsonl FIRE.jsonl --out PRUNED.jsonl [--workers 2]
YGORL_ROOT picks the checkout whose src/ is imported (as handtrap_labels_graded.py).
"""

import argparse, json, multiprocessing, os, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("YGORL_ROOT", "/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review"))
sys.path.insert(0, str(ROOT / "src"))

_CTX = {}


def _init():
    from ygorl.cards.cdb import CardVocab
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    cards = default_cards()
    _CTX.update(env=load_environment("md-2026-09"), cards=cards, vocab=CardVocab.from_db(cards))


def work(item):
    plain_json, fire_jsons = item
    from ygorl.solver import Demonstration
    from ygorl.solver.demo import DemoError
    from ygorl.train.lines import recovery_line

    env, cards, vocab = _CTX["env"], _CTX["cards"], _CTX["vocab"]
    base = Demonstration.from_json(plain_json)
    seen = set()
    out, tried, kept = [], 0, 0
    for fj in fire_jsons:
        fire = Demonstration.from_json(fj)
        keep = []
        for k in range(len(fire.lines)):
            tried += 1
            try:
                rec = recovery_line(fire, k, base, vocab, cards=cards, env=env)
            except (DemoError, ValueError):
                continue
            if rec.key in seen:
                continue
            seen.add(rec.key)
            keep.append(k)
        if keep:
            fj = dict(fj)
            fj["lines"] = [fj["lines"][k] for k in keep]
            out.append(fj)
            kept += len(keep)
    return out, tried, kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plain")
    p.add_argument("fire")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    plain = {}
    for l in open(a.plain):
        r = json.loads(l)
        if r["variant"] == "plain" and r["status"] == "solved" and r["lines"]:
            plain.setdefault((r["deck"]["name"], r["hand_index"]), r)
    groups = {}
    for l in open(a.fire):
        r = json.loads(l)
        if r["variant"] != "fire" or r["status"] != "solved" or not r["lines"]:
            continue
        key = (r["deck"]["name"], r["hand_index"])
        if key in plain:
            groups.setdefault((*key, r["fire"]), []).append(r)
    items = [(plain[k[:2]], v) for k, v in groups.items()]
    t0 = time.time()
    tried = kept = 0
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(a.out, "w") as f:
        for n, (recs, t, k) in enumerate(pool.imap(work, items, chunksize=4), 1):
            tried += t
            kept += k
            f.write("".join(json.dumps(r) + "\n" for r in recs))
            if n % 200 == 0:
                print(f"[{n}/{len(items)} {time.time() - t0:.0f}s] lines tried {tried} kept {kept}", flush=True)
    print(f"PRUNE-DONE groups {len(items)} lines tried {tried} kept {kept} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
