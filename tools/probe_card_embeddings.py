"""How much game-mechanics information does a card representation carry? Linear probes (docs/nets.md「卡片表示探针」).

  uv run --no-sync python tools/probe_card_embeddings.py out/text/e5-base-v2 out/text/bge-large-en-v1.5 ... \\
      [--device cuda] [--out out/text/probe.json]

Every card of the database (alternate artworks dropped) gets labels from the card data and the card scripts:

- ``archetype``: its archetype codes (cdb setcode, low 12 bits; archetypes with >= 8 cards);
- ``category``: the cdb effect-category bits (>= 50 cards);
- ``query``: what its scripts take from where, (action, location) pairs mined by ``ygorl.build.scripts``
  (e.g. to_hand from the Deck, special_summon from the GY; >= 50 cards).

Each representation -- the structured card columns the network already sees, each text directory (card-text vector
concatenated with the mean of the card's effect-string vectors), and structured + text -- gets one linear
multi-label probe per task (logistic, trained on a fixed 80% of the cards, macro ROC-AUC on the other 20%), plus
archetype retrieval: the share of a card's 10 nearest neighbours (cosine) that share an archetype with it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

LOCATIONS = {0x1: "deck", 0x2: "hand", 0x4: "mzone", 0x8: "szone", 0x10: "grave", 0x20: "banished", 0x40: "extra"}


def structured(cards, pws) -> np.ndarray:
    rows = []
    for pw in pws:
        c = cards[pw]
        bits = lambda x, n: [(x >> i) & 1 for i in range(n)]  # noqa: E731
        rows.append([*bits(c.type, 27), *bits(c.attribute, 7), *bits(c.race, 32), *bits(c.link_marker, 9),
                     c.level / 12, c.lscale / 13, c.rscale / 13,
                     math.log1p(max(c.attack, 0)) / 9, math.log1p(max(c.defense, 0)) / 9])  # fmt: skip
    return np.asarray(rows, dtype=np.float32)


def text_features(d: Path, pws) -> np.ndarray:
    card = np.load(d / "card_text.npy")
    cpw = np.load(d / "card_text_passwords.npy")
    eff = np.load(d / "effect_text.npy")
    keys = np.load(d / "effect_text_keys.npy")
    row = {int(p): i for i, p in enumerate(cpw)}
    out_c = np.zeros((len(pws), card.shape[1]), np.float32)
    for k, pw in enumerate(pws):
        if pw in row:
            out_c[k] = card[row[pw]]
    sums = np.zeros((len(pws), eff.shape[1]), np.float32)
    counts = np.zeros(len(pws), np.float32)
    index = {pw: k for k, pw in enumerate(pws)}
    for (pw, _), v in zip(keys.tolist(), eff, strict=True):
        k = index.get(pw)
        if k is not None:
            sums[k] += v
            counts[k] += 1
    return np.concatenate([out_c, sums / np.maximum(counts, 1)[:, None]], 1)


def labels(cards, pws, facts) -> dict[str, tuple[np.ndarray, list[str]]]:
    def table(sets, min_count):
        counts: dict = {}
        for s in sets:
            for x in s:
                counts[x] = counts.get(x, 0) + 1
        keep = sorted(x for x, n in counts.items() if n >= min_count)
        col = {x: i for i, x in enumerate(keep)}
        y = np.zeros((len(sets), len(keep)), np.float32)
        for r, s in enumerate(sets):
            for x in s:
                if x in col:
                    y[r, col[x]] = 1
        return y, [str(x) for x in keep]

    arche = [{s & 0x0FFF for s in cards[pw].setcodes if s} for pw in pws]
    cat = [{i for i in range(64) if cards[pw].category >> i & 1} for pw in pws]
    query = []
    for pw in pws:
        f = facts.get(pw)
        q = set()
        for x in f.queries if f else ():
            for bit, name in LOCATIONS.items():
                if x.locations & bit:
                    q.add(f"{x.action}:{name}")
        query.append(q)
    return {"archetype": table(arche, 8), "category": table(cat, 50), "query": table(query, 50)}


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe(x: np.ndarray, y: np.ndarray, train: np.ndarray, device: str, steps: int = 400) -> float:
    import torch

    xt = torch.as_tensor(x, device=device)
    mu, sd = xt[train].mean(0), xt[train].std(0) + 1e-6
    xt = (xt - mu) / sd
    yt = torch.as_tensor(y, device=device)
    lin = torch.nn.Linear(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    tr = torch.as_tensor(train, device=device)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(lin(xt[tr]), yt[tr]).backward()
        opt.step()
    with torch.no_grad():
        s = lin(xt).cpu().numpy()
    test = ~train
    aucs = [auc(s[test, j], y[test, j]) for j in range(y.shape[1]) if y[test, j].sum() > 0 and y[train, j].sum() > 0]
    return float(np.nanmean(aucs))


def retrieval(x: np.ndarray, arche: np.ndarray, k: int = 10) -> float:
    import torch

    has = arche.sum(1) > 0
    xt = torch.as_tensor(x[has])
    xt = torch.nn.functional.normalize(xt - xt.mean(0), dim=1)
    a = torch.as_tensor(arche[has])
    hits = []
    for s in range(0, len(xt), 2048):
        sim = xt[s : s + 2048] @ xt.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -2
        nn = sim.topk(k, dim=1).indices
        share = (a[s : s + 2048, None, :] * a[nn]).sum(-1) > 0
        hits.append(share.float().mean(1))
    return float(torch.cat(hits).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text_dirs", nargs="*", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("card", "archetype"), default="archetype",
                    help="archetype: whole archetypes go to train or test (a new archetype is what generalization "
                         "needs; the card split lets siblings leak); cards without an archetype split at random")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ygorl import paths
    from ygorl.build.synergy_graph import analyze_scripts
    from ygorl.engine.duel import default_cards

    cards = default_cards()
    pws = [pw for pw in sorted(cards.keys()) if not cards[pw].alias]
    facts = {}
    for d in reversed(paths.script_directories()):  # later directories first so the first match wins
        facts.update(analyze_scripts(d, workers=args.workers))
    tasks = labels(cards, pws, facts)
    rng = np.random.default_rng(args.seed)
    train = rng.random(len(pws)) < 0.8
    if args.split == "archetype":
        codes = sorted({s & 0x0FFF for pw in pws for s in cards[pw].setcodes if s})
        held = {c for c in codes if rng.random() < 0.2}
        for k, pw in enumerate(pws):
            mine = {s & 0x0FFF for s in cards[pw].setcodes if s}
            if mine:
                train[k] = not (mine & held)
    base = structured(cards, pws)
    views = {"structured": base}
    for d in args.text_dirs:
        t = text_features(d, pws)
        views[d.name] = t
        views[f"structured+{d.name}"] = np.concatenate([base, t], 1)
    arche_y = tasks["archetype"][0]
    print(f"{len(pws)} cards ({args.split} split, {int(train.sum())} train); labels: " + ", ".join(f"{k} {v[0].shape[1]}" for k, v in tasks.items()), flush=True)
    header = f"{'representation':40s} {'dim':>5s} " + " ".join(f"{k:>10s}" for k in tasks) + f" {'arch@10':>8s}"
    print(header, flush=True)
    rows = {}
    for name, x in views.items():
        r = {k: probe(x, y, train, args.device) for k, (y, _) in tasks.items()}
        r["archetype_retrieval@10"] = retrieval(x, arche_y)
        r["dim"] = x.shape[1]
        rows[name] = r
        print(f"{name:40s} {x.shape[1]:5d} " + " ".join(f"{r[k]:10.3f}" for k in tasks)
              + f" {r['archetype_retrieval@10']:8.3f}", flush=True)  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps({"cards": len(pws), "labels": {k: v[1] for k, v in tasks.items()},
                                        "results": rows}, indent=1) + "\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
