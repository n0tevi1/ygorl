"""M3 noise ceiling: how well can ANY signal rank cards when the LOO truth itself has 1000-pair noise? Split the pairs
in random halves: spearman(LOO half A, LOO half B) is the truth's own reliability; spearman(opening half A, LOO half B)
is the signal on independent games. Averaged over 200 random splits.

Usage: tools/deckevo_m3_ceiling.py M2.npz"""

import json
import sys
from pathlib import Path

import numpy as np

from ygorl.build.diagnose import opening_effects


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


npz = np.load(sys.argv[1])
summary = json.loads(Path(sys.argv[1]).with_suffix(".json").read_text())
rng = np.random.default_rng(0)
rows = {d["type"]: [] for d in summary}
pooled, pooled_ex = [], []
for _ in range(200):
    pl = {"tt": ([], []), "st": ([], [])}
    ple = {"tt": ([], []), "st": ([], [])}
    for di, deck in enumerate(summary):
        s = npz[f"d{di}_scores"]
        n = s.shape[1]
        hands = npz[f"d{di}_hands"]
        main = npz[f"d{di}_main"].tolist()
        chosen = npz[f"d{di}_chosen"].tolist()
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2 :]

        def loo(idx, s=s, k=len(chosen)):
            return np.array([np.nanmean(s[0, idx].mean(-1) - s[c, idx].mean(-1)) for c in range(1, k + 1)])

        la, lb = loo(A), loo(B)
        eff = {
            e.card: e.effect
            for e in opening_effects([tuple(h) for h in hands[A].reshape(-1, 5)], s[0, A].reshape(-1), main)
        }
        oa = np.array([eff[c] for c in chosen])
        rows[deck["type"]].append((spearman(la, lb), spearman(oa, lb)))
        for tgt in (pl, ple) if deck["type"] != "Exodia" else (pl,):
            tgt["tt"][0].extend(z(la))
            tgt["tt"][1].extend(z(lb))
            tgt["st"][0].extend(z(oa))
            tgt["st"][1].extend(z(lb))
    pooled.append([spearman(np.array(pl[k][0]), np.array(pl[k][1])) for k in ("tt", "st")])
    pooled_ex.append([spearman(np.array(ple[k][0]), np.array(ple[k][1])) for k in ("tt", "st")])
print("half-split spearman (1000 pairs -> 500/500):   LOO vs LOO | opening vs LOO")
for t, r in rows.items():
    r = np.array(r)
    print(f"  {t:28s} {r[:, 0].mean():+.2f}        | {r[:, 1].mean():+.2f}")
print(f"  pooled                       {np.mean(pooled, 0)[0]:+.2f}        | {np.mean(pooled, 0)[1]:+.2f}")
print(f"  pooled without Exodia        {np.mean(pooled_ex, 0)[0]:+.2f}        | {np.mean(pooled_ex, 0)[1]:+.2f}")
