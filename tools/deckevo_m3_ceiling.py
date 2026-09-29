"""M3 noise ceiling: how well can ANY signal rank cards when the LOO truth itself has 1000-pair noise? Split the pairs
in random halves: spearman(LOO half A, LOO half B) is the half truth's reliability r; a noiseless signal would reach
about sqrt(r) against a half, sqrt(2r / (1 + r)) (Spearman-Brown) against the full 1000-pair truth; spearman(opening half A, LOO half B)
is the signal on independent games. Averaged over 200 random splits.

Usage: tools/deckevo_m3_ceiling.py M2.npz"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckevo_m3_signals import spearman  # noqa: E402

from ygorl.build.diagnose import opening_effects  # noqa: E402


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


def line(t, r, s):
    c = np.sqrt(max(r, 0))
    full = np.sqrt(max(2 * r / (1 + r), 0))
    print(f"  {t:28s} r {r:+.2f}  ceiling half {c:.2f} / full {full:.2f}  | opening vs LOO half {s:+.2f}")


print("half-split spearman (1000 pairs -> 500/500)")
for t, r in rows.items():
    r = np.array(r)
    line(t, r[:, 0].mean(), r[:, 1].mean())
line("pooled", *np.mean(pooled, 0))
line("pooled without Exodia", *np.mean(pooled_ex, 0))
