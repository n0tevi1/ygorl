"""M3 (docs/spikes/deck-evolution.md): does a per-card signal predict the leave-one-out truth of M2? For each deck, the
opening-hand effect of each chosen card from the parent's own games (ygorl.build.diagnose) against its leave-one-out
value (parent minus blank-swapped child, paired); Spearman within deck and pooled (z-scored per deck).

Usage: tools/deckevo_m3_signals.py M2.npz [CRITIC_OPEN.json]"""

import json
import sys
from pathlib import Path

import numpy as np

from ygorl.build.diagnose import opening_effects


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    npz = np.load(sys.argv[1])
    summary = json.loads(Path(sys.argv[1]).with_suffix(".json").read_text())
    critic = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    pooled_sig, pooled_true, pooled_crit, pooled_ct = [], [], [], []
    for di, deck in enumerate(summary):
        s = npz[f"d{di}_scores"]  # [decks, pairs, 2]
        hands = npz[f"d{di}_hands"].reshape(-1, 5)
        main = npz[f"d{di}_main"].tolist()
        eff = {e.card: e for e in opening_effects([tuple(h) for h in hands], s[0].reshape(-1), main)}
        loo = {r["card"]: r for r in deck["loo"]}
        cards = [c for c in loo if c in eff]
        sig = np.array([eff[c].effect for c in cards])
        tru = np.array([loo[c]["value"] for c in cards])
        rho = spearman(sig, tru)
        crit = {}
        if critic:
            crit = {int(k): v for k, v in critic[di]["critic_opening"].items() if v is not None}
            cc = [c for c in cards if c in crit]
            cs, ct = np.array([crit[c] for c in cc]), np.array([loo[c]["value"] for c in cc])
            rc = spearman(cs, ct)
            pooled_crit.extend(((cs - cs.mean()) / (cs.std() + 1e-9)).tolist())
            pooled_ct.extend(((ct - ct.mean()) / (ct.std() + 1e-9)).tolist())
        print(
            f"{deck['type']:28s} parent {deck['parent']:.3f}  spearman(opening effect, LOO) {rho:+.2f}  "
            + (f"spearman(critic, LOO) {rc:+.2f}  " if critic else "")
            + f"LOO range {tru.min():+.3f}..{tru.max():+.3f}"
        )
        for c in sorted(cards, key=lambda c: -loo[c]["value"]):
            print(
                f"    {loo[c]['name'][:26]:26s} x{loo[c]['copies']}  LOO {loo[c]['value']:+.3f} (±{loo[c]['ci']:.3f})  "
                f"opening {eff[c].effect:+.3f} (±{1.96 * eff[c].stderr:.3f}, n_in {eff[c].games_in})"
                + (f"  critic {crit[c]:+.3f}" if c in crit else "")
            )
        pooled_sig.extend(((sig - sig.mean()) / (sig.std() + 1e-9)).tolist())
        pooled_true.extend(((tru - tru.mean()) / (tru.std() + 1e-9)).tolist())
    print(
        f"pooled spearman (per-deck z-scores): {spearman(np.array(pooled_sig), np.array(pooled_true)):+.2f} "
        f"over {len(pooled_sig)} cards"
    )
    if critic:
        print(
            f"pooled spearman critic (per-deck z-scores): {spearman(np.array(pooled_crit), np.array(pooled_ct)):+.2f} "
            f"over {len(pooled_crit)} cards"
        )


if __name__ == "__main__":
    main()
