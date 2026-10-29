"""print one line per panel summary: arm update mean strong first second gap turns teacher(first/second)"""

import json
import sys
import glob

for pat in sys.argv[1:]:
    for p in sorted(glob.glob(pat)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        o = d.get("overall", {})
        t = d.get("teacher", {})
        parts = p.split("/")
        arm = parts[-5]
        upd = parts[-2][-4:]
        print(
            f"{arm:22s} u{upd} mean {d['mean']:.3f} strong {d.get('mean_strong', float('nan')):.3f} first {o.get('first', float('nan')):.3f} second {o.get('second', float('nan')):.3f} gap {o.get('second', 0) - o.get('first', 0):+.3f} turns {o.get('mean_turns', float('nan')):.2f} teacher {t.get('score', float('nan')):.3f} (1st {t.get('first', float('nan')):.3f})"
        )
