"""Comparable panel rows after the E2016 loss: strong = G576 alone; first/second/mean_turns recomputed over all
opponents except E2016 from each summary.json's per-opponent entries (old and new rows alike).
usage: python panel_g576.py ARM[/uNNNN] ...   (ARM = dir name under ab1 or colab_arms; prints one row per checkpoint)"""

import glob, json, sys

D = "/home/ya0guang/Code/ygorl/out/claude-scale"


def row(path):
    s = json.load(open(path))
    opp = {k: v for k, v in s.items() if isinstance(v, dict) and "score" in v and k != "E2016"}
    g = s.get("G576", {}).get("score", float("nan"))

    def w(key):
        num = sum(v[key] * v["games"] for v in opp.values() if key in v)
        den = sum(v["games"] for v in opp.values() if key in v)
        return num / den if den else float("nan")

    return g, w("score"), w("first"), w("second")


for arg in sys.argv[1:]:
    arm, _, u = arg.partition("/u")
    pat = (
        f"{D}/*/{arm}/checkpoints/panel/update_*{u}/summary.json"
        if u
        else f"{D}/*/{arm}/checkpoints/panel/update_*/summary.json"
    )
    for p in sorted(glob.glob(pat)):
        upd = int(p.split("update_")[1].split("/")[0])
        g, m, f, sec = row(p)
        print(f"{arm:12s} u{upd:<5d} G576 {g:.3f}  mean {m:.3f}  first {f:.3f}  second {sec:.3f}")
