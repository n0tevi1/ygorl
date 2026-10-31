"""Command line: export a checkpoint's actor as the traced graph ygorl._infer loads.

uv run python tools/export_snapshot_actor.py CKPT [--out PATH]   # default: alongside the checkpoint
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    from ygorl.train.checkpoint import load_policy
    from ygorl.train.export_actor import export_actor

    ckpt = Path(args.checkpoint)
    meta = export_actor(load_policy(ckpt), Path(args.out) if args.out else ckpt.parent.parent / "actor_actpath.pt")
    (Path(meta["path"]).with_suffix(".json")).write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
