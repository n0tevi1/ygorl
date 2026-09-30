"""Build the full deck dataset of an environment from the masterduelmeta deck-list history (docs/data.md「牌组数据集」).

  uv run python tools/build_deck_dataset.py md-2026-09        # -> out/deck_dataset/md-2026-09/{decks.npz,meta.json}

Reads the raw history that ``tools/build_deck_corpus.py --fetch`` downloaded (``--raw-dir``, default
``out/deck_corpus/raw``): every list whose cards all map, deduplicated by main + extra counts, with its type, date,
source and legality in the environment (illegal lists kept, flagged). Load it with
``ygorl.data.deck_dataset.load_deck_dataset``. Too large for git: it stays under ``out/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env", help="environment version or directory (legality)")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "out" / "deck_corpus" / "raw")
    p.add_argument("--out", type=Path, default=None, help="output directory (default out/deck_dataset/<version>)")
    args = p.parse_args()

    from ygorl.data import masterduelmeta as mdm
    from ygorl.data.cardmap import CardMapper
    from ygorl.data.deck_dataset import DECKS_FILE, META_FILE, build_deck_dataset
    from ygorl.data.environment import load_environment
    from ygorl.data.fetch import provenance, read_raw
    from ygorl.engine.duel import default_cards

    t0 = time.time()
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    raw = args.raw_dir
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    records = mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, CardMapper(cards), since="")

    def problems(r):
        return [v.code for v in env.validate_deck(mdm.to_deck(r, r.type), cards)]

    source = provenance(raw / mdm.TOP_DECKS_FILE)
    meta = {"environment": env.stamp(), "source": f"masterduelmeta.com {mdm.BASE}/top-decks",
            "retrieved": source.get("retrieved"), "since": source.get("since"),
            "sha256": source.get("sha256")}  # fmt: skip
    ds = build_deck_dataset(records, problems, meta)
    out = ds.save(args.out or ROOT / "out" / "deck_dataset" / env.version)
    ds.meta["stats"]["bytes"] = (out / DECKS_FILE).stat().st_size
    ds.meta["stats"]["build_s"] = round(time.time() - t0, 1)
    (out / META_FILE).write_text(json.dumps(ds.meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(ds.meta["stats"], ensure_ascii=False, indent=1))
    print(f"wrote {len(ds)} lists to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
