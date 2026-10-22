"""Build the Master Duel card id -> password table for the memory reader (design 10, T7.6).

Usage: uv run python tools/build_md_cid.py [md-2026-09] [--raw cardinfo.json]

MD's internal card id is the Konami card database id, which YGOPRODECK lists as ``misc_info.konami_id``.
Fetches ``cardinfo.php?misc=yes`` (every card, one request) unless ``--raw`` gives a saved response, maps each
card to a canonical BabelCDB password and writes ``environments/<version>/artifacts/md_cid.json`` (stamped with
``Environment.stamp()``). Unmapped cards are listed in the file, never dropped silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ygorl.data.cardmap import CardMapper
from ygorl.data.environment import load_environment
from ygorl.data.fetch import Http, now_utc
from ygorl.data.ygoprodeck import API, REQUEST_INTERVAL, md_card_ids
from ygorl.engine.duel import default_cards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("env", nargs="?", default="md-2026-09")
    ap.add_argument("--raw", type=Path, help="saved cardinfo.php?misc=yes response instead of fetching")
    args = ap.parse_args()
    env = load_environment(args.env)
    if args.raw:
        data, source = json.loads(args.raw.read_text(encoding="utf-8")), str(args.raw)
    else:
        http = Http(REQUEST_INTERVAL)
        data = http.get_json(API, {"misc": "yes"})
        source = http.urls[-1]
    ids, unmapped = md_card_ids(data, CardMapper(default_cards()))
    out = env.artifact_path("md_cid.json")
    doc = {**env.stamp(), "source": source, "fetched": now_utc(), "cards": {str(k): v for k, v in sorted(ids.items())},
           "unmapped": unmapped}  # fmt: skip
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{out}: {len(ids)} card ids, {len(unmapped)} unmapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
