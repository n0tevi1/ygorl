"""Cross-check tests/data/card_samples.json and our cdb decoding against YGOPRODECK.

Usage: uv run python tools/check_ygoprodeck.py

Needs network access to db.ygoprodeck.com (API v7). Prints one line per
mismatch and exits non-zero if any field disagrees.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from ygorl.cards.cdb import CardDB

ROOT = Path(__file__).resolve().parents[1]
API = "https://db.ygoprodeck.com/api/v7/cardinfo.php?id="
LINK = {"Top": "top", "Bottom": "bottom", "Left": "left", "Right": "right", "Top-Left": "top_left",
        "Top-Right": "top_right", "Bottom-Left": "bottom_left", "Bottom-Right": "bottom_right"}  # fmt: skip


def main() -> int:
    samples = json.loads((ROOT / "tests" / "data" / "card_samples.json").read_text())["cards"]
    ids = ",".join(str(s["password"]) for s in samples)
    with urllib.request.urlopen(API + ids, timeout=60) as resp:
        remote = {c["id"]: c for c in json.load(resp)["data"]}
    db = CardDB.load()
    bad = 0
    for s in samples:
        r, card = remote.get(s["password"]), db[s["password"]]
        if r is None:
            print(f"{s['password']}: not found on YGOPRODECK")
            bad += 1
            continue
        checks = {"name": (card.name, r["name"])}
        if card.is_monster:
            checks["attack"] = (card.attack, r.get("atk"))
            checks["attribute"] = (card.attribute_names[0].upper(), r.get("attribute"))
            if card.is_link:
                checks["link"] = (card.link, r.get("linkval"))
                checks["markers"] = (sorted(card.link_marker_names), sorted(LINK[m] for m in r.get("linkmarkers", [])))
            else:
                checks["defense"] = (card.defense, r.get("def"))
                checks["level"] = (card.level, r.get("level"))
            if card.is_pendulum:
                checks["scale"] = (card.lscale, r.get("scale"))
        for field, (ours, theirs) in checks.items():
            if ours != theirs:
                print(f"{s['password']} {s['name']}: {field} ours={ours!r} ygoprodeck={theirs!r}")
                bad += 1
    print("OK" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
