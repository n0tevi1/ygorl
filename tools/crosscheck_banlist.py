"""Cross-check an environment's Master Duel banlist against sources independent of masterduelmeta.

Usage:
    uv run python tools/crosscheck_banlist.py md-2026-09 [--root DIR] [--save DIR | --from DIR]
        [--yugipedia-page "September 2026 Lists (Master Duel)"] [--date YYYY-MM-DD]

``banlist.lflist.conf`` is generated from masterduelmeta's ``banStatus`` (docs/data.md). This tool compares it,
by canonical password (alternate artworks fold to the original, names go through ``CardMapper``), with:

- **YGOPRODECK**: the banlist API behind https://ygoprodeck.com/banlist/ (``list=Master Duel``; passwords and
  status). Its ``getBanListDates.php`` also tells whether a newer MD list than ``--date`` exists.
- **Yugipedia**: the ``{{Master Duel Limitation status list}}`` on a "<Month> <Year> Lists (Master Duel)" page
  (names and status); the page's infobox gives the effective date and the ``next`` list, if any.

It prints each source's counts and a markdown table of every card on which a source disagrees with the
environment, and exits 1 when there is a disagreement, an unmappable entry or a newer list. ``--save DIR`` keeps
the downloaded responses (with ``.source.json`` provenance sidecars); ``--from DIR`` re-runs offline on them.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ygorl.cards.lflist import load_lflist
from ygorl.data.cardmap import CardMapper
from ygorl.data.fetch import Http, read_raw, write_raw

YGOPRODECK_LIST = "https://ygoprodeck.com/api/banlist/getBanList.php"
YGOPRODECK_DATES = "https://ygoprodeck.com/api/banlist/getBanListDates.php"
YUGIPEDIA_API = "https://yugipedia.com/api.php"
FILES = {"ygoprodeck": "ygoprodeck-md.json", "dates": "ygoprodeck-md-dates.json", "yugipedia": "yugipedia-md.json"}
STATUS = {"forbidden": 0, "limited": 1, "semi-limited": 2, "unlimited": 3}
LABELS = {0: "Forbidden", 1: "Limited", 2: "Semi-Limited", 3: "Unlimited"}


@dataclass(frozen=True)
class Entry:
    name: str
    limit: int
    password: int | None = None


@dataclass
class MappedList:
    limits: dict[int, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    unmapped: list[Entry] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def status_limit(text: str) -> int:
    key = text.strip().casefold().replace(" ", "-")
    if key not in STATUS:
        raise ValueError(f"unknown status {text!r}")
    return STATUS[key]


def parse_ygoprodeck(entries: Any) -> list[Entry]:
    """``getBanList.php`` JSON: ``[{"name", "id", "status_text"}, ...]``."""
    if not isinstance(entries, list):
        raise ValueError("YGOPRODECK banlist: expected a JSON list")
    return [Entry(str(e["name"]), status_limit(e["status_text"]), int(e["id"])) for e in entries]


def latest_ygoprodeck_date(dates: Any, list_type: str = "Master Duel") -> str | None:
    return max((d["date"] for d in dates if d.get("type") == list_type), default=None)


def parse_yugipedia(wikitext: str) -> tuple[list[Entry], dict[str, str]]:
    """Entries of ``{{Master Duel Limitation status list | cards = Name; Status[; Previous status] ...}}`` and the
    infobox fields (``effective_date``, ``prev``, ``next``)."""
    info = {k: v.strip() for k, v in re.findall(r"^\|\s*(\w+)\s*=([^\n]*)$", wikitext, re.M)}
    m = re.search(r"\{\{\s*Master Duel Limitation status list(.*?)\n\}\}", wikitext, re.S)
    if m is None:
        raise ValueError("Yugipedia page: no {{Master Duel Limitation status list}}")
    body = m.group(1).split("| cards", 1)[-1].split("=", 1)[-1]
    out: list[Entry] = []
    for line in body.splitlines():
        parts = [p.strip() for p in html.unescape(line).split(";")]
        if len(parts) >= 2 and parts[0] and not parts[0].startswith("|"):
            out.append(Entry(parts[0], status_limit(parts[1])))
    return out, {k: info.get(k, "") for k in ("effective_date", "prev", "next")}


def map_entries(entries: list[Entry], mapper: CardMapper) -> MappedList:
    """Restricted entries (limit < 3) keyed by canonical password; the stricter one wins on a clash."""
    out = MappedList()
    for e in entries:
        if e.limit >= 3:
            continue
        hit = mapper.map(e.password, e.name)
        if hit is None:
            out.unmapped.append(e)
            continue
        pw = hit.password
        if pw in out.limits and out.limits[pw] != e.limit:
            out.conflicts.append(f"{e.name} ({pw}): {LABELS[out.limits[pw]]} and {LABELS[e.limit]}")
        out.limits[pw] = min(e.limit, out.limits.get(pw, e.limit))
        out.names.setdefault(pw, e.name)
    return out


def diff(ours: Mapping[int, int], sources: Mapping[str, Mapping[int, int]]) -> list[tuple[int, int, dict[str, int]]]:
    """``(password, our limit, {source: its limit})`` for every card some source disagrees on (absent = 3)."""
    rows = []
    for pw in sorted(set(ours).union(*sources.values())):
        mine = ours.get(pw, 3)
        theirs = {name: lim.get(pw, 3) for name, lim in sources.items()}
        if any(v != mine for v in theirs.values()):
            rows.append((pw, mine, theirs))
    return rows


def counts(limits: Mapping[int, int]) -> str:
    return ", ".join(f"{LABELS[k]} {sum(1 for v in limits.values() if v == k)}" for k in (0, 1, 2))


def fetch(page: str, save: Path | None) -> dict[str, Any]:
    http = Http(interval=1.0)
    raw = {
        "ygoprodeck": http.get_json(YGOPRODECK_LIST, {"list": "Master Duel"}),
        "dates": http.get_json(YGOPRODECK_DATES),
        "yugipedia": http.get_json(YUGIPEDIA_API, {"action": "parse", "page": page, "prop": "wikitext",
                                                   "format": "json", "formatversion": "2"}),  # fmt: skip
    }
    if save is not None:
        for (key, data), url in zip(raw.items(), http.urls, strict=True):
            write_raw(save / FILES[key], data, [url])
    return raw


def main(argv: list[str] | None = None, cards: Mapping[int, object] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--root", type=Path, default=Path("environments"))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--save", type=Path, help="keep the downloaded responses in DIR")
    src.add_argument("--from", dest="offline", type=Path, help="read saved responses from DIR instead of fetching")
    ap.add_argument("--yugipedia-page", default="September 2026 Lists (Master Duel)")
    ap.add_argument("--date", help="effective date the environment's list should match (YYYY-MM-DD)")
    args = ap.parse_args(argv)

    ours = load_lflist(args.root / args.version / "banlist.lflist.conf")[0]
    if cards is None:
        from ygorl.cards.cdb import CardDB

        cards = CardDB.load()
    mapper = CardMapper(cards)
    if args.offline:
        raw = {key: read_raw(args.offline / name) for key, name in FILES.items()}
    else:
        raw = fetch(args.yugipedia_page, args.save)

    problems = 0
    yp = raw["yugipedia"]
    if "error" in yp:
        raise SystemExit(f"Yugipedia: {yp['error']}")
    yp_entries, info = parse_yugipedia(yp["parse"]["wikitext"])
    latest = latest_ygoprodeck_date(raw["dates"])
    print(f"environment {args.version}: {ours.name}: {counts(ours.limits)}")
    print(f"Yugipedia {args.yugipedia_page!r}: effective {info['effective_date'] or '?'}, "
          f"next list: {info['next'] or 'none'}")  # fmt: skip
    print(f"YGOPRODECK newest Master Duel list: {latest}")
    if info["next"]:
        print(f"  NEWER LIST on Yugipedia: {info['next']}")
        problems += 1
    if args.date and latest and latest > args.date:
        print(f"  NEWER LIST on YGOPRODECK: {latest} > {args.date}")
        problems += 1

    mapped = {"YGOPRODECK": map_entries(parse_ygoprodeck(raw["ygoprodeck"]), mapper),
              "Yugipedia": map_entries(yp_entries, mapper)}  # fmt: skip
    for name, m in mapped.items():
        print(f"{name}: {counts(m.limits)}")
        for e in m.unmapped:
            print(f"  UNMAPPED: {e.name} ({e.password}, {LABELS[e.limit]})")
        for c in m.conflicts:
            print(f"  CONFLICT: {c}")
        problems += len(m.unmapped) + len(m.conflicts)

    rows = diff(ours.limits, {n: m.limits for n, m in mapped.items()})
    if rows:
        names = mapper.names(pw for pw, _, _ in rows)
        print("\n| password | card | environment | " + " | ".join(mapped) + " |")
        print("|---|---|---|" + "---|" * len(mapped))
        for pw, mine, theirs in rows:
            name = names.get(pw) or next((m.names[pw] for m in mapped.values() if pw in m.names), "?")
            print(f"| {pw} | {name} | {LABELS[mine]} | " + " | ".join(LABELS[v] for v in theirs.values()) + " |")
    print(f"\n{len(rows)} disagreements")
    return 1 if rows or problems else 0


if __name__ == "__main__":
    sys.exit(main())
