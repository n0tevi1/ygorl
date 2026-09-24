"""Build an environment's deck corpus from the masterduelmeta deck-list history (docs/data.md「牌组语料」).

  uv run python tools/build_deck_corpus.py md-2026-09 --fetch            # download, select, write, smoke-test
  uv run python tools/build_deck_corpus.py md-2026-09 --per-type 3       # reselect from the downloaded lists

Writes ``environments/<v>/artifacts/decks/<type>[-k].ydk`` (the old ones are replaced) and
``artifacts/deck_corpus.json`` (one entry per list: type, role, source, date, URL; plus counts). Raw downloads go
to ``--raw-dir`` (default ``out/deck_corpus/raw``, not in git). ``--smoke`` plays every deck once per side against
the environment's first meta deck (greedy vs random) and fails on script errors or unknown messages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env", help="environment version or directory")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "out" / "deck_corpus" / "raw")
    p.add_argument("--fetch", action="store_true", help="download the card table and the deck-list history first")
    p.add_argument("--since", default="2021-01-01", help="oldest deck list to download (ISO date)")
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--min-distance", type=int, default=8, help="cards a further list must differ by")
    p.add_argument("--smoke", action="store_true", help="play every deck once per side (greedy vs random)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    from ygorl.data import masterduelmeta as mdm
    from ygorl.data.cardmap import CardMapper
    from ygorl.data.corpus import select_lists
    from ygorl.data.environment import load_environment
    from ygorl.data.fetch import Http, read_raw
    from ygorl.engine.duel import default_cards

    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    raw = args.raw_dir
    if args.fetch:
        http = Http(mdm.REQUEST_INTERVAL)
        mdm.fetch_cards(raw, http)
        mdm.fetch_top_decks(raw, args.since, http)
        print(f"downloaded {len(http.urls)} pages into {raw}")
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    records = mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, CardMapper(cards), args.since)

    def problems(r):
        return [v.code for v in env.validate_deck(mdm.to_deck(r, r.type), cards)]

    chosen = select_lists(records, problems, per_type=args.per_type, min_distance=args.min_distance)

    out_dir = env.artifact_path("decks", "x").parent
    for old in out_dir.glob("*.ydk"):
        old.unlink()
    entries, used, counts = [], set(), Counter()
    for c in chosen:
        base = re.sub(r"[^a-z0-9]+", "-", c.type.casefold()).strip("-") or "deck"
        counts[base] += 1
        slug = base if counts[base] == 1 else f"{base}-{counts[base]}"
        while slug in used:
            slug += "x"
        used.add(slug)
        r = c.record
        body = mdm.to_deck(r, c.type).to_ydk().split("\n", 1)[1]
        (out_dir / f"{slug}.ydk").write_text(
            f"#created by tools/build_deck_corpus.py: {c.type} ({c.role}), masterduelmeta list of {r.created[:10]}\n"
            f"#source {r.url} ({r.source or 'unknown source'}); {c.legal_lists} legal list(s) of this type\n{body}",
            encoding="utf-8")  # fmt: skip
        entries.append({"file": f"decks/{slug}.ydk", "type": c.type, "role": c.role, "source": r.source,
                        "created": r.created[:10], "url": r.url, "legal_lists": c.legal_lists,
                        "distance": c.distance, "main": len(r.main), "extra": len(r.extra)})  # fmt: skip

    meta_names = {m.name for m in env.meta_decks}
    stats = {"lists": len(records), "types": len({r.type for r in records}), "legal_types": len({c.type for c in chosen}),
             "decks": len(entries), "meta_types_in_corpus": len(meta_names & {c.type for c in chosen}),
             "meta_types": len(meta_names),
             "by_source": dict(Counter(e["source"] or "unknown" for e in entries).most_common())}  # fmt: skip
    smoke = None
    if args.smoke:
        from ygorl.agents.registry import AgentSpec
        from ygorl.cards.ydk import load_ydk
        from ygorl.engine.duel import DuelConfig
        from ygorl.eval.arena import Arena

        opponent = env.meta_decks[0].deck
        decks = [load_ydk(env.artifacts_dir / e["file"]) for e in entries]
        arena = Arena(AgentSpec("greedy"), AgentSpec("random"), env=env, config=DuelConfig.from_environment(env),
                      workers=args.workers)  # fmt: skip
        reports = arena.run_many([(d, opponent) for d in decks], 1)
        bad = []
        for e, rep in zip(entries, reports, strict=True):
            n_script = sum(g.script_errors for g in rep.records)
            if rep.errors or rep.unknown_messages or n_script:
                bad.append({"file": e["file"], "errors": rep.errors, "unknown_messages": rep.unknown_messages,
                            "script_errors": n_script})  # fmt: skip
        smoke = {"games": sum(r.games for r in reports), "opponent": env.meta_decks[0].name, "problems": bad}
        stats["smoke"] = smoke
    doc = {"source": f"masterduelmeta.com {mdm.BASE}/top-decks", "since": args.since,
           "method": "per deck type: medoid of its newest distinct lists legal in this environment, then farthest-point "
                     f"lists >= {args.min_distance} cards apart, up to {args.per_type} (docs/data.md)",
           "stats": stats, "decks": entries}  # fmt: skip
    path = env.artifact_path("deck_corpus.json")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"wrote {len(entries)} decks to {out_dir} and {path}")
    return 1 if smoke and smoke["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
