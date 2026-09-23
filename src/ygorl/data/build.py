"""One-command build of a Master Duel environment ``environments/md-<YYYY>-<MM>/`` (T5.1).

Steps (docs/data.md):

1. **fetch** (unless ``offline``): every raw file that is missing (or all, with ``refresh``) is downloaded
   into ``<env>/raw/`` by the source's fetcher; raw files are git-ignored and can be replaced by hand.
2. **parse**: pool (YGOPRODECK), banlist (masterduelmeta ban statuses + ``review/banlist-overrides.lflist.conf``),
   meta decks (masterduelmeta top decks since the last banlist update: share per deck type, medoid list),
   archetype relations (Yugipedia).
3. **write** ``environment.json``, ``pool.json``, ``banlist.lflist.conf``, ``meta.json``, ``meta/*.ydk``,
   ``relations.json`` and the review report ``review/report.md``.
4. **validate** with :func:`ygorl.data.load_environment` and the card database (meta decks must be legal).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from ygorl.cards.legality import DeckRules, validate_deck
from ygorl.cards.lflist import Banlist, load_lflist, parse_lflist, select
from ygorl.data import masterduelmeta as mdm
from ygorl.data import yugipedia, ygoprodeck
from ygorl.data.cardmap import CardMapper
from ygorl.data.environment import BANLIST, MANIFEST, META, META_DIR, POOL, VERSION_RE, load_environment
from ygorl.data.fetch import Http, provenance, read_raw, sidecar

RAW_DIR = "raw"
REVIEW_DIR = "review"
REPORT = "report.md"
OVERRIDES = "banlist-overrides.lflist.conf"
CROSSCHECK = "banlist-crosscheck.md"  # hand-written record of a cross-check against other sources
RELATIONS = "relations.json"
MD_VERSION_RE = re.compile(r"^md-(\d{4})-(\d{2})(?:-[a-z0-9.]+)?$")
LIMIT_LABELS = {0: "禁止", 1: "限制", 2: "准限制", 3: "无限制"}
LIMIT_KEYS = {0: "forbidden", 1: "limited", 2: "semi_limited"}


class BuildError(RuntimeError):
    """The environment cannot be built (missing raw file offline, bad version, invalid result)."""


@dataclass
class BuildOptions:
    version: str
    root: Path
    raw_dir: Path | None = None  # default <root>/<version>/raw
    offline: bool = False  # parse existing raw files only
    refresh: bool = False  # download every raw file again
    # None = the value recorded by the previous build of this version (environment.json sources.build), else
    # the default: last MD banlist update date, 0.01, 20
    since: str | None = None  # meta window start (ISO date)
    min_share: float | None = None
    max_decks: int | None = None
    relations: bool = True
    reviewed_by: str | None = None

    @property
    def out(self) -> Path:
        return self.root / self.version

    @property
    def raw(self) -> Path:
        return self.raw_dir if self.raw_dir is not None else self.out / RAW_DIR


@dataclass
class BuildResult:
    path: Path
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


DEFAULT_MIN_SHARE = 0.01
DEFAULT_MAX_DECKS = 20


def resolve(opts: BuildOptions) -> BuildOptions:
    """Fill unset window / selection options from the previous build of this version, then defaults."""
    previous = (_read_json(opts.out / MANIFEST).get("sources") or {}).get("build") or {}
    if not isinstance(previous, dict):
        previous = {}
    return replace(opts, since=opts.since or previous.get("since"),
                   min_share=_first(opts.min_share, previous.get("min_share"), DEFAULT_MIN_SHARE),
                   max_decks=_first(opts.max_decks, previous.get("max_decks"), DEFAULT_MAX_DECKS))  # fmt: skip


def _first(*values):
    return next(v for v in values if v is not None)


def banlist_name(version: str) -> str:
    m = MD_VERSION_RE.match(version)
    return f"{m.group(1)}.{m.group(2)} MD" if m else version


# --------------------------------------------------------------------------- fetch


def fetch(opts: BuildOptions, log: Callable[[str], None] = print) -> None:
    """Download the raw files the build needs (skips existing ones unless ``refresh``)."""
    raw = opts.raw
    mdm_http, wiki_http = Http(mdm.REQUEST_INTERVAL), Http(yugipedia.REQUEST_INTERVAL)  # one rate limit per site

    def need(*rel: Path) -> bool:
        return opts.refresh or not all((raw / r).is_file() for r in rel)

    if need(ygoprodeck.RAW_FILE):
        log(f"fetch  YGOPRODECK Master Duel card list -> {raw / ygoprodeck.RAW_FILE}")
        ygoprodeck.fetch_md_cards(raw)
    if need(mdm.CARDS_FILE, mdm.BANLIST_FILE):
        log(f"fetch  masterduelmeta cards and ban statuses -> {raw / mdm.CARDS_FILE}")
        mdm.fetch_cards(raw, mdm_http)
    if need(mdm.ARTICLES_FILE):
        log(f"fetch  masterduelmeta articles (banlist update date) -> {raw / mdm.ARTICLES_FILE}")
        mdm.fetch_articles(raw, mdm_http)
    since = meta_since(opts)
    decks = raw / mdm.TOP_DECKS_FILE
    fetched_since = _sidecar(decks).get("since") if decks.is_file() else None
    if opts.refresh or not decks.is_file() or (fetched_since and fetched_since > since):
        log(f"fetch  masterduelmeta top decks since {since} -> {decks}")
        mdm.fetch_top_decks(raw, since, mdm_http)
    if opts.relations:
        for key in yugipedia.PROPERTIES:
            if need(yugipedia.raw_file(key)):
                log(f"fetch  Yugipedia '{yugipedia.PROPERTIES[key]}' -> {raw / yugipedia.raw_file(key)}")
                yugipedia.fetch_property(raw, key, wiki_http)


def _sidecar(path: Path) -> dict[str, Any]:
    side = sidecar(path)
    return json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}


def meta_since(opts: BuildOptions) -> str:
    """Start of the meta window: ``--since``, else the last banlist update, else the version's first day."""
    if opts.since:
        return opts.since
    articles = opts.raw / mdm.ARTICLES_FILE
    if articles.is_file():
        update = mdm.last_banlist_update(read_raw(articles))
        if update:
            return update["date"]
    m = MD_VERSION_RE.match(opts.version)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    raise BuildError("cannot tell where the meta window starts: pass --since YYYY-MM-DD")


# --------------------------------------------------------------------------- build


def build(opts: BuildOptions, cards: Mapping[int, Any] | None = None, log: Callable[[str], None] = print) -> BuildResult:
    if not VERSION_RE.match(opts.version) or not MD_VERSION_RE.match(opts.version):
        raise BuildError(f"version {opts.version!r} must look like md-YYYY-MM[-revision] (only Master Duel "
                         "environments can be built so far)")  # fmt: skip
    opts = resolve(opts)
    if not opts.offline:
        fetch(opts, log)
    if cards is None:
        from ygorl.cards.cdb import CardDB

        cards = CardDB.load()
    raw = opts.raw
    for rel in _required_raw(opts):
        if not (raw / rel).is_file():
            raise BuildError(f"raw file missing: {raw / rel} (run without --offline, or put a hand-made file there)")

    mapper = CardMapper(cards)
    result = BuildResult(opts.out)
    out = opts.out

    # pool
    pool_res = ygoprodeck.parse_md_pool(read_raw(raw / ygoprodeck.RAW_FILE), mapper)
    pool = frozenset(pool_res.passwords)
    if not pool:
        raise BuildError("the card pool is empty")

    # banlist (+ human overrides)
    name = banlist_name(opts.version)
    ban_res = mdm.parse_banlist(read_raw(raw / mdm.BANLIST_FILE), mapper, name)
    overrides = _load_overrides(out / REVIEW_DIR / OVERRIDES)
    limits = dict(ban_res.banlist.limits)
    for pw, limit in overrides.items():
        pw = mapper.canonical(pw)
        if limit >= 3:
            limits.pop(pw, None)
        else:
            limits[pw] = limit
    banlist = Banlist(name, limits)

    # meta decks
    since = meta_since(opts)
    update = mdm.last_banlist_update(read_raw(raw / mdm.ARTICLES_FILE))
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    records = mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, mapper, since)
    rules = DeckRules()

    memo: dict[mdm.DeckRecord, list[str]] = {}

    def problems(r: mdm.DeckRecord) -> list[str]:
        if r not in memo:
            deck = mdm.to_deck(r, r.type)
            memo[r] = [v.code for v in validate_deck(deck, cards=cards, banlist=banlist, pool=pool, rules=rules)]
        return memo[r]

    summaries = mdm.summarize_types(records, problems)
    chosen = [s for s in summaries if s.share >= opts.min_share and s.chosen is not None][: opts.max_decks]
    skipped = [s for s in summaries if s.share >= opts.min_share and s.chosen is None]

    # relations
    rel_data = None
    if opts.relations:
        parsed, rel_unmapped = {}, {}
        for key in yugipedia.PROPERTIES:
            parsed[key], rel_unmapped[key] = yugipedia.parse_property(read_raw(raw / yugipedia.raw_file(key)), mapper, key)
        rel_data = {"relations": yugipedia.relations(parsed, pool), "unmapped": rel_unmapped,
                    "cards": len({p for table in parsed.values() for p in table if p in pool})}  # fmt: skip

    # write
    prov = {rel.as_posix(): provenance(raw / rel) for rel in _required_raw(opts)}
    old_manifest = _read_json(out / MANIFEST)
    old_banlist = (out / BANLIST).read_text(encoding="utf-8") if (out / BANLIST).is_file() else None
    retrieved = _date(prov[ygoprodeck.RAW_FILE.as_posix()])
    meta_retrieved = _date(prov[mdm.TOP_DECKS_FILE.as_posix()])
    out.mkdir(parents=True, exist_ok=True)

    banlist_text = _banlist_text(banlist, cards, ban_res, prov[mdm.BANLIST_FILE.as_posix()], update)
    (out / BANLIST).write_text(banlist_text, encoding="utf-8")
    pool_json = {"source": f"YGOPRODECK API v7 {ygoprodeck.API}?format={ygoprodeck.FORMAT}", "retrieved": retrieved,
                 "note": "canonical BabelCDB passwords; alternate artworks count as their original",
                 "cards": sorted(pool)}  # fmt: skip
    (out / POOL).write_text(json.dumps(pool_json, indent=0) + "\n", encoding="utf-8")

    meta_dir = out / META_DIR
    meta_dir.mkdir(exist_ok=True)
    for old in meta_dir.glob("*.ydk"):
        old.unlink()
    decks_json, used = [], set()
    for s in chosen:
        slug = _slug(s.name, used)
        rel = f"{META_DIR}/{slug}.ydk"
        (out / rel).write_text(_ydk(s, since), encoding="utf-8")
        decks_json.append({"name": s.name, "file": rel, "share": _floor(s.share), "lists": s.decks,
                           "legal_lists": s.legal, "representative": s.chosen.url})  # fmt: skip
    total_weight = sum(s.weight for s in summaries)
    meta_json = {"source": f"masterduelmeta.com {mdm.BASE}/top-decks", "retrieved": meta_retrieved,
                 "window": {"since": since, "until": meta_retrieved},
                 "method": "share = statsWeight-weighted fraction of the window's deck lists of this deck type; "
                           "list = the type's medoid among its lists that are legal here (docs/data.md)",
                 "lists": len(records), "types": len(summaries), "min_share": opts.min_share, "decks": decks_json}  # fmt: skip
    (out / META).write_text(json.dumps(meta_json, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if rel_data is not None:
        _write_relations(out / RELATIONS, rel_data["relations"], prov)

    review = _review_status(opts, old_manifest, old_banlist, banlist_text)
    manifest = {
        "version": opts.version,
        "format": "md",
        "description": f"Master Duel: YGOPRODECK card pool and masterduelmeta banlist retrieved {retrieved}, "
                       f"meta deck lists {since} to {meta_retrieved}",
        "rules": {"mode": "MR5", "extra_flags": []},
        "player": {"starting_lp": 8000, "starting_hand": 5, "draw_per_turn": 1},
        "deck": {"main_min": 40, "main_max": 60, "extra_max": 15, "side_max": 15, "max_copies": 3},
        "banlist_name": name,
        "sources": {"raw": {k: _compact(v) for k, v in prov.items()}, "banlist_update": update,
                    "meta_window": {"since": since, "until": meta_retrieved},
                    "build": {"command": "ygorl env build (docs/data.md)", "since": since,
                              "min_share": opts.min_share, "max_decks": opts.max_decks}},
        "review": review,
    }  # fmt: skip
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    stats = {
        "pool": len(pool),
        "pool_source_cards": pool_res.source_cards,
        "pool_tokens_skipped": pool_res.tokens,
        "pool_by_name": pool_res.by_name,
        "pool_unmapped": pool_res.unmapped,
        "banlist": {name: sum(1 for v in banlist.limits.values() if v == k) for k, name in LIMIT_KEYS.items()},
        "banlist_unmapped": ban_res.unmapped,
        "banlist_merged": ban_res.merged,
        "banlist_by_name": ban_res.by_name,
        "banlist_conflicts": ban_res.conflicts,
        "banlist_overrides": len(overrides),
        "banlist_outside_pool": sorted(p for p in banlist.limits if p not in pool),
        "meta_lists": len(records),
        "meta_types": len(summaries),
        "meta_weight": total_weight,
        "meta_counted": sum(1 for r in records if r.weight > 0),
        "meta_decks": len(chosen),
        "meta_share": round(sum(d["share"] for d in decks_json), 4),
        "meta_unmapped": _unmapped_deck_cards(records),
        "meta_skipped": [(s.name, round(s.share, 4), dict(s.problems)) for s in skipped],
        "relations": rel_data["cards"] if rel_data else None,
        "meta_illegal_by_day": _illegal_by_day(records, problems),
        "relations_unmapped": {k: len(v) for k, v in rel_data["unmapped"].items()} if rel_data else None,
        "review": review["banlist"]["status"],
    }
    result.stats = stats
    (out / REVIEW_DIR).mkdir(exist_ok=True)
    (out / REVIEW_DIR / REPORT).write_text(
        _report(opts, stats, summaries, chosen, banlist, ban_res, overrides, cards, pool, update, since), encoding="utf-8"
    )

    # validate: format, then legality of every meta deck under the new pool / banlist
    try:
        load_environment(out, cards=cards)
    except ValueError as exc:
        raise BuildError(f"the built environment does not validate:\n{exc}") from None
    if stats["pool_unmapped"]:
        result.warnings.append(f"{len(stats['pool_unmapped'])} pool card(s) not in BabelCDB (see {REVIEW_DIR}/{REPORT})")
    if stats["banlist_unmapped"]:
        result.warnings.append(f"{len(stats['banlist_unmapped'])} banlist card(s) not in BabelCDB")
    late = _late_start(stats["meta_illegal_by_day"])
    if late:
        result.warnings.append(f"every list illegal under the new banlist predates {late}: the banlist probably took "
                               f"effect then; rebuild with --since {late} (see {REVIEW_DIR}/{REPORT})")  # fmt: skip
    if review["banlist"]["status"] != "reviewed":
        result.warnings.append(f"banlist not reviewed yet: check {REVIEW_DIR}/{REPORT} against the in-game list")
    return result


def _required_raw(opts: BuildOptions) -> list[Path]:
    files = [ygoprodeck.RAW_FILE, mdm.CARDS_FILE, mdm.BANLIST_FILE, mdm.ARTICLES_FILE, mdm.TOP_DECKS_FILE]
    if opts.relations:
        files += [yugipedia.raw_file(k) for k in yugipedia.PROPERTIES]
    return files


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _date(prov: Mapping[str, Any]) -> str:
    return str(prov.get("retrieved") or "unknown")[:10]


def _floor(share: float) -> float:
    return int(share * 10_000) / 10_000


def _slug(name: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "deck"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def _load_overrides(path: Path) -> dict[int, int]:
    """``<password> <limit> [--reason]`` lines (EDOPro syntax, no ``!name`` needed); limit 3 lifts a restriction."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    lists = parse_lflist(text if re.search(r"^\s*!", text, re.M) else "!overrides\n" + text, source=str(path))
    return dict(lists[0].limits) if lists else {}


def _review_status(opts: BuildOptions, old_manifest: Mapping[str, Any], old_banlist: str | None,
                   new_banlist: str) -> dict[str, Any]:  # fmt: skip
    if opts.reviewed_by:
        return {"banlist": {"status": "reviewed", "by": opts.reviewed_by, "date": date.today().isoformat()}}
    old = (old_manifest.get("review") or {}).get("banlist") or {}
    if old.get("status") == "reviewed" and old_banlist == new_banlist:
        return {"banlist": dict(old)}
    return {"banlist": {"status": "pending"}}


def _card_name(cards: Mapping[int, Any], pw: int, fallback: str = "") -> str:
    card = cards.get(pw)
    return getattr(card, "name", "") or fallback or str(pw)


def _banlist_text(banlist: Banlist, cards, ban_res: mdm.BanlistResult, prov: Mapping[str, Any],
                  update: Mapping[str, str] | None) -> str:  # fmt: skip
    lines = [f"#[{banlist.name}]",
             f"# generated by `ygorl env build` from masterduelmeta.com ban statuses (retrieved {_date(prov)})",
             f"# last Master Duel banlist update: {update['date']} {update['url']}" if update else "#",
             "# human review: see review/report.md; corrections go to review/banlist-overrides.lflist.conf",
             f"!{banlist.name}"]  # fmt: skip
    for limit in (0, 1, 2):
        lines.append(f"--{LIMIT_LABELS[limit]} / {('Forbidden', 'Limited', 'Semi-Limited')[limit]}")
        entries = sorted((p for p, n in banlist.limits.items() if n == limit), key=lambda p: _card_name(cards, p))
        lines += [f"{p} {limit} --{_card_name(cards, p, ban_res.names.get(p, ''))}" for p in entries]
    return "\n".join(lines) + "\n"


def _ydk(s: mdm.TypeSummary, since: str) -> str:
    r = s.chosen
    deck = mdm.to_deck(r, s.name)
    body = deck.to_ydk().split("\n", 1)[1]  # drop to_ydk's own "#created by" line
    return (f"#created by ygorl env build: {s.name}, masterduelmeta list of {r.created[:10]}\n"
            f"#source {r.url}\n#medoid of {s.decks} list(s) since {since}, {s.legal} legal\n{body}")  # fmt: skip


def _write_relations(path: Path, rel: Mapping[str, Mapping[str, list[int]]], prov: Mapping[str, Any]) -> None:
    """``relations.json``: one line per (property, page name) so version diffs stay readable."""
    keys = [yugipedia.raw_file(k).as_posix() for k in yugipedia.PROPERTIES]
    head = {"source": "Yugipedia Semantic MediaWiki (action=ask)", "retrieved": min(_date(prov[k]) for k in keys),
            "properties": dict(yugipedia.PROPERTIES),
            "note": "property -> Yugipedia page name (archetype or card group) -> passwords of pool cards "
                    "that carry it (docs/data.md)"}  # fmt: skip
    lines = ["{", *(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items())]
    for i, (key, table) in enumerate(rel.items()):
        lines.append(f" {json.dumps(key)}: {{")
        items = list(table.items())
        lines += [f"  {json.dumps(n, ensure_ascii=False)}: {json.dumps(pws)}{',' if j < len(items) - 1 else ''}"
                  for j, (n, pws) in enumerate(items)]  # fmt: skip
        lines.append(" }" + ("," if i < len(rel) - 1 else ""))
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact(prov: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance for environment.json: the first URL and the request count instead of every URL."""
    urls = prov.get("urls") or []
    out = {k: v for k, v in prov.items() if k not in ("urls", "file", "user_agent")}
    out.update(url=urls[0] if urls else None, requests=len(urls))
    return out


def _illegal_by_day(records: list[mdm.DeckRecord], problems: Callable[[mdm.DeckRecord], list[str]]) -> dict[str, list[int]]:
    """``{day: [lists, lists illegal under the new banlist]}``: shows when the banlist took effect."""
    out: dict[str, list[int]] = {}
    for r in records:
        day = out.setdefault(r.created[:10], [0, 0])
        day[0] += 1
        day[1] += bool(not r.unmapped and problems(r))
    return dict(sorted(out.items()))


def _late_start(by_day: Mapping[str, list[int]]) -> str | None:
    """First day after the last day with an illegal list, if illegal lists only occur early in the window."""
    bad = [d for d, (_, n) in by_day.items() if n]
    if not bad:
        return None
    later = [d for d in by_day if d > bad[-1]]
    total = sum(n for n, _ in by_day.values())
    after = sum(by_day[d][0] for d in later)
    return later[0] if later and after >= 0.5 * total else None


def _unmapped_deck_cards(records: list[mdm.DeckRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        for name in r.unmapped:
            out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _other_lists() -> dict[str, Banlist]:
    """TCG and OCG lists from third_party/LFLists for the review table (empty if unavailable)."""
    from ygorl.paths import lflists

    out = {}
    for label, fname in (("TCG", "0TCG.lflist.conf"), ("OCG", "OCG.lflist.conf")):
        try:
            out[label] = select(load_lflist(lflists() / fname, strict=False))
        except (OSError, ValueError):
            continue
    return out


def _report(opts, stats, summaries, chosen, banlist: Banlist, ban_res, overrides, cards, pool, update, since) -> str:
    others = _other_lists()
    prev = _previous_banlist(opts)
    L: list[str] = [f"# {opts.version} 构建与校对报告", "",
                    "由 `ygorl env build` 生成，重新构建时覆盖。流程见 [docs/data.md](../../../docs/data.md)。", ""]  # fmt: skip
    status = stats["review"]
    L += ["## 禁限表校对", "",
          f"状态：**{'已校对' if status == 'reviewed' else '待校对'}**。"
          "对照游戏内「禁止・限制卡一览」逐条核对下表；不一致的卡写进 "
          f"`review/{OVERRIDES}`（`<password> <张数> --原因`，张数 3 表示解除），然后 "
          f"`ygorl env build {opts.version} --offline --reviewed-by <名字>` 重新生成。", ""]  # fmt: skip
    if update:
        L += [f"最近一次 MD 禁限更新：{update['date']}，[{update['title']}]({update['url']})。", ""]
    if (opts.out / REVIEW_DIR / CROSSCHECK).is_file():
        L += [f"与独立来源（tools/crosscheck_banlist.py）的交叉核对记录：[{CROSSCHECK}]({CROSSCHECK})。", ""]
    counts = stats["banlist"]
    L += [f"共 {sum(counts.values())} 张：" + "、".join(f"{LIMIT_LABELS[k]} {counts[n]}" for k, n in LIMIT_KEYS.items()) +
          f"；人工修正 {len(overrides)} 条。", ""]  # fmt: skip
    if prev is not None:
        name, old = prev
        changed = sorted(set(old.limits) | set(banlist.limits), key=lambda p: _card_name(cards, p))
        diff = [(p, old.limit(p), banlist.limit(p)) for p in changed if old.limit(p) != banlist.limit(p)]
        L += [f"与上一版本 `{name}` 相比变化 {len(diff)} 张：", ""]
        L += [f"- {_card_name(cards, p)} ({p})：{LIMIT_LABELS[a]} → {LIMIT_LABELS[b]}" for p, a, b in diff] or ["- 无"]
        L.append("")
    hdr = "| 密码 | 卡名 | MD | " + " | ".join(others) + " | 备注 |"
    L += [hdr, "|" + "---|" * (4 + len(others))]
    notes: dict[int, list[str]] = {}
    for e in ban_res.merged:
        notes.setdefault(e["password"], []).append(f"异画 {e['konamiID']} 合并")
    for e in ban_res.by_name:
        notes.setdefault(e["password"], []).append(f"按卡名匹配（源密码 {e['konamiID']}）")
    for p in overrides:
        notes.setdefault(p, []).append("人工修正")
    for p in banlist.limits:
        if p not in pool:
            notes.setdefault(p, []).append("不在卡池")
    for limit in (0, 1, 2):
        for p in sorted((p for p, n in banlist.limits.items() if n == limit), key=lambda p: _card_name(cards, p)):
            other = " | ".join(LIMIT_LABELS[lst.limit(p)] for lst in others.values())
            L.append(f"| {p} | {_card_name(cards, p)} | {LIMIT_LABELS[limit]} | {other} | {'；'.join(notes.get(p, []))} |")
    L.append("")
    if ban_res.unmapped or ban_res.conflicts:
        L += ["无法对应到 BabelCDB 的禁限条目（未写入禁限表，需人工处理）：", ""]
        L += [f"- {e['name']}（konamiID {e['konamiID']}，{LIMIT_LABELS[e['limit']]}）" for e in ban_res.unmapped]
        L += [f"- 冲突：{c}" for c in ban_res.conflicts]
        L.append("")

    L += ["## 卡池", "",
          f"YGOPRODECK `format=master duel` 列出 {stats['pool_source_cards']} 张，跳过衍生物 {stats['pool_tokens_skipped']} 张，"
          f"卡池 {stats['pool']} 个密码（异画归并到原卡）。", ""]  # fmt: skip
    if stats["pool_by_name"]:
        L += ["按卡名对应（YGOPRODECK 的 id 不在 BabelCDB）：", ""]
        L += [f"- {e['name']}：{e['id']} → {e['password']}" for e in stats["pool_by_name"]]
        L.append("")
    if stats["pool_unmapped"]:
        L += ["无法对应、未进卡池：", ""]
        L += [f"- {e['name']}（{e['id']}，{e['frameType']}）" for e in stats["pool_unmapped"]]
        L.append("")

    L += ["## Meta 卡组", "",
          f"窗口 {since} 起的 masterduelmeta 卡表 {stats['meta_lists']} 份（计入份额的 {stats['meta_counted']} 份，"
          f"其余是 statsWeight 为 0 的活动 / 特殊赛事卡表），{stats['meta_types']} 个卡组类型；"
          f"份额 ≥ {opts.min_share:.1%} 且有合法卡表的前 {opts.max_decks} 个类型入选，共 {stats['meta_decks']} 套，"
          f"份额合计 {stats['meta_share']:.1%}。", "",
          "| 卡组 | 份额 | 卡表数 | 合法卡表 | 入选 | 不合法原因（卡表数） |", "|---|---|---|---|---|---|"]  # fmt: skip
    chosen_names = {s.name for s in chosen}
    for s in summaries:
        if s.share < opts.min_share / 2:
            continue
        why = "、".join(f"{k} {v}" for k, v in s.problems.most_common())
        L.append(f"| {s.name} | {s.share:.1%} | {s.decks} | {s.legal} | {'是' if s.name in chosen_names else ''} | {why} |")
    L.append("")
    by_day = stats["meta_illegal_by_day"]
    L += ["按日期统计的卡表数与在新禁限表下不合法的卡表数。新禁限表生效后仍有不合法卡表，说明禁限表可能有误（需校对）；"
          "不合法卡表只出现在窗口开头，说明禁限表在那之后才生效，应以 `--since` 推迟窗口起点：", "",
          "| 日期 | 卡表 | 不合法 |", "|---|---|---|"]  # fmt: skip
    L += [f"| {d} | {n} | {bad} |" for d, (n, bad) in by_day.items()]
    L.append("")
    if stats["meta_unmapped"]:
        L += ["卡表中无法对应到 BabelCDB 的卡（含这些卡的卡表不参与代表卡表的选择）：", ""]
        L += [f"- {n}（{k} 份卡表）" for n, k in stats["meta_unmapped"].items()]
        L.append("")

    if stats["relations"] is not None:
        L += ["## Yugipedia 关系", "",
              f"`relations.json` 覆盖卡池中 {stats['relations']} 张卡。各属性无法对应到卡片数据库的页面数："
              + "、".join(f"{k} {v}" for k, v in stats["relations_unmapped"].items()) + "。", ""]  # fmt: skip
    return "\n".join(L)


def _previous_banlist(opts: BuildOptions) -> tuple[str, Banlist] | None:
    """Banlist of the newest ``md-*`` environment older than this version under the same root."""
    older = sorted(p for p in opts.root.glob("md-*") if p.is_dir() and p.name < opts.version and (p / BANLIST).is_file())
    for path in reversed(older):
        try:
            manifest = _read_json(path / MANIFEST)
            return path.name, select(load_lflist(path / BANLIST), manifest.get("banlist_name"))
        except ValueError:
            continue
    return None

