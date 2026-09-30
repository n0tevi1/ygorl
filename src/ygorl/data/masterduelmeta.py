"""masterduelmeta.com (unofficial JSON API): Master Duel banlist, meta deck lists and their shares.

Fetchers (about one request per second, identifying user agent) write four raw files under
``<raw>/masterduelmeta/``:

``cards.json``
    every card: ``[{"_id", "name", "konamiID", "banStatus"?}, ...]`` from ``/api/v1/cards`` (paged).
    ``konamiID`` is the card's password; deck lists refer to cards by ``_id`` only.
``banlist.json``
    the cards of ``cards.json`` that carry a ``banStatus`` (the data behind the site's Forbidden &
    Limited page): ``[{"konamiID", "name", "banStatus"}, ...]`` with ``banStatus`` one of
    ``Forbidden`` / ``Limited 1`` / ``Limited 2`` (``Limited`` / ``Semi-Limited`` or an integer
    ``limit`` are accepted too, for hand-made files).
``articles.json``
    recent article titles and dates, used to find the date of the last Master Duel banlist update.
``top-decks.json``
    the ``/api/v1/top-decks`` entries created since the meta window's start, newest first:
    ``[{"deckType": {"name"}, "created", "url", "main"/"extra": [{"card": {"_id", "name"}, "amount"}],
    "rankedType"/"tournamentType": {"statsWeight"}?}, ...]``.

Parsers turn them into an EDOPro banlist and into one representative ``.ydk`` per deck type with the type's
share of the window (see docs/data.md for the method).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ygorl.cards.lflist import Banlist
from ygorl.cards.ydk import Deck
from ygorl.data.cardmap import CardMapper
from ygorl.data.fetch import Http, write_raw

BASE = "https://www.masterduelmeta.com/api/v1"
SITE = "https://www.masterduelmeta.com"
RAW_DIR = Path("masterduelmeta")
CARDS_FILE = RAW_DIR / "cards.json"
BANLIST_FILE = RAW_DIR / "banlist.json"
ARTICLES_FILE = RAW_DIR / "articles.json"
TOP_DECKS_FILE = RAW_DIR / "top-decks.json"
REQUEST_INTERVAL = 1.0
CARDS_PAGE = 3000
DECKS_PAGE = 500
MAX_PAGES = 200
DEFAULT_WEIGHT = 10  # masterduelmeta's statsWeight of an ordinary ranked / tournament deck
BANLIST_TITLE = re.compile(r"master duel.*forbidden.*limited", re.I)

STATUS_LIMITS = {"forbidden": 0, "banned": 0, "limited": 1, "limited 1": 1, "semi-limited": 2, "limited 2": 2,
                 "semi limited": 2, "unlimited": 3}  # fmt: skip


# --------------------------------------------------------------------------- fetchers


def fetch_cards(raw_dir: Path, http: Http | None = None) -> tuple[Path, Path]:
    """Every card (id map) and the banlist subset; returns the two raw file paths."""
    http = http or Http(REQUEST_INTERVAL)
    start = len(http.urls)
    fields = "_id,name,konamiID,banStatus,alternateArt"
    cards: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        batch = http.get_json(f"{BASE}/cards", {"fields": fields, "limit": CARDS_PAGE, "page": page, "sort": "_id"})
        if not isinstance(batch, list):
            raise ValueError(f"{BASE}/cards: expected a JSON list, got {type(batch).__name__}")
        cards.extend(batch)
        if len(batch) < CARDS_PAGE:
            break
    ids = [c.get("_id") for c in cards]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{BASE}/cards: paging returned duplicate cards; the API's paging changed")
    banned = [{k: c[k] for k in ("konamiID", "name", "banStatus") if k in c} for c in cards if c.get("banStatus")]
    urls = http.urls[start:]
    return (write_raw(raw_dir / CARDS_FILE, cards, urls, http.user_agent),
            write_raw(raw_dir / BANLIST_FILE, banned, urls, http.user_agent,
                      note="cards with a banStatus, taken from cards.json"))  # fmt: skip


def fetch_articles(raw_dir: Path, http: Http | None = None, limit: int = 400) -> Path:
    http = http or Http(REQUEST_INTERVAL)
    data = http.get_json(f"{BASE}/articles", {"limit": limit, "sort": "-date", "fields": "title,date,url,category"})
    if not isinstance(data, list):
        raise ValueError(f"{BASE}/articles: expected a JSON list")
    return write_raw(raw_dir / ARTICLES_FILE, data, http.urls[-1:], http.user_agent)


def fetch_top_decks(raw_dir: Path, since: str, http: Http | None = None) -> Path:
    """Deck lists created on or after ``since`` (ISO date), newest first."""
    http = http or Http(REQUEST_INTERVAL)
    start = len(http.urls)
    decks: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        batch = http.get_json(f"{BASE}/top-decks", {"sort": "-created", "limit": DECKS_PAGE, "page": page})
        if not isinstance(batch, list):
            raise ValueError(f"{BASE}/top-decks: expected a JSON list")
        decks.extend(d for d in batch if str(d.get("created", "")) >= since)
        if len(batch) < DECKS_PAGE or min(str(d.get("created", "")) for d in batch) < since:
            break
    urls = http.urls[start:]
    return write_raw(raw_dir / TOP_DECKS_FILE, decks, urls, http.user_agent, since=since)


# --------------------------------------------------------------------------- parsers


def last_banlist_update(articles: Iterable[Mapping[str, Any]]) -> dict[str, str] | None:
    """The newest ``Master Duel: Forbidden / Limited List Update`` article: ``{"date", "title", "url"}``."""
    hits = [a for a in articles if BANLIST_TITLE.search(str(a.get("title", ""))) and a.get("date")]
    if not hits:
        return None
    a = max(hits, key=lambda a: a["date"])
    return {"date": str(a["date"])[:10], "title": str(a["title"]), "url": SITE + str(a.get("url", ""))}


def status_limit(entry: Mapping[str, Any]) -> int:
    if isinstance(entry.get("limit"), int) and not isinstance(entry.get("limit"), bool):
        return entry["limit"]
    status = str(entry.get("banStatus", "")).strip().lower()
    if status not in STATUS_LIMITS:
        raise ValueError(f"unknown banStatus {entry.get('banStatus')!r} for {entry.get('name')!r}")
    return STATUS_LIMITS[status]


@dataclass
class BanlistResult:
    banlist: Banlist
    names: dict[int, str] = field(default_factory=dict)  # password -> source name, for the report
    merged: list[dict[str, Any]] = field(default_factory=list)  # alternate artworks folded into the original
    by_name: list[dict[str, Any]] = field(default_factory=list)
    unmapped: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def parse_banlist(entries: Any, mapper: CardMapper, name: str, source: str = str(BANLIST_FILE)) -> BanlistResult:
    """EDOPro banlist (canonical passwords) from masterduelmeta ban statuses."""
    if not isinstance(entries, list):
        raise ValueError(f"{source}: expected a JSON list of cards")
    limits: dict[int, int] = {}
    out = BanlistResult(Banlist(name))
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: [{i}] is not an object")
        try:
            limit = status_limit(entry)
        except ValueError as exc:
            raise ValueError(f"{source}: [{i}]: {exc}") from None
        if limit >= 3:
            continue
        mapped = mapper.map(entry.get("konamiID"), entry.get("name"))
        if mapped is None:
            out.unmapped.append({"konamiID": entry.get("konamiID"), "name": entry.get("name"), "limit": limit})
            continue
        pw = mapped.password
        note = {"konamiID": entry.get("konamiID"), "name": entry.get("name"), "password": pw}
        if mapped.how == "alias" and note not in out.merged:
            out.merged.append(note)
        elif mapped.how == "name" and note not in out.by_name:
            out.by_name.append(note)
        if pw in limits and limits[pw] != limit:
            out.conflicts.append(f"{entry.get('name')} ({pw}): limits {limits[pw]} and {limit}; kept the lower")
        limits[pw] = min(limit, limits.get(pw, limit))
        out.names.setdefault(pw, str(entry.get("name", "")))
    out.banlist = Banlist(name, limits)
    return out


@dataclass(frozen=True)
class DeckRecord:
    type: str
    url: str
    created: str
    weight: float
    main: tuple[int, ...]
    extra: tuple[int, ...]
    unmapped: tuple[str, ...] = ()
    source: str = ""  # rankedType / tournamentType name: "Master I", "Rating Duels", an event or tournament
    kind: str = ""  # "ranked" (rankedType) or "tournament" (tournamentType)
    event: str = ""  # a tournament's customTournamentName ("Master Cup"), if any

    def counts(self) -> Counter[int]:
        return Counter(self.main) + Counter(self.extra)


def deck_source(deck: Mapping[str, Any]) -> str:
    for key in ("rankedType", "tournamentType"):
        info = deck.get(key)
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
    return ""


def deck_kind(deck: Mapping[str, Any]) -> str:
    for key, kind in (("rankedType", "ranked"), ("tournamentType", "tournament")):
        if isinstance(deck.get(key), dict):
            return kind
    return ""


def deck_weight(deck: Mapping[str, Any]) -> float:
    for key in ("rankedType", "tournamentType"):
        info = deck.get(key)
        if isinstance(info, dict):
            if info.get("includeInStats") is False:
                return 0.0
            weight = info.get("statsWeight")
            return float(weight) if isinstance(weight, int | float) else float(DEFAULT_WEIGHT)
    return float(DEFAULT_WEIGHT)


def card_ids(cards: Any, source: str = str(CARDS_FILE)) -> dict[str, tuple[str | None, str]]:
    """masterduelmeta card ``_id`` -> (konamiID, name)."""
    if not isinstance(cards, list):
        raise ValueError(f"{source}: expected a JSON list of cards")
    return {
        str(c["_id"]): (c.get("konamiID"), str(c.get("name", ""))) for c in cards if isinstance(c, dict) and "_id" in c
    }


def parse_top_decks(decks: Any, ids: Mapping[str, tuple[str | None, str]], mapper: CardMapper, since: str,
                    until: str | None = None, source: str = str(TOP_DECKS_FILE)) -> list[DeckRecord]:  # fmt: skip
    """Deck lists created in ``[since, until)`` with cards mapped to canonical passwords (side decks dropped)."""
    if not isinstance(decks, list):
        raise ValueError(f"{source}: expected a JSON list of decks")
    out: list[DeckRecord] = []
    for i, d in enumerate(decks):
        if not isinstance(d, dict):
            raise ValueError(f"{source}: [{i}] is not an object")
        created = str(d.get("created", ""))
        if created < since or (until is not None and created >= until):
            continue
        dtype = (d.get("deckType") or {}).get("name") if isinstance(d.get("deckType"), dict) else None
        if not dtype:
            continue
        sections: dict[str, list[int]] = {"main": [], "extra": []}
        unmapped: list[str] = []
        for section in sections:
            for slot in d.get(section) or []:
                card = slot.get("card") or {}
                konami, name = ids.get(str(card.get("_id")), (card.get("konamiID"), str(card.get("name", ""))))
                mapped = mapper.map(konami, name or card.get("name"))
                amount = int(slot.get("amount", 1))
                if mapped is None:
                    unmapped.append(name or str(card.get("name") or card.get("_id")))
                else:
                    sections[section].extend([mapped.password] * amount)
        out.append(DeckRecord(dtype, SITE + str(d.get("url", "")), created, deck_weight(d),
                              tuple(sections["main"]), tuple(sections["extra"]), tuple(unmapped),
                              deck_source(d), deck_kind(d), str(d.get("customTournamentName") or "")))  # fmt: skip
    return out


@dataclass
class TypeSummary:
    name: str
    decks: int
    weight: float
    share: float
    legal: int = 0
    chosen: DeckRecord | None = None
    problems: Counter[str] = field(default_factory=Counter)


def _distance(a: Counter[int], b: Counter[int]) -> int:
    return sum(abs(a[k] - b[k]) for k in a.keys() | b.keys())


def summarize_types(records: Iterable[DeckRecord], validate: Callable[[DeckRecord], list[str]]) -> list[TypeSummary]:
    """Share of every deck type (weighted by ``statsWeight``) and its representative list.

    The representative is the *medoid* of the type: among the type's lists that map completely and are legal
    (``validate`` returns no problem codes), the one with the smallest total card-count distance to all the
    type's lists; ties go to the newest list. Types are returned by share, largest first.
    """
    by_type: dict[str, list[DeckRecord]] = defaultdict(list)
    for r in records:
        by_type[r.type].append(r)
    total = sum(r.weight for rs in by_type.values() for r in rs)
    out: list[TypeSummary] = []
    for name, rs in by_type.items():
        weight = sum(r.weight for r in rs)
        summary = TypeSummary(name, len(rs), weight, weight / total if total else 0.0)
        rs.sort(key=lambda r: r.created, reverse=True)  # newest first, so ties keep the newest list
        counts = [r.counts() for r in rs]
        best: int | None = None
        for r, c in zip(rs, counts, strict=True):
            problems = ["unmapped_card"] if r.unmapped else validate(r)
            if problems:
                summary.problems.update(set(problems))
                continue
            summary.legal += 1
            key = sum(_distance(c, o) for o in counts)
            if best is None or key < best:
                best, summary.chosen = key, r
        out.append(summary)
    out.sort(key=lambda s: (-s.weight, s.name))
    return out


def to_deck(record: DeckRecord, name: str) -> Deck:
    return Deck(main=record.main, extra=record.extra, side=(), name=name)
