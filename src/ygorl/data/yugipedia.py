"""Yugipedia Semantic MediaWiki: hand-maintained archetype relations of cards.

Fetcher: the MediaWiki ``action=ask`` API, one query per property (``Archetype support``, ``Anti-support``,
``Archseries related``, and ``Archseries`` itself so archetype names can be resolved to their members),
restricted to pages with a ``Password``. SMW silently restarts at offset 0 past offset 5000, so every query
is split by card type and attribute and a partition that would need a larger offset is an error. Requests
are spaced two seconds apart. Raw file per property: ``<raw>/yugipedia/<key>.json`` shaped like an ``ask``
response, ``{"query": {"results": {<page>: {"printouts": {"Password": [...], <property>: [...]}}}}}``;
a property value may be ``{"fulltext": name}`` or a plain string.

Parser: ``password -> [archetype or concept names]`` per property; :func:`relations` inverts it to
``{property: {name: [passwords]}}`` over the card pool, the shape of ``relations.json``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ygorl.data.cardmap import CardMapper
from ygorl.data.fetch import FetchError, Http, write_raw

API = "https://yugipedia.com/api.php"
RAW_DIR = Path("yugipedia")
REQUEST_INTERVAL = 2.0
PAGE = 1000
MAX_OFFSET = 5000  # SMW's query offset cap on Yugipedia: larger offsets silently restart at 0

# key in our files -> SMW property name
PROPERTIES: Mapping[str, str] = {
    "archetype_support": "Archetype support",
    "anti_support": "Anti-support",
    "archseries_related": "Archseries related",
    "archseries": "Archseries",
}
ATTRIBUTES = ("DARK", "LIGHT", "EARTH", "WATER", "FIRE", "WIND", "DIVINE")
PARTITIONS = ("[[Card type::Spell Card]]", "[[Card type::Trap Card]]",
              *(f"[[Card type::Monster Card]][[Attribute::{a}]]" for a in ATTRIBUTES))  # fmt: skip


def raw_file(key: str) -> Path:
    return RAW_DIR / f"{key.replace('_', '-')}.json"


def fetch_property(raw_dir: Path, key: str, http: Http | None = None) -> Path:
    """Every card page with a value for property ``key`` (see :data:`PROPERTIES`)."""
    http = http or Http(REQUEST_INTERVAL)
    prop = PROPERTIES[key]
    start = len(http.urls)
    results: dict[str, Any] = {}
    for part in PARTITIONS:
        offset = 0
        while True:
            query = f"[[{prop}::+]][[Password::+]]{part}|?Password|?{prop}|limit={PAGE}|offset={offset}"
            data = http.get_json(API, {"action": "ask", "query": query, "format": "json"})
            q = data.get("query") if isinstance(data, dict) else None
            if not isinstance(q, dict) or "error" in data:
                raise FetchError(f"{API}: ask {query!r} failed: {str(data)[:200]}")
            if int((q.get("meta") or {}).get("offset", offset)) != offset:
                raise FetchError(f"{API}: ask {query!r} ignored offset {offset} (SMW offset cap)")
            page = q.get("results") or {}
            if isinstance(page, dict):
                results.update(page)
            nxt = data.get("query-continue-offset")
            if not nxt:
                break
            if int(nxt) > MAX_OFFSET:
                raise FetchError(f"{API}: partition {part} of {prop!r} has more than {MAX_OFFSET} pages; split it")
            offset = int(nxt)
    return write_raw(raw_dir / raw_file(key), {"query": {"results": results}}, http.urls[start:], http.user_agent,
                     property=prop)  # fmt: skip


def _values(items: Iterable[Any]) -> list[str]:
    out = []
    for item in items:
        name = item.get("fulltext") if isinstance(item, dict) else item
        if isinstance(name, str) and name:
            out.append(name)
    return out


def parse_property(data: Any, mapper: CardMapper, key: str, source: str = "") -> tuple[dict[int, list[str]], list[str]]:
    """(canonical password -> values, page titles that map to no card) for one property's raw file."""
    prop = PROPERTIES[key]
    try:
        results = data["query"]["results"]
    except (TypeError, KeyError):
        raise ValueError(f"{source or raw_file(key)}: expected {{'query': {{'results': ...}}}}") from None
    if isinstance(results, list):  # the API serializes an empty result set as []
        results = {}
    out: dict[int, set[str]] = {}
    unmapped: list[str] = []
    for title, page in results.items():
        printouts = (page or {}).get("printouts", {})
        values = _values(printouts.get(prop, []))
        if not values:
            continue
        passwords = printouts.get("Password") or [None]
        hits = {m.password for m in (mapper.map(p, title) for p in passwords) if m is not None}
        if not hits:
            unmapped.append(title)
        for pw in hits:
            out.setdefault(pw, set()).update(values)
    return {pw: sorted(v) for pw, v in out.items()}, sorted(unmapped)


def relations(parsed: Mapping[str, Mapping[int, list[str]]], pool: Iterable[int]) -> dict[str, dict[str, list[int]]]:
    """``{key: {page name: [passwords]}}`` over the cards of ``pool`` (inverted: one entry per archetype)."""
    pool = set(pool)
    out: dict[str, dict[str, list[int]]] = {}
    for key, table in parsed.items():
        inverted: dict[str, list[int]] = {}
        for pw in sorted(p for p in table if p in pool):
            for value in table[pw]:
                inverted.setdefault(value, []).append(pw)
        out[key] = dict(sorted(inverted.items()))
    return out
