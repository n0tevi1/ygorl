"""YGOPRODECK API v7: the Master Duel card pool.

Fetcher: one request to ``cardinfo.php?format=master duel`` (the API's limit is 20 requests/s; we stay
well below it) -> ``<raw>/ygoprodeck/cardinfo.json``, the response exactly as served.

Parser: needs ``{"data": [{"id": <password>, "name": ..., "frameType": ...}, ...]}`` and nothing else, so a
hand-made file with those three keys per card can stand in for the API. Every card is mapped to a
BabelCDB password (:class:`ygorl.data.cardmap.CardMapper`); tokens are skipped; cards that cannot be
mapped are returned for the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ygorl.data.cardmap import CardMapper
from ygorl.data.fetch import Http, write_raw

API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
FORMAT = "master duel"
RAW_FILE = Path("ygoprodeck") / "cardinfo.json"
REQUEST_INTERVAL = 0.1  # 10 requests/s, half the documented limit of 20/s


def fetch_md_cards(raw_dir: Path, http: Http | None = None) -> Path:
    """Download every card YGOPRODECK lists for Master Duel into ``raw_dir``."""
    http = http or Http(REQUEST_INTERVAL)
    data = http.get_json(API, {"format": FORMAT})
    if not isinstance(data, dict) or not isinstance(data.get("data"), list) or not data["data"]:
        raise ValueError(f"{API}: unexpected response shape (no 'data' list)")
    return write_raw(raw_dir / RAW_FILE, data, http.urls[-1:], http.user_agent)


@dataclass
class PoolResult:
    passwords: set[int] = field(default_factory=set)  # canonical BabelCDB passwords
    source_cards: int = 0
    tokens: int = 0
    by_name: list[dict[str, Any]] = field(default_factory=list)  # mapped through the name, not the id
    unmapped: list[dict[str, Any]] = field(default_factory=list)


def parse_md_pool(data: Any, mapper: CardMapper, source: str = str(RAW_FILE)) -> PoolResult:
    """The card pool from a ``cardinfo.php`` response (see the module docstring for the required keys)."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(f"{source}: expected an object with a 'data' list")
    out = PoolResult()
    for i, card in enumerate(data["data"]):
        if not isinstance(card, dict) or "id" not in card or "name" not in card:
            raise ValueError(f"{source}: data[{i}] needs 'id' and 'name'")
        out.source_cards += 1
        if card.get("frameType") == "token" or card.get("type") == "Token":
            out.tokens += 1
            continue
        mapped = mapper.map(card["id"], card["name"])
        if mapped is None:
            out.unmapped.append({"id": card["id"], "name": card["name"], "frameType": card.get("frameType")})
            continue
        if mapped.how == "name":
            out.by_name.append({"id": card["id"], "name": card["name"], "password": mapped.password})
        out.passwords.add(mapped.password)
    return out
