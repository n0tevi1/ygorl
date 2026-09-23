"""Network side of data acquisition: polite HTTP GETs and the raw-file cache.

Fetchers (``ygorl.data.ygoprodeck`` / ``masterduelmeta`` / ``yugipedia``) download a source into one raw
file under ``<raw>/<source>/<name>.json`` and record where it came from in a sidecar
``<name>.json.source.json`` (URLs, retrieval time, SHA-256, user agent). Parsers only ever read raw files,
so when a source breaks, a raw file can be replaced by hand (same JSON shape, see docs/data.md) and the
environment rebuilt with ``--offline``; :func:`provenance` then reports the file as edited by hand.

Only the standard library is used: ``urllib`` honours ``HTTPS_PROXY`` and ``ssl`` the system CA bundle
(``SSL_CERT_FILE``); TLS verification is never disabled.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "ygorl-data/0.1 (Yu-Gi-Oh! RL research; +https://github.com/n0tevi1/ygorl)"
SIDECAR_SUFFIX = ".source.json"
RETRY_STATUS = (429, 500, 502, 503, 504)


class FetchError(RuntimeError):
    """A source could not be downloaded (after retries) or returned something unusable."""


class RateLimiter:
    """At most one request per ``interval`` seconds (per limiter, i.e. per source)."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._next = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval


class Http:
    """GET JSON with a user agent, a rate limit and retries on 429 / 5xx / network errors."""

    def __init__(self, interval: float, *, retries: int = 3, timeout: float = 120.0, user_agent: str = USER_AGENT):
        self.limiter = RateLimiter(interval)
        self.retries = retries
        self.timeout = timeout
        self.user_agent = user_agent
        self.urls: list[str] = []

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        delay = 2.0
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read()
                self.urls.append(url)
                try:
                    return json.loads(body)
                except json.JSONDecodeError as exc:
                    raise FetchError(f"{url}: response is not JSON ({exc}); first bytes {body[:80]!r}") from None
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt == self.retries:
                    raise FetchError(f"{url}: HTTP {exc.code} {exc.reason}") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.retries:
                    raise FetchError(f"{url}: {exc}") from None
                wait = delay
            time.sleep(wait)
            delay *= 2
        raise AssertionError("unreachable")


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def write_raw(path: Path, data: Any, urls: list[str], user_agent: str = USER_AGENT, **extra: Any) -> Path:
    """Write ``data`` as JSON to ``path`` and its provenance sidecar; returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    meta = {"urls": urls, "retrieved": now_utc(), "sha256": sha256_file(path), "user_agent": user_agent, **extra}
    sidecar(path).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def read_raw(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"raw source file not found: {path} (fetch it, or put a hand-made file there)")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: not a UTF-8 JSON file: {exc}") from None


def provenance(path: Path) -> dict[str, Any]:
    """Where a raw file came from: its sidecar, with ``manual: true`` if the file was replaced or edited."""
    digest = sha256_file(path)
    side = sidecar(path)
    if not side.is_file():
        return {"file": path.name, "sha256": digest, "manual": True, "retrieved": None, "urls": []}
    meta = json.loads(side.read_text(encoding="utf-8"))
    out = {"file": path.name, **meta, "sha256": digest}
    if meta.get("sha256") != digest:
        out["manual"] = True
    return out
