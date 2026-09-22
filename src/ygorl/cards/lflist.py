"""Parser for EDOPro / Project Ignis ``.lflist.conf`` banlists.

Format (one file may hold several lists)::

    #[2026.09 TCG]          comment (ignored)
    !2026.09 TCG            starts a new list with this name
    $whitelist              optional: cards not listed are forbidden
    21044178 0 --Abyss Dweller
    <password> <limit> [--comment]

``limit`` is the number of copies allowed (0 forbidden, 1 limited,
2 semi-limited; 3 is used by whitelists to allow a card unrestricted).
Lines starting with ``#`` or ``--`` are comments.

Upstream Project Ignis files contain a few quirks that EDOPro tolerates
(a password listed twice, where the last entry wins; sentinel entries with
a negative limit). ``strict=False`` mirrors that behaviour: duplicates keep
the last value and out-of-range entries are skipped. ``strict=True`` (the
default, used for our own environment banlists) rejects both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

MAX_COPIES = 3


class LflistError(ValueError):
    """Malformed banlist file."""


@dataclass(frozen=True)
class Banlist:
    name: str
    limits: Mapping[int, int] = field(default_factory=dict)
    whitelist: bool = False

    def limit(self, password: int) -> int:
        """Copies of ``password`` allowed by this list (0..3)."""
        if password in self.limits:
            return self.limits[password]
        return 0 if self.whitelist else MAX_COPIES

    def forbidden(self) -> frozenset[int]:
        return frozenset(p for p, n in self.limits.items() if n == 0)

    def to_text(self) -> str:
        lines = [f"#[{self.name}]", f"!{self.name}"]
        if self.whitelist:
            lines.append("$whitelist")
        lines.extend(f"{p} {n}" for p, n in sorted(self.limits.items(), key=lambda kv: (kv[1], kv[0])))
        return "\n".join(lines) + "\n"


def parse_lflist(text: str, source: str = "<string>", strict: bool = True) -> list[Banlist]:
    """Parse every list in ``text``; raises :class:`LflistError` with a line number."""
    lists: list[Banlist] = []
    name: str | None = None
    limits: dict[int, int] = {}
    whitelist = False

    def flush() -> None:
        if name is not None:
            lists.append(Banlist(name=name, limits=dict(limits), whitelist=whitelist))

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if line.startswith("!"):
            flush()
            name, limits, whitelist = line[1:].strip(), {}, False
            if not name:
                raise LflistError(f"{source}:{lineno}: empty list name after '!'")
            continue
        if line.startswith("$"):
            if name is None:
                raise LflistError(f"{source}:{lineno}: '{line}' before any '!name' line")
            if line.lower() == "$whitelist":
                whitelist = True
            continue  # other $-directives are ignored by EDOPro as well
        if name is None:
            raise LflistError(f"{source}:{lineno}: card entry before any '!name' line")
        body = line.split("--", 1)[0].split()
        if len(body) < 2:
            raise LflistError(f"{source}:{lineno}: expected '<password> <limit>', got {raw!r}")
        try:
            password, limit = int(body[0]), int(body[1])
        except ValueError:
            raise LflistError(f"{source}:{lineno}: non-integer password or limit in {raw!r}") from None
        if not 0 <= limit <= MAX_COPIES:
            if not strict:
                continue
            raise LflistError(f"{source}:{lineno}: limit {limit} out of range 0..{MAX_COPIES}")
        if strict and password in limits and limits[password] != limit:
            raise LflistError(f"{source}:{lineno}: password {password} listed twice with different limits")
        limits[password] = limit
    flush()
    return lists


def load_lflist(path: str | Path, strict: bool = True) -> list[Banlist]:
    path = Path(path)
    return parse_lflist(path.read_text(encoding="utf-8"), source=str(path), strict=strict)


def select(lists: Iterable[Banlist], name: str | None = None) -> Banlist:
    """Pick the list called ``name`` (or the only/first list when ``name`` is None)."""
    lists = list(lists)
    if not lists:
        raise LflistError("banlist file contains no '!name' list")
    if name is None:
        return lists[0]
    for lst in lists:
        if lst.name == name:
            return lst
    raise LflistError(f"no list named {name!r}; available: {[lst.name for lst in lists]}")
