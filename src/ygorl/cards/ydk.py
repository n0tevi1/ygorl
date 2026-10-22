"""Parser and writer for ``.ydk`` deck files (YGOPro / EDOPro / Master Duel exports) and ``ydke://`` links.

``.ydk`` format::

    #created by ...        comment
    #main
    89631139               one password per line, one line per copy
    #extra
    ...
    !side
    ...

``ydke://`` link (the deck-sharing URL of EDOPro, YGOPRODECK and the deck-transfer tools): ``ydke://``
followed by three ``!``-terminated base64 sections, main / extra / side, each the section's passwords as
little-endian 32-bit integers.
"""

from __future__ import annotations

import base64
import binascii
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MAX_PASSWORD = 99_999_999  # card passwords have at most 8 digits
YDKE_PREFIX = "ydke://"


class YdkError(ValueError):
    """Malformed .ydk file."""


def valid_password(password: object) -> bool:
    """True for an int card password: 1 to 99999999 (at most 8 digits)."""
    return isinstance(password, int) and not isinstance(password, bool) and 0 < password <= MAX_PASSWORD


@dataclass(frozen=True)
class Deck:
    main: tuple[int, ...] = ()
    extra: tuple[int, ...] = ()
    side: tuple[int, ...] = ()
    name: str = ""

    def counts(self, include_side: bool = False) -> Counter[int]:
        c = Counter(self.main) + Counter(self.extra)
        if include_side:
            c += Counter(self.side)
        return c

    def to_ydk(self) -> str:
        lines = [
            "#created by ygorl",
            "#main",
            *map(str, self.main),
            "#extra",
            *map(str, self.extra),
            "!side",
            *map(str, self.side),
        ]
        return "\n".join(lines) + "\n"

    def to_ydke(self) -> str:
        def section(passwords: tuple[int, ...]) -> str:
            return base64.b64encode(struct.pack(f"<{len(passwords)}I", *passwords)).decode("ascii")

        return YDKE_PREFIX + "".join(section(s) + "!" for s in (self.main, self.extra, self.side))


def parse_ydk(text: str, name: str = "", source: str = "<string>") -> Deck:
    sections: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    current: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in ("#main", "#extra", "!side"):
            current = lowered[1:]
            continue
        if line.startswith("#") or line.startswith("!"):
            continue  # "#created by", other comments
        if current is None:
            raise YdkError(f"{source}:{lineno}: card before '#main' header")
        try:
            password = int(line.split()[0])
        except ValueError:
            raise YdkError(f"{source}:{lineno}: expected a card password, got {raw!r}") from None
        if not valid_password(password):
            raise YdkError(f"{source}:{lineno}: invalid card password {password} (expected 1 to {MAX_PASSWORD})")
        sections[current].append(password)
    return Deck(tuple(sections["main"]), tuple(sections["extra"]), tuple(sections["side"]), name=name)


def is_ydke(text: str) -> bool:
    return text.startswith(YDKE_PREFIX)


def parse_ydke(url: str, name: str = "ydke") -> Deck:
    """A deck from a ``ydke://`` link (a missing side or extra section is empty)."""
    if not is_ydke(url):
        raise YdkError(f"not a ydke:// link: {url[:40]!r}")
    parts = url[len(YDKE_PREFIX) :].strip().split("!")
    if len(parts) > 4 or (len(parts) == 4 and parts[3]):
        raise YdkError("ydke:// link has more than three sections (main!extra!side!)")
    sections: list[tuple[int, ...]] = []
    for label, part in zip(("main", "extra", "side"), [*parts, "", ""][:3]):
        try:
            raw = base64.b64decode(part, validate=True)
        except binascii.Error as exc:
            raise YdkError(f"ydke:// {label} section is not valid base64 ({exc})") from None
        if len(raw) % 4:
            raise YdkError(f"ydke:// {label} section has {len(raw)} bytes, not a multiple of 4")
        passwords = struct.unpack(f"<{len(raw) // 4}I", raw)
        bad = [p for p in passwords if not valid_password(p)]
        if bad:
            raise YdkError(f"ydke:// {label} section has invalid card password {bad[0]} (expected 1 to {MAX_PASSWORD})")
        sections.append(passwords)
    return Deck(sections[0], sections[1], sections[2], name=name)


def load_ydk(path: str | Path) -> Deck:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise YdkError(f"{path}: not valid UTF-8 ({exc.reason} at byte {exc.start})") from None
    return parse_ydk(text, name=path.stem, source=str(path))
