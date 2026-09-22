"""Parser and writer for ``.ydk`` deck files (YGOPro / EDOPro / Master Duel exports).

Format::

    #created by ...        comment
    #main
    89631139               one password per line, one line per copy
    #extra
    ...
    !side
    ...
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


class YdkError(ValueError):
    """Malformed .ydk file."""


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
        lines = ["#created by ygorl", "#main", *map(str, self.main), "#extra", *map(str, self.extra), "!side", *map(str, self.side)]
        return "\n".join(lines) + "\n"


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
        if password <= 0:
            raise YdkError(f"{source}:{lineno}: invalid card password {password}")
        sections[current].append(password)
    return Deck(tuple(sections["main"]), tuple(sections["extra"]), tuple(sections["side"]), name=name)


def load_ydk(path: str | Path) -> Deck:
    path = Path(path)
    return parse_ydk(path.read_text(encoding="utf-8"), name=path.stem, source=str(path))
