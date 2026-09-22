"""The ``Environment``: a versioned, first-class game configuration.

An environment bundles format + card pool + banlist + rule flags + meta decks
(with shares) + version. It lives in ``environments/<version>/``::

    environment.json        manifest: version, format, rules, player/deck rules
    pool.json               legal card passwords
    banlist.lflist.conf     EDOPro-format banlist
    meta.json               meta deck list with shares, pointing into meta/
    meta/*.ydk              meta decks
    artifacts/              outputs bound to this version (matrices, models, ...)

See docs/environments.md for the full specification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ygorl.cards.legality import DeckRules, Violation, validate_deck
from ygorl.cards.lflist import Banlist, LflistError, load_lflist, select
from ygorl.cards.ydk import MAX_PASSWORD, Deck, YdkError, load_ydk, valid_password
from ygorl.engine import constants as C

MANIFEST = "environment.json"
POOL = "pool.json"
BANLIST = "banlist.lflist.conf"
META = "meta.json"
META_DIR = "meta"
ARTIFACTS_DIR = "artifacts"
REQUIRED_FILES = (MANIFEST, POOL, BANLIST, META)

VERSION_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9.]+)*$")
SHARE_TOLERANCE = 1e-6
MAX_RULE_VALUE = 2**31 - 1  # player and deck rule values must fit the core's 32-bit (signed LP) fields

DUEL_MODES: Mapping[str, int] = {
    "MR1": C.DUEL_MODE_MR1,
    "MR2": C.DUEL_MODE_MR2,
    "MR3": C.DUEL_MODE_MR3,
    "MR4": C.DUEL_MODE_MR4,
    "MR5": C.DUEL_MODE_MR5,
    "SPEED": C.DUEL_MODE_SPEED,
    "RUSH": C.DUEL_MODE_RUSH,
    "GOAT": C.DUEL_MODE_GOAT,
}


class EnvironmentConfigError(ValueError):
    """An environment directory is malformed or inconsistent."""


class EnvironmentFileMissing(EnvironmentConfigError, FileNotFoundError):
    """A required file of an environment directory does not exist."""


@dataclass(frozen=True)
class PlayerRules:
    starting_lp: int = 8000
    starting_hand: int = 5
    draw_per_turn: int = 1


@dataclass(frozen=True)
class MetaDeck:
    name: str
    share: float
    deck: Deck
    path: Path


@dataclass(frozen=True)
class Environment:
    version: str
    format: str
    card_pool: frozenset[int]
    banlist: Banlist
    rule_flags: int
    meta_decks: tuple[MetaDeck, ...]
    player: PlayerRules = PlayerRules()
    deck_rules: DeckRules = DeckRules()
    description: str = ""
    root: Path | None = None
    fingerprint: str = ""
    manifest: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # --- artifacts --------------------------------------------------------
    @property
    def artifacts_dir(self) -> Path:
        if self.root is None:
            raise EnvironmentConfigError(f"environment {self.version} has no directory for artifacts")
        path = self.root / ARTIFACTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, *parts: str) -> Path:
        """Path inside ``artifacts/``; parent directories are created.

        Raises :class:`EnvironmentConfigError` for a path that would leave ``artifacts/`` (absolute parts,
        ``..``, or a symlink pointing elsewhere).
        """
        artifacts = self.artifacts_dir
        path = artifacts.joinpath(*parts)
        if not parts or any(Path(p).is_absolute() or ".." in Path(p).parts for p in parts) or (
            artifacts.resolve() not in path.resolve().parents
        ):
            raise EnvironmentConfigError(f"artifact path {'/'.join(map(str, parts))!r} escapes {artifacts}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def stamp(self) -> dict[str, str]:
        """Identity to embed in every artifact produced under this environment."""
        return {"environment": self.version, "fingerprint": self.fingerprint}

    def check_stamp(self, stamp: Mapping[str, Any]) -> None:
        """Raise if ``stamp`` (from an artifact) was produced under another environment."""
        if stamp.get("environment") != self.version:
            raise EnvironmentConfigError(
                f"artifact belongs to environment {stamp.get('environment')!r}, not {self.version!r}"
            )
        if stamp.get("fingerprint") and self.fingerprint and stamp["fingerprint"] != self.fingerprint:
            raise EnvironmentConfigError(
                f"artifact was produced under a different revision of {self.version!r} "
                f"(fingerprint {stamp['fingerprint'][:12]} != {self.fingerprint[:12]})"
            )

    # --- queries ----------------------------------------------------------
    def meta_deck(self, name: str) -> MetaDeck:
        for md in self.meta_decks:
            if md.name == name:
                return md
        raise KeyError(f"no meta deck {name!r} in {self.version}; have {[m.name for m in self.meta_decks]}")

    def in_pool(self, password: int) -> bool:
        return password in self.card_pool

    def validate_deck(self, deck: Deck, cards: Mapping[int, Any]) -> list[Violation]:
        """Violations of ``deck`` under this environment's pool, banlist and deck rules."""
        return validate_deck(deck, cards=cards, banlist=self.banlist, pool=self.card_pool, rules=self.deck_rules)

    def check_meta_decks(self, cards: Mapping[int, Any]) -> None:
        """Raise :class:`EnvironmentConfigError` naming every illegal meta deck and all its violations."""
        problems = []
        for md in self.meta_decks:
            violations = self.validate_deck(md.deck, cards)
            if violations:
                lines = "".join(f"\n  - {v.message}" for v in violations)
                problems.append(f"{md.path}: meta deck {md.name!r} is illegal in environment {self.version}:{lines}")
        if problems:
            raise EnvironmentConfigError("\n".join(problems))


# --- loading ---------------------------------------------------------------


def default_root() -> Path:
    """Directory holding all environments (``$YGORL_ENVIRONMENTS`` or ./environments)."""
    return Path(os.environ.get("YGORL_ENVIRONMENTS", "environments"))


def load_environment(path_or_version: str | os.PathLike[str], root: str | os.PathLike[str] | None = None, *,
                     cards: Mapping[int, Any] | None = None) -> Environment:  # fmt: skip
    """Load and validate an environment from a directory path or a version name.

    With ``cards`` (password -> card, e.g. a :class:`ygorl.cards.cdb.CardDB`) every meta deck must also be
    legal under the environment's pool, banlist and deck rules; the command line always passes the card
    database.
    """
    path = Path(path_or_version)
    if not path.is_dir():
        candidate = Path(root) / path if root is not None else default_root() / path
        if not candidate.is_dir():
            raise EnvironmentFileMissing(f"environment directory not found: {path} (also tried {candidate})")
        path = candidate
    env = _load_dir(path.resolve())
    if cards is not None:
        env.check_meta_decks(cards)
    return env


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise EnvironmentConfigError(f"{path}: not valid UTF-8 ({exc.reason} at byte {exc.start})") from None
    except json.JSONDecodeError as exc:
        raise EnvironmentConfigError(f"{path}: invalid JSON: {exc}") from None


def _require(obj: Mapping[str, Any], key: str, typ: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(obj, dict):
        raise EnvironmentConfigError(f"{where}: expected a JSON object, not {type(obj).__name__}")
    if key not in obj:
        raise EnvironmentConfigError(f"{where}: missing key {key!r}")
    value = obj[key]
    if not isinstance(value, typ) or isinstance(value, bool) and typ is not bool:
        raise EnvironmentConfigError(f"{where}: key {key!r} has wrong type {type(value).__name__}")
    return value


def _optional(obj: Mapping[str, Any], key: str, typ: type | tuple[type, ...], default: Any, where: str) -> Any:
    """``obj[key]`` checked like :func:`_require`, or ``default`` when the key is absent or null."""
    return default if obj.get(key) is None else _require(obj, key, typ, where)


def _object(manifest: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = manifest.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EnvironmentConfigError(f"{where}: {key} must be an object, not {type(value).__name__}")
    return value


def _rule_flags(rules: Mapping[str, Any], where: str) -> int:
    mode = rules.get("mode", "MR5")
    if not isinstance(mode, str) or mode not in DUEL_MODES:
        raise EnvironmentConfigError(f"{where}: unknown rules.mode {mode!r}; expected one of {sorted(DUEL_MODES)}")
    flags = DUEL_MODES[mode]
    extra = rules.get("extra_flags", [])
    if not isinstance(extra, list):
        raise EnvironmentConfigError(f"{where}: rules.extra_flags must be a list of flag names, "
                                     f"not {type(extra).__name__}")  # fmt: skip
    for name in extra:
        const = getattr(C, f"DUEL_{name}", None) if isinstance(name, str) else None
        if not isinstance(const, int) or name.startswith("MODE_"):
            raise EnvironmentConfigError(f"{where}: unknown rules.extra_flags entry {name!r}")
        flags |= const
    return flags


def _dataclass_from(cls: type, data: Mapping[str, Any] | None, where: str) -> Any:
    data = dict(data or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise EnvironmentConfigError(f"{where}: unknown keys {sorted(unknown)}")
    for k, v in data.items():
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise EnvironmentConfigError(f"{where}: {k} must be a non-negative integer")
        if v > MAX_RULE_VALUE:
            raise EnvironmentConfigError(f"{where}: {k} must be at most {MAX_RULE_VALUE}")
    return cls(**data)


def _check_rules(player: PlayerRules, deck: DeckRules, where: str) -> None:
    """Reject rules under which no game can be played (see docs/environments.md)."""
    if player.starting_lp < 1:
        raise EnvironmentConfigError(f"{where}: player.starting_lp must be at least 1")
    for key in ("starting_hand", "draw_per_turn"):
        if getattr(player, key) > deck.main_min:
            raise EnvironmentConfigError(f"{where}: player.{key} {getattr(player, key)} is larger than deck.main_min "
                                         f"{deck.main_min} (a legal deck could not draw it)")  # fmt: skip
    if deck.main_min > deck.main_max:
        raise EnvironmentConfigError(f"{where}: deck.main_min {deck.main_min} is larger than deck.main_max "
                                     f"{deck.main_max}")  # fmt: skip
    if deck.max_copies < 1:
        raise EnvironmentConfigError(f"{where}: deck.max_copies must be at least 1")


def _load_pool(path: Path) -> frozenset[int]:
    data = _read_json(path)
    cards = _require(data, "cards", list, str(path))
    passwords: list[int] = []
    for i, entry in enumerate(cards):
        pw = entry.get("password") if isinstance(entry, dict) else entry
        if not valid_password(pw):
            raise EnvironmentConfigError(f"{path}: cards[{i}] is not a valid card password (1 to {MAX_PASSWORD}): "
                                         f"{entry!r}")  # fmt: skip
        passwords.append(pw)
    pool = frozenset(passwords)
    if len(pool) != len(passwords):
        raise EnvironmentConfigError(f"{path}: duplicate card passwords in pool")
    if not pool:
        raise EnvironmentConfigError(f"{path}: card pool is empty")
    return pool


def _load_meta(root: Path) -> tuple[MetaDeck, ...]:
    path = root / META
    data = _read_json(path)
    entries = _require(data, "decks", list, str(path))
    decks: list[MetaDeck] = []
    seen: set[str] = set()
    total = 0.0
    for i, entry in enumerate(entries):
        where = f"{path}: decks[{i}]"
        if not isinstance(entry, dict):
            raise EnvironmentConfigError(f"{where}: expected an object")
        name = _require(entry, "name", str, where)
        rel = _require(entry, "file", str, where)
        share = _require(entry, "share", (int, float), where)
        if name in seen:
            raise EnvironmentConfigError(f"{where}: duplicate deck name {name!r}")
        seen.add(name)
        if not 0.0 <= share <= 1.0:
            raise EnvironmentConfigError(f"{where}: share {share} outside [0, 1]")
        total += share
        ydk_path = (root / rel).resolve()
        if root not in ydk_path.parents:
            raise EnvironmentConfigError(f"{where}: file {rel!r} escapes the environment directory")
        if not ydk_path.is_file():
            raise EnvironmentFileMissing(f"{where}: meta deck file not found: {ydk_path}")
        try:
            deck = load_ydk(ydk_path)
        except YdkError as exc:
            raise EnvironmentConfigError(str(exc)) from None
        decks.append(MetaDeck(name=name, share=float(share), deck=deck, path=ydk_path))
    if total > 1.0 + SHARE_TOLERANCE:
        raise EnvironmentConfigError(f"{path}: meta shares sum to {total:.4f} > 1")
    return tuple(decks)


def _fingerprint(root: Path, meta_decks: tuple[MetaDeck, ...]) -> str:
    h = hashlib.sha256()
    files = [root / name for name in REQUIRED_FILES] + sorted(md.path for md in meta_decks)
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _load_dir(root: Path) -> Environment:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise EnvironmentFileMissing(f"environment {root} is missing required file(s): {', '.join(missing)}")

    manifest_path = root / MANIFEST
    manifest = _read_json(manifest_path)
    where = str(manifest_path)
    if not isinstance(manifest, dict):
        raise EnvironmentConfigError(f"{where}: expected a JSON object")
    version = _require(manifest, "version", str, where)
    if not VERSION_RE.match(version):
        raise EnvironmentConfigError(f"{where}: version {version!r} must match {VERSION_RE.pattern}")
    if version != root.name:
        raise EnvironmentConfigError(f"{where}: version {version!r} does not match directory name {root.name!r}")
    fmt = _require(manifest, "format", str, where)
    rules = _object(manifest, "rules", where)
    player = _dataclass_from(PlayerRules, _object(manifest, "player", where), f"{where}: player")
    deck_rules = _dataclass_from(DeckRules, _object(manifest, "deck", where), f"{where}: deck")
    _check_rules(player, deck_rules, where)
    description = _optional(manifest, "description", str, "", where)
    banlist_name = _optional(manifest, "banlist_name", str, None, where)

    try:
        banlist = select(load_lflist(root / BANLIST), banlist_name)
    except LflistError as exc:
        raise EnvironmentConfigError(str(exc)) from None

    meta_decks = _load_meta(root)
    return Environment(
        version=version,
        format=fmt,
        card_pool=_load_pool(root / POOL),
        banlist=banlist,
        rule_flags=_rule_flags(rules, where),
        meta_decks=meta_decks,
        player=player,
        deck_rules=deck_rules,
        description=description,
        root=root,
        fingerprint=_fingerprint(root, meta_decks),
        manifest=manifest,
    )
