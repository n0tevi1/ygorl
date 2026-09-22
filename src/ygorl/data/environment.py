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

from ygorl.cards.lflist import Banlist, LflistError, load_lflist, select
from ygorl.cards.ydk import Deck, YdkError, load_ydk
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
class DeckRules:
    main_min: int = 40
    main_max: int = 60
    extra_max: int = 15
    side_max: int = 15
    max_copies: int = 3


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
        """Path inside ``artifacts/``; parent directories are created."""
        path = self.artifacts_dir.joinpath(*parts)
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


# --- loading ---------------------------------------------------------------


def default_root() -> Path:
    """Directory holding all environments (``$YGORL_ENVIRONMENTS`` or ./environments)."""
    return Path(os.environ.get("YGORL_ENVIRONMENTS", "environments"))


def load_environment(path_or_version: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Environment:
    """Load and validate an environment from a directory path or a version name."""
    path = Path(path_or_version)
    if not path.is_dir():
        candidate = Path(root) / path if root is not None else default_root() / path
        if not candidate.is_dir():
            raise EnvironmentFileMissing(f"environment directory not found: {path} (also tried {candidate})")
        path = candidate
    return _load_dir(path.resolve())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvironmentConfigError(f"{path}: invalid JSON: {exc}") from None


def _require(obj: Mapping[str, Any], key: str, typ: type | tuple[type, ...], where: str) -> Any:
    if key not in obj:
        raise EnvironmentConfigError(f"{where}: missing key {key!r}")
    value = obj[key]
    if not isinstance(value, typ) or isinstance(value, bool) and typ is not bool:
        raise EnvironmentConfigError(f"{where}: key {key!r} has wrong type {type(value).__name__}")
    return value


def _rule_flags(rules: Mapping[str, Any], where: str) -> int:
    mode = rules.get("mode", "MR5")
    if mode not in DUEL_MODES:
        raise EnvironmentConfigError(f"{where}: unknown rules.mode {mode!r}; expected one of {sorted(DUEL_MODES)}")
    flags = DUEL_MODES[mode]
    for name in rules.get("extra_flags", []):
        const = getattr(C, f"DUEL_{name}", None)
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
    return cls(**data)


def _load_pool(path: Path) -> frozenset[int]:
    data = _read_json(path)
    cards = _require(data, "cards", list, str(path))
    passwords: list[int] = []
    for i, entry in enumerate(cards):
        pw = entry.get("password") if isinstance(entry, dict) else entry
        if not isinstance(pw, int) or isinstance(pw, bool) or pw <= 0:
            raise EnvironmentConfigError(f"{path}: cards[{i}] is not a valid card password: {entry!r}")
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
    rules = manifest.get("rules", {})
    if not isinstance(rules, dict):
        raise EnvironmentConfigError(f"{where}: rules must be an object")

    try:
        banlist = select(load_lflist(root / BANLIST), manifest.get("banlist_name"))
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
        player=_dataclass_from(PlayerRules, manifest.get("player"), f"{where}: player"),
        deck_rules=_dataclass_from(DeckRules, manifest.get("deck"), f"{where}: deck"),
        description=manifest.get("description", ""),
        root=root,
        fingerprint=_fingerprint(root, meta_decks),
        manifest=manifest,
    )
