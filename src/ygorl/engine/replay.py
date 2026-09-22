"""Replays: record, store, reload and re-run duels; export EDOPro ``.yrpX``.

A replay is everything needed to reproduce a duel bit for bit (see
docs/replays.md): environment version + fingerprint, seed (and the four core
seed words derived from it), who went first, rule flags, player rules, both
decks, and every ``set_response`` payload. Optionally it keeps the per-step
candidate lists, chosen index and policy probabilities (``record_steps``).

Storage is JSON (``.json``) or gzipped JSON (``.json.gz``); responses are hex.
Playing a replay under a different environment version, or under the same
version whose files changed (fingerprint), raises
:class:`ReplayEnvironmentMismatch`.

``to_yrpx`` writes an EDOPro replay: a streamed packet list (MSG_START plus
every engine message a spectator would see) and an embedded ``yrp1`` replay
(seed, decks in load order, responses) that EDOPro's "yrp" mode re-simulates
with its own core. Layout transcribed from edo9300/edopro gframe/replay.cpp
and generic_duel.cpp.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, DuelResult, expand_seed

FORMAT = "ygorl-replay"
FORMAT_VERSION = 1

# EDOPro replay constants (gframe/replay.h, gframe/common.h)
REPLAY_YRP1 = 0x31707279
REPLAY_YRPX = 0x58707279
REPLAY_LUA64 = 0x10
REPLAY_NEWREPLAY = 0x20
REPLAY_64BIT_DUELFLAG = 0x100
REPLAY_EXTENDED_HEADER = 0x200
OLD_REPLAY_MODE = 231
EDOPRO_VERSION = (41, 0)  # client version written in the header (EDOPro 41.0)

# Hint types the EDOPro host sends only to one player and leaves out of replays.
_PRIVATE_HINTS = {1, 2, 3, 5}


class ReplayEnvironmentMismatch(ValueError):
    """A replay is played under a different environment than it was recorded in."""


def _deck_to_dict(deck: Deck) -> dict:
    return {"name": deck.name, "main": list(deck.main), "extra": list(deck.extra), "side": list(deck.side)}


def _deck_from_dict(d: dict) -> Deck:
    return Deck(tuple(d["main"]), tuple(d["extra"]), tuple(d.get("side", ())), d.get("name", ""))


@dataclass
class Replay:
    seed: int
    first: int
    rule_flags: int
    player: dict
    shuffle_decks: bool
    decks: dict  # {"a": {...}, "b": {...}}
    responses: list[bytes]
    environment: dict | None = None  # {"version", "fingerprint"} or None
    engine: dict = field(default_factory=dict)
    max_turns: int = 200
    max_decisions: int = 20000
    result: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    curriculum: str = "full"  # DuelConfig.curriculum / learner / augmented_start (T2.6); absent in older files
    learner: int = 0
    augmented_start: bool = False

    # -- construction -----------------------------------------------------
    @classmethod
    def from_duel(cls, duel: Duel, result: DuelResult) -> Replay:
        cfg = duel.config
        major, minor = _core.ocg_version()
        return cls(
            seed=duel.seed,
            first=duel.first,
            rule_flags=cfg.rule_flags,
            player=asdict(cfg.player),
            shuffle_decks=cfg.shuffle_decks,
            decks={"a": _deck_to_dict(duel.decks[0]), "b": _deck_to_dict(duel.decks[1])},
            responses=list(result.responses),
            environment={"version": duel.env.version, "fingerprint": duel.env.fingerprint} if duel.env is not None else None,
            engine={"ocgcore": [major, minor]},
            max_turns=cfg.max_turns,
            max_decisions=cfg.max_decisions,
            result={"winner": result.winner, "reason": result.reason, "win_reason": result.win_reason,
                    "turns": result.turns, "lp": list(result.lp), "decisions": result.decisions},  # fmt: skip
            steps=list(result.steps),
            curriculum=cfg.curriculum,
            learner=cfg.learner,
            augmented_start=cfg.augmented_start,
        )

    @property
    def core_seed(self) -> list[int]:
        return expand_seed(self.seed)

    def config(self) -> DuelConfig:
        return DuelConfig(rule_flags=self.rule_flags, player=PlayerRules(**self.player), max_turns=self.max_turns,
                          max_decisions=self.max_decisions, shuffle_decks=self.shuffle_decks, curriculum=self.curriculum,
                          learner=self.learner, augmented_start=self.augmented_start)  # fmt: skip

    def duel(self, env: Environment | None = None, **kwargs) -> Duel:
        """A fresh Duel set up exactly like the recorded one (after the environment check)."""
        self.check_environment(env)
        decks = (_deck_from_dict(self.decks["a"]), _deck_from_dict(self.decks["b"]))
        return Duel(self.seed, env, decks[0], decks[1], config=self.config(), first=self.first, **kwargs)

    def check_environment(self, env: Environment | None) -> None:
        if self.environment is None:
            return
        want = self.environment["version"]
        if env is None:
            raise ReplayEnvironmentMismatch(f"replay requires environment {want}; pass env=load_environment({want!r})")
        if env.version != want:
            raise ReplayEnvironmentMismatch(f"replay was recorded in environment {want}, not {env.version}")
        if self.environment.get("fingerprint") and env.fingerprint != self.environment["fingerprint"]:
            raise ReplayEnvironmentMismatch(
                f"environment {want} has changed since the replay was recorded "
                f"(fingerprint {env.fingerprint[:12]} != {self.environment['fingerprint'][:12]})"
            )

    def play(self, env: Environment | None = None, **kwargs) -> DuelResult:
        """Re-run the duel from the response log (``kwargs`` go to :class:`Duel`)."""
        return self.duel(env, **kwargs).replay(self.responses)

    # -- storage ----------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["responses"] = [r.hex() for r in self.responses]
        return {"format": FORMAT, "format_version": FORMAT_VERSION, **d}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Replay:
        if data.get("format") != FORMAT:
            raise ValueError(f"not a ygorl replay (format={data.get('format')!r})")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported replay format_version {data.get('format_version')} (expected {FORMAT_VERSION})")
        fields = {k: v for k, v in data.items() if k not in ("format", "format_version")}
        fields["responses"] = [bytes.fromhex(r) for r in fields["responses"]]
        return cls(**fields)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        text = json.dumps(self.to_json(), separators=(",", ":"))
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        else:
            path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Replay:
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        return cls.from_json(json.loads(raw))

    # -- EDOPro export ----------------------------------------------------
    def to_yrpx(self, path: str | Path, names: tuple[str, str] = ("Player A", "Player B"), env: Environment | None = None) -> None:
        """Write an EDOPro ``.yrpX`` (streamed packets + embedded ``yrp1``).

        ``names`` are given in (a, b) order. The duel is re-run to collect the
        message stream, so the environment check applies.
        """
        duel = self.duel(env, record_messages=True)
        result = duel.replay(self.responses)
        seat_names = (names[duel.deck_of(0)], names[duel.deck_of(1)])
        loaded = duel.loaded_decks()
        yrp1 = self._yrp1(seat_names, loaded)

        body = bytearray(_names_block(seat_names))
        body += struct.pack("<Q", self.rule_flags)
        lp = self.player["starting_lp"]
        start = struct.pack("<BIIHHHH", 0, lp, lp, len(loaded[0][0]), len(loaded[0][1]), len(loaded[1][0]), len(loaded[1][1]))
        body += _packet(C.MSG_START, start)
        for buf in result.message_log:
            for record in M.split_messages(buf):
                msg_type, payload = record[0], record[1:]
                if msg_type in M.DECISION_TYPES or msg_type == C.MSG_RETRY:
                    continue
                if msg_type == C.MSG_HINT and payload[:1] and payload[0] in _PRIVATE_HINTS:
                    continue
                body += _packet(msg_type, payload)
        body += _packet(OLD_REPLAY_MODE, yrp1)
        Path(path).write_bytes(_header(REPLAY_YRPX, self.core_seed, len(body), self.seed) + bytes(body))

    def _yrp1(self, seat_names: tuple[str, str], loaded) -> bytes:
        p = self.player
        body = bytearray(_names_block(seat_names))
        body += struct.pack("<IIIQ", p["starting_lp"], p["starting_hand"], p["draw_per_turn"], self.rule_flags)
        for main, extra in loaded:
            body += struct.pack(f"<I{len(main)}I", len(main), *main)
            body += struct.pack(f"<I{len(extra)}I", len(extra), *extra)
        body += struct.pack("<I", 0)  # no extra rule cards
        for r in self.responses:
            if not 0 < len(r) < 256:
                raise ValueError(f"response of {len(r)} bytes cannot be stored in a yrp1 replay")
            body += bytes([len(r)]) + r
        return _header(REPLAY_YRP1, self.core_seed, len(body), self.seed) + bytes(body)


def _names_block(names: tuple[str, str]) -> bytes:
    out = bytearray()
    for name in names:
        encoded = name.encode("utf-16-le")[:38]
        out += struct.pack("<I", 1) + encoded + b"\0" * (40 - len(encoded))
    return bytes(out)


def _packet(msg_type: int, payload: bytes) -> bytes:
    return struct.pack("<BI", msg_type, len(payload)) + payload


def _header(ident: int, seed_words: list[int], datasize: int, timestamp: int) -> bytes:
    major, minor = _core.ocg_version()
    version = EDOPRO_VERSION[0] | (EDOPRO_VERSION[1] << 8) | (major << 16) | (minor << 24)
    flag = REPLAY_LUA64 | REPLAY_NEWREPLAY | REPLAY_64BIT_DUELFLAG | REPLAY_EXTENDED_HEADER
    base = struct.pack("<6I8s", ident, version, flag, timestamp & 0xFFFFFFFF, datasize, 0, b"\0" * 8)
    return base + struct.pack("<Q4Q", 1, *seed_words)


__all__ = ["Replay", "ReplayEnvironmentMismatch"]
