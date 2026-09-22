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
import lzma
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
    seed_words: list[int] | None = None  # explicit core seed words (replays from other hosts); None = expand_seed(seed)

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
            seed_words=None if duel.core_seed == expand_seed(duel.seed) else list(duel.core_seed),
        )

    @classmethod
    def from_yrp(cls, source: str | Path | bytes | YrpFile) -> Replay:
        """A replay of the duel an EDOPro ``.yrp`` / ``.yrpX`` (or its embedded ``yrp1``) records.

        Only 1-vs-1 duels with the extended header (four core seed words) are
        supported. The decks are taken in load order, so the host shuffle is
        off; engine player 0 (the first ``yrp1`` deck) is deck ``a`` and moves
        first. Such a replay carries no environment: it re-runs under the rule
        flags and player rules of the file.
        """
        yrp = source if isinstance(source, YrpFile) else (parse_yrp(source) if isinstance(source, bytes) else load_yrp(source))
        yrp = yrp.replayable()
        if yrp.seed is None:
            raise YrpError("replay without the extended header (four core seed words) is not supported")
        if (yrp.home_count, yrp.opposing_count) != (1, 1) or yrp.flag & REPLAY_HAND_TEST:
            raise YrpError("only 1-vs-1 duels can be re-run (tag duels and hand tests are not supported)")
        lp, hand, draw = yrp.player
        names = yrp.names + ("", "")
        decks = {side: {"name": names[i], "main": list(yrp.decks[i][0]), "extra": list(yrp.decks[i][1]), "side": []}
                 for i, side in enumerate("ab")}  # fmt: skip
        return cls(seed=yrp.timestamp, first=0, rule_flags=yrp.rule_flags,
                   player={"starting_lp": lp, "starting_hand": hand, "draw_per_turn": draw}, shuffle_decks=False,
                   decks=decks, responses=list(yrp.responses), engine={"ocgcore": list(_core.ocg_version())},
                   seed_words=list(yrp.seed))  # fmt: skip

    @property
    def core_seed(self) -> list[int]:
        return list(self.seed_words) if self.seed_words is not None else expand_seed(self.seed)

    def config(self) -> DuelConfig:
        return DuelConfig(rule_flags=self.rule_flags, player=PlayerRules(**self.player), max_turns=self.max_turns,
                          max_decisions=self.max_decisions, shuffle_decks=self.shuffle_decks, curriculum=self.curriculum,
                          learner=self.learner, augmented_start=self.augmented_start)  # fmt: skip

    def duel(self, env: Environment | None = None, **kwargs) -> Duel:
        """A fresh Duel set up exactly like the recorded one (after the environment check)."""
        self.check_environment(env)
        decks = (_deck_from_dict(self.decks["a"]), _deck_from_dict(self.decks["b"]))
        return Duel(self.seed, env, decks[0], decks[1], config=self.config(), first=self.first, core_seed=self.seed_words,
                    **kwargs)  # fmt: skip

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
        if self.seed_words is None:
            del d["seed_words"]  # files of ordinary duels stay as before
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
    def to_yrpx(self, path: str | Path, names: tuple[str, str] = ("Player A", "Player B"), env: Environment | None = None,
                **kwargs) -> None:  # fmt: skip
        """Write an EDOPro ``.yrpX`` (streamed packets + embedded ``yrp1``).

        ``names`` are given in (a, b) order. The duel is re-run to collect the
        message stream, so the environment check applies; ``kwargs`` (``cards``,
        ``scripts``) go to :class:`Duel`.
        """
        duel = self.duel(env, record_messages=True, **kwargs)
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


# ------------------------------------------------------------------ reading .yrp / .yrpX

REPLAY_COMPRESSED = 0x1
REPLAY_TAG = 0x2
REPLAY_SINGLE_MODE = 0x8
REPLAY_HAND_TEST = 0x40
_NAME_BYTES = 40  # 20 UTF-16 code units


class YrpError(ValueError):
    """A file is not a readable EDOPro replay, or not one this host can re-run."""


@dataclass(frozen=True)
class YrpFile:
    """The contents of an EDOPro replay, as ``gframe/replay.cpp`` reads it.

    ``yrp1`` files carry the duel parameters, the decks in load order and the
    responses; ``yrpX`` files carry the broadcast packet stream and, as the
    last packet (``OLD_REPLAY_MODE``), an embedded ``yrp1`` (``embedded``).
    """

    id: int
    version: int
    flag: int
    timestamp: int
    seed: tuple[int, ...] | None  # four core seed words (extended header), else None
    names: tuple[str, ...]  # home players then opposing players
    home_count: int
    opposing_count: int
    rule_flags: int
    player: tuple[int, int, int] | None = None  # yrp1: starting LP, hand, draw per turn
    decks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()  # yrp1: (main, extra) per duelist, load order
    rule_cards: tuple[int, ...] = ()
    responses: tuple[bytes, ...] = ()  # yrp1
    packets: tuple[tuple[int, bytes], ...] = ()  # yrpX: (message type, payload), OLD_REPLAY_MODE included
    embedded: YrpFile | None = None  # yrpX: the embedded yrp1

    @property
    def kind(self) -> str:
        return "yrpX" if self.id == REPLAY_YRPX else "yrp1"

    def replayable(self) -> YrpFile:
        """The ``yrp1`` that re-runs this duel: the file itself, or the one a ``yrpX`` embeds."""
        if self.id == REPLAY_YRP1:
            return self
        if self.embedded is None:
            raise YrpError("yrpX replay has no embedded yrp1: it cannot be re-simulated")
        return self.embedded


class _Cursor:
    def __init__(self, data: bytes, what: str) -> None:
        self.data, self.pos, self.what = data, 0, what

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise YrpError(f"truncated replay: {self.what} ends at byte {len(self.data)}, needed {self.pos + n}")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def name(self) -> str:
        return self.take(_NAME_BYTES).decode("utf-16-le", "replace").split("\0", 1)[0]

    @property
    def eof(self) -> bool:
        return self.pos >= len(self.data)


def load_yrp(path: str | Path) -> YrpFile:
    """Read an EDOPro replay file (``.yrp`` or ``.yrpX``)."""
    path = Path(path)
    try:
        return parse_yrp(path.read_bytes())
    except YrpError as exc:
        raise YrpError(f"{path}: {exc}") from None


def parse_yrp(data: bytes) -> YrpFile:
    """Parse the bytes of an EDOPro replay (layout of ``gframe/replay.cpp``; LZMA bodies are decompressed)."""
    if len(data) < 32:
        raise YrpError(f"truncated header ({len(data)} bytes)")
    ident, version, flag, timestamp, datasize, _hash = struct.unpack_from("<6I", data, 0)
    props = data[24:29]
    if ident not in (REPLAY_YRP1, REPLAY_YRPX):
        raise YrpError(f"not an EDOPro replay (identifier 0x{ident:08x})")
    header_len, seed = 32, None
    if flag & REPLAY_EXTENDED_HEADER:
        if len(data) < 72:
            raise YrpError(f"truncated header ({len(data)} bytes, extended header needs 72)")
        seed = struct.unpack_from("<4Q", data, 40)
        header_len = 72
    raw = data[header_len:]
    if flag & REPLAY_COMPRESSED:
        # EDOPro stores the 5 LZMA property bytes in the header and the raw stream after it (LzmaUncompress);
        # rebuilding the classic .lzma header lets the standard library decode it.
        try:
            body = lzma.decompress(props + struct.pack("<Q", datasize) + raw, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError as exc:
            raise YrpError(f"LZMA decompression failed: {exc}") from None
    else:
        body = raw
    if len(body) < datasize:
        raise YrpError(f"truncated body: {len(body)} bytes, header announces {datasize}")
    cur = _Cursor(body[:datasize], "body")
    if flag & REPLAY_SINGLE_MODE:
        names = (cur.name(), cur.name())
        home = opposing = 1
    else:
        counts, names_list = [], []
        for _ in range(2):
            n = cur.u32() if flag & REPLAY_NEWREPLAY else (2 if flag & REPLAY_TAG else 1)
            counts.append(n)
            names_list += [cur.name() for _ in range(n)]
        home, opposing = counts
        names = tuple(names_list)
    fields: dict = {"id": ident, "version": version, "flag": flag, "timestamp": timestamp, "seed": seed, "names": names,
                    "home_count": home, "opposing_count": opposing}  # fmt: skip
    if ident == REPLAY_YRP1:
        fields["player"] = (cur.u32(), cur.u32(), cur.u32())
    fields["rule_flags"] = cur.u64() if flag & REPLAY_64BIT_DUELFLAG else cur.u32()
    if ident == REPLAY_YRPX:
        packets, embedded = [], None
        while not cur.eof:
            msg = cur.u8()
            payload = cur.take(cur.u32())
            packets.append((msg, payload))
            if msg == OLD_REPLAY_MODE and embedded is None:
                embedded = parse_yrp(payload)
        return YrpFile(**fields, packets=tuple(packets), embedded=embedded)
    if flag & REPLAY_SINGLE_MODE:
        cur.take(cur.u16())  # puzzle script name
        if not flag & REPLAY_HAND_TEST:
            raise YrpError("single-mode (puzzle) replays carry no decks and are not supported")
    decks = []
    for _ in range(home + opposing):
        main = struct.unpack(f"<{(n := cur.u32())}I", cur.take(4 * n))
        extra = struct.unpack(f"<{(n := cur.u32())}I", cur.take(4 * n))
        decks.append((main, extra))
    rule_cards: tuple[int, ...] = ()
    if flag & REPLAY_NEWREPLAY and not flag & REPLAY_HAND_TEST:
        rule_cards = struct.unpack(f"<{(n := cur.u32())}I", cur.take(4 * n))
    responses = []
    while not cur.eof:
        n = cur.u8()
        if n == 0:  # optional terminator (the combo solver writes one, EDOPro stops at the end of the data)
            break
        responses.append(cur.take(n))
    return YrpFile(**fields, decks=tuple(decks), rule_cards=rule_cards, responses=tuple(responses))


__all__ = ["REPLAY_YRP1", "REPLAY_YRPX", "Replay", "ReplayEnvironmentMismatch", "YrpError", "YrpFile", "load_yrp",
           "parse_yrp"]  # fmt: skip
