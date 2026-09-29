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

``to_yrpx`` writes an EDOPro replay: a streamed packet list (what EDOPro's
host records: MSG_START, every engine message a spectator would see and the
MSG_UPDATE_DATA / MSG_UPDATE_CARD refreshes it queries from the core around
them) and an embedded ``yrp1`` replay (seed, decks in load order, responses)
that EDOPro's "yrp" mode re-simulates with its own core, both LZMA-compressed as
EDOPro saves them. Layout transcribed from edo9300/edopro gframe/replay.cpp,
generic_duel.cpp and core_utils.cpp.
"""

from __future__ import annotations

import gzip
import json
import lzma
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, DuelResult, expand_seed, seat_of_deck

FORMAT = "ygorl-replay"
FORMAT_VERSION = 1

# EDOPro replay constants (gframe/replay.h, gframe/common.h)
REPLAY_YRP1 = 0x31707279
REPLAY_YRPX = 0x58707279
REPLAY_COMPRESSED = 0x1
REPLAY_LUA64 = 0x10
REPLAY_NEWREPLAY = 0x20
REPLAY_64BIT_DUELFLAG = 0x100
REPLAY_EXTENDED_HEADER = 0x200
OLD_REPLAY_MODE = 231
EDOPRO_VERSION = (41, 0)  # client version written in the header (EDOPro 41.0)
# EDOPro's encoder settings (gframe/replay.cpp, Replay::EndRecord): LzmaCompress(..., level 5, dictSize 1 << 24,
# lc 3, lp 0, pb 2, fb 32, 1 thread). liblzma's preset 5 is the same match finder (bt4, depth 16 + fb / 2).
_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "preset": 5, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 2, "nice_len": 32}
]

# Hint types the EDOPro host sends only to one player and leaves out of replays.
_PRIVATE_HINTS = {1, 2, 3, 5}
# MSG_WIN reason of the packet written for a game stopped by a turn / decision limit: EDOPro shows it as
# "[loser] Time limit up" (its host writes the same reason when a player runs out of time).
WIN_REASON_LIMIT = 0x3
_LIMIT_REASONS = ("turn_limit", "decision_limit")


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
    recorded_at: int | None = None  # unix time the duel was recorded (the .yrpX date); None in older files

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
            environment={"version": duel.env.version, "fingerprint": duel.env.fingerprint}
            if duel.env is not None
            else None,
            engine={"ocgcore": [major, minor]},
            max_turns=cfg.max_turns,
            max_decisions=cfg.max_decisions,
            result={
                "winner": result.winner,
                "reason": result.reason,
                "win_reason": result.win_reason,
                "turns": result.turns,
                "lp": list(result.lp),
                "decisions": result.decisions,
            },  # fmt: skip
            steps=list(result.steps),
            curriculum=cfg.curriculum,
            learner=cfg.learner,
            augmented_start=cfg.augmented_start,
            seed_words=None if duel.core_seed == expand_seed(duel.seed) else list(duel.core_seed),
            recorded_at=int(time.time()),
        )

    @classmethod
    def from_yrp(cls, source: str | Path | bytes | YrpFile) -> Replay:
        """A replay of the duel an EDOPro ``.yrp`` / ``.yrpX`` (or its embedded ``yrp1``) records.

        Only 1-vs-1 duels with the extended header (four core seed words) are
        supported. The decks are taken in load order, so the host shuffle is
        off; engine player 0 (the first ``yrp1`` deck) is deck ``a`` and moves
        first. Such a replay carries no environment: it re-runs under the rule
        flags and player rules of the file. The header timestamp is the
        recording date (``recorded_at``); the seed words are explicit, so
        ``seed`` is 0.
        """
        yrp = (
            source
            if isinstance(source, YrpFile)
            else (parse_yrp(source) if isinstance(source, bytes) else load_yrp(source))
        )
        yrp = yrp.replayable()
        if yrp.seed is None:
            raise YrpError("replay without the extended header (four core seed words) is not supported")
        if (yrp.home_count, yrp.opposing_count) != (1, 1) or yrp.flag & REPLAY_HAND_TEST:
            raise YrpError("only 1-vs-1 duels can be re-run (tag duels and hand tests are not supported)")
        lp, hand, draw = yrp.player
        names = yrp.names + ("", "")
        decks = {side: {"name": names[i], "main": list(yrp.decks[i][0]), "extra": list(yrp.decks[i][1]), "side": []}
                 for i, side in enumerate("ab")}  # fmt: skip
        return cls(seed=0, first=0, rule_flags=yrp.rule_flags,
                   player={"starting_lp": lp, "starting_hand": hand, "draw_per_turn": draw}, shuffle_decks=False,
                   decks=decks, responses=list(yrp.responses), engine={"ocgcore": list(_core.ocg_version())},
                   seed_words=list(yrp.seed), recorded_at=yrp.timestamp or None)  # fmt: skip

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
        for key in ("seed_words", "recorded_at"):
            if d[key] is None:
                del d[key]  # optional keys: files without them stay as before
        return {"format": FORMAT, "format_version": FORMAT_VERSION, **d}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Replay:
        """Replay from its JSON form; raises ValueError naming the first malformed field."""
        if not isinstance(data, dict):
            raise ValueError(f"not a ygorl replay (expected a JSON object, not {type(data).__name__})")
        if data.get("format") != FORMAT:
            raise ValueError(f"not a ygorl replay (format={data.get('format')!r})")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported replay format_version {data.get('format_version')} (expected {FORMAT_VERSION})"
            )
        values = {k: v for k, v in data.items() if k not in ("format", "format_version")}
        _check_replay_fields(values)
        values["responses"] = [bytes.fromhex(r) for r in values["responses"]]
        return cls(**values)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        text = json.dumps(self.to_json(), separators=(",", ":"))
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        else:
            path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Replay:
        """Read a ``.json`` / ``.json.gz`` replay; a malformed file raises ValueError (a missing one OSError)."""
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            try:
                raw = gzip.decompress(raw)
            except (EOFError, OSError, zlib.error) as exc:
                raise ValueError(f"{path}: truncated or corrupt gzip data ({exc})") from None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})") from None
        return cls.from_json(data)

    # -- EDOPro export ----------------------------------------------------
    def to_yrpx(self, path: str | Path, names: tuple[str, str] = ("Player A", "Player B"), env: Environment | None = None,
                compress: bool = True, **kwargs) -> None:  # fmt: skip
        """Write an EDOPro ``.yrpX`` (streamed packets + embedded ``yrp1``).

        ``names`` are given in (a, b) order. The duel is re-run to collect the
        packet stream (the core is queried along the way, as EDOPro's host
        does), so the environment check applies; ``kwargs`` (``cards``,
        ``scripts``) go to :class:`Duel`. A game the recorded result says was
        stopped by the turn or decision limit ends with a MSG_WIN packet
        (winner from ``result``, reason :data:`WIN_REASON_LIMIT`). The header
        date is ``recorded_at`` (0 when unknown), so exports are reproducible.
        Both replays are LZMA-compressed as EDOPro writes them (``REPLAY_COMPRESSED``);
        ``compress=False`` stores them uncompressed, which EDOPro reads as well.
        """
        duel = self.duel(env, **kwargs)
        seat_names = (names[duel.deck_of(0)], names[duel.deck_of(1)])
        loaded = duel.loaded_decks()
        lp = self.player["starting_lp"]
        host = _HostStream(struct.pack("<BIIHHHH", 0, lp, lp, len(loaded[0][0]), len(loaded[0][1]), len(loaded[1][0]),
                                       len(loaded[1][1])))  # fmt: skip
        duel.replay(self.responses, observer=host)
        if not host.won and self.result.get("reason") in _LIMIT_REASONS:
            winner = self.result.get("winner")
            seat = 2 if winner is None else seat_of_deck(duel.first, winner)
            host.packets += _packet(
                C.MSG_WIN, bytes([seat, WIN_REASON_LIMIT])
            )  # the host's own [player, reason] packet

        timestamp = self.recorded_at or 0
        body = bytearray(_names_block(seat_names))
        body += struct.pack("<Q", self.rule_flags)
        body += host.packets
        body += _packet(OLD_REPLAY_MODE, self._yrp1(seat_names, loaded, timestamp, compress))
        Path(path).write_bytes(_replay_file(REPLAY_YRPX, self.core_seed, bytes(body), timestamp, compress))

    def _yrp1(self, seat_names: tuple[str, str], loaded, timestamp: int, compress: bool) -> bytes:
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
        return _replay_file(REPLAY_YRP1, self.core_seed, bytes(body), timestamp, compress)


# ---------------------------------------------------------------- EDOPro host packet stream

# Query flags of the host's refreshes (gframe/generic_duel.h default arguments).
_Q_MZONE, _Q_SZONE, _Q_HAND, _Q_GRAVE, _Q_EXTRA = 0x3981FFF, 0x3F81FFF, 0x3781FFF, 0x381FFF, 0x381FFF
_Q_SINGLE, _Q_DECK, _Q_SET_CARD = 0x3F81FFF, 0x1181FFF, 0x3181FFF
_Q_TAG_MZONE, _Q_TAG_SZONE = 0x3181FFF, 0x3781FFF
_AFTER_FIELD = frozenset({C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END, C.MSG_SUMMONED, C.MSG_SPSUMMONED,
                          C.MSG_FLIPSUMMONED, C.MSG_NEW_PHASE, C.MSG_CHAINED, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END})  # fmt: skip
_KNOWN_QUERIES = frozenset(1 << i for i in range(26)) | {C.QUERY_END}


def _loc_info(payload: bytes, pos: int) -> tuple[int, int, int, int]:
    """(controller, location, sequence, position) of a 10-byte loc_info (CoreUtils::ReadLocInfo)."""
    con, loc, seq, position = struct.unpack_from("<BBII", payload, pos)
    return con, loc, seq, position


def _host_query(raw: bytes, pos: int) -> tuple[bytes, int]:
    """One card's query as EDOPro's host re-serialises it (CoreUtils::Query::Parse + GenerateBuffer(false, false)).

    The core writes ``[u16 size][u32 flag][data]`` records up to QUERY_END (a lone ``u16 0`` is an empty
    slot). The host writes the flags it read in ascending order, drops a reason / equip card without a
    location and keeps unknown flags without their data. Returns the bytes and the position after the card.
    """
    (size,) = struct.unpack_from("<H", raw, pos)
    if size == 0:
        return b"\0\0", pos + 2
    fields: dict[int, bytes] = {}
    while True:
        (size, flag) = struct.unpack_from("<HI", raw, pos)
        fields[flag] = raw[pos + 6 : pos + 2 + size] if flag in _KNOWN_QUERIES else b""
        pos += 2 + size
        if flag == C.QUERY_END:
            break
    out = bytearray()
    for flag in sorted(fields):
        data = fields[flag]
        if flag in (C.QUERY_REASON_CARD, C.QUERY_EQUIP_CARD) and data[1] == 0:
            continue
        out += struct.pack("<HI", len(data) + 4, flag) + data
    return bytes(out), pos


class _HostStream:
    """The packets EDOPro's host (gframe/generic_duel.cpp) records into a ``.yrpX``, rebuilt from a live core.

    ``GenericDuel::Analyze``: every engine message but decisions and private hints is recorded;
    ``BeforeParsing`` / ``AfterParsing`` add MSG_UPDATE_DATA / MSG_UPDATE_CARD refreshes queried from the
    core, and for the messages ``BeforeParsing`` handles the message itself goes after them
    (``record_last``). The host stops reading a buffer at a decision and stops recording at MSG_WIN.
    Unlike the host we also leave out MSG_RETRY (the host ends the duel there; a recorded game that
    retried a response is still worth watching).
    """

    def __init__(self, start_payload: bytes) -> None:
        self.start_payload = start_payload
        self.packets = bytearray()
        self.won = False
        self._core = None
        self._out: list[bytes] = []

    # -- observer protocol (Duel.replay) --------------------------------
    def on_start(self, core) -> None:
        self._core = core
        self.packets += _packet(C.MSG_START, self.start_payload)
        self._out = []
        self._deck(0)
        self._deck(1)
        self._location(0, C.LOCATION_EXTRA, _Q_EXTRA)
        self._location(1, C.LOCATION_EXTRA, _Q_EXTRA)
        self._flush()

    def on_buffer(self, core, buf: bytes) -> None:
        if self.won:
            return
        self._core = core
        for record in M.split_messages(buf):
            msg, payload = record[0], record[1:]
            self._out = []
            last = self._before(msg, payload)
            record_it = not (msg in M.DECISION_TYPES or msg == C.MSG_RETRY
                             or (msg == C.MSG_HINT and payload[:1] and payload[0] in _PRIVATE_HINTS))  # fmt: skip
            self._after(msg, payload)
            if record_it:
                packet = _packet(msg, payload)
                self._out.insert(len(self._out) if last else 0, packet)
            self._flush()
            if msg == C.MSG_WIN:
                self.won = True
                return
            if msg in M.DECISION_TYPES:
                return

    # -- GenericDuel::BeforeParsing / AfterParsing ----------------------
    def _before(self, msg: int, p: bytes) -> bool:
        if msg in (C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD):
            self._field(hands=True)
        elif msg in (C.MSG_SELECT_CHAIN, C.MSG_NEW_TURN):
            self._field()
        elif msg == C.MSG_FLIPSUMMONING:
            con, loc, seq, _ = _loc_info(p, 4)
            self._single(con, loc, seq)
        else:
            return False
        return True

    def _after(self, msg: int, p: bytes) -> None:
        if msg in (C.MSG_SHUFFLE_HAND, C.MSG_DRAW):
            self._location(p[0], C.LOCATION_HAND, _Q_HAND)
        elif msg == C.MSG_SHUFFLE_EXTRA:
            self._location(p[0], C.LOCATION_EXTRA, _Q_EXTRA)
        elif msg == C.MSG_SWAP_GRAVE_DECK:
            self._location(p[0], C.LOCATION_GRAVE, _Q_GRAVE)
        elif msg == C.MSG_REVERSE_DECK:
            self._deck(0)
            self._deck(1)
        elif msg == C.MSG_SHUFFLE_SET_CARD:
            self._location(0, p[0], _Q_SET_CARD)
            self._location(1, p[0], _Q_SET_CARD)
        elif msg in _AFTER_FIELD:
            if msg == C.MSG_CHAIN_END:
                self._deck(0)
                self._deck(1)
            self._location(0, C.LOCATION_MZONE, _Q_MZONE)
            self._location(1, C.LOCATION_MZONE, _Q_MZONE)
            if msg not in (C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END):
                self._location(0, C.LOCATION_SZONE, _Q_SZONE)
                self._location(1, C.LOCATION_SZONE, _Q_SZONE)
            if msg in (C.MSG_NEW_PHASE, C.MSG_CHAINED, C.MSG_CHAIN_END):
                self._location(0, C.LOCATION_HAND, _Q_HAND)
                self._location(1, C.LOCATION_HAND, _Q_HAND)
        elif msg == C.MSG_MOVE:
            prev_con, prev_loc, _, _ = _loc_info(p, 4)
            con, loc, seq, _ = _loc_info(p, 14)
            if loc and not loc & C.LOCATION_OVERLAY and (loc != prev_loc or con != prev_con):
                self._single(con, loc, seq)
        elif msg == C.MSG_POS_CHANGE:
            con, loc, seq, prev_pos, pos = p[4], p[5], p[6], p[7], p[8]
            if prev_pos & C.POS_FACEDOWN and pos & C.POS_FACEUP:
                self._single(con, loc, seq)
        elif msg == C.MSG_SWAP:
            first, second = _loc_info(p, 4), _loc_info(p, 18)
            self._single(*first[:3])
            self._single(*second[:3])
        elif msg == C.MSG_TAG_SWAP:
            self._deck(p[0])
            self._location(p[0], C.LOCATION_EXTRA, _Q_EXTRA)
            self._location(0, C.LOCATION_MZONE, _Q_TAG_MZONE)
            self._location(1, C.LOCATION_MZONE, _Q_TAG_MZONE)
            self._location(0, C.LOCATION_SZONE, _Q_TAG_SZONE)
            self._location(1, C.LOCATION_SZONE, _Q_TAG_SZONE)
            self._location(0, C.LOCATION_HAND, _Q_HAND)
            self._location(1, C.LOCATION_HAND, _Q_HAND)
        elif msg == C.MSG_RELOAD_FIELD:
            self._location(0, C.LOCATION_EXTRA, _Q_EXTRA)
            self._location(1, C.LOCATION_EXTRA, _Q_EXTRA)

    def _field(self, hands: bool = False) -> None:
        for loc, flags in ((C.LOCATION_MZONE, _Q_MZONE), (C.LOCATION_SZONE, _Q_SZONE)):
            self._location(0, loc, flags)
            self._location(1, loc, flags)
        if hands:
            self._location(0, C.LOCATION_HAND, _Q_HAND)
            self._location(1, C.LOCATION_HAND, _Q_HAND)

    # -- GenericDuel::RefreshLocation / RefreshSingle / PseudoRefreshDeck --
    def _location(self, player: int, location: int, flags: int) -> None:
        raw = self._core.query_location(flags, player, location)
        if not raw:
            return
        (size,) = struct.unpack_from("<I", raw, 0)
        body, pos = bytearray(), 4
        while pos < 4 + size:
            card, pos = _host_query(raw, pos)
            body += card
        self._out.append(_packet(C.MSG_UPDATE_DATA, struct.pack("<BBI", player, location & 0xFF, len(body)) + body))

    def _single(self, player: int, location: int, sequence: int) -> None:
        raw = self._core.query(_Q_SINGLE, player, location & 0xFF, sequence & 0xFF, 0)
        if not raw:
            return
        card, _ = _host_query(raw, 0)
        self._out.append(_packet(C.MSG_UPDATE_CARD, bytes([player, location & 0xFF, sequence & 0xFF]) + card))

    def _deck(self, player: int) -> None:
        raw = self._core.query_location(_Q_DECK, player, C.LOCATION_DECK)
        if raw:
            self._out.append(_packet(C.MSG_UPDATE_DATA, bytes([player, C.LOCATION_DECK]) + raw))

    def _flush(self) -> None:
        for packet in self._out:
            self.packets += packet
        self._out = []


MAX_CARD_CODE = (1 << 32) - 1  # the core's card codes are 32-bit
MAX_WORD = (1 << 64) - 1


def _is_int(v: object, lo: int | None = None, hi: int | None = None) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and (lo is None or v >= lo) and (hi is None or v <= hi)


def _need_int(obj: dict, key: str, name: str, lo: int | None = None, hi: int | None = None) -> None:
    v = obj[key]
    if not _is_int(v, lo, hi):
        bounds = "" if lo is None else f" >= {lo}" if hi is None else f" in [{lo}, {hi}]"
        got = repr(v) if _is_int(v) else type(v).__name__
        raise ValueError(f"replay field {name!r} must be an integer{bounds}, not {got}")


def _check_replay_fields(d: dict[str, Any]) -> None:
    """Type-check a replay's JSON fields (``format`` keys removed) so that loading, ``--verify`` and export
    fail with a ValueError naming the field instead of a TypeError deep in the engine."""
    unknown = sorted(set(d) - {f.name for f in fields(Replay)})
    if unknown:
        raise ValueError(f"unknown replay field(s) {unknown}")
    for key in ("seed", "first", "rule_flags", "player", "shuffle_decks", "decks", "responses"):
        if key not in d:
            raise ValueError(f"replay is missing field {key!r}")
    _need_int(d, "seed", "seed")
    if not _is_int(d["first"], 0, 1):
        raise ValueError(f"replay field 'first' must be 0 or 1, not {d['first']!r}")
    _need_int(d, "rule_flags", "rule_flags", 0, MAX_WORD)
    player = d["player"]
    if not isinstance(player, dict):
        raise ValueError(f"replay field 'player' must be an object, not {type(player).__name__}")
    keys = {f.name for f in fields(PlayerRules)}
    if set(player) != keys:
        raise ValueError(f"replay field 'player' needs keys {sorted(keys)}, has {sorted(player)}")
    for key in sorted(keys):
        _need_int(player, key, f"player.{key}", 0, (1 << 31) - 1)
    for key in ("shuffle_decks", "augmented_start"):
        if key in d and not isinstance(d[key], bool):
            raise ValueError(f"replay field {key!r} must be true or false, not {d[key]!r}")
    decks = d["decks"]
    if not isinstance(decks, dict) or set(decks) != {"a", "b"}:
        raise ValueError("replay field 'decks' must be an object with decks 'a' and 'b'")
    for side, deck in decks.items():
        if not isinstance(deck, dict) or not {"main", "extra"} <= set(deck):
            raise ValueError(f"replay field 'decks.{side}' must be an object with 'main' and 'extra' lists")
        if not isinstance(deck.get("name", ""), str):
            raise ValueError(f"replay field 'decks.{side}.name' must be a string")
        for section in ("main", "extra", "side"):
            cards = deck.get(section, [])
            if not isinstance(cards, list):
                raise ValueError(f"replay field 'decks.{side}.{section}' must be a list of card passwords")
            for i, code in enumerate(cards):
                if not _is_int(code, 1, MAX_CARD_CODE):
                    raise ValueError(f"replay field 'decks.{side}.{section}[{i}]' is not a card password: {code!r}")
    responses = d["responses"]
    if not isinstance(responses, list):
        raise ValueError("replay field 'responses' must be a list of hex strings")
    for i, r in enumerate(responses):
        try:
            bytes.fromhex(r)
        except (TypeError, ValueError):
            raise ValueError(f"replay field 'responses[{i}]' is not a hex string: {r!r}") from None
    env = d.get("environment")
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError(f"replay field 'environment' must be null or an object, not {type(env).__name__}")
        if not isinstance(env.get("version"), str):
            raise ValueError("replay field 'environment.version' must be a string")
        if not isinstance(env.get("fingerprint", ""), str):
            raise ValueError("replay field 'environment.fingerprint' must be a string")
    words = d.get("seed_words")
    four_words = isinstance(words, list) and len(words) == 4 and all(_is_int(w, 0, MAX_WORD) for w in words)
    if words is not None and not four_words:
        raise ValueError(f"replay field 'seed_words' must be null or four 64-bit words, not {words!r}")
    at = d.get("recorded_at")
    if at is not None and not _is_int(at, 0, 0xFFFFFFFF):
        raise ValueError(f"replay field 'recorded_at' must be null or unix seconds (32-bit), not {at!r}")
    for key in ("max_turns", "max_decisions", "learner"):
        if key in d:
            _need_int(d, key, key, 0)
    if not isinstance(d.get("curriculum", ""), str):
        raise ValueError("replay field 'curriculum' must be a string")
    for key, typ, what in (("engine", dict, "an object"), ("result", dict, "an object"), ("steps", list, "a list")):
        if key in d and not isinstance(d[key], typ):
            raise ValueError(f"replay field {key!r} must be {what}, not {type(d[key]).__name__}")
    ocg = d.get("engine", {}).get("ocgcore")
    if ocg is not None and not (isinstance(ocg, list) and len(ocg) == 2 and all(_is_int(x) for x in ocg)):
        raise ValueError(f"replay field 'engine.ocgcore' must be [major, minor], not {ocg!r}")
    result = d.get("result", {})
    lp = result.get("lp")
    if lp is not None and not (isinstance(lp, list) and len(lp) == 2 and all(_is_int(x) for x in lp)):
        raise ValueError(f"replay field 'result.lp' must be [lp_a, lp_b], not {lp!r}")
    if result.get("winner") is not None and not _is_int(result["winner"], 0, 1):
        raise ValueError(f"replay field 'result.winner' must be 0, 1 or null, not {result['winner']!r}")
    for key in ("turns", "decisions", "win_reason"):
        if result.get(key) is not None and not _is_int(result[key]):
            raise ValueError(f"replay field 'result.{key}' must be an integer, not {result[key]!r}")


def _names_block(names: tuple[str, str]) -> bytes:
    out = bytearray()
    for name in names:
        encoded = name.encode("utf-16-le")[:38]
        out += struct.pack("<I", 1) + encoded + b"\0" * (40 - len(encoded))
    return bytes(out)


def _packet(msg_type: int, payload: bytes) -> bytes:
    return struct.pack("<BI", msg_type, len(payload)) + payload


def _replay_file(ident: int, seed_words: list[int], body: bytes, timestamp: int, compress: bool) -> bytes:
    """Extended header + body, LZMA-compressed like EDOPro's Replay::EndRecord when ``compress``.

    A compressed body keeps its uncompressed length in ``datasize`` and its 5 LZMA property bytes in the
    header's ``props``; what follows the header is the bare LZMA stream (no ``.lzma`` header).
    """
    major, minor = _core.ocg_version()
    version = EDOPRO_VERSION[0] | (EDOPRO_VERSION[1] << 8) | (major << 16) | (minor << 24)
    flag = REPLAY_LUA64 | REPLAY_NEWREPLAY | REPLAY_64BIT_DUELFLAG | REPLAY_EXTENDED_HEADER
    props, data = b"", body
    if compress:
        flag |= REPLAY_COMPRESSED
        # The .lzma container is 5 property bytes, a u64 size (unknown here) and the stream. liblzma ends the stream
        # with an end marker, which EDOPro's LzmaUncompress (LZMA_FINISH_ANY, stops at datasize) never reaches.
        alone = lzma.compress(body, format=lzma.FORMAT_ALONE, filters=_LZMA_FILTERS)
        props, data = alone[:5], alone[13:]
    base = struct.pack("<6I8s", ident, version, flag, timestamp & 0xFFFFFFFF, len(body), 0, props)
    return base + struct.pack("<Q4Q", 1, *seed_words) + data


# ------------------------------------------------------------------ reading .yrp / .yrpX

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


__all__ = ["REPLAY_COMPRESSED", "REPLAY_YRP1", "REPLAY_YRPX", "Replay", "ReplayEnvironmentMismatch", "YrpError",
           "YrpFile", "load_yrp", "parse_yrp"]  # fmt: skip
