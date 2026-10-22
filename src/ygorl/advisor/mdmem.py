"""Read-only Master Duel state reader (design 10, T7.6): game memory -> :class:`BoardSnapshot`.

Linux + Proton only. The reader finds the game process, locates ``duel.dll`` in ``/proc/<pid>/maps``, derives the
duel-state layout from that DLL (:mod:`ygorl.advisor.mdoffsets`) and copies the state out with
``process_vm_readv``. It never writes to the process, never sends input, never injects code.

What the advisor cannot see on screen is never put in a snapshot, even though the local ``duel.dll`` may hold it:
the opponent's hand and deck are counts only, the opponent's face-down cards are unknown (password 0). Until a
build's flag semantics are checked against the screen (``Offsets.verified``), :func:`to_snapshot` refuses to run and
:func:`format_raw` hides every opponent card outside the GY.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ygorl.advisor.mdoffsets import FIELD_SLOTS, PILES, Offsets, load_or_derive
from ygorl.advisor.snapshot import UNKNOWN, BoardSnapshot, SnapshotCard, SnapshotError, SnapshotSide

PROCESS_NAME = "masterduel.exe"
DLL_NAME = "duel.dll"
MD_APP_ID = 1449850
PHASES = ("draw", "standby", "main1", "battle", "main2", "end")  # DLL_DuelGetCurrentPhase values
STATE_SIZE = 0x3C40  # S .. past the turn/phase fields; covers both player blocks and the uid table


class MemoryReadError(OSError):
    """The game process is missing or cannot be read."""


class Memory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


class _IoVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


class ProcessMemory:
    """Reads another process's memory with ``process_vm_readv`` (needs ptrace access, see :func:`ptrace_hint`)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self._readv = libc.process_vm_readv
        self._readv.argtypes = [ctypes.c_int, ctypes.POINTER(_IoVec), ctypes.c_ulong, ctypes.POINTER(_IoVec),
                                ctypes.c_ulong, ctypes.c_ulong]  # fmt: skip
        self._readv.restype = ctypes.c_ssize_t

    def read(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        local, remote = _IoVec(ctypes.addressof(buf), size), _IoVec(address, size)
        n = self._readv(self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        if n != size:
            err = ctypes.get_errno()
            msg = f"process_vm_readv({self.pid}, {address:#x}, {size}): {os.strerror(err) if n < 0 else 'short read'}"
            if err == errno.EPERM:
                msg += "; " + ptrace_hint()
            raise MemoryReadError(err, msg)
        return buf.raw


class BufferMemory:
    """Memory made of ``{address: bytes}`` regions (tests and saved dumps)."""

    def __init__(self, regions: Mapping[int, bytes]) -> None:
        self.regions = dict(regions)

    def read(self, address: int, size: int) -> bytes:
        for start, data in self.regions.items():
            if start <= address and address + size <= start + len(data):
                return data[address - start : address - start + size]
        raise MemoryReadError(errno.EFAULT, f"no region holds {address:#x}+{size}")


def ptrace_hint() -> str:
    try:
        scope = Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip()
    except OSError:
        return "reading another process needs ptrace permission"
    if scope == "0":
        return "ptrace_scope is 0; is the game running as another user?"
    return (f"kernel.yama.ptrace_scope is {scope}: only a parent process may read the game. Either run "
            "`sudo sysctl kernel.yama.ptrace_scope=0` (until reboot) or run the reader with sudo")  # fmt: skip


# --- locating the game ------------------------------------------------------------------------------------------


def find_pid(proc: Path = Path("/proc")) -> int:
    """PID of the running Master Duel client; :class:`MemoryReadError` if there is none."""
    found = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            comm = (d / "comm").read_text().strip()
        except OSError:
            continue
        if comm.lower() == PROCESS_NAME:
            found.append(int(d.name))
    if not found:
        raise MemoryReadError(errno.ESRCH, f"{PROCESS_NAME} is not running")
    return min(found)


@dataclass(frozen=True)
class Module:
    base: int
    path: Path  # the DLL on disk (a Unix path under the Steam library)


def find_module(pid: int, name: str = DLL_NAME, proc: Path = Path("/proc")) -> Module:
    """Load address of ``name`` in ``pid`` from its memory map (the mapping of file offset 0)."""
    try:
        lines = (proc / str(pid) / "maps").read_text().splitlines()
    except OSError as exc:
        raise MemoryReadError(exc.errno or errno.EIO, f"cannot read the memory map of {pid}: {exc}") from None
    for line in lines:
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].lower().endswith("/" + name) and int(parts[2], 16) == 0:
            return Module(int(parts[0].split("-")[0], 16), Path(parts[5]))
    raise MemoryReadError(errno.ENOENT, f"{name} is not loaded in process {pid} (is a duel open?)")


def steam_buildid(dll: Path) -> str:
    """Steam ``buildid`` of the install holding ``dll`` ('' if the app manifest is not found)."""
    for parent in dll.parents:
        if parent.name == "steamapps":
            manifest = parent / f"appmanifest_{MD_APP_ID}.acf"
            if manifest.is_file():
                m = re.search(r'"buildid"\s+"(\d+)"', manifest.read_text(errors="replace"))
                return m.group(1) if m else ""
    return ""


# --- raw state --------------------------------------------------------------------------------------------------


def decode_uid(word: int) -> int:
    return (word & 1) | ((word >> 8) << 1)


@dataclass(frozen=True)
class RawCard:
    cid: int  # MD card id (= Konami database id)
    uid: int  # unique id of this card in the duel
    flags: int  # low byte of the uid word without the uid bit
    face: int = 0  # field slots only: the DLL_DuelGetCardFace byte (face-up)
    sideways: int = 0  # field slots only: the DLL_DuelGetCardTurn byte (turned sideways = Defense Position)
    facedown: bool = False  # banished cards only
    materials: tuple[int, ...] = ()  # Xyz materials, MD card ids


@dataclass(frozen=True)
class RawPlayer:
    lp: int
    slots: tuple[RawCard | None, ...]  # FIELD_SLOTS entries: 0-4 Main Monster, 5-6 Extra Monster, 7-11 S/T, 12 Field
    piles: dict[str, tuple[RawCard, ...]]  # PILES -> cards in index order
    counts: dict[str, int]


@dataclass(frozen=True)
class RawDuel:
    turn: int
    phase: int
    turn_player: int
    me: int  # which of players[0..1] the client plays
    players: tuple[RawPlayer, RawPlayer]
    offsets: Offsets = field(repr=False)


def _u16(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]


def _u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def read_duel(mem: Memory, base: int, off: Offsets) -> RawDuel:
    """Copy the duel state out of a process whose ``duel.dll`` is loaded at ``base``."""
    state = struct.unpack("<Q", mem.read(base + off.state_ptr, 8))[0]
    key_struct = struct.unpack("<Q", mem.read(base + off.key_ptr, 8))[0]
    if not state or not key_struct:
        raise MemoryReadError(errno.EAGAIN, "no duel in progress (duel-state pointer is null)")
    s = mem.read(state, STATE_SIZE)
    key = mem.read(key_struct, 8)
    xor_key, me = _u16(key, 0), _u32(key, off.my_player) & 1

    def node(index: int) -> bytes:
        rel = off.effect_table + index * off.effect_stride
        return s[rel : rel + off.effect_stride] if rel + off.effect_stride <= len(s) else mem.read(state + rel, off.effect_stride)  # fmt: skip

    def card_id(uid: int) -> int:
        return _u16(s, off.uid_table + (uid & off.uid_mask) * 8)

    def materials(head: int) -> tuple[int, ...]:
        out, seen = [], set()
        while head and head not in seen and len(seen) < 4096:
            seen.add(head)
            n = node(head)
            if _u16(n, 0) == off.overlay_code and n[off.effect_kind] & 0xF == off.overlay_kind:
                out.append(card_id(_u32(n, off.effect_uid)))
            head = _u16(n, off.effect_next)
        return tuple(out)

    players = []
    for p in (0, 1):
        blk = p * off.player_stride
        slots: list[RawCard | None] = []
        for i in range(FIELD_SLOTS):
            at = blk + off.slot_base + i * off.slot_stride
            cid = _u16(s, at)
            if not cid:
                slots.append(None)
                continue
            w = _u16(s, at + off.slot_uid)
            slots.append(RawCard(cid=cid, uid=decode_uid(w), flags=w & 0xFE, face=s[at + off.slot_face],
                                 sideways=s[at + off.slot_turn], materials=materials(_u16(s, at + off.slot_chain))))  # fmt: skip
        counts = {z: _u32(s, blk + off.count[z]) for z in PILES}
        piles = {}
        for z in PILES:
            n = min(counts[z], 255)
            entries = [blk + off.pile[z] + i * off.pile_stride for i in range(n)]
            piles[z] = tuple(RawCard(cid=_u16(s, e), uid=decode_uid(_u16(s, e + 2)), flags=_u16(s, e + 2) & 0xFE,
                                     facedown=z == "removed" and bool(s[blk + off.removed_face + 2 * i] & off.removed_facedown))
                             for i, e in enumerate(entries))  # fmt: skip
        lp = (_u32(s, blk + off.lp) ^ xor_key) & 0xFFFFFFFF
        players.append(RawPlayer(lp=lp, slots=tuple(slots), piles=piles, counts=counts))
    return RawDuel(turn=_u32(s, off.turn), phase=_u32(s, off.phase), turn_player=_u32(s, off.turn_player) & 1,
                   me=me, players=(players[0], players[1]), offsets=off)  # fmt: skip


@dataclass(frozen=True)
class Attached:
    pid: int
    module: Module
    offsets: Offsets
    buildid: str
    memory: Memory

    def read(self) -> RawDuel:
        return read_duel(self.memory, self.module.base, self.offsets)


def attach(pid: int | None = None) -> Attached:
    """Find the game and its ``duel.dll``; derive (or load the recorded) offsets for that exact DLL."""
    pid = pid or find_pid()
    module = find_module(pid)
    return Attached(pid, module, load_or_derive(module.path), steam_buildid(module.path), ProcessMemory(pid))


# --- presentation and conversion --------------------------------------------------------------------------------

SLOT_NAMES = [f"M{i + 1}" for i in range(5)] + ["EMZ-L", "EMZ-R"] + [f"S{i + 1}" for i in range(5)] + ["FIELD"]


def format_raw(raw: RawDuel, names: Mapping[int, str] | None = None) -> str:
    """Human-readable dump for checking a build against the screen: the advisor's own cards in full, the
    opponent's field as hidden placeholders (flag semantics may be unknown), GY by name."""
    names = names or {}

    def label(c: RawCard, hidden: bool) -> str:
        if hidden:
            return f"(hidden) uid={c.uid} flags={c.flags:#04x}"
        return f"{c.cid} {names.get(c.cid, '?')} uid={c.uid} flags={c.flags:#04x}"

    phase = PHASES[raw.phase] if 0 <= raw.phase < len(PHASES) else str(raw.phase)
    lines = [f"turn {raw.turn} {phase}, turn player {'me' if raw.turn_player == raw.me else 'opponent'}"
             f" (MD player {raw.turn_player}; I am {raw.me})"]  # fmt: skip
    for p in (raw.me, 1 - raw.me):
        pl = raw.players[p]
        mine = p == raw.me
        lines.append(f"{'ME' if mine else 'OPPONENT'}: LP {pl.lp}, " + ", ".join(f"{z} {pl.counts[z]}" for z in PILES))
        for i, c in enumerate(pl.slots):
            if c is not None:
                hide = not mine
                extra = f" face={c.face:#04x} sideways={c.sideways}"
                mats = f" materials={len(c.materials)}" if c.materials else ""
                lines.append(f"  {SLOT_NAMES[i]:6} {label(c, hide)}{extra if not hide else ''}{mats}")
        for z in ("hand", "grave", "removed") if mine else ("grave",):
            for c in pl.piles[z]:
                lines.append(f"  {z:6} {label(c, False)}")
    return "\n".join(lines)


def to_snapshot(raw: RawDuel, passwords: Mapping[int, int]) -> BoardSnapshot:
    """The advisor's view (seat 0 = the client) as a :class:`BoardSnapshot`; MD card ids -> passwords."""
    off = raw.offsets
    if not off.verified:
        raise SnapshotError(f"duel.dll {off.dll_sha256[:16]} is not verified against the screen yet; "
                            "use --raw and record the flag semantics first")  # fmt: skip
    if raw.turn_player != raw.me or PHASES[raw.phase] not in ("main1", "main2"):
        raise SnapshotError("the advisor only handles its own Main Phase")

    def pw(cid: int) -> int:
        if cid not in passwords:
            raise SnapshotError(f"MD card id {cid} has no password in md_cid.json")
        return passwords[cid]

    sides = []
    for p in (raw.me, 1 - raw.me):
        pl, mine = raw.players[p], p == raw.me
        cards: list[SnapshotCard] = []
        for i, c in enumerate(pl.slots):
            if c is None:
                continue
            loc, seq = ("mzone", i) if i < 7 else ("szone", i - 7)
            faceup = bool(c.face & off.semantics["faceup"])
            if loc == "mzone":
                defense = bool(c.sideways & off.semantics["defense"])
                pos = ("faceup_" if faceup else "facedown_") + ("defense" if defense else "attack")
            else:
                pos = "faceup" if faceup else "facedown"
            known = mine or faceup
            cards.append(SnapshotCard(location=loc, sequence=seq, position=pos,
                                      password=pw(c.cid) if known else UNKNOWN,
                                      materials=tuple(pw(m) for m in c.materials) if known else ()))  # fmt: skip
        cards += [SnapshotCard("grave", pw(c.cid)) for c in pl.piles["grave"]]
        for c in pl.piles["removed"]:
            down = c.facedown
            cards.append(SnapshotCard("removed", pw(c.cid) if mine or not down else UNKNOWN,
                                      position="facedown" if down else ""))  # fmt: skip
        if mine:
            cards += [SnapshotCard("hand", pw(c.cid)) for c in pl.piles["hand"]]
        else:
            cards += [SnapshotCard("hand") for _ in range(pl.counts["hand"])]
        sides.append(SnapshotSide(lp=pl.lp, cards=tuple(cards), deck_count=pl.counts["deck"],
                                  extra_count=pl.counts["extra"]))  # fmt: skip
    went_first = (raw.turn % 2 == 1) == (raw.turn_player == raw.me)
    return BoardSnapshot(turn=raw.turn, phase=PHASES[raw.phase], sides=(sides[0], sides[1]), turn_player=0,
                         went_first=went_first, source=f"mdmem:{off.dll_sha256[:16]}").check()  # fmt: skip


__all__ = ["Attached", "BufferMemory", "Memory", "MemoryReadError", "Module", "ProcessMemory", "RawCard", "RawDuel",
           "RawPlayer", "attach", "decode_uid", "find_module", "find_pid", "format_raw", "ptrace_hint", "read_duel",
           "load_md_cid", "steam_buildid", "to_snapshot"]  # fmt: skip


def load_md_cid(path: str | Path) -> dict[int, int]:
    """MD card id -> password from an ``md_cid.json`` artifact (tools/build_md_cid.py)."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in data["cards"].items()}
