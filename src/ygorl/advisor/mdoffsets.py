"""Derive the Master Duel duel-state layout from ``duel.dll`` itself (design 10, T7.6).

MD's rule engine ``duel.dll`` exports small query functions (``DLL_DuelGetLP``, ``DLL_DuelGetCardNum``, ...) that
read one global duel-state struct. Their names survive game updates even when every address and field moves, so
instead of hand-maintained offsets per build we disassemble those exports (GNU ``objdump``) and pattern-match the
displacements they use. After an update, :func:`derive` re-runs on the new DLL; only the *meaning* of flag bits
needs a human check, recorded per build in ``offsets/<sha256 prefix>.json`` (``verified``).

Layout found in build 23020425 (``S`` = ``*(duel.dll + state_ptr)``, ``p`` = player 0/1)::

    S + p*player_stride + lp            LP, XOR the u16 at *(duel.dll + key_ptr)
    S + p*player_stride + count[zone]   u32 card counts (hand, deck, extra, grave, removed)
    S + p*player_stride + slot_base + i*slot_stride   field slot i (0-6 Monster Zones, 7-11 S/T, 12 Field)
        +0 u16 card id, +2 u16 uid word, +slot_turn u8 (sideways = Defense Position), +slot_face u8 (face-up),
        +slot_chain u16 effect-list head
    S + p*player_stride + zone_base[zone] + i*4       pile entry: u16 card id, u16 uid word
    S + p*player_stride + removed_face + i*2          banished entry i: bit removed_facedown set = face-down
    S + uid_table + uid*8                             u16 card id of a unique id (uid < 512)
    S + effect_table + n*16                           effect-list node: u16 code, u8 kind, u32 uid, ..., u16 next

A uid word encodes ``uid = (w & 1) | (w >> 8) << 1``; its other low bits are flags. Xyz materials are the nodes on a
monster's effect list with ``code == overlay_code`` and ``kind & 0xF == overlay_kind``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

OFFSETS_DIR = Path(__file__).with_name("offsets")
PILES = ("hand", "extra", "deck", "grave", "removed")  # MD locate 13..17, in this order
FIELD_SLOTS = 13
SEMANTICS = {"faceup", "defense", "locked_byte", "locked", "cpu_type", "turn_base", "normal_summons", "set_turn"}


class OffsetError(RuntimeError):
    """``duel.dll`` does not look like the code the deriver understands (an update changed a query function)."""


@dataclass(frozen=True)
class Offsets:
    state_ptr: int  # RVA of the pointer to the duel state S
    key_ptr: int  # RVA of the pointer to the LP XOR key (u16 at +0) and my player number (u32 at +4)
    my_player: int
    player_type: int  # u32 per player at key struct + player_type + p*4; 0 = human (DLL_DuelIsHuman)
    duel_finish: int  # u32 at key struct + duel_finish: non-zero once the duel is decided (DLL_DuelGetDuelFinish)
    duel_result: int  # u32 at key struct + duel_result (DLL_DuelGetDuelResult)
    turn: int
    turn_player: int
    phase: int
    player_stride: int
    lp: int
    count: dict[str, int]  # pile -> u32 offset in the player block
    pile: dict[str, int]  # pile -> first entry in the player block
    pile_stride: int
    slot_base: int
    slot_stride: int
    slot_uid: int
    slot_turn: int  # DLL_DuelGetCardTurn: the card is turned sideways (Defense Position), not a turn number
    slot_face: int
    slot_chain: int
    removed_face: int  # per banished card, 2 bytes each
    removed_facedown: int  # bit mask in that byte
    uid_table: int
    uid_mask: int
    effect_table: int
    effect_stride: int
    effect_kind: int
    effect_uid: int
    effect_next: int
    overlay_code: int
    overlay_kind: int
    # filled by callers, not derived
    dll_sha256: str = ""
    buildid: str = ""
    verified: bool = False  # flag semantics checked against the screen in a Solo duel
    # read off the screen, not derived: "faceup" mask of a slot's face byte, "defense" mask of its turn byte,
    # "locked" mask of the slot byte at "locked_byte" (summoned, Set or repositioned this turn),
    # "cpu_type" the player type of the computer opponent in a Solo duel, "turn_base" what to add to the turn
    # counter to get the turn number on screen, "normal_summons" the u32 in the player block counting this turn's
    # Normal Summons, "set_turn" the mask in the slot byte at "locked_byte" for a card Set this turn
    semantics: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        def h(v):
            if isinstance(v, dict):
                return {k: h(x) for k, x in v.items()}
            return hex(v) if isinstance(v, int) and not isinstance(v, bool) else v

        return {k: h(v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict) -> Offsets:
        def num(v):
            return int(v, 16) if isinstance(v, str) and v.startswith("0x") else v

        out = {k: ({kk: num(vv) for kk, vv in v.items()} if isinstance(v, dict) else num(v)) for k, v in data.items()}
        offsets = cls(**out)
        if offsets.verified and set(offsets.semantics) != SEMANTICS:
            raise OffsetError(f"verified offsets need semantics {sorted(SEMANTICS)}")
        return offsets


# --- PE exports ---------------------------------------------------------------------------------------------


def pe_exports(data: bytes) -> tuple[int, dict[str, int]]:
    """(image base, export name -> RVA) of a PE32+ image."""
    if data[:2] != b"MZ":
        raise OffsetError("not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise OffsetError("not a PE file")
    opt = pe + 24
    if struct.unpack_from("<H", data, opt)[0] != 0x20B:
        raise OffsetError("not a 64-bit PE image")
    image_base = struct.unpack_from("<Q", data, opt + 24)[0]
    export_rva = struct.unpack_from("<I", data, opt + 112)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    sections = [struct.unpack_from("<8sIIII", data, opt + opt_size + 40 * i) for i in range(nsec)]

    def off(rva: int) -> int:
        for _, vsize, va, rsize, raw in sections:
            if va <= rva < va + max(vsize, rsize):
                return rva - va + raw
        raise OffsetError(f"RVA {rva:#x} is outside every section")

    e = off(export_rva)
    count, funcs, names, ords = struct.unpack_from("<IIII", data, e + 24)
    exports = {}
    for i in range(count):
        name_off = off(struct.unpack_from("<I", data, off(names) + 4 * i)[0])
        name = data[name_off : data.index(b"\0", name_off)].decode("ascii")
        ordinal = struct.unpack_from("<H", data, off(ords) + 2 * i)[0]
        exports[name] = struct.unpack_from("<I", data, off(funcs) + 4 * ordinal)[0]
    return image_base, exports


# --- disassembly ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Insn:
    addr: int
    op: str  # mnemonic
    args: str  # Intel-syntax operands, lower case
    target: int | None  # the address objdump resolved (rip-relative operand or branch target)


_LINE = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*([^#<]*?)\s*(?:#\s*0x([0-9a-f]+).*|<[^>]*>)?\s*$")


def parse_objdump(text: str) -> list[Insn]:
    out = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        addr, op, args, comment = m.groups()
        target = int(comment, 16) if comment else None
        if target is None and op.startswith("j") and re.fullmatch(r"0x[0-9a-f]+", args):
            target = int(args, 16)
        out.append(Insn(int(addr, 16), op, args.lower(), target))
    return out


Disassembler = Callable[[int, int], list[Insn]]


def objdump_disassembler(path: Path) -> Disassembler:
    """``dis(va, length)`` over ``path`` with GNU objdump (binutils)."""
    exe = shutil.which("objdump")
    if exe is None:
        raise OffsetError("objdump (GNU binutils) is required to derive duel.dll offsets")

    def dis(va: int, length: int) -> list[Insn]:
        cmd = [exe, "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={va:#x}",
               f"--stop-address={va + length:#x}", str(path)]  # fmt: skip
        return parse_objdump(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)

    return dis


# --- derivation -------------------------------------------------------------------------------------------------

_HEX = r"0x([0-9a-f]+)"


class _Code:
    """Function bodies by export name; follows the one-``jmp`` thunks MSVC emits for exports."""

    def __init__(self, dis: Disassembler, image_base: int, exports: dict[str, int]) -> None:
        self.dis, self.base, self.exports = dis, image_base, exports

    def at(self, va: int, length: int = 0x200) -> list[Insn]:
        insns = self.dis(va, length)
        if insns and insns[0].op == "jmp" and insns[0].target is not None:
            return self.dis(insns[0].target, length)
        return insns

    def fn(self, name: str, length: int = 0x200) -> list[Insn]:
        if name not in self.exports:
            raise OffsetError(f"duel.dll has no export {name}")
        return self.at(self.base + self.exports[name], length)


def _until_ret(insns: list[Insn]) -> list[Insn]:
    for i, ins in enumerate(insns):
        if ins.op == "ret" or (ins.op == "jmp" and i > 0):
            return insns[: i + 1]
    return insns


def _find(insns: list[Insn], pattern: str, what: str) -> re.Match:
    rx = re.compile(pattern)
    for ins in insns:
        m = rx.search(f"{ins.op} {ins.args}")
        if m:
            return m
    raise OffsetError(f"cannot find {what} ({pattern!r})")


def _rip_loads(insns: list[Insn], base: int) -> list[int]:
    return [i.target - base for i in insns if "[rip+" in i.args and i.target is not None]


def _locate_chain(insns: list[Insn]) -> dict[int, int]:
    """The ``locate`` switch: ``sub edx,0xd; je A; sub edx,1; je B; ...; cmp edx,1; je E`` -> {13: A, 14: B, ...}
    (the last case may be ``cmp edx,1; jne default`` falling through)."""
    out: dict[int, int] = {}
    value = 0
    for a, b, c in zip(insns, insns[1:], insns[2:], strict=False):
        m = re.fullmatch(r"(\w+),(0x[0-9a-f]+)", a.args)
        if not m or a.op not in ("sub", "cmp") or b.op not in ("je", "jne") or b.target is None:
            continue
        value += int(m.group(2), 16)
        out[value] = b.target if b.op == "je" else c.addr  # `cmp edx,1; jne default` falls through to the case
        if a.op == "cmp":
            break
    return out


def derive(path: str | Path, dis: Disassembler | None = None) -> Offsets:
    """Offsets for the ``duel.dll`` at ``path``; :class:`OffsetError` if an export no longer matches."""
    path = Path(path)
    data = path.read_bytes()
    image_base, exports = pe_exports(data)
    code = _Code(dis or objdump_disassembler(path), image_base, exports)
    disp = r"\+" + _HEX + r"\]"

    def single_field(name: str) -> tuple[int, int]:
        body = _until_ret(code.fn(name, 0x40))
        (ptr,) = _rip_loads(body, image_base)[:1] or [None]
        if ptr is None:
            raise OffsetError(f"{name}: no global load")
        return ptr, int(_find(body, r"mov\s+\S+,dword ptr \[\w+" + disp, f"{name} field").group(1), 16)

    state_ptr, turn = single_field("DLL_DuelGetTurnNum")
    _, turn_player = single_field("DLL_DuelWhichTurnNow")
    _, phase = single_field("DLL_DuelGetCurrentPhase")
    key_ptr, my_player = single_field("DLL_DuelGetMyPlayerNum")

    human = _until_ret(code.fn("DLL_DuelIsHuman", 0x40))
    if _rip_loads(human, image_base)[:1] != [key_ptr]:
        raise OffsetError("DLL_DuelIsHuman: expected a load of the key struct pointer")
    player_type = int(_find(human, r"cmp\s+dword ptr \[\w+\+\w+\*4" + disp, "player type").group(1), 16)

    finish_ptr, duel_finish = single_field("DLL_DuelGetDuelFinish")
    result_ptr, duel_result = single_field("DLL_DuelGetDuelResult")
    if {finish_ptr, result_ptr} != {key_ptr}:
        raise OffsetError("DLL_DuelGetDuelFinish / Result: expected the key struct")

    lp_body = _until_ret(code.fn("DLL_DuelGetLP", 0x40))
    if set(_rip_loads(lp_body, image_base)) != {state_ptr, key_ptr}:
        raise OffsetError("DLL_DuelGetLP: expected loads of the state and key pointers")
    player_stride = int(_find(lp_body, r"imul\s+\w+,\w+," + _HEX, "player stride").group(1), 16)
    m = re.search(
        r"xor\s+\w+,dword ptr \[[\w*+]+?(?:\+" + _HEX + r")?\]", "\n".join(f"{i.op} {i.args}" for i in lp_body)
    )
    if not m:
        raise OffsetError("DLL_DuelGetLP: no XOR with the LP field")
    lp = int(m.group(1), 16) if m.group(1) else 0

    # DLL_DuelGetCardNum(p, locate): field slots for locate <= 12, a u32 count per pile otherwise
    num = code.fn("DLL_DuelGetCardNum", 0x100)
    strides = [int(m.group(1), 16) for i in num if i.op == "imul" and (m := re.search(_HEX + "$", i.args))]
    slot_stride = next((v for v in strides if v != player_stride), None)
    if slot_stride is None:
        raise OffsetError("DLL_DuelGetCardNum: no field slot stride")
    slot_base = int(_find(num, r"movzx\s+\w+,word ptr \[[\w*+]+" + disp, "slot card id").group(1), 16)
    counts = {}
    for locate, va in _locate_chain(num).items():
        body = _until_ret(code.at(va, 0x40))
        counts[PILES[locate - 13]] = int(
            _find(body, r"mov\s+\w+,dword ptr \[[\w*+]+" + disp, "pile count").group(1), 16
        )

    # DLL_DuelGetCardUniqueID(p, locate, index): uid word of a slot / pile entry
    uidf = code.fn("DLL_DuelGetCardUniqueID", 0x180)
    slot_uid = (
        int(_find(_until_ret(uidf), r"movzx\s+\w+,word ptr \[[\w*+]+" + disp, "slot uid").group(1), 16) - slot_base
    )
    pile, pile_stride = {}, 0
    for locate, va in _locate_chain(uidf).items():
        body = _until_ret(code.at(va, 0x40))
        quarter = int(_find(body, r"imul\s+\w+,\w+," + _HEX, "pile player stride").group(1), 16)
        if quarter * 4 != player_stride:
            raise OffsetError("DLL_DuelGetCardUniqueID: pile entries are no longer 4 bytes")
        pile_stride = 4
        pile[PILES[locate - 13]] = int(_find(body, r"word ptr \[\w+\+\w+\*4" + disp, "pile uid").group(1), 16) - 2
    if set(pile) != set(PILES) or set(counts) != set(PILES):
        raise OffsetError(f"locate switch decoded {sorted(pile)} / {sorted(counts)}, expected {PILES}")

    def slot_byte(name: str) -> int:
        body = _until_ret(code.fn(name, 0x80))
        return int(_find(body, r"movzx\s+\w+,byte ptr \[[\w*+]+" + disp, f"{name} byte").group(1), 16) - slot_base

    slot_turn, slot_face = slot_byte("DLL_DuelGetCardTurn"), slot_byte("DLL_DuelGetCardFace")

    # DLL_DuelGetCardFace(p, 17, i): banished cards keep their face in a separate 2-byte-per-card array
    face_fn = code.fn("DLL_DuelGetCardFace", 0x200)
    removed_va = _locate_chain(face_fn).get(17)
    if removed_va is None:
        raise OffsetError("DLL_DuelGetCardFace: no banished-card case")
    body = _until_ret(code.at(removed_va, 0x80))
    removed_face = int(_find(body, r"byte ptr \[\w+\+\w+\*2" + disp, "banished face array").group(1), 16)
    removed_facedown = 1 << int(_find(body, r"shr\s+\w+," + _HEX + "$", "banished face bit").group(1), 16)

    ids = _until_ret(code.fn("DLL_DuelGetCardIDByUniqueID2", 0x80))
    uid_mask = int(_find(ids, r"and\s+\w+," + _HEX, "uid mask").group(1), 16)
    # two u16 loads per entry: the card id and (at +4) an alias index; the card id comes first in the entry
    loads = [int(m.group(1), 16) for i in ids if (m := re.search(r"word ptr \[\w+\+\w+\*8" + disp, i.args))]
    if not loads:
        raise OffsetError("DLL_DuelGetCardIDByUniqueID2: no uid table load")
    uid_table = min(loads)

    ov = _until_ret(code.fn("DLL_DuelGetThisCardOverlayNum", 0x80))
    slot_chain = int(_find(ov, r"movzx\s+\w+,word ptr \[[\w*+]+" + disp, "effect-list head").group(1), 16) - slot_base
    overlay_code = int(_find(ov, r"mov\s+\w+," + _HEX + "$", "overlay code").group(1), 16)
    effect_next = int(_find(ov, r"movzx\s+\w+,word ptr \[\w+\+\w+\*8" + disp, "next node").group(1), 16)
    effect_table = int(_find(ov, r"cmp\s+word ptr \[\w+\+\w+\*8" + disp, "node code").group(1), 16)
    kind_at = int(_find(ov, r"movzx\s+\w+,byte ptr \[\w+\+\w+\*8" + disp, "node kind").group(1), 16)
    masked = [b for a, b in zip(ov, ov[1:], strict=False) if a.op == "and" and a.args.endswith(",0xf")]
    overlay_kind = int(_find(masked, r"cmp\s+\w+," + _HEX + "$", "overlay kind").group(1), 16)
    if not any(i.op in ("add", "shl") for i in ov):
        raise OffsetError("DLL_DuelGetThisCardOverlayNum: node indexing changed")

    search = code.fn("DLL_DuelSearchCardByUniqueID", 0x100)
    effect_uid = int(_find(search, r"cmp\s+dword ptr \[\w+\+\w+\*8" + disp, "node uid").group(1), 16) - effect_table

    return Offsets(state_ptr=state_ptr, key_ptr=key_ptr, my_player=my_player, player_type=player_type,
                   duel_finish=duel_finish, duel_result=duel_result, turn=turn, turn_player=turn_player,
                   phase=phase, player_stride=player_stride, lp=lp, count=counts, pile=pile, pile_stride=pile_stride,
                   slot_base=slot_base, slot_stride=slot_stride, slot_uid=slot_uid, slot_turn=slot_turn,
                   slot_face=slot_face, slot_chain=slot_chain, removed_face=removed_face,
                   removed_facedown=removed_facedown, uid_table=uid_table, uid_mask=uid_mask,
                   effect_table=effect_table, effect_stride=16, effect_kind=kind_at - effect_table,
                   effect_uid=effect_uid, effect_next=effect_next - effect_table, overlay_code=overlay_code,
                   overlay_kind=overlay_kind, dll_sha256=hashlib.sha256(data).hexdigest())  # fmt: skip


def offsets_file(sha256: str) -> Path:
    return OFFSETS_DIR / f"{sha256[:16]}.json"


def load_or_derive(dll: str | Path) -> Offsets:
    """The recorded offsets of this exact ``duel.dll`` if any (they carry ``verified``), else freshly derived ones."""
    dll = Path(dll)
    sha = hashlib.sha256(dll.read_bytes()).hexdigest()
    recorded = offsets_file(sha)
    if recorded.is_file():
        return Offsets.from_json(json.loads(recorded.read_text(encoding="utf-8")))
    return derive(dll)


__all__ = ["FIELD_SLOTS", "OFFSETS_DIR", "OffsetError", "Offsets", "PILES", "derive", "load_or_derive",
           "offsets_file", "parse_objdump", "pe_exports"]  # fmt: skip
