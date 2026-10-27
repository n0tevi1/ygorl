"""What the Solo player reads and checks, as pure helpers (design 10, Solo self-play): the duel state's input
fields and MD's fixed screen spots, :func:`classify` (what is being asked), :class:`Expectation` and
:func:`check_expectation` (did the action happen), the board / progress keys and the action keys of the masks.
No clicks and no decisions; :mod:`ygorl.selfplay.loop` and its mixins build on these."""

from __future__ import annotations

import functools
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ygorl.engine import constants as C
from ygorl.md.memory import PHASES, RawDuel
from ygorl.md.client import CommandBit

# --- input state (S + ...): written by duel.dll's command layer, read by DLL_DuelComGetCommandMask --------------
INPUT_MODE, INPUT_ACCEPT, INPUT_PLAYER, FOCUS_CID = 0x3CD0, 0x3CDC, 0x3D00, 0x3CF0
OPTION_DIALOG = 6  # the dialog type of MD's "which effect?" choice (DLL_DuelDlg*: tabs and an OK button)
OPTION_OK = (1024, 788)


PHASE_CHANGES = ("battle_phase", "main2", "end_phase")  # the policy's phase-change actions (``Player.context``)
LIST_NEXT = (1496, 1120)  # the card list's ">" arrow (shown when the list has more cards than fit)
LIST_PREV = (552, 1120)  # its "<" arrow, mirrored (run s65's capture)
RA = 10000010  # The Winged Dragon of Ra
FIELD_ICON_STEP, FIELD_ICON_DY = 172, -150  # a field card's command icons: apart, and above the card
SLIFER = 10000020  # Slifer the Sky Dragon
HAND_LOCATION = 13  # MD's location number of the hand (the client's lists)
# the command bits of each action kind (MD's CommandBit; Pendulum activation counts as activate)
COMMAND_BITS = {"summon": CommandBit.SUMMON, "mset": CommandBit.SETMONST, "sset": CommandBit.SET,
                "spsummon": CommandBit.SPSUMMON, "activate": CommandBit.ACTION | CommandBit.PENDULUM,
                "attack": CommandBit.ATTACK,
                "reposition": CommandBit.TURNATK | CommandBit.TURNDEF | CommandBit.REVERSE}  # fmt: skip
LIST_SHOWN = 6  # cards a list shows at once
LIST_PAGES = 40  # the arrow scrolls one card at a time (the Deck's 40 at most)


def list_more(img: np.ndarray) -> bool:
    """The card list's ">" arrow is lit: more cards to the right (0.37 of its box white measured, 0 when off)."""
    x, y = LIST_NEXT
    return bool((img[y - 20 : y + 20, x - 20 : x + 20].min(axis=2) > 200).mean() > 0.15)


def option_tab(index: int, count: int, y: int = 446) -> tuple[int, int]:
    """The tab of choice ``index`` among ``count`` in the effect-choice dialog (tabs span x 520-1540; row ``y``)."""
    return round(520 + (index + 0.5) * 1020 / count), y


def option_tabs(img: np.ndarray, count: int) -> tuple[int, int | None]:
    """The effect-choice dialog's tab row (y) and which tab is selected (the lime one), read from the screen. The
    row was once taken as y 418, the dialog's title: tab clicks did nothing and OK took the preselected effect
    (Millennium Revelation's first effect instead of the second)."""
    rows: dict[int, list[int]] = {}
    for y in range(400, 540, 2):
        for i in range(count):
            x = option_tab(i, count)[0]
            r = img[y, x - 60 : x + 60].astype(int)
            if ((r[:, 1] > 160) & (r[:, 0] > 110) & (r[:, 2] < 60)).mean() > 0.6:
                rows.setdefault(i, []).append(y)
    if not rows:
        return 446, None
    i = max(rows, key=lambda k: len(rows[k]))
    return (min(rows[i]) + max(rows[i])) // 2, i  # the middle of the lime tab


RESULT_WIN = 1  # DLL_DuelGetDuelResult for the client: 1 won (VICTORY, Confirm -> menus), 2 lost (Retry offered)
NEUTRAL_SPOT = (1300, 640)  # empty board between the rows: clicking it closes a card left open
RESULT_RETRY = (743, 1093)  # MD's result screen: Retry (left) and OK (right)
# the settings gear in a duel, its Retry and the confirmation's Yes: restart a duel the player gave up on
SETTINGS_GEAR, SETTINGS_RETRY, SETTINGS_RETRY_YES = (66, 95), (298, 979), (1218, 743)
MAIN, BATTLE, OPEN_PROMPT, CHAIN_PROMPT, SUMMON_PICK, PLACE, FIELD_PICK = 2, 3, 4, 5, 6, 7, 8  # observed modes
FIELD_PICK_TRIBUTE = 9  # "choose a monster to Tribute" (an effect's tribute: the explorer learned pick + Confirm)


def _progress(raw: RawDuel, inp) -> tuple:
    """What changes in memory while a duel moves on: turn, phase, LP, who holds the input, and where every card is
    (by uid only: nothing about hidden cards is kept)."""
    return (
        raw.turn,
        raw.phase,
        inp.mode,
        inp.player,
        tuple(
            (
                p.lp,
                tuple(c.uid if c else 0 for c in p.slots),
                tuple((k, tuple(c.uid for c in v)) for k, v in sorted(p.piles.items())),
            )
            for p in raw.players
        ),
    )


@dataclass
class Input:
    mode: int
    accepting: bool
    player: int


@dataclass
class Expectation:
    """What the policy chose and the board it chose on, checked against memory once the action has resolved."""

    action: object
    before: RawDuel
    phase: str


def _cards(pl, location: str) -> list:
    if location == "hand":
        return list(pl.piles["hand"])
    if location == "mzone":
        return [c for c in pl.slots[:7] if c]
    if location == "szone":
        return [c for c in pl.slots[7:12] if c]
    return list(pl.piles[location])


def check_expectation(exp: Expectation, now: RawDuel, passwords: dict[int, int]) -> str | None:
    """None when ``now`` shows the chosen action happened, else what is inconsistent."""
    act, before = exp.action, exp.before
    me0, me1 = before.players[before.me], now.players[now.me]
    op0, op1 = before.players[1 - before.me], now.players[1 - now.me]
    kind = act.kind
    if kind in ("battle_phase", "main2", "end_phase"):
        want = {"battle_phase": "battle", "main2": "main2"}.get(kind)
        phase = PHASES[now.phase] if now.phase < len(PHASES) else str(now.phase)
        if now.turn != before.turn:
            return None if kind == "end_phase" else f"{kind}: the turn ended instead"
        if want is not None and phase != want:
            return f"{kind}: the phase is {phase}"
        if kind == "end_phase" and phase in ("main1", "battle"):
            return f"end_phase: still in {phase}"
        return None
    cid = next((c for c, p in passwords.items() if act.card is not None and p == act.card.code), None)

    def count(pl, loc, **kw):
        return sum(c.cid == cid and all(getattr(c, k) == v for k, v in kw.items()) for c in _cards(pl, loc))

    left_hand = count(me1, "hand") < count(me0, "hand")
    if kind == "summon" and act.card is not None and act.card.loc.location == C.LOCATION_MZONE:
        return None  # a Gemini monster summoned again in place: nothing on the board the memory read shows changes
    # a hand copy arrived in the zone, told by uid (a Tribute Summon of Dark Magician that tributed the Dark
    # Magician on the field kept the count of Dark Magicians there at 1: a false alarm)
    hand_uids = {c.uid for c in me0.piles["hand"] if c.cid == cid and c.uid}

    def arrived(loc, **kw):
        # on either field: Ra - Sphere Mode is summoned to the opponent's (a false alarm once)
        return any(c.uid in hand_uids and all(getattr(c, k) == v for k, v in kw.items())
                   for c in _cards(me1, loc) + _cards(op1, loc))  # fmt: skip

    if kind == "summon":
        ok = left_hand and (count(me1, "mzone", face=1, sideways=0) > count(me0, "mzone", face=1, sideways=0)
                            or arrived("mzone", face=1, sideways=0))  # fmt: skip
        return None if ok else "summon: no new face-up Attack Position monster from the hand"
    if kind == "mset":
        ok = left_hand and (count(me1, "mzone", face=0) > count(me0, "mzone", face=0) or arrived("mzone", face=0))
        return None if ok else "mset: no new face-down monster from the hand"
    if kind == "sset":
        ok = left_hand and (count(me1, "szone", face=0) > count(me0, "szone", face=0) or arrived("szone", face=0))
        return None if ok else "sset: no new face-down Spell/Trap from the hand"
    if kind == "attack":
        # an attack on a face-down monster with equal DEF only flips it: that counts (a false alarm once)
        up = lambda pl: sum(1 for c in _cards(pl, "mzone") if c.face)  # noqa: E731
        changed = (op1.lp, me1.lp, len(_cards(op1, "mzone")), len(_cards(me1, "mzone")), up(op1)) != (
            op0.lp, me0.lp, len(_cards(op0, "mzone")), len(_cards(me0, "mzone")), up(op0))  # fmt: skip
        if not changed and act.card is not None and 0 <= act.card.loc.sequence < 7:
            # a 0-ATK direct attack (Slifer with an empty hand) changes no LP: the attacker's own slot still records
            # the attack (its turn flags: attack declared)
            a0, a1 = me0.slots[act.card.loc.sequence], me1.slots[act.card.loc.sequence]
            changed = a0 is not None and a1 is not None and a0.uid == a1.uid and a0.slot != a1.slot
        return None if changed else "attack: nothing changed (no damage, no monster left the field or turned up)"
    if kind in ("activate", "spsummon"):
        if _board_key(me0, op0) == _board_key(me1, op1):
            return f"{kind}: nothing changed on the board"
        loc = act.card.loc if act.card is not None else None
        if kind == "activate" and loc is not None and loc.location == C.LOCATION_MZONE:
            s0, s1 = me0.slots[loc.sequence], me1.slots[loc.sequence]
            turned = s0 is not None and s1 is not None and s0.cid == s1.cid and s0.sideways != s1.sideways
            rest0 = _board_key(me0, op0)[1:]
            rest1 = _board_key(me1, op1)[1:]
            if turned and rest0 == rest1 and [c for i, c in enumerate(me0.slots) if i != loc.sequence] == [
                    c for i, c in enumerate(me1.slots) if i != loc.sequence]:  # fmt: skip
                return "activate: only the monster's position changed (the Change Position icon was clicked)"
        return None
    return None  # reposition and others: not checked yet


EFFECT_LIST_END = 0x3C80  # S + uid_table (0x1bec) .. here: the uid table and the effect list (S + 0x2bec)
POSITION_DIALOG, RADIO_DIALOG = 10, 13  # DLL_DuelDlg* dialog types (6 is OPTION_DIALOG)
NEUTRAL = ("notice", "minimised")  # screens that belong to nobody's input: always answered


def own_effect_nodes(state: bytes, me: int, off) -> dict[int, bytes]:
    """{offset in S: node bytes} of the effect-list nodes reachable from our player list head and our field slots'
    list heads: what our own effects are (kept in the recorded memory; the rest of the table is blanked)."""
    blk = me * off.player_stride
    heads = [struct.unpack_from("<H", state, blk + off.player_effects)[0]]
    for i in range(13):
        at = blk + off.slot_base + i * off.slot_stride
        if struct.unpack_from("<H", state, at)[0]:
            heads.append(struct.unpack_from("<H", state, at + off.slot_chain)[0])
    out: dict[int, bytes] = {}
    for head in heads:
        while 0 < head < 0x100:
            at = off.effect_table + head * off.effect_stride
            if at in out or at + off.effect_stride > len(state):
                break
            out[at] = bytes(state[at : at + off.effect_stride])
            head = struct.unpack_from("<H", state, at + off.effect_next)[0]
    return out


def classify(*, mine: bool, ours: bool, accepting: bool, mode: int, dialog_type: int, choices: int,
             screen: str | None, context: str = "", zone_pending: bool = False) -> str:  # fmt: skip
    """What the step must answer (design 10 「识别与执行的系统化」): memory first (who holds the input, the input
    mode, MD's dialog type), the screen class only where memory cannot tell (MD's own dialogs while the engine
    input is closed) and as the locator. ``mine``: the engine waits for our input; ``ours``: the input (or MD's
    dialog) is the client's, accepting or not. Returns a key of :data:`Player.HANDLERS`, "wait" or "over"."""
    if screen == "over":
        return "over"
    if screen in NEUTRAL:
        return screen
    if not ours:
        return "wait"  # the opponent holds the input: whatever is on screen is theirs
    if not mine and not accepting and dialog_type == POSITION_DIALOG and screen in ("banner", "position"):
        # MD's own summon placement while the engine takes no input (a Special Summon by an effect, runs s69 / s71):
        # the zone banner, then the position buttons
        return "place" if screen == "banner" else "position"
    if screen == "phase_menu" and mine and mode in (MAIN, BATTLE):
        screen = None  # a phase change under way: the command decision uses (or closes) the open menu
    if screen == "panel":  # a dark band without yes / no labels: the effect choice when memory says so
        screen = "option" if dialog_type == OPTION_DIALOG and choices >= 2 else None
    if screen in ("yesno", "panel") and dialog_type == OPTION_DIALOG and choices >= 2:
        screen = "option"
    if screen == "radio" and dialog_type != RADIO_DIALOG:
        if not accepting:
            return "settle"  # with no input taken: an animation (run s65: a Fusion Summon's big card, explored)
        screen = None  # rings without MD's radio dialog: selection frames or zone outlines (clicked a set card)
    if mine and mode == SUMMON_PICK and screen in (None, "banner"):
        return "summon_pick"
    if mine and mode in (OPEN_PROMPT, CHAIN_PROMPT) and screen in (None, "list"):
        return "prompt"  # an activate-or-pass prompt panel looks like a list
    if mine and mode == PLACE and (screen == "position" or dialog_type == POSITION_DIALOG):
        return "position"
    if mine and mode == PLACE and screen in ("banner", "panel", None):
        return "place"  # the zone question can come while a chain's resolve animation still dims the board
    if mine and mode == MAIN and screen == "banner" and zone_pending:
        return "zone_again"  # MD still asks for the zone of our own Summon / Set
    if screen == "position" and dialog_type != POSITION_DIALOG and not (mine and mode == PLACE):
        # a position question only when memory asks one: a drawn card's reveal and an opponent's summon cut-in
        # passed for it; with no input taken it is an animation to wait out, not a screen to explore
        if not accepting:
            return "settle"
        screen = None
    if mine and mode in (MAIN, BATTLE) and screen == "banner":
        screen = None  # the engine waits for a command: no banner question is pending (the opponent's ATK / DEF
        # numbers on a dark board passed for a banner's text)
    if mine and mode in (FIELD_PICK, FIELD_PICK_TRIBUTE) and context == "tribute":
        return "tributes"
    if mine and mode in (FIELD_PICK, FIELD_PICK_TRIBUTE) and screen not in ("list", "yesno", "option", "notice"):
        return "field_pick"  # memory says a field pick; the selection frames can pass for other screens (radio)
    if screen == "banner" and not accepting:
        return "settle"  # MD's own text while the engine takes no input (a Summon animating)
    if screen:
        return screen  # MD's own dialogs: yesno, option, list, radio, position, phase_menu, declare, ...
    if mode == MAIN:
        return "idle"
    if mode == BATTLE:
        return "battle"
    if mode == 1:
        return "draw"
    return "explore"


@functools.cache
def declares_a_card(password: int) -> bool:
    """The card's script asks to declare a card name (``Duel.AnnounceCard``): MD's answer is a search box that needs
    typed text, and window-only input cannot type (The True Name stalled the duel)."""
    from ygorl import paths

    for d in paths.script_directories():
        f = Path(d) / f"c{password}.lua"
        if f.is_file():
            return "AnnounceCard" in f.read_text(errors="replace")
    return False


def plain_name(name: str) -> bool:
    """A card name of ASCII letters, digits and spaces only: typed whole, MD's search finds exactly it."""
    return all((ch.isascii() and ch.isalnum()) or ch == " " for ch in name)


def searchable(name: str) -> str:
    """The longest stretch of a card name that the keyboard can type (search matches substrings). Apostrophes and
    quotes split it: MD's names use other glyphs for them, and a search with the ASCII one finds nothing."""
    from ygorl.md.input import GatedMouse

    ok = lambda ch: ((ch.isascii() and ch.isalnum()) or ch in GatedMouse.KEYSYMS) and ch not in "'\""  # noqa: E731
    parts, cur = [], ""
    for ch in name:
        if ok(ch):
            cur += ch
        else:
            parts.append(cur)
            cur = ""
    parts.append(cur)
    return max(parts, key=len).strip() or name


def clickable(action) -> bool:
    """Whether the player can carry ``action`` out: not yet effects activated from the banished cards (the
    Graveyard's are, through its viewer: :meth:`Player.graveyard_activate`)."""
    card = action.card
    if action.kind == "activate" and card is not None and card.loc.location == C.LOCATION_REMOVED:
        return False
    return True


ATTACKS_MASKED = ("attack", "*")  # a forbidden entry masking every attack of a phase


def _action_key(action) -> tuple:
    card = action.card
    return (action.kind, card.code if card else 0, card.loc.location if card else 0, card.loc.sequence if card else 0)


def _own_cards(raw: RawDuel) -> tuple:
    """Our field (with chain marks) and hand: what an activation of ours would change."""
    me = raw.players[raw.me]
    return (tuple((c.cid, c.face, c.sideways, c.flags) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]))  # fmt: skip


def _board_key(me, op) -> tuple:
    """Everything an action of ours can visibly change; the piles' counts too (a search moves a card from a Deck
    to a hand without touching the field: Millennium Seeker's effect read as "nothing changed")."""
    return (tuple((c.cid, c.face, c.sideways) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]), tuple(c.cid for c in me.piles["grave"]),
            tuple((c.cid, c.face) if c else None for c in op.slots), op.lp, me.lp,
            len(me.piles["removed"]), len(op.piles["grave"]), tuple(sorted(me.counts.items())),
            tuple(sorted(op.counts.items())))  # fmt: skip
