"""T2.3: the step-wise feasible sets equal the sets the engine accepts.

Each scenario is a :class:`~ygorl.engine.puzzle.Puzzle` (cards placed directly
on the field / in hand) that leads to a material or tribute selection. From the
summon command on, every decision is explored:

* SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM are answered in one core response
  but split into steps by the environment. For each such decision every subset
  of the offered cards (and ``-1``) is sent to the real engine, which either
  accepts it or answers ``MSG_RETRY``. The accepted set must equal the set of
  responses reachable through the environment's step-wise actions (every path
  of ``select`` actions ending in ``finish`` or auto-completion), for both the
  Python (``ygorl.engine.actions``) and the C++ (``_core.DecisionState``) state
  machines, and no path may dead-end.
* SELECT_UNSELECT_CARD is asked card by card by the core itself (the Synchro,
  Xyz and Link procedures of CardScripts use ``Group.SelectUnselect``): all
  actions are explored, as for any other decision a procedure asks in between
  (e.g. ANNOUNCE_NUMBER for how many materials Drake Shark counts as).
* Chain windows, zones and positions take a fixed default (pass / first).

Every reachable summon is recorded by the cards it used as material, which
must equal the material sets the rules allow (``expected`` of each scenario,
enumerated by hand from the card texts).

Cards (8-digit passwords) are listed in ``CARDS`` below.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from itertools import combinations

import pytest

from ygorl import _core
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import default_cards, default_scripts
from ygorl.engine.puzzle import Placement, Puzzle, last_decision, run_until_stop
from ygorl.engine.query import parse_query_location
from tests.test_cpp_host import rand_record

# ------------------------------------------------------------------ cards

CARDS = {
    # normal monsters (vanilla), distinct passwords so materials can be told apart by code
    "warwolf": 69247929,  # Gene-Warped Warwolf, level 4 EARTH
    "alexandrite": 43096270,  # Alexandrite Dragon, level 4 LIGHT
    "luster": 11091375,  # Luster Dragon, level 4 WIND
    "heliotrope": 77542832,  # Evilswarm Heliotrope, level 4 DARK
    "archfiend": 49881766,  # Archfiend Soldier, level 4 DARK
    "frostosaurus": 6631034,  # Frostosaurus, level 6 WATER
    "orion": 2971090,  # Orion the Battle King, level 5
    "mountain_warrior": 4931562,  # Mountain Warrior, level 3
    "dark_plant": 13193642,  # Dark Plant, level 1 DARK
    "blue_eyes": 89631139,  # Blue-Eyes White Dragon, level 8 LIGHT
    "dark_magician": 46986414,  # Dark Magician, level 7 DARK
    # effect monsters
    "coston": 44436472,  # Double Coston, level 4 DARK: counts as 2 Tributes for a DARK monster
    "kuriboh": 40640057,  # Kuriboh, level 1
    "star_drawing": 24610207,  # Star Drawing, level 4: can be treated as level 5 for an Xyz Summon
    "drake_shark": 81096431,  # Drake Shark, level 4 WATER: 2 materials for a WATER Xyz needing 3+
    "junk_synchron": 63977008,  # Junk Synchron, level 3 Tuner
    "yamatako": 4632019,  # Yamatako Orochi, level 1 Tuner: can be treated as level 8 for a Synchro Summon
    "tuningware": 92676637,  # Tuningware, level 1: can be treated as level 2 for a Synchro Summon
    "ritual_raven": 34334692,  # Ritual Raven, level 1 DARK: whole Tribute for a DARK Ritual Monster
    "miracle_raven": 18988396,  # Miracle Raven, level 1: on the field, whole Tribute for any Ritual Monster
    # Ritual
    "black_luster_ritual": 55761792,  # Black Luster Ritual: total levels >= 8 (SELECT_SUM, at-least mode)
    "black_luster_soldier": 5405694,  # Black Luster Soldier, level 8 Ritual
    "contract_abyss": 69035382,  # Contract with the Abyss: DARK Ritual, total levels exactly equal
    "demise": 72426662,  # Demise, King of Armageddon, level 8 DARK Ritual
    # extra deck
    "utopia": 84013237,  # Number 39: Utopia, rank 4: 2 level 4 monsters
    "volcasaurus": 29669359,  # Number 61: Volcasaurus, rank 5: 2 level 5 monsters
    "shark_drake": 65676461,  # Number 32: Shark Drake, WATER rank 4: 3 level 4 monsters
    "stardust": 44508094,  # Stardust Dragon, level 8 Synchro: 1 Tuner + 1+ non-Tuner monsters
    "decode_talker": 1861629,  # Decode Talker, Link-3: 2+ Effect Monsters
    "proxy_dragon": 22862454,  # Proxy Dragon, Link-2 Effect
    "linkuriboh": 41999284,  # Linkuriboh, Link-1 Effect
    # SELECT_CARD
    "dmg_apprentice": 2501624,  # Dark Magician Girl the Magician's Apprentice: Special Summon by discarding 0..1
    "twin_twisters": 43898403,  # Twin Twisters: discard 1, then target up to 2 Spells/Traps
    "pot_of_greed": 55144522,  # set face-down by the opponent as targets
    "monster_reborn": 83764718,
    "raigeki": 12580477,
}
NAMES = {v: k for k, v in CARDS.items()}

H, MZ, SZ, EX, DK = C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_EXTRA, C.LOCATION_DECK
MULTI = (M.SelectCard, M.SelectTribute, M.SelectSum)
CANCEL = "cancel"


def P(name, loc, seq=0, team=0, pos=None):
    return Placement(CARDS[name], loc, seq, team, pos)


def filler():
    """A few deck cards per player so nothing draws from an empty deck."""
    return tuple(Placement(CARDS["blue_eyes"], DK, 0, t) for t in (0, 1) for _ in range(3))


# ------------------------------------------------------------------ engine helpers


@pytest.fixture(scope="module")
def native():
    return default_cards().to_core()


class Replayer:
    """Replays a puzzle from scratch with a list of responses (determinism makes this exact)."""

    def __init__(self, puzzle: Puzzle, native):
        self.puzzle, self.native, self.scripts = puzzle, native, default_scripts()

    def advance(self, responses) -> tuple[_core.Duel, bytes]:
        core = self.puzzle.start(self.native, self.scripts)
        _, records = run_until_stop(core)
        for r in responses:
            core.set_response(r)
            _, records = run_until_stop(core)
            assert not any(rec[0] == C.MSG_RETRY for rec in records), f"replay rejected {r.hex()}"
        record = last_decision(records)
        assert record is not None, "puzzle ended unexpectedly"
        return core, record


def cards_response(indices) -> bytes:
    return struct.pack(f"<iI{len(indices)}I", 0, len(indices), *indices)


def normalize(response: bytes):
    """A multi-select response as ``CANCEL`` or the frozenset of chosen indices."""
    (kind,) = struct.unpack_from("<i", response)
    if kind == -1:
        return CANCEL
    assert kind == 0, response.hex()
    (n,) = struct.unpack_from("<I", response, 4)
    return frozenset(struct.unpack_from(f"<{n}I", response, 8))


def engine_accepted(rep: Replayer, prefix, record: bytes, n: int) -> set:
    """Every subset of the ``n`` offered cards (and -1) the engine accepts, sending one per try.

    A rejected response leaves the core waiting on the same decision (it only
    writes ``MSG_RETRY``), so rejected candidates are tried on the same duel and
    only an accepted one costs a fresh replay.
    """
    candidates = [(CANCEL, struct.pack("<i", -1))]
    candidates += [(frozenset(s), cards_response(s)) for k in range(n + 1) for s in combinations(range(n), k)]
    accepted, core = set(), None
    for key, raw in candidates:
        if core is None:
            core, again = rep.advance(prefix)
            assert again == record
        core.set_response(raw)
        _, records = run_until_stop(core)
        if any(rec[0] == C.MSG_RETRY for rec in records):
            continue
        accepted.add(key)
        core = None
    return accepted


def env_responses(new_state) -> set[bytes]:
    """All complete responses reachable through step-wise actions; asserts there is no dead end."""
    out: set[bytes] = set()

    def dfs(path):
        state = new_state()
        for i in path:
            state.step(i)
        n = len(state.actions())
        assert n > 0, f"dead end after actions {path}"
        for i in range(n):
            s = new_state()
            response = None
            for j in (*path, i):
                response = s.step(j)
            if s.done:
                out.add(bytes(response))
            else:
                dfs((*path, i))

    dfs(())
    return out


def in_play(core) -> list[int]:
    """Codes of every card in a hand or on the field (Xyz materials excluded), both players."""
    codes = []
    for team in (0, 1):
        for loc in (H, MZ, SZ):
            codes += [c["code"] for c in parse_query_location(core.query_location(C.QUERY_CODE, team, loc)) if c]
    return sorted(codes)


def left_play(before: list[int], after: list[int]) -> list[int]:
    """Codes in ``before`` but not in ``after`` (multisets): cards used as material, discarded, destroyed..."""
    rest = list(after)
    out = []
    for code in before:
        if code in rest:
            rest.remove(code)
        else:
            out.append(code)
    return out


# ------------------------------------------------------------------ exploration


@dataclass
class Scenario:
    puzzle: Puzzle
    lead: tuple[str, str]  # (IDLECMD action kind, card name) that starts the selection
    ignore: frozenset[str] = frozenset()  # cards that reach the GY but are not materials (the spell itself)


@dataclass
class Report:
    outcomes: set = field(default_factory=set)
    checked: dict = field(default_factory=dict)  # decision type -> number of decisions compared with the engine
    cancelled: bool = False  # some path cancelled the summon (nothing was used)


DEFAULTED = (M.SelectChain, M.SelectPlace, M.SelectDisfield, M.SelectPosition)


def default_response(record: bytes) -> bytes:
    """Answer a decision that does not choose materials: pass, else the first option, step by step."""
    state = make_decision(M.decode_message(record), cards=default_cards())
    response = None
    while not state.done:
        acts = state.actions()
        response = state.step(next((i for i, a in enumerate(acts) if a.kind == "pass"), 0))
    return response


def explore(scenario: Scenario, native) -> Report:
    """Explore every choice from the lead command until control returns to the idle command."""
    rep = Replayer(scenario.puzzle, native)
    report = Report()
    opening: list[bytes] = []
    while True:  # pass any opening chain window (e.g. a Quick-Play Spell in hand)
        core, record = rep.advance(opening)
        msg = M.decode_message(record)
        if isinstance(msg, M.SelectIdleCmd):
            break
        opening.append(default_response(record))
    kind, name = scenario.lead
    state = make_decision(msg, cards=default_cards())
    lead = next(i for i, a in enumerate(state.actions()) if a.kind == kind and a.card and a.card.code == CARDS[name])
    before = in_play(core)
    seen: set = set()

    def visit(prefix, ctx=frozenset()):
        # ctx: the latest answer to each earlier decision other than SELECT_UNSELECT_CARD (e.g. how
        # many materials Drake Shark counts as), which the next message alone does not show; keyed by
        # the decision message so that re-answering it (after unselect / reselect) keeps ctx bounded
        core, record = rep.advance(prefix)
        msg = M.decode_message(record)
        if isinstance(msg, M.SelectIdleCmd):
            used = frozenset(NAMES[c] for c in left_play(before, in_play(core))) - scenario.ignore
            if used:
                report.outcomes.add(used)
            else:
                report.cancelled = True  # the selection was cancelled (-1 on a cancelable decision)
            return
        if isinstance(msg, DEFAULTED):
            visit([*prefix, default_response(record)], ctx)
            return
        if isinstance(msg, M.SelectUnselectCard):
            # a card-by-card selection state is its message (offered and already selected cards), what
            # has moved and ctx; select-then-unselect cycles come back to a seen state
            key = (record, tuple(in_play(core)), ctx)
            if key in seen:
                return
            seen.add(key)
        py = env_responses(lambda: make_decision(msg, cards=default_cards()))
        cpp = env_responses(lambda: _core.DecisionState(record, native))
        assert py == cpp, f"{msg}: python {sorted(py)} != c++ {sorted(cpp)}"
        if isinstance(msg, MULTI):
            accepted = engine_accepted(rep, prefix, record, len(msg.cards))
            env = {normalize(r) for r in py}
            assert env == accepted, (
                f"{type(msg).__name__} {msg}: env-only {sorted(env - accepted, key=str)}, "
                f"engine-only {sorted(accepted - env, key=str)}"
            )
            report.checked[type(msg).__name__] = report.checked.get(type(msg).__name__, 0) + 1
        for r in sorted(py):
            nxt = ctx if isinstance(msg, M.SelectUnselectCard) else frozenset({*(e for e in ctx if e[0] != record), (record, r)})
            visit([*prefix, r], nxt)

    visit([*opening, state.step(lead)])
    return report


def names(*sets):
    return {frozenset(s) for s in sets}


# ------------------------------------------------------------------ scenarios


def test_puzzle_starts_in_main_phase(native):
    puzzle = Puzzle((P("dark_magician", H), P("coston", MZ, 0), *filler()))
    core = puzzle.start(native, default_scripts())
    _, records = run_until_stop(core)
    msg = M.decode_message(last_decision(records))
    assert isinstance(msg, M.SelectIdleCmd) and msg.player == 0
    assert [c.code for c in msg.summonable] == [CARDS["dark_magician"]]


TRIBUTE = Puzzle((P("dark_magician", H), P("blue_eyes", H, 1), P("coston", MZ, 0), P("warwolf", MZ, 1),
                  P("alexandrite", MZ, 2), P("luster", MZ, 3), *filler()))  # fmt: skip


def test_tribute_summon_with_double_tribute_monster(native):
    """Dark Magician (DARK, level 7) needs 2 Tributes; Double Coston counts as 2 for it."""
    others = ("warwolf", "alexandrite", "luster")
    expected = names(["coston"], *(["coston", o] for o in others), *combinations(others, 2))
    report = explore(Scenario(TRIBUTE, ("summon", "dark_magician")), native)
    assert report.outcomes == expected
    assert report.checked["SelectTribute"] == 1


def test_tribute_summon_double_tribute_not_applicable(native):
    """For Blue-Eyes (LIGHT) Double Coston is a single Tribute: any 2 of the 4 monsters."""
    mats = ("coston", "warwolf", "alexandrite", "luster")
    expected = names(*combinations(mats, 2))
    report = explore(Scenario(TRIBUTE, ("summon", "blue_eyes")), native)
    assert report.outcomes == expected


def test_select_card_min_max(native):
    """Twin Twisters: SELECT_CARD 1 of 2 to discard, then 1..2 of 3 face-down Spells as targets."""
    puzzle = Puzzle((P("twin_twisters", H), P("warwolf", H, 1), P("alexandrite", H, 2),
                     P("pot_of_greed", SZ, 0, 1, C.POS_FACEDOWN), P("monster_reborn", SZ, 1, 1, C.POS_FACEDOWN),
                     P("raigeki", SZ, 2, 1, C.POS_FACEDOWN), *filler()))  # fmt: skip
    targets = ("pot_of_greed", "monster_reborn", "raigeki")
    expected = {frozenset((d, *t)) for d in ("warwolf", "alexandrite") for k in (1, 2) for t in combinations(targets, k)}
    report = explore(Scenario(puzzle, ("activate", "twin_twisters"), frozenset({"twin_twisters"})), native)
    assert report.outcomes == expected
    assert report.checked["SelectCard"] == 3  # the discard, then the targets after each discard


def test_select_card_min_zero(native):
    """SELECT_CARD with min 0: the engine accepts an explicit empty selection as well as -1.

    Dark Magician Girl the Magician's Apprentice is Special Summoned from the hand by
    discarding 1 card, chosen with ``SelectMatchingCard(..., 0, 1, ...)``; choosing
    nothing aborts the summon (the script checks ``#g>0``).
    """
    puzzle = Puzzle((P("dmg_apprentice", H), P("warwolf", H, 1), P("alexandrite", H, 2), *filler()))
    expected = names(["warwolf"], ["alexandrite"])
    report = explore(Scenario(puzzle, ("spsummon", "dmg_apprentice")), native)
    assert report.outcomes == expected and report.cancelled
    assert report.checked["SelectCard"] == 1


LEVELS = {"warwolf": (4,), "alexandrite": (4,), "heliotrope": (4,), "archfiend": (4,), "frostosaurus": (6,),
          "orion": (5,), "mountain_warrior": (3,), "dark_plant": (1,)}  # fmt: skip


def ritual_sets(pool: dict[str, tuple[int, ...]], target: int, exact: bool) -> set[frozenset[str]]:
    """Material sets the core accepts for a Ritual SELECT_SUM (field::process(SelectSum)).

    ``pool`` maps each candidate to its possible ritual levels (a whole-Tribute
    card has two). Exact mode: some choice of levels sums to ``target``.
    At-least mode: the sum of the larger levels reaches ``target`` while the sum
    of the smaller levels minus the smallest one stays below it (no material is
    superfluous).
    """
    out = set()
    names_ = sorted(pool)
    for k in range(1, len(names_) + 1):
        for group in combinations(names_, k):
            if exact:
                sums = {0}
                for n in group:
                    sums = {s + v for s in sums for v in pool[n]}
                ok = target in sums
            else:
                lows = [min(pool[n]) for n in group]
                ok = sum(max(pool[n]) for n in group) >= target and sum(lows) - min(lows) < target
            if ok:
                out.add(frozenset(group))
    return out


def test_ritual_at_least_mode(native):
    """Black Luster Ritual (total levels >= 8) with Miracle Raven (whole Tribute: 1 or 8) on the field."""
    puzzle = Puzzle((P("black_luster_ritual", H), P("black_luster_soldier", H, 1), P("warwolf", H, 2),
                     P("frostosaurus", H, 3), P("miracle_raven", MZ, 0), P("alexandrite", MZ, 1),
                     P("mountain_warrior", MZ, 2), P("dark_plant", MZ, 3), *filler()))  # fmt: skip
    pool = {n: LEVELS[n] for n in ("warwolf", "frostosaurus", "alexandrite", "mountain_warrior", "dark_plant")}
    pool["miracle_raven"] = (1, 8)
    expected = ritual_sets(pool, 8, exact=False)
    report = explore(Scenario(puzzle, ("activate", "black_luster_ritual"), frozenset({"black_luster_ritual"})), native)
    assert report.outcomes == expected
    assert report.checked["SelectSum"] == 1


def test_ritual_exact_mode(native):
    """Contract with the Abyss (total levels exactly 8) with Ritual Raven (whole Tribute for DARK: 1 or 8)."""
    puzzle = Puzzle((P("contract_abyss", H), P("demise", H, 1), P("ritual_raven", H, 2), P("heliotrope", H, 3),
                     P("frostosaurus", H, 4), P("archfiend", MZ, 0), P("mountain_warrior", MZ, 1),
                     P("dark_plant", MZ, 2), P("orion", MZ, 3), *filler()))  # fmt: skip
    pool = {n: LEVELS[n] for n in ("heliotrope", "frostosaurus", "archfiend", "mountain_warrior", "dark_plant", "orion")}
    pool["ritual_raven"] = (1, 8)
    expected = ritual_sets(pool, 8, exact=True)
    report = explore(Scenario(puzzle, ("activate", "contract_abyss"), frozenset({"contract_abyss"})), native)
    assert report.outcomes == expected
    assert report.checked["SelectSum"] == 1


XYZ = Puzzle((P("star_drawing", MZ, 0), P("drake_shark", MZ, 1), P("warwolf", MZ, 2), P("alexandrite", MZ, 3),
              P("orion", MZ, 4), P("utopia", EX), P("volcasaurus", EX), P("shark_drake", EX), *filler()))  # fmt: skip
LEVEL4 = ("star_drawing", "drake_shark", "warwolf", "alexandrite")


@pytest.mark.parametrize("target, expected", [
    ("utopia", names(*combinations(LEVEL4, 2))),  # 2 level 4 monsters
    ("volcasaurus", names(["star_drawing", "orion"])),  # 2 level 5: Star Drawing as level 5
    # 3 level 4 monsters, or Drake Shark as 2 materials plus one more
    ("shark_drake", names(*combinations(LEVEL4, 3), *(["drake_shark", o] for o in LEVEL4 if o != "drake_shark"))),
])  # fmt: skip
def test_xyz_materials(native, target, expected):
    report = explore(Scenario(XYZ, ("spsummon", target)), native)
    assert report.outcomes == expected


def test_synchro_materials(native):
    """Stardust Dragon (level 8, 1 Tuner + 1+ non-Tuners) with two-level materials.

    Yamatako Orochi (Tuner, level 1 or 8) and Tuningware (level 1 or 2).
    """
    puzzle = Puzzle((P("junk_synchron", MZ, 0), P("yamatako", MZ, 1), P("tuningware", MZ, 2), P("warwolf", MZ, 3),
                     P("orion", MZ, 4), P("mountain_warrior", MZ, 5), P("stardust", EX), *filler()))  # fmt: skip
    expected = names(["junk_synchron", "orion"], ["junk_synchron", "warwolf", "tuningware"],
                     ["junk_synchron", "mountain_warrior", "tuningware"], ["yamatako", "warwolf", "mountain_warrior"],
                     ["yamatako", "orion", "tuningware"])  # fmt: skip
    report = explore(Scenario(puzzle, ("spsummon", "stardust")), native)
    assert report.outcomes == expected


def test_link_materials(native):
    """Decode Talker (Link-3, 2+ Effect Monsters): Proxy Dragon counts as 1 or 2, the normal monster never."""
    puzzle = Puzzle((P("proxy_dragon", MZ, 0), P("linkuriboh", MZ, 1), P("kuriboh", MZ, 2), P("coston", MZ, 3),
                     P("warwolf", MZ, 4), P("decode_talker", EX), *filler()))  # fmt: skip
    ones = ("linkuriboh", "kuriboh", "coston")
    expected = names(*(["proxy_dragon", o] for o in ones), *combinations(("proxy_dragon", *ones), 3))
    report = explore(Scenario(puzzle, ("spsummon", "decode_talker")), native)
    assert report.outcomes == expected


# ------------------------------------------------------------------ random decisions vs the core's checks
#
# The puzzles above cover the decisions real scripts produce. Edge cases they do
# not reach (must-select cards, min 0, two-value cards in every mode, ...) are
# covered by random messages checked against a transcription of the core's
# response checks in playerop.cpp (field::process for SelectCard,
# SelectTribute and SelectSum).


def core_accepts(msg, response) -> bool:
    """Would the core accept ``response`` (CANCEL or a frozenset of indices) for ``msg``?"""
    if response == CANCEL:
        # parse_response_cards: -1 is accepted when the message says cancelable; SELECT_SUM never
        return not isinstance(msg, M.SelectSum) and msg.cancelable
    if isinstance(msg, M.SelectCard):
        return msg.min <= len(response) <= msg.max
    if isinstance(msg, M.SelectTribute):
        return len(response) <= msg.max and sum(msg.cards[i].release_param for i in response) >= msg.min
    opts = [*msg.must, *(msg.cards[i] for i in response)]
    if msg.exact:
        if not msg.min <= len(response) <= max(msg.max, msg.min):
            return False
        sums = {0}  # select_sum_check1 with positive values: some choice of values adds up to the target
        for o in opts:
            sums = {s + v for s in sums for v in o.values}
        return msg.target > 0 and msg.target in sums
    lo_hi = [(o.param & 0xFFFF, o.param >> 16) for o in opts]
    ms = [hi if hi and hi < lo else lo for lo, hi in lo_hi]
    mx = [max(lo, hi) for lo, hi in lo_hi]
    return bool(opts) and sum(mx) >= msg.target and sum(ms) - min(ms) < msg.target


@pytest.mark.parametrize("kind", [C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM],
                         ids=lambda k: M.MESSAGE_NAMES[k])  # fmt: skip
def test_random_decisions_match_core_checks(native, kind):
    rng = random.Random(1000 + kind)
    checked = 0
    for _ in range(400):
        record = rand_record(rng, kind)
        msg = M.decode_message(record)
        n = len(msg.cards)
        candidates = [CANCEL, *(frozenset(s) for k in range(n + 1) for s in combinations(range(n), k))]
        accepted = {c for c in candidates if core_accepts(msg, c)}
        if not accepted:
            continue  # no valid answer: the core never asks such a decision (scripts check first)
        checked += 1
        py = env_responses(lambda: make_decision(msg))
        assert py == env_responses(lambda: _core.DecisionState(record, native))
        env = {normalize(r) for r in py}
        assert env == accepted, f"{msg}: env-only {sorted(env - accepted, key=str)}, core-only {sorted(accepted - env, key=str)}"
    assert checked > 150
