"""C++ audit findings: API misuse and re-entrancy must raise clean Python errors, never crash the process.

Each scenario runs in a subprocess, because before the fixes they segfaulted (or corrupted another
duel's memory) and would take the test runner down with them.
"""

import subprocess
import sys
import textwrap

import pytest

PRELUDE = """
import struct, sys
from pathlib import Path
from ygorl import _core
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed

db = CardDB.load()
vocab = CardVocab.from_db(db)
PASSWORDS = [vocab.password(i) for i in range(2, len(vocab))]
DECKS = {p.stem: load_ydk(p) for p in sorted(Path("tests/decks").glob("*.ydk"))}
P = (8000, 5, 1)

def cards_fn(code):
    c = db._cards.get(code)
    return None if c is None else c.to_core_tuple()

def core(cards=None, snapshots=False, seed=7):
    d = _core.Duel(expand_seed(seed), DuelConfig().rule_flags, P, P, cards or db.to_core(), default_scripts(),
                   snapshots=snapshots)
    assert d.load_script("constant.lua") and d.load_script("utility.lua")
    return d

def loaded(seed=1):
    duel = Duel(seed, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db)
    return [(list(m), list(e)) for m, e in duel.loaded_decks()]

def expect(exc, fn):
    try:
        fn()
    except exc as e:
        print("raised", type(e).__name__, e)
        return
    raise SystemExit(f"expected {exc.__name__}")
"""


def run(body: str) -> str:
    out = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(body)], capture_output=True, text=True,
                         timeout=300)  # fmt: skip
    assert out.returncode == 0, f"exit {out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-3000:]}"
    return out.stdout


@pytest.mark.parametrize("mode", ["zero_seed", "no_scripts"])
def test_host_pool_reset_failure_is_reported_not_fatal(mode):
    out = run(f"""
        scripts = default_scripts() if "{mode}" == "zero_seed" else _core.ScriptDirectory([])
        seed = [0, 0, 0, 0] if "{mode}" == "zero_seed" else expand_seed(1)
        pool = _core.HostPool(1, 1, db.to_core(), scripts, PASSWORDS)
        pool.reset(0, seed, DuelConfig().rule_flags, P, P, loaded(), 100, 1000)
        ((env, done, player, obs, result, priv),) = pool.recv(1, -1)
        assert done and obs is None and result["reason"] == "error" and result["error"], result
        print("error:", result["error"])
        pool.reset(0, expand_seed(1), DuelConfig().rule_flags, P, P, loaded(), 100, 1000)  # the slot is reusable
        ((env, done, player, obs, result, priv),) = pool.recv(1, -1)
        print("ok" if "{mode}" == "no_scripts" or not done else "unexpected")
    """)
    assert "error:" in out


def test_a_plain_duel_called_from_a_snapshot_duels_callback_stays_out_of_its_arena():
    """Scope(nullptr) used to keep the caller's suspended arena, so B's Lua state landed in A's slot."""
    out = run("""
        b = core(seed=5)
        holder = {}
        def cards(code):
            if holder.pop("armed", False):
                before = holder["a"].arena_bytes()
                b.load_script("c14558127.lua")  # runs Lua in B (returns False: no card context)
                holder["grew"] = holder["a"].arena_bytes() - before
            return cards_fn(code)
        a = core(cards=cards, snapshots=True)
        holder["a"] = a
        holder["armed"] = True
        a.new_card(0, 0, 55144522, 0, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
        print("grew", holder["grew"])
        assert holder["grew"] == 0
        a.close()
        for team, (main, extra) in enumerate(loaded(5)):
            for code in main:
                b.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
        b.start()
        for _ in range(50):
            if b.process() != _core.DUEL_STATUS_CONTINUE:
                break
            b.get_message()
        print("B survived")
    """)
    assert "B survived" in out


@pytest.mark.parametrize("snapshots", [False, True])
@pytest.mark.parametrize("call", ["close()", "snapshot()", "load_script('c14558127.lua')", "process()"])
def test_reentrant_calls_from_a_callback_are_rejected(snapshots, call):
    if call == "snapshot()" and not snapshots:
        pytest.skip("snapshot() needs snapshots=True")
    out = run(f"""
        holder = {{}}
        def cards(code):
            if holder.pop("armed", False):
                holder["core"].{call}
            return cards_fn(code)
        d = core(cards=cards, snapshots={snapshots})
        holder["core"] = d
        holder["armed"] = True
        expect(RuntimeError, lambda: d.new_card(0, 0, 89631139, 0, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE))
        d.new_card(0, 0, 89631139, 0, C.LOCATION_DECK, 1, C.POS_FACEDOWN_DEFENSE)  # the duel is still usable
        print("closed", d.closed)
    """)
    assert "re-entrant" in out and "closed False" in out


@pytest.mark.parametrize(
    "call",
    [
        "d.new_card(0, 0, 89631139, 2, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)",  # controller 2
        "d.new_card(2, 0, 89631139, 0, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)",  # team 2
        "d.new_card(0, 0, 89631139, 0, C.LOCATION_MZONE, 9, C.POS_FACEUP_ATTACK)",  # 7 monster zones
        "d.new_card(0, 0, 89631139, 0, C.LOCATION_SZONE, 8, C.POS_FACEDOWN)",  # 8 spell/trap zones
        "d.new_card(0, 0, 89631139, 0, C.LOCATION_MZONE | C.LOCATION_HAND, 0, C.POS_FACEUP_ATTACK)",  # two locations
        "d.query(0x1, 7, C.LOCATION_MZONE, 0)",
        "d.query_location(0x1, 2, C.LOCATION_MZONE)",
        "d.query_count(5, C.LOCATION_DECK)",
    ],
)
def test_out_of_range_arguments_raise_value_error(call):
    out = run(f"""
        d = core()
        expect(ValueError, lambda: {call})
        d.new_card(0, 0, 89631139, 0, C.LOCATION_MZONE, 6, C.POS_FACEUP_ATTACK)  # extra monster zone is valid
        print("still fine")
    """)
    assert "raised ValueError" in out and "still fine" in out


@pytest.mark.parametrize("call", ["h.act(0)", "h.observe()", "h.observe_privileged()", "h.result()", "h.actions()"])
def test_host_duel_before_start_raises(call):
    out = run(f"""
        h = _core.HostDuel(db.to_core(), default_scripts(), PASSWORDS)
        expect(RuntimeError, lambda: {call})
    """)
    assert "raised RuntimeError" in out


def test_select_sum_with_a_huge_max_is_cheap_and_matches_python():
    out = run("""
        from ygorl.engine import messages as M
        from ygorl.engine.actions import make_decision
        rec = bytes([C.MSG_SELECT_SUM, 0]) + struct.pack("<BIII", 0, 8, 1, 0x7FFFFFFF)
        rec += struct.pack("<I", 0) + struct.pack("<I", 2)
        rec += struct.pack("<IBBII", 1, 0, 2, 0, 0) + struct.pack("<I", 4)
        rec += struct.pack("<IBBII", 2, 0, 2, 1, 0) + struct.pack("<I", 4)
        cpp = _core.DecisionState(rec, db.to_core()).actions()
        py = make_decision(M.decode_message(rec), db).actions()
        print(len(cpp), len(py))
        assert len(cpp) == len(py) == 2
        rec = bytes([C.MSG_SELECT_TRIBUTE, 0, 0]) + struct.pack("<II", 0x80000000, 0xFFFFFFFF) + struct.pack("<I", 1)
        rec += struct.pack("<IBBIB", 1, 0, 4, 0, 1)
        cpp = _core.DecisionState(rec, db.to_core()).actions()
        py = make_decision(M.decode_message(rec), db).actions()
        print(len(cpp), len(py))
        assert len(cpp) == len(py)
    """)
    assert out.strip()
