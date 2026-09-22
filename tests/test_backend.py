"""Tests for the pybind11 CoreBackend (T1.1)."""

import threading

import pytest

from ygorl import _core, paths
from ygorl.cards.cdb import CardDB
from ygorl.engine import constants as C
from ygorl.engine.messages import decode_buffer

SEED = [1, 2, 3, 4]
PLAYER = (8000, 5, 1)
DECK = [89631139] * 3 + [14558127] * 3 + [38517737] * 3 + [1861629]  # 10 cards x4 = 40


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def scripts():
    return _core.ScriptDirectory([str(p) for p in paths.script_directories()])


def setup_duel(duel):
    assert duel.load_script("constant.lua") and duel.load_script("utility.lua")
    for team in (0, 1):
        for code in DECK * 4:
            duel.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
    duel.start()


def test_card_database_roundtrip():
    db = _core.CardDatabase()
    db.add(1, 0, [0xDD, 0], 0x11, 8, 0x10, 0x2000, 3000, 2500, 0, 0, 0)
    assert len(db) == 1 and 1 in db and 2 not in db
    assert db.get(1) == (1, 0, [0xDD], 0x11, 8, 0x10, 0x2000, 3000, 2500, 0, 0, 0)
    assert db.get(2) is None


def test_script_directory_first_match_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "c1.lua").write_text("-- a")
    (b / "c1.lua").write_text("-- b")
    (b / "c2.lua").write_text("-- b2")
    sd = _core.ScriptDirectory([str(a), str(b), str(tmp_path / "missing")])
    assert len(sd) == 2
    assert sd.read("c1.lua") == b"-- a"
    assert sd.read("./script/c2.lua") == b"-- b2"  # only the file name matters
    assert sd.read("c3.lua") is None


def test_create_duel_rejects_zero_seed(db, scripts):
    with pytest.raises(RuntimeError, match="OCG_CreateDuel failed"):
        _core.Duel([0, 0, 0, 0], C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), scripts)


def test_bad_sources_are_type_errors(scripts):
    with pytest.raises(TypeError):
        _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, object(), scripts)


def test_duel_runs_to_first_decision_with_native_sources(db, scripts):
    duel = _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), scripts)
    setup_duel(duel)
    # Decode Talker is a Link monster: the core redirects it to the Extra Deck.
    assert duel.query_count(0, C.LOCATION_DECK) == 36 and duel.query_count(0, C.LOCATION_EXTRA) == 4
    names = []
    for _ in range(50):
        status = duel.process()
        names += [m.name for m in decode_buffer(duel.get_message())]
        if status == _core.DUEL_STATUS_AWAITING:
            break
    assert status == _core.DUEL_STATUS_AWAITING
    assert "MSG_DRAW" in names and names[-1].startswith("MSG_SELECT")
    assert duel.query_count(0, C.LOCATION_HAND) == 5
    assert len(duel.query_location(C.QUERY_CODE, 0, C.LOCATION_HAND)) > 0
    assert len(duel.query_field()) > 0
    assert not [t for t, _ in duel.pop_logs() if t == _core.LOG_TYPE_ERROR]
    duel.close()
    assert duel.closed
    with pytest.raises(RuntimeError, match="closed"):
        duel.process()


def test_python_sources(db, scripts):
    requested_cards, requested_scripts = [], []

    def card_source(code):
        requested_cards.append(code)
        return db[code].to_core_tuple() if code in db else None

    def script_source(name):
        requested_scripts.append(name)
        return scripts.read(name)

    duel = _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, card_source, script_source)
    setup_duel(duel)
    duel.process()
    assert set(DECK) <= set(requested_cards)
    assert "constant.lua" in requested_scripts and f"c{DECK[0]}.lua" in requested_scripts


def test_callback_exception_propagates(db, scripts):
    def script_source(name):
        if name == "constant.lua":
            raise ValueError(f"boom {name}")
        return scripts.read(name)

    duel = _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), script_source)
    with pytest.raises(ValueError, match="boom constant.lua"):
        duel.load_script("constant.lua")
    assert duel.load_script("utility.lua") is False  # constant.lua missing -> utility fails, no stale error


def test_callback_exception_during_creation_propagates(db):
    def script_source(name):
        raise KeyError(name)  # the core already asks for scripts while creating the duel

    with pytest.raises(KeyError):
        _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), script_source)


def test_script_errors_are_logged(db, tmp_path):
    (tmp_path / "bad.lua").write_text("this is not lua")
    duel = _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), _core.ScriptDirectory([str(tmp_path)]))
    assert not duel.load_script("bad.lua")
    assert any(t == _core.LOG_TYPE_ERROR for t, _ in duel.pop_logs())


def test_gil_released_while_core_runs(db, tmp_path):
    """While one thread is inside the core, another thread keeps running Python."""
    (tmp_path / "busy.lua").write_text("local x = 0 for i = 1, 30000000 do x = x + i end")
    duel = _core.Duel(SEED, C.DUEL_MODE_MR5, PLAYER, PLAYER, db.to_core(), _core.ScriptDirectory([str(tmp_path)]))
    entered = threading.Event()

    def worker():
        entered.set()
        duel.load_script("busy.lua")

    t = threading.Thread(target=worker)
    t.start()
    entered.wait()
    iterations = 0
    while t.is_alive():
        iterations += 1
    t.join()
    assert iterations > 1000
