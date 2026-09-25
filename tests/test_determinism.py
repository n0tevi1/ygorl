"""Determinism and response-log replay (T1.6)."""

from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, ScriptedAgent

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
PAIRS = [("snake_eye", "kashtira"), ("labrynth", "tenpai"), ("tearlaments", "branded_despia")]


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def play(db, seed, a, b, first=0, agent_seeds=(1, 2)):
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, record_messages=True)
    return duel.run(RandomAgent(agent_seeds[0]), RandomAgent(agent_seeds[1]))


@pytest.mark.parametrize("a, b", PAIRS)
def test_same_seed_and_responses_give_identical_byte_stream(db, a, b):
    r1, r2 = play(db, 42, a, b), play(db, 42, a, b)
    assert r1.message_log and r1.message_log == r2.message_log
    assert r1.responses == r2.responses


def test_different_seed_changes_stream(db):
    assert play(db, 1, "yubel", "purrely").message_log != play(db, 2, "yubel", "purrely").message_log


@pytest.mark.parametrize("a, b", PAIRS)
@pytest.mark.parametrize("first", [0, 1])
def test_response_log_replays_to_same_end(db, a, b, first):
    original = play(db, 7, a, b, first=first)
    replayed = Duel(7, None, DECKS[a], DECKS[b], cards=db, first=first, record_messages=True).replay(original.responses)
    assert replayed.message_log == original.message_log
    assert (replayed.winner, replayed.reason, replayed.win_reason, replayed.turns, replayed.lp) == (
        original.winner, original.reason, original.win_reason, original.turns, original.lp)  # fmt: skip


def test_action_log_replays_with_scripted_agent(db):
    original = play(db, 9, "voiceless_voice", "fiendsmith_ryzeal")
    agent = ScriptedAgent(original.actions)
    again = Duel(9, None, DECKS["voiceless_voice"], DECKS["fiendsmith_ryzeal"], cards=db, record_messages=True).run(
        agent, agent
    )
    assert again.message_log == original.message_log and again.responses == original.responses


def test_partial_replay_stops_when_log_is_exhausted(db):
    original = play(db, 3, "snake_eye", "yubel")
    cut = len(original.responses) // 2
    partial = Duel(3, None, DECKS["snake_eye"], DECKS["yubel"], cards=db).replay(original.responses[:cut])
    assert partial.reason == "log_exhausted" and partial.winner is None
    assert partial.responses == original.responses[:cut]


def test_replay_detects_divergence(db):
    original = play(db, 4, "snake_eye", "yubel")
    with pytest.raises(ValueError, match="differs from the reference at message buffer 0"):
        Duel(5, None, DECKS["snake_eye"], DECKS["yubel"], cards=db).replay(
            original.responses, reference_log=original.message_log
        )
    # the right seed replays cleanly against its own reference
    Duel(4, None, DECKS["snake_eye"], DECKS["yubel"], cards=db).replay(
        original.responses, reference_log=original.message_log
    )


def test_lua_string_keyed_pairs_order_is_stable(db, tmp_path):
    """Lua 5.4 seeds string hashing from the clock and heap addresses unless
    luai_makeseed is pinned; then pairs() order over string keys differs per duel."""
    from ygorl import _core
    from ygorl.engine import constants as C
    from ygorl.engine import messages as M

    (tmp_path / "probe.lua").write_text(
        'local t = {} for i = 1, 200 do t["key" .. i] = i end '
        "local out = {} for k in pairs(t) do out[#out + 1] = k end "
        'Debug.ShowHint(table.concat(out, ","))'
    )
    scripts = _core.ScriptDirectory([str(tmp_path)])
    orders = []
    for seed in ([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]):
        duel = _core.Duel(seed, C.DUEL_MODE_MR5, (8000, 5, 1), (8000, 5, 1), db.to_core(), scripts)
        assert duel.load_script("probe.lua")
        hints = [m for m in M.decode_buffer(duel.get_message()) if isinstance(m, M.ShowHint)]
        orders.append(hints[0].text)
        duel.close()
    assert orders[0] == orders[1] == orders[2]
