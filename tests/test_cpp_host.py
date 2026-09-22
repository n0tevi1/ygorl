"""C++ host (decision decoding, action states, tracker, encoder) vs the Python reference (T2.2)."""

import random
import struct
from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.encoding import ACTION_KINDS, ObservationEncoder

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def py_action(a):
    card = a.card
    loc = card.loc if card is not None else None
    return (ACTION_KINDS.index(a.kind), a.index, card is not None, card.code if card else 0,
            (loc.controller, loc.location, loc.sequence, loc.position) if loc else (0, 0, 0, 0), a.description, a.value)  # fmt: skip


# ------------------------------------------------------------------ differential fuzzing of decision states

LOC = struct.Struct("<BBII")


def loc_info(rng, con=None):
    return LOC.pack(rng.randrange(2) if con is None else con, rng.choice([2, 4, 8, 16, 32]), rng.randrange(7), rng.choice([1, 4, 8, 10]))


def rand_record(rng, kind):
    p = rng.randrange(2)
    code = lambda: rng.choice([89631139, 14558127, 1861629, 46986414])  # noqa: E731
    if kind == C.MSG_SELECT_CARD:
        n = rng.randrange(1, 7)
        mx = rng.randrange(1, n + 1)
        mn = rng.randrange(0, mx + 1)
        body = struct.pack("<BBIII", p, rng.randrange(2), mn, mx, n) + b"".join(struct.pack("<I", code()) + loc_info(rng) for _ in range(n))
    elif kind == C.MSG_SELECT_TRIBUTE:
        n = rng.randrange(1, 6)
        mx = rng.randrange(1, 4)
        body = struct.pack("<BBIII", p, rng.randrange(2), rng.randrange(0, mx + 1), mx, n)
        body += b"".join(struct.pack("<IBBIB", code(), p, 4, i, rng.choice([1, 1, 2])) for i in range(n))
    elif kind == C.MSG_SELECT_SUM:
        must = rng.randrange(0, 2)
        n = rng.randrange(1, 7)
        exact = rng.randrange(2)
        mn = rng.randrange(1, 3)
        mx = rng.randrange(mn, 5) if exact else 0

        def param():
            lo = rng.randrange(1, 9)
            return lo | ((rng.randrange(1, 9) << 16) if rng.random() < 0.3 else 0)

        body = struct.pack("<BBIIII", p, 0 if exact else 1, rng.randrange(4, 17), mn, mx, must)
        body += b"".join(struct.pack("<I", code()) + loc_info(rng) + struct.pack("<I", param()) for _ in range(must))
        body += struct.pack("<I", n) + b"".join(struct.pack("<I", code()) + loc_info(rng) + struct.pack("<I", param()) for _ in range(n))
    elif kind == C.MSG_SELECT_UNSELECT_CARD:
        n, u = rng.randrange(0, 5), rng.randrange(0, 3)
        body = struct.pack("<BBBII", p, rng.randrange(2), rng.randrange(2), 1, 3) + struct.pack("<I", n)
        body += b"".join(struct.pack("<I", code()) + loc_info(rng) for _ in range(n)) + struct.pack("<I", u)
        body += b"".join(struct.pack("<I", code()) + loc_info(rng) for _ in range(u))
    elif kind == C.MSG_SELECT_COUNTER:
        n = rng.randrange(1, 4)
        counters = [rng.randrange(1, 4) for _ in range(n)]
        body = struct.pack("<BHHI", p, 0x1, rng.randrange(1, sum(counters) + 1), n)
        body += b"".join(struct.pack("<IBBBH", code(), p, 8, i, c) for i, c in enumerate(counters))
    elif kind in (C.MSG_SORT_CARD, C.MSG_SORT_CHAIN):
        n = rng.randrange(1, 5)
        body = struct.pack("<BI", p, n) + b"".join(struct.pack("<IBII", code(), p, 4, i) for i in range(n))
    elif kind in (C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD):
        body = struct.pack("<BBI", p, rng.randrange(1, 3), rng.getrandbits(32) | 0x00400000 | rng.getrandbits(32))
    elif kind == C.MSG_ANNOUNCE_RACE:
        body = struct.pack("<BBQ", p, rng.randrange(1, 3), rng.getrandbits(8) | 0x3)
    elif kind == C.MSG_ANNOUNCE_ATTRIB:
        body = struct.pack("<BBI", p, rng.randrange(1, 3), rng.getrandbits(7) | 0x3)
    elif kind == C.MSG_ANNOUNCE_NUMBER:
        n = rng.randrange(1, 5)
        body = struct.pack(f"<BB{n}Q", p, n, *(rng.randrange(1, 13) for _ in range(n)))
    elif kind == C.MSG_ANNOUNCE_CARD:
        ops = rng.choice([[C.TYPE_MONSTER, C.OPCODE_ISTYPE], [0xDD, C.OPCODE_ISSETCARD], [89631139, C.OPCODE_ISCODE],
                          [C.RACE_DRAGON, C.OPCODE_ISRACE, C.ATTRIBUTE_LIGHT, C.OPCODE_ISATTRIBUTE, C.OPCODE_AND]])  # fmt: skip
        body = struct.pack(f"<BB{len(ops)}Q", p, len(ops), *ops)
    elif kind == C.MSG_SELECT_POSITION:
        body = struct.pack("<BIB", p, code(), rng.randrange(1, 16))
    elif kind == C.MSG_SELECT_OPTION:
        n = rng.randrange(1, 4)
        body = struct.pack(f"<BB{n}Q", p, n, *(rng.getrandbits(40) for _ in range(n)))
    elif kind == C.MSG_SELECT_CHAIN:
        n = rng.randrange(0, 4)
        body = struct.pack("<BBBIII", p, 0, rng.randrange(2) if n else 0, 0, 0, n)
        body += b"".join(struct.pack("<I", code()) + loc_info(rng) + struct.pack("<QB", rng.getrandbits(40), 0) for _ in range(n))
    elif kind in (C.MSG_SELECT_YESNO,):
        body = struct.pack("<BQ", p, rng.getrandbits(40))
    elif kind == C.MSG_SELECT_EFFECTYN:
        body = struct.pack("<BI", p, code()) + loc_info(rng) + struct.pack("<Q", rng.getrandbits(40))
    elif kind == C.MSG_ROCK_PAPER_SCISSORS:
        body = struct.pack("<B", p)
    else:
        raise AssertionError(kind)
    return bytes([kind]) + body


FUZZ_KINDS = [C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM, C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_COUNTER,
              C.MSG_SORT_CARD, C.MSG_SORT_CHAIN, C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD, C.MSG_ANNOUNCE_RACE,
              C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_NUMBER, C.MSG_ANNOUNCE_CARD, C.MSG_SELECT_POSITION,
              C.MSG_SELECT_OPTION, C.MSG_SELECT_CHAIN, C.MSG_SELECT_YESNO, C.MSG_SELECT_EFFECTYN,
              C.MSG_ROCK_PAPER_SCISSORS]  # fmt: skip


@pytest.mark.parametrize("kind", FUZZ_KINDS, ids=lambda k: M.MESSAGE_NAMES[k])
def test_decision_states_match_python(db, kind):
    rng = random.Random(kind)
    native = db.to_core()
    for _ in range(60):
        record = rand_record(rng, kind)
        py = make_decision(M.decode_message(record), cards=db)
        cpp = _core.DecisionState(record, native)
        while True:
            assert cpp.actions() == [py_action(a) for a in py.actions()]
            if not py.actions():
                break
            i = rng.randrange(len(py.actions()))
            assert cpp.step(i) == py.step(i)
            if py.done:
                assert cpp.done
                break


def test_decision_state_rejects_garbage(db):
    with pytest.raises(ValueError):
        _core.DecisionState(bytes([C.MSG_SELECT_CARD, 0]), db.to_core())
    with pytest.raises(ValueError):
        _core.DecisionState(bytes([C.MSG_DRAW, 0, 0, 0, 0, 0]), db.to_core())


# ------------------------------------------------------------------ lockstep on real games


class Lockstep:
    """Plays a Python duel while stepping a C++ HostDuel with the same choices, comparing everything."""

    def __init__(self, seed, host, encoder, duel, checks):
        self.rng, self.host, self.encoder, self.duel, self.checks = RandomAgent(seed), host, encoder, duel, checks

    def act(self, point):
        host = self.host
        assert host.player() == point.player
        assert host.actions() == [py_action(a) for a in point.actions]
        py_obs = self.encoder.encode(point, self.duel._core)
        cpp_obs = host.observe()
        for k in py_obs:
            np.testing.assert_array_equal(cpp_obs[k], py_obs[k], err_msg=f"{k} at decision {point.index}")
        idx = self.rng.act(point)
        host.act(idx)
        self.checks.append(1)
        return idx


def lockstep_game(db, vocab, seed, a, b, first=0, max_decisions=20000):
    cfg = DuelConfig(max_decisions=max_decisions)
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=cfg)
    host = _core.HostDuel(db.to_core(), default_scripts(), vocab_passwords(vocab))
    decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
    host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1), decks, cfg.max_turns, cfg.max_decisions)
    checks = []
    encoder = ObservationEncoder(db, vocab)
    seats = [Lockstep(seed, host, encoder, duel, checks), Lockstep(seed + 1, host, encoder, duel, checks)]
    result = duel.run(*(seats if first == 0 else seats[::-1]))
    r = host.result()
    assert host.done()
    assert (r["winner"], r["reason"], r["win_reason"], r["turns"], tuple(r["lp"]), r["decisions"]) == (
        result.winner if first == 0 else (None if result.winner is None else 1 - result.winner),
        result.reason, result.win_reason, result.turns, result.lp if first == 0 else result.lp[::-1], result.decisions)  # fmt: skip
    assert r["responses"] == result.responses
    return len(checks)


def vocab_passwords(vocab):
    return [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]


@pytest.mark.parametrize("seed, a, b, first", [(1, "snake_eye", "kashtira", 0), (2, "labrynth", "tenpai", 1),
                                               (3, "branded_despia", "purrely", 0), (4, "voiceless_voice", "yubel", 1),
                                               (5, "fiendsmith_ryzeal", "tearlaments", 0)])  # fmt: skip
def test_host_duel_matches_python_step_by_step(db, vocab, seed, a, b, first):
    assert lockstep_game(db, vocab, seed, a, b, first) > 100
