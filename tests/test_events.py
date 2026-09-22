"""Event token stream with response-window / abstain tokens (T2.4, docs/encoding.md)."""

import random
import struct
from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.events import DEFAULT_EVENT_LENGTH, E_EVENT, EVENT_TYPES, TRIGGERS, EventHistory

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
COL = {name: i for i, name in enumerate(
    ["type", "player", "card", "card2", "from_controller", "from_location", "from_sequence", "from_position",
     "to_controller", "to_location", "to_sequence", "to_position", "value1", "value2", "value3", "turn", "phase",
     "my_turn", "my_lp", "op_lp"])}  # fmt: skip
T = {name: i + 1 for i, name in enumerate(EVENT_TYPES)}

ROTA, CELTIC, ASH, BEWD = 32807846, 91152256, 14558127, 89631139  # Reinforcement of the Army, Celtic Guardian, ...


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def passwords(vocab):
    return [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]


def rows(obs):
    return obs["events"][obs["event_mask"] == 1]


def of_type(obs, name):
    return [r for r in rows(obs) if r[COL["type"]] == T[name]]


# ------------------------------------------------------------------ hand-built scenario: Ash Blossom could respond


class Scripted:
    """Both seats: activate / select / pass (or chain when ``chain``), feeding one shared history."""

    def __init__(self, history, chain=False):
        self.history, self.chain = history, chain
        self.seen = []  # (point, obs)

    def act(self, point):
        self.history.feed(point.events)
        self.seen.append((point, self.history.encode(point.player)))
        kinds = [a.kind for a in point.actions]
        for pref in (("chain",) if self.chain else ()) + ("activate", "select", "pass", "no", "end_phase", "finish"):
            if pref in kinds:
                return kinds.index(pref)
        return 0


def scenario(db, vocab, b_card, chain=False, length=64):
    """Player 0 (A) holds Reinforcement of the Army and searches; player 1 (B) holds five copies of ``b_card``."""
    deck_a = Deck(main=(CELTIC,) * 20 + (ROTA,) * 20)  # unshuffled: the last five cards are drawn
    deck_b = Deck(main=(b_card,) * 40)
    duel = Duel(1, None, deck_a, deck_b, cards=db, config=DuelConfig(max_turns=2, shuffle_decks=False))
    agent = Scripted(EventHistory(db, vocab, length=length), chain)
    duel.run(agent, agent)
    return agent.seen


def first_after_chain_end(seen, player):
    """The first decision of ``player`` whose stream contains a chain_end."""
    return next(obs for point, obs in seen if point.player == player and of_type(obs, "chain_end"))


def test_ash_blossom_could_respond_but_passes(db, vocab):
    seen = scenario(db, vocab, ASH)
    # B was really offered Ash Blossom against the search and passed
    offer = next(p for p, _ in seen if p.player == 1 and isinstance(p.decision, M.SelectChain))
    assert any(c.code == ASH for c in offer.decision.chains)
    for viewer, me in ((0, 2), (1, 1)):  # B is the opponent of viewer 0 and the viewer itself for viewer 1
        (tok,) = [r for r in of_type(first_after_chain_end(seen, viewer), "abstain") if r[COL["player"]] == me]
        assert tok[COL["card"]] == vocab.index(ROTA)
        assert tok[COL["value1"]] == TRIGGERS["search"]
        assert tok[COL["value2"]] == 0  # B's field
        assert tok[COL["value3"]] == 5  # B's hand
        assert tok[COL["from_controller"]] == (1 if viewer == 0 else 2) and tok[COL["from_location"]] == 4
        assert (tok[COL["turn"]], tok[COL["my_turn"]], tok[COL["my_lp"]], tok[COL["op_lp"]]) == (
            1,
            int(viewer == 0),
            8000,
            8000,
        )
        assert tok[COL["phase"]] == (C.PHASE_MAIN1).bit_length()


def test_chained_ash_gives_no_abstain_for_that_link(db, vocab):
    seen = scenario(db, vocab, ASH, chain=True)
    obs = first_after_chain_end(seen, 0)
    assert not [r for r in of_type(obs, "abstain") if r[COL["player"]] == 2]
    (ash,) = [r for r in of_type(obs, "chaining") if r[COL["card"]] == vocab.index(ASH)]
    assert ash[COL["player"]] == 2 and ash[COL["value1"]] == 2
    assert [r[COL["value1"]] for r in of_type(obs, "chain_disabled")] == [1]
    # A did not answer Ash either: that is an abstain token too, with no deck movement ("other")
    (mine,) = [r for r in of_type(obs, "abstain") if r[COL["player"]] == 1]
    assert mine[COL["card"]] == vocab.index(ASH) and mine[COL["value1"]] == TRIGGERS["other"]


def test_abstain_stream_is_public(db, vocab):
    """A's stream cannot tell "B had Ash Blossom and passed" from "B had nothing": both are identical."""
    with_ash = [obs for p, obs in scenario(db, vocab, ASH) if p.player == 0]
    without = [obs for p, obs in scenario(db, vocab, BEWD) if p.player == 0]
    assert len(with_ash) == len(without) > 5
    for a, b in zip(with_ash, without):
        np.testing.assert_array_equal(a["events"], b["events"])
    assert any(r[COL["player"]] == 2 for r in of_type(with_ash[-1], "abstain"))


def test_search_confirm_reveals_only_the_searched_card(db, vocab):
    obs = first_after_chain_end(scenario(db, vocab, BEWD), 1)  # B's view of A's search
    (search,) = [r for r in of_type(obs, "move") if r[COL["from_location"]] == 1 and r[COL["to_location"]] == 2]
    assert search[COL["card"]] == 1 and search[COL["from_sequence"]] == 0  # hidden; no deck order
    assert [r[COL["card"]] for r in of_type(obs, "confirm_cards")] == [vocab.index(CELTIC)]
    draws = of_type(obs, "draw")
    assert {r[COL["card"]] for r in draws if r[COL["player"]] == 2} == {1}  # A's opening hand is hidden from B
    assert {r[COL["card"]] for r in draws if r[COL["player"]] == 1} == {vocab.index(BEWD)}


# ------------------------------------------------------------------ synthetic event windows


def loc(con, location, seq, pos=0):
    return M.Location(con, location, seq, pos)


IDLE = M.SelectIdleCmd(0, (), (), (), (), (), (), False, True, False)


def summons(n, con=0):
    return [M.Summoning(CELTIC, loc(con, C.LOCATION_MZONE, i, C.POS_FACEUP_ATTACK)) for i in range(n)]


def test_fifth_summon_window(db, vocab):
    h = EventHistory(db, vocab)
    h.feed([M.NewTurn(0), M.NewPhase(C.PHASE_MAIN1), *summons(5), IDLE])
    (tok,) = of_type(h.encode(1), "abstain")
    assert tok[COL["player"]] == 1 and tok[COL["value1"]] == TRIGGERS["fifth_summon"]
    assert tok[COL["card"]] == vocab.index(CELTIC) and tok[COL["from_sequence"]] == 4
    h = EventHistory(db, vocab)  # four summons: no window; the opponent responding: no token
    h.feed([M.NewTurn(0), *summons(4), IDLE])
    assert not of_type(h.encode(1), "abstain")
    h = EventHistory(db, vocab)
    h.feed([M.NewTurn(0), *summons(5), M.Chaining(ASH, loc(1, C.LOCATION_HAND, 0), 1, C.LOCATION_HAND, 0, 0, 1), IDLE])
    assert not of_type(h.encode(1), "abstain")


def test_attack_window_closes_at_the_next_open_state(db, vocab):
    h = EventHistory(db, vocab)
    attacker = loc(0, C.LOCATION_MZONE, 2, C.POS_FACEUP_ATTACK)
    h.feed([M.NewTurn(0), M.Move(CELTIC, loc(0, C.LOCATION_HAND, 0), attacker, 0), M.NewPhase(C.PHASE_BATTLE_STEP),
            M.Attack(attacker, loc(0, 0, 0))])  # fmt: skip
    assert not of_type(h.encode(1), "abstain")  # still open
    h.feed([M.Damage(1, 1400), M.SelectBattleCmd(0, (), (), True, True)])
    (tok,) = of_type(h.encode(1), "abstain")
    assert (
        tok[COL["player"]] == 1 and tok[COL["value1"]] == TRIGGERS["attack"] and tok[COL["card"]] == vocab.index(CELTIC)
    )
    assert tok[COL["op_lp"]] == 8000 and tok[COL["my_lp"]] == 6600
    (atk,) = of_type(h.encode(1), "attack")
    assert atk[COL["value1"]] == 1 and atk[COL["to_location"]] == 0


def test_hidden_set_and_field_map(db, vocab):
    h = EventHistory(db, vocab)
    zone = loc(1, C.LOCATION_SZONE, 1, C.POS_FACEDOWN)
    h.feed([M.Draw(1, ((ASH, C.POS_FACEDOWN),)), M.Move(ASH, loc(1, C.LOCATION_HAND, 0, C.POS_FACEDOWN), zone, 0),
            M.SetCard(ASH, zone), M.BecomeTarget((zone,)),
            M.PosChange(ASH, loc(1, C.LOCATION_SZONE, 1), C.POS_FACEDOWN, C.POS_FACEUP),
            M.BecomeTarget((zone,))])  # fmt: skip
    ash = vocab.index(ASH)
    assert [r[COL["card"]] for r in rows(h.encode(0))] == [1, 1, 1, 1, ash, ash]  # the opponent's view
    assert [r[COL["card"]] for r in rows(h.encode(1))] == [ash] * 6
    assert h.field_count(1) == 1 and h.hand_count(1) == 0 and h.field_count(0) == h.hand_count(0) == 0


# ------------------------------------------------------------------ real games: visibility, tracking, length


def history_game(db, vocab, seed, a, b, length=DEFAULT_EVENT_LENGTH, check=None, max_decisions=600):
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=max_decisions))
    histories = [EventHistory(db, vocab, length=length)]

    class Agent:
        def __init__(self, s):
            self.inner = RandomAgent(s)

        def act(self, point):
            for h in histories:
                h.feed(point.events)
            if check:
                check(point, duel, histories[0])
            return self.inner.act(point)

    duel.run(Agent(seed), Agent(seed + 1))
    return histories[0]


def test_opponent_draws_and_sets_are_hidden(db, vocab):
    h = history_game(db, vocab, 7, "snake_eye", "kashtira", length=100_000)
    for viewer in (0, 1):
        obs = h.encode(viewer)
        draws = of_type(obs, "draw")
        assert draws and all(r[COL["card"]] == 1 for r in draws if r[COL["player"]] == 2)
        assert all(r[COL["card"]] >= 2 for r in draws if r[COL["player"]] == 1)
        assert all(r[COL["card"]] == 1 for r in of_type(obs, "set") if r[COL["player"]] == 2)
        for r in of_type(obs, "move"):  # into the opponent's hand or deck from a hidden place
            if r[COL["to_controller"]] == 2 and r[COL["to_location"]] in (1, 2) and r[COL["from_location"]] in (1, 2):
                assert r[COL["card"]] == 1
            if r[COL["from_location"]] == 1 or r[COL["to_location"]] == 1:
                assert (r[COL["from_location"]] != 1 or r[COL["from_sequence"]] == 0) and (
                    r[COL["to_location"]] != 1 or r[COL["to_sequence"]] == 0)  # fmt: skip


@pytest.mark.parametrize("seed, a, b", [(11, "labrynth", "tenpai"), (12, "branded_despia", "purrely")])
def test_tracked_counts_match_the_core(db, vocab, seed, a, b):
    checked = []

    def check(point, duel, h):
        core = duel._core
        for p in (0, 1):
            assert h.hand_count(p) == core.query_count(p, C.LOCATION_HAND), point.index
            assert h.field_count(p) == core.query_count(p, C.LOCATION_MZONE) + core.query_count(p, C.LOCATION_SZONE)
        checked.append(1)

    history_game(db, vocab, seed, a, b, check=check)
    assert len(checked) > 100


def test_length_is_configurable(db, vocab):
    full = history_game(db, vocab, 5, "yubel", "voiceless_voice", length=100_000)
    for length in (0, 1, 8, 300):
        h = history_game(db, vocab, 5, "yubel", "voiceless_voice", length=length)
        for viewer in (0, 1):
            obs, ref = h.encode(viewer), rows(full.encode(viewer))
            assert len(ref) > 300
            assert obs["events"].shape == (length, E_EVENT) and obs["event_mask"].shape == (length,)
            assert obs["events"].dtype == np.int32 and obs["event_mask"].dtype == np.int32
            n = min(length, len(ref))
            assert obs["event_mask"].sum() == n and not obs["events"][n:].any()
            np.testing.assert_array_equal(obs["events"][:n], ref[len(ref) - n :])


# ------------------------------------------------------------------ C++ port


def cpp_history(db, vocab, length):
    return _core.EventHistory(db.to_core(), passwords(vocab), length, 8000)


LOCS = [0, C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_GRAVE, C.LOCATION_REMOVED,
        C.LOCATION_EXTRA, C.LOCATION_OVERLAY | C.LOCATION_MZONE, C.LOCATION_ONFIELD]  # fmt: skip
POSS = [0, 1, 2, 4, 8, 5, 10]


def rand_record(rng, t):
    code = lambda: rng.choice([0, ROTA, CELTIC, ASH, BEWD, 12345678])  # noqa: E731
    p = lambda: rng.choice([0, 1, 0, 1, 2])  # noqa: E731
    li = lambda: struct.pack("<BBII", p(), rng.choice(LOCS), rng.randrange(9), rng.choice(POSS))  # noqa: E731
    n = rng.randrange(4)
    body = {
        C.MSG_DRAW: lambda: (
            struct.pack("<BI", p(), n) + b"".join(struct.pack("<II", code(), rng.choice(POSS)) for _ in range(n))
        ),
        C.MSG_MOVE: lambda: (
            struct.pack("<I", code()) + li() + li() + struct.pack("<I", rng.choice([0, 0x40, 0x80000400]))
        ),
        C.MSG_POS_CHANGE: lambda: struct.pack(
            "<IBBBBB", code(), p(), rng.choice(LOCS), rng.randrange(9), rng.choice(POSS), rng.choice(POSS)
        ),
        C.MSG_SWAP: lambda: struct.pack("<I", code()) + li() + struct.pack("<I", code()) + li(),
        C.MSG_CHAINING: lambda: (
            struct.pack("<I", code())
            + li()
            + struct.pack(
                "<BBIQI",
                p(),
                rng.choice(LOCS),
                rng.randrange(9),
                rng.choice([0, 1160, ROTA << 20 | 1, 99 << 20]),
                rng.randrange(5),
            )
        ),
        C.MSG_NEW_TURN: lambda: struct.pack("<B", p()),
        C.MSG_NEW_PHASE: lambda: struct.pack("<H", rng.choice([C.PHASE_MAIN1, C.PHASE_BATTLE_STEP, C.PHASE_END, 0])),
        C.MSG_BATTLE: lambda: (
            li() + struct.pack("<iiB", rng.randrange(-5, 70000), 1000, 1) + li() + struct.pack("<iiB", 500, -1, 0)
        ),
        C.MSG_UNEQUIP: li,
        C.MSG_ADD_COUNTER: lambda: struct.pack(
            "<HBBBH", 0x1001, p(), rng.choice(LOCS), rng.randrange(9), rng.randrange(5)
        ),
        C.MSG_REMOVE_COUNTER: lambda: struct.pack(
            "<HBBBH", 0x1001, p(), rng.choice(LOCS), rng.randrange(9), rng.randrange(5)
        ),
        C.MSG_RANDOM_SELECTED: lambda: struct.pack("<BI", p(), n) + b"".join(li() for _ in range(n)),
        C.MSG_DECK_TOP: lambda: struct.pack("<BIII", p(), 0, code(), rng.choice(POSS)),
        C.MSG_SHUFFLE_DECK: lambda: struct.pack("<B", p()),
        C.MSG_SHUFFLE_HAND: lambda: struct.pack("<BI", p(), n) + b"".join(struct.pack("<I", code()) for _ in range(n)),
        C.MSG_SHUFFLE_EXTRA: lambda: struct.pack("<BI", p(), n) + b"".join(struct.pack("<I", code()) for _ in range(n)),
        C.MSG_SHUFFLE_SET_CARD: lambda: struct.pack("<BB", rng.choice(LOCS), n) + b"".join(li() for _ in range(2 * n)),
        C.MSG_SWAP_GRAVE_DECK: lambda: struct.pack("<BII", p(), 3, 1) + b"\x01",
        C.MSG_FIELD_DISABLED: lambda: struct.pack("<I", rng.getrandbits(32)),
        C.MSG_TOSS_COIN: lambda: struct.pack("<BB", p(), n) + bytes(rng.randrange(2) for _ in range(n)),
        C.MSG_TOSS_DICE: lambda: struct.pack("<BB", p(), n + 9) + bytes(rng.randrange(1, 7) for _ in range(n + 9)),
        C.MSG_HAND_RES: lambda: struct.pack("<B", rng.randrange(16)),
        C.MSG_HINT: lambda: struct.pack("<BBQ", 1, p(), 5),
        C.MSG_WIN: lambda: struct.pack("<BB", p(), 1),
        C.MSG_SELECT_IDLECMD: lambda: struct.pack("<B6IBBB", p(), 0, 0, 0, 0, 0, 0, 1, 1, 0),
        C.MSG_SELECT_BATTLECMD: lambda: struct.pack("<BIIBB", p(), 0, 0, 1, 1),
        C.MSG_SELECT_CHAIN: lambda: struct.pack("<BBBIII", p(), 0, 0, 0, 0, 0),
    }
    for t2 in (C.MSG_SET, C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
        body[t2] = lambda: struct.pack("<I", code()) + li()
    for t2 in (C.MSG_CHAINED, C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
        body[t2] = lambda: struct.pack("<B", rng.randrange(5))
    for t2 in (C.MSG_CHAIN_END, C.MSG_SUMMONED, C.MSG_SPSUMMONED, C.MSG_FLIPSUMMONED, C.MSG_ATTACK_DISABLED,
               C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END, C.MSG_REVERSE_DECK, C.MSG_RETRY):  # fmt: skip
        body[t2] = lambda: b""
    for t2 in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_LPUPDATE, C.MSG_PAY_LPCOST):
        body[t2] = lambda: struct.pack("<BI", p(), rng.randrange(9000))
    for t2 in (C.MSG_ATTACK, C.MSG_EQUIP, C.MSG_CARD_TARGET, C.MSG_CANCEL_TARGET):
        body[t2] = lambda: li() + li()
    for t2 in (C.MSG_BECOME_TARGET, C.MSG_CARD_SELECTED):
        body[t2] = lambda: struct.pack("<I", n) + b"".join(li() for _ in range(n))
    for t2 in (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP):
        body[t2] = lambda: struct.pack("<BI", p(), n) + b"".join(
            struct.pack("<IBBI", code(), p(), rng.choice(LOCS), rng.randrange(9)) for _ in range(n))  # fmt: skip
    if t is None:
        return sorted(body)
    rec = bytes([t]) + body[t]()
    if rng.random() < 0.05:  # truncated / padded payloads are skipped / tolerated like the Python decoder does
        rec = rec[: rng.randrange(1, len(rec) + 1)] if rng.random() < 0.7 else rec + b"\x00\x07"
    return rec


def frame(records):
    return b"".join(struct.pack("<I", len(r)) + r for r in records)


def test_cpp_matches_python_on_random_messages(db, vocab):
    rng = random.Random(4)
    types = rand_record(rng, None)
    for trial in range(30):
        py, cpp = EventHistory(db, vocab, length=48), cpp_history(db, vocab, 48)
        for _ in range(40):
            buf = frame([rand_record(rng, rng.choice(types)) for _ in range(rng.randrange(1, 8))])
            py.feed(M.decode_buffer(buf))
            cpp.feed(buf)
            for viewer in (0, 1):
                events, mask = cpp.encode(viewer)
                ref = py.encode(viewer)
                np.testing.assert_array_equal(events, ref["events"], err_msg=f"trial {trial}")
                np.testing.assert_array_equal(mask, ref["event_mask"])
            assert [cpp.hand_count(p) for p in (0, 1)] == [py.hand_count(p) for p in (0, 1)]
            assert [cpp.field_count(p) for p in (0, 1)] == [py.field_count(p) for p in (0, 1)]


@pytest.mark.parametrize("seed, a, b, first", [(21, "snake_eye", "kashtira", 0), (22, "tearlaments", "yubel", 1),
                                               (23, "fiendsmith_ryzeal", "labrynth", 0)])  # fmt: skip
def test_host_duel_events_match_python(db, vocab, seed, a, b, first):
    cfg = DuelConfig(max_decisions=1500)
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=cfg)
    host = _core.HostDuel(db.to_core(), default_scripts(), passwords(vocab), event_length=96)
    host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    history = EventHistory(db, vocab, length=96)
    points = []

    class Lockstep:
        def __init__(self, s):
            self.inner = RandomAgent(s)

        def act(self, point):
            history.feed(point.events)
            ref, obs = history.encode(point.player), host.observe()
            np.testing.assert_array_equal(obs["events"], ref["events"], err_msg=f"decision {point.index}")
            np.testing.assert_array_equal(obs["event_mask"], ref["event_mask"])
            idx = self.inner.act(point)
            host.act(idx)
            points.append(1)
            return idx

    duel.run(Lockstep(seed), Lockstep(seed + 1))
    assert len(points) > 100


def test_encoded_vec_env_exposes_events(db, vocab):
    spec = GameSpec(seed=31, deck_a=DECKS["snake_eye"], deck_b=DECKS["yubel"])
    for length, expect in ((16, True), (0, False), (None, True)):
        kwargs = {} if length is None else {"event_length": length}
        env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, **kwargs)
        env.reset(0, spec)
        (ev,) = env.recv(1)
        assert ("events" in ev.obs) == expect and ("event_mask" in ev.obs) == expect
        if expect:
            n = DEFAULT_EVENT_LENGTH if length is None else length
            assert ev.obs["events"].shape == (n, E_EVENT) and ev.obs["event_mask"].shape == (n,)
            assert ev.obs["event_mask"].sum() > 0 and of_type(ev.obs, "draw")


def test_shuffled_set_cards_keep_the_field_map_consistent(db, vocab):
    """MSG_SHUFFLE_SET_CARD hides which set card is in which zone: the zones stay occupied, their
    identities become unknown, and a later move out of one of them clears it (audit finding)."""

    def locb(con, location, seq, pos):
        return struct.pack("<BBII", con, location, seq, pos)

    fd = C.POS_FACEDOWN_DEFENSE
    a, b = locb(1, C.LOCATION_MZONE, 0, fd), locb(1, C.LOCATION_MZONE, 1, fd)
    grave = locb(1, C.LOCATION_GRAVE, 0, C.POS_FACEUP)
    records = [
        bytes([C.MSG_SET]) + struct.pack("<I", ASH) + a,
        bytes([C.MSG_SET]) + struct.pack("<I", CELTIC) + b,
        bytes([C.MSG_SHUFFLE_SET_CARD, C.LOCATION_MZONE, 2]) + a + b + locb(0, 0, 0, 0) * 2,
        bytes([C.MSG_MOVE]) + struct.pack("<I", CELTIC) + a + grave + struct.pack("<I", C.REASON_DESTROY),
    ]
    py, cpp = EventHistory(db, vocab, length=16), cpp_history(db, vocab, 16)
    py.feed(M.decode_buffer(frame(records)))
    cpp.feed(frame(records))
    assert py.field_count(1) == cpp.field_count(1) == 1  # the core has one set card left
    for viewer in (0, 1):
        events, _ = cpp.encode(viewer)
        np.testing.assert_array_equal(events, py.encode(viewer)["events"])
