"""Training-mode ground truth (privileged tensors) and the inference-mode switch (T2.5, docs/encoding.md)."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.engine.query import parse_query_location
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv, chooser
from ygorl.env.encoding import ObservationEncoder
from ygorl.env.privileged import (
    P_WIDTHS,
    PRIVILEGED_KEYS,
    CandidateCards,
    belief_targets,
    copy_counts,
    encode_privileged,
)
from ygorl.eval.beliefs import BeliefBatch, Head

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)
ACTOR_KEYS = {"cards", "globals", "actions", "action_mask"}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def passwords(vocab):
    return [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]


def specs(n):
    return [GameSpec(seed=7000 + i, deck_a=DECKS[NAMES[i % 10]], deck_b=DECKS[NAMES[(i * 3 + 1) % 10]], first=i % 2)
            for i in range(n)]  # fmt: skip


# ------------------------------------------------------------------ inference mode: nothing privileged


def test_encoded_env_defaults_to_inference_mode(db, vocab):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab)
    assert env.privileged is False
    env.reset(0, specs(1)[0])
    for _ in range(20):
        (ev,) = env.recv(1)
        assert ev.privileged is None
        if ev.result is not None:
            break
        assert set(ev.obs) == ACTOR_KEYS
        env.step(0, 0)


def test_python_encoder_refuses_privileged_in_inference_mode(db, vocab):
    duel = Duel(1, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db)
    enc = ObservationEncoder(db, vocab)
    assert enc.privileged is False
    seen = []

    class Probe:
        def act(self, point):
            with pytest.raises(RuntimeError, match="privileged"):
                enc.encode_privileged(point, duel._core)
            seen.append(1)
            raise StopIteration

    with pytest.raises(StopIteration):
        duel.run(Probe(), Probe())
    assert seen


class SpyCore:
    """Wraps a core and records every (controller, location) the encoder queries."""

    def __init__(self, core):
        self.core, self.calls = core, []

    def query_location(self, flags, con, loc):
        self.calls.append((con, loc))
        return self.core.query_location(flags, con, loc)

    def __getattr__(self, name):
        return getattr(self.core, name)


def test_actor_encoding_never_queries_the_opponent_deck(db, vocab):
    duel = Duel(2, None, DECKS["labrynth"], DECKS["tenpai"], cards=db, config=DuelConfig(max_decisions=150))
    enc = ObservationEncoder(db, vocab, privileged=True)
    n = [0]

    class Agent(RandomAgent):
        def act(self, point):
            spy = SpyCore(duel._core)
            enc.encode(point, spy)
            assert (1 - point.player, C.LOCATION_DECK) not in spy.calls
            n[0] += 1
            return super().act(point)

    duel.run(Agent(2), Agent(3))
    assert n[0] > 50


# ------------------------------------------------------------------ training mode == engine queries


def query_slot(core, con, loc, seq):
    buf = core.query(C.QUERY_CODE | C.QUERY_POSITION, con, loc, seq)
    if not buf:
        return None
    (card,) = parse_query_location(b"\0\0\0\0" + buf)
    return card


def expected_privileged(core, viewer, vocab):
    """Ground truth rebuilt from single-card ``core.query`` calls (independent of the location parser)."""
    op = 1 - viewer

    def listing(loc, keep=lambda c: True):
        out = []
        for seq in range(core.query_count(op, loc)):
            card = query_slot(core, op, loc, seq)
            if card is not None and keep(card):
                out.append((vocab.index(card["code"]), int(bool(card["public"])), seq))
        return out

    facedown = lambda c: bool(c.get("position", 0) & C.POS_FACEDOWN)  # noqa: E731
    exp = {
        "op_hand": listing(C.LOCATION_HAND),
        "op_deck": sorted((i, p, 0) for i, p, _ in listing(C.LOCATION_DECK)),
        "op_extra": sorted((i, p, 0) for i, p, _ in listing(C.LOCATION_EXTRA)),
        "op_removed": listing(C.LOCATION_REMOVED, facedown),
    }
    field = [(0, 0, 0)] * 15
    for loc, base, n in ((C.LOCATION_MZONE, 0, 7), (C.LOCATION_SZONE, 7, 8)):
        for seq in range(n):
            card = query_slot(core, op, loc, seq)
            if card is not None and facedown(card):
                field[base + seq] = (vocab.index(card["code"]), int(bool(card["public"])), seq)
    exp["op_set"] = field
    return exp


def check_against_engine(priv, core, viewer, vocab):
    exp = expected_privileged(core, viewer, vocab)
    assert set(priv) == set(PRIVILEGED_KEYS)
    for key in ("op_hand", "op_deck", "op_extra", "op_removed"):
        rows = priv[key]
        assert rows.shape == (P_WIDTHS[key], 3) and rows.dtype == np.int32
        n = len(exp[key])
        assert n <= len(rows), f"{key} truncated in a test game"
        assert [tuple(r) for r in rows[:n]] == exp[key], key
        assert not rows[n:].any(), key
    assert priv["op_set"].shape == (15, 3)
    assert [tuple(r) for r in priv["op_set"]] == exp["op_set"]
    set_count = sum(1 for r in exp["op_set"] if r[0])
    counts = [len(exp["op_hand"]), len(exp["op_deck"]), len(exp["op_extra"]), set_count, len(exp["op_removed"])]
    assert priv["counts"].tolist() == counts
    assert counts[0] == core.query_count(1 - viewer, C.LOCATION_HAND)
    assert counts[1] == core.query_count(1 - viewer, C.LOCATION_DECK)
    return counts


@pytest.mark.parametrize("seed, a, b", [(11, "kashtira", "snake_eye"), (12, "labrynth", "voiceless_voice"),
                                        (13, "tearlaments", "purrely")])  # fmt: skip
def test_python_privileged_matches_engine_queries(db, vocab, seed, a, b):
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=600))
    enc = ObservationEncoder(db, vocab, privileged=True)
    totals = np.zeros(5, dtype=int)
    n = [0]

    class Agent(RandomAgent):
        def act(self, point):
            priv = enc.encode_privileged(point, duel._core)
            totals[:] += check_against_engine(priv, duel._core, point.player, vocab)
            n[0] += 1
            return super().act(point)

    duel.run(Agent(seed), Agent(seed + 1))
    assert n[0] > 100
    assert totals[0] and totals[1] and totals[2]  # hands, decks and extra decks were actually checked


def test_set_cards_and_facedown_banish_are_seen(db, vocab):
    """Kashtira games exercise the per-zone and face-down banish tensors (not just padding) against the engine."""
    enc = ObservationEncoder(db, vocab, privileged=True)
    seen = {"set": 0, "removed": 0}
    for seed in range(20, 26):
        duel = Duel(seed, None, DECKS["kashtira"], DECKS[NAMES[seed % 10]], cards=db, first=seed % 2)

        class Agent(RandomAgent):
            def act(self, point):
                counts = check_against_engine(enc.encode_privileged(point, duel._core), duel._core, point.player, vocab)
                seen["set"] += int(counts[3] > 0)
                seen["removed"] += int(counts[4] > 0)
                return super().act(point)

        duel.run(Agent(seed), Agent(seed + 1))
    assert seen["set"] > 0 and seen["removed"] > 0


# ------------------------------------------------------------------ C++ == Python


def lockstep(db, vocab, seed, a, b, first):
    cfg = DuelConfig()
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=cfg)
    host = _core.HostDuel(db.to_core(), default_scripts(), passwords(vocab))
    host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    enc = ObservationEncoder(db, vocab, privileged=True)
    stats = {"points": 0, "set": 0, "removed": 0}

    class Agent(RandomAgent):
        def act(self, point):
            py = enc.encode_privileged(point, duel._core)
            cpp = host.observe_privileged()
            assert set(cpp) == set(py)
            for k in py:
                assert cpp[k].dtype == np.int32
                np.testing.assert_array_equal(cpp[k], py[k], err_msg=f"{k} at decision {point.index}")
            stats["points"] += 1
            stats["set"] += int(py["counts"][3] > 0)
            stats["removed"] += int(py["counts"][4] > 0)
            idx = super().act(point)
            host.act(idx)
            return idx

    duel.run(Agent(seed), Agent(seed + 1))
    assert host.done()
    return stats


@pytest.mark.parametrize("seed, a, b, first", [(31, "kashtira", "snake_eye", 0), (32, "labrynth", "tenpai", 1),
                                               (33, "branded_despia", "kashtira", 0), (34, "voiceless_voice", "yubel", 1),
                                               (35, "fiendsmith_ryzeal", "tearlaments", 0)])  # fmt: skip
def test_cpp_privileged_matches_python(db, vocab, seed, a, b, first):
    assert lockstep(db, vocab, seed, a, b, first)["points"] > 100


# ------------------------------------------------------------------ pool: training vs inference mode


def play_collect(db, vocab, spec, privileged):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=privileged)
    env.reset(0, spec)
    obs, priv, step = [], [], 0
    while True:
        (ev,) = env.recv(1)
        if ev.result is not None:
            assert ev.privileged is None
            return obs, priv, ev.result
        obs.append(ev.obs)
        priv.append(ev.privileged)
        env.step(0, chooser(spec.seed, step, int(ev.obs["action_mask"].sum())))
        step += 1


def sequential_privileged(db, vocab, spec):
    duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
    host = _core.HostDuel(db.to_core(), default_scripts(), passwords(vocab))
    cfg = spec.config
    host.start(expand_seed(spec.seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    out, step = [], 0
    while not host.done() and host.actions():
        out.append(host.observe_privileged())
        host.act(chooser(spec.seed, step, int(host.observe()["action_mask"].sum())))
        step += 1
    return out


def test_training_mode_pool_outputs_privileged_separately(db, vocab):
    spec = specs(1)[0]
    obs_t, priv_t, res_t = play_collect(db, vocab, spec, privileged=True)
    obs_i, priv_i, res_i = play_collect(db, vocab, spec, privileged=False)
    assert res_t["responses"] == res_i["responses"]
    assert all(p is None for p in priv_i)
    assert len(obs_t) == len(obs_i) > 50
    for a, b in zip(obs_t, obs_i, strict=True):  # the switch never changes what the actor sees
        assert set(a) == set(b) == ACTOR_KEYS
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    seq = sequential_privileged(db, vocab, spec)
    assert len(seq) == len(priv_t)
    for got, want in zip(priv_t, seq, strict=True):
        assert set(got) == set(PRIVILEGED_KEYS)
        for k in want:
            np.testing.assert_array_equal(got[k], want[k])


# ------------------------------------------------------------------ belief-head targets


def rows(*entries, width):
    out = np.zeros((width, 3), dtype=np.int32)
    for i, e in enumerate(entries):
        out[i] = e
    return out


def test_copy_counts_and_belief_targets(vocab):
    pw = [vocab.password(i) for i in range(2, 8)]  # six real cards, indices 2..7
    cands = CandidateCards(vocab, [pw[0], pw[1], pw[2], pw[3]])  # index 2..5 -> columns 0..3
    assert len(cands) == 4
    priv = {
        "op_hand": rows((2, 0, 0), (3, 1, 1), (7, 0, 2), width=P_WIDTHS["op_hand"]),
        "op_deck": rows(*[(2, 0, 0)] * 2, (4, 0, 0), *[(5, 0, 0)] * 5, width=P_WIDTHS["op_deck"]),
        "op_extra": rows((4, 1, 0), (5, 0, 0), width=P_WIDTHS["op_extra"]),
        "op_set": rows((0, 0, 0), (3, 0, 1), width=15),
        "op_removed": rows(width=P_WIDTHS["op_removed"]),
    }
    priv["op_set"][7 + 2] = (7, 0, 2)  # a non-candidate set spell/trap
    priv["op_set"][7 + 3] = (4, 1, 3)  # a revealed face-down card
    priv["counts"] = np.array([3, 8, 2, 3, 0], dtype=np.int32)

    assert copy_counts(priv["op_deck"], cands).tolist() == [2, 0, 1, 5]
    assert copy_counts(priv["op_deck"], vocab).shape == (len(vocab),)
    assert copy_counts(priv["op_deck"], vocab)[[0, 1, 2, 4, 5]].tolist() == [0, 0, 2, 1, 5]

    t = belief_targets(priv, cands)
    assert t["hand"].targets.tolist() == [1, 1, 0, 0]
    assert t["hand"].mask.tolist() == [True, False, True, True]  # public copy in hand: set to 1 by construction
    assert t["remaining_copies"].targets.tolist() == [2, 0, 1, 3]  # public extra copy excluded, clipped at 3
    assert t["remaining_copies"].mask.all()
    assert t["set_cards"].targets.tolist() == [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2] + [-1] * 4
    assert np.flatnonzero(t["set_cards"].mask).tolist() == [1]

    # batched input lines up with ygorl.eval.beliefs heads
    batch = {k: np.stack([v, v]) for k, v in priv.items()}
    tb = belief_targets(batch, cands)
    c, s = len(cands), 15
    BeliefBatch(
        hand=Head(np.full((2, c), 0.5), tb["hand"].targets, tb["hand"].mask),
        remaining_copies=Head(np.full((2, c, 4), 0.25), tb["remaining_copies"].targets, tb["remaining_copies"].mask),
        set_cards=Head(np.full((2, s, c), 1 / c), tb["set_cards"].targets, tb["set_cards"].mask),
    )
    np.testing.assert_array_equal(tb["hand"].targets[1], t["hand"].targets)


def test_encode_privileged_is_viewer_relative(db, vocab):
    core = Duel(5, None, DECKS["snake_eye"], DECKS["yubel"], cards=db)._setup()
    deck_counts = [core.query_count(p, C.LOCATION_DECK) for p in (0, 1)]
    for viewer in (0, 1):
        priv = encode_privileged(core, viewer, vocab)
        assert priv["counts"][1] == deck_counts[1 - viewer]
    core.close()
