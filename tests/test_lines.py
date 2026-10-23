"""Scripted-line pool opponents (ygorl.train.lines; docs/training.md §8.3 「脚本线对手」).

``tests/data/line_demos.jsonl`` holds three solved plain records of ``out/claude-scale/handtrap/full_fire8.jsonl``
(blue-eyes hands 1, 2 and 7; the environment binding removed so the test does not depend on the environment's
fingerprint): a 23-step line with chain windows for the opponent and two 4-step lines (set a trap, end the turn).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.engine.constants import DUEL_PSEUDO_SHUFFLE, MSG_SELECT_CHAIN  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.encoding import ACTION_KINDS  # noqa: E402
from ygorl.nets import NetConfig  # noqa: E402
from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.solver.demo import PASSIVE_KINDS, iter_steps, read_jsonl  # noqa: E402
from ygorl.train.bc import action_key, line_steps  # noqa: E402
from ygorl.train.lines import LineLibrary, LinePlayer, LineScript, LineStep, encoded_action_key  # noqa: E402
from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer, line_stats  # noqa: E402

HERE = Path(__file__).parent
DEMOS = HERE / "data" / "line_demos.jsonl"
PAIR = (str(HERE / "decks" / "snake_eye.ydk"), str(HERE / "decks" / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}
PASSIVE = [ACTION_KINDS.index(k) + 1 for k in PASSIVE_KINDS]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


@pytest.fixture(scope="module")
def library(db, vocab):
    lib = LineLibrary.load([DEMOS], vocab, cards=db)
    assert not lib.problems and len(lib) == 3
    return lib


def passive(obs) -> int:
    """The learner passes on everything: the first legal end-phase / pass / no / cancel / finish row, else the first."""
    legal = np.flatnonzero(obs["action_mask"])
    kinds = obs["actions"][:, 0]
    return int(next((i for k in PASSIVE for i in legal if kinds[i] == k), legal[0]))


def play_scripted(env, script, learner_deck, seed=123):
    """Drive one scripted game: the line for seat 0, passive answers for seat 1; stops when the line is over."""
    env.reset(0, script.spec(learner_deck, seed, DuelConfig(max_decisions=400)))
    player = LinePlayer(script, env.skip_forced)
    chosen = []
    while True:
        (ev,) = env.recv(1)
        if ev.result is not None:
            return player, chosen, None
        if ev.player == 0:
            row = player.act(ev.obs)
            if row is None:
                return player, chosen, int(ev.obs["globals"][3])
            chosen.append(encoded_action_key(ev.obs, row))
            env.step(0, row)
        else:
            env.step(0, passive(ev.obs))


# ------------------------------------------------------------------------------------------ lines


def test_scripts_start_from_the_record_and_key_every_step_of_the_demo_player(library, db, vocab):
    demos = {d.hand_index: d for d in read_jsonl(DEMOS)}
    for script in library.scripts:
        demo = demos[script.hand_index]
        assert script.deck.main == tuple(demo.start["decks"]["a"]["main"]) and script.rule_flags & DUEL_PSEUDO_SHUFFLE
        assert script.core_seed == tuple(demo.start["core_seed"]) and script.name == f"blue-eyes#{script.hand_index}"
        steps = [s for s in line_steps(demo, 0, cards=db) if s.player == 0]
        assert len(script.steps) == len(steps)
        # a forced step is one the C++ loop plays itself: one legal action, or only equivalent copies of one
        assert all(st.forced <= (s.n_legal >= 1) for st, s in zip(script.steps, steps))
        assert all(st.forced for st, s in zip(script.steps, steps) if s.n_legal == 1)
    spec = library[0].spec(load_ydk(PAIR[1]), 7, DuelConfig(max_decisions=50))
    assert spec.first == 0 and spec.core_seed == library[0].core_seed and not spec.config.shuffle_decks
    assert spec.config.max_decisions == 50 and sorted(spec.deck_b.main) == sorted(load_ydk(PAIR[1]).main)


@pytest.mark.parametrize("skip_forced", [True, False])
def test_a_scripted_game_reproduces_the_line_when_the_learner_passes(library, db, vocab, skip_forced):
    """The opponent seat's choices are the demonstration's steps (as action keys), up to the start of turn 2."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=skip_forced)
    learner = load_ydk(PAIR[1])
    for script in library.scripts:
        player, chosen, turn = play_scripted(env, script, learner)
        expected = [st.key for st in script.steps if not (skip_forced and st.forced)]
        assert player.status == "completed" and chosen == expected and player.played == len(expected)
        assert turn is not None and turn >= 2  # the line ends at the first decision of turn 2: the snapshot takes over
        assert player.finish() == {"id": script.id, "name": script.name, "status": "completed", "played": len(expected),
                                   "steps": len(script.steps), "broken_at": None}  # fmt: skip


def test_encoded_keys_tell_choices_apart_like_action_key(library, db, vocab):
    """The encoded keys of the demonstrated actions induce the same partition as bc.action_key (reference encoder)."""
    demo = next(d for d in read_jsonl(DEMOS) if d.hand_index == 1)
    points = list(iter_steps(demo, 0, cards=db))
    steps = [s for s in line_steps(demo, 0, vocab, cards=db, player=0) if s.player == 0 and s.obs is not None]
    ref = [action_key(*points[s.step]) for s in steps]
    enc = [encoded_action_key(s.obs, s.action) for s in steps]
    assert len(steps) > 5 and len(set(ref)) > 1
    for i in range(len(steps)):
        for j in range(len(steps)):
            assert (ref[i] == ref[j]) == (enc[i] == enc[j])
    keyed = [st.key for st in library[0].steps]
    assert all(k in keyed for k in enc)


# ------------------------------------------------------------------------------------------ collector


def tiny_model(vocab):
    torch.manual_seed(0)
    return ActorCritic(NetConfig(vocab_size=len(vocab), **TINY), privileged=True, privileged_dim=8, critic_hidden=16)


def scripted_collector(db, vocab, scripts, *, steps=12, max_decisions=60, num_envs=2):
    """Every game is a scripted pool game on the next script (cycling), the tiny model on both the learner seat and
    the snapshot that takes the opponent seat over."""
    model = tiny_model(vocab)
    pool = SnapshotPool(2)
    sid = pool.add(model, update=0)
    learner = load_ydk(PAIR[1])
    it = iter(range(10**6))

    def next_game():
        k = next(it)
        script = scripts[k % len(scripts)]
        spec = script.spec(learner, 100 + k, DuelConfig(max_decisions=max_decisions))
        return Assignment(spec, sid, 1, {"line": script.id}, script)

    env = EncodedVecEnv(num_envs, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    return RolloutCollector(env, model, next_game, steps, opponents=pool.get, seed=5)


def test_a_broken_line_falls_back_to_the_snapshot_and_the_game_finishes(library, db, vocab):
    bogus = replace(library[1], steps=(LineStep(("no such", "key"), False),) + library[1].steps[1:])
    col = scripted_collector(db, vocab, [bogus], steps=40, max_decisions=30)
    ro = col.collect()
    assert ro.games and all(g.line is not None for g in ro.games)
    for g in ro.games:
        assert g.line["status"] == "broken" and g.line["broken_at"] == 0 and g.line["played"] == 0
        assert g.result["decisions"] > g.learner_rows > 0  # the snapshot played the opponent seat after the break
    assert ro.opponent_decisions > 0 and (ro.players == 1).all()
    stats = line_stats(ro.games)
    assert stats["line_games"] == len(ro.games) == stats["line_broken"] and stats["line_broken_step"] == 0
    assert stats["line_completed"] == stats["line_unfinished"] == 0


def test_learner_rows_are_tagged_with_the_line_and_their_chain_window_count(library, db, vocab):
    col = scripted_collector(db, vocab, library.scripts, steps=16, max_decisions=120)
    ro = col.collect()
    T, B = ro.shape
    assert ro.line_id.shape == ro.line_window.shape == (T, B)
    ids = {g.line["id"] for g in ro.games} | {a.line.id for a in (s.assignment for s in col._slots)}
    assert set(ro.line_id.unique().tolist()) <= ids and (ro.line_id >= 0).all()
    msg = ro.obs["globals"].reshape(T, B, -1)[:, :, 18]
    assert ((ro.line_window > 0) == (msg == MSG_SELECT_CHAIN)).all()
    assert (ro.line_window > 0).any(), "the lines leave the learner some chain windows"
    # the effect answered and its occurrence: on chain windows with a real choice only, counted per game and effect
    choice = (ro.action_mask.sum(-1) > 1) & (msg == MSG_SELECT_CHAIN)
    assert ((ro.line_occurrence > 0) == choice).all() and ((ro.line_effect >= 0) == choice).all()
    for b in range(B):
        counts = {}
        for t in range(T):
            if choice[t, b]:
                e = int(ro.line_effect[t, b])
                counts[e] = counts.get(e, 0) + 1
                assert int(ro.line_occurrence[t, b]) == counts[e]
            if ro.dones[t, b]:
                counts = {}
    for b in range(B):  # 1, 2, 3 ... within a game, restarting with the next game
        expected = 0
        for t in range(T):
            if msg[t, b] == MSG_SELECT_CHAIN:
                expected += 1
                assert int(ro.line_window[t, b]) == expected
            if ro.dones[t, b]:
                expected = 0
    # every scripted game either completed its line or broke it on the learner's interruption, never "playing" at
    # the end unless cut by the decision limit
    for g in ro.games:
        assert g.line["status"] in ("completed", "broken") or g.truncated


def test_a_selfplay_rollout_tags_nothing(db, vocab):
    model = tiny_model(vocab)
    schedule = SelfPlaySchedule(DeckPool([load_ydk(p) for p in PAIR], "cross"), SnapshotPool(2),
                                DuelConfig(max_decisions=30), selfplay_fraction=1.0, seed=5)  # fmt: skip
    env = EncodedVecEnv(2, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    ro = RolloutCollector(env, model, schedule, 8, opponents=schedule.opponent, seed=5).collect()
    assert (ro.line_id == -1).all() and all(g.line is None for g in ro.games)
    assert ((ro.line_window > 0) == (ro.obs["globals"].reshape(8, 2, -1)[:, :, 18] == MSG_SELECT_CHAIN)).all()
    assert ((ro.line_occurrence > 0) <= (ro.line_window > 0)).all() and (
        (ro.line_effect >= 0) == (ro.line_occurrence > 0)
    ).all()


# ------------------------------------------------------------------------------------------ schedule


def test_the_schedule_deals_scripted_pairs_only_with_a_share_and_a_pool(library, vocab):
    decks = [load_ydk(p) for p in PAIR]
    model = tiny_model(vocab)

    def deals(**kw):
        pool = SnapshotPool(2)
        pool.add(model, update=0)
        schedule = SelfPlaySchedule(DeckPool(decks, "cross"), pool, selfplay_fraction=0.0, seed=5, **kw)
        return [schedule() for _ in range(40)]

    plain = deals()
    assert all(a.line is None for a in plain)
    same = deals(lines=library, line_share=0.0)  # no share: no extra random draw, the same deals
    assert [(a.spec.seed, a.spec.first, a.learner_seat, dict(a.info)) for a in same] == [
        (a.spec.seed, a.spec.first, a.learner_seat, dict(a.info)) for a in plain
    ]
    scripted = deals(lines=library, line_share=1.0)
    assert all(a.line is not None and a.opponent is not None for a in scripted)
    for a in scripted:
        assert a.learner_seat == 1 and a.spec.first == 0 and a.spec.deck_a is a.line.deck
        assert a.spec.core_seed == a.line.core_seed and not a.spec.config.shuffle_decks
        assert (
            a.info["line"] == a.line.id
            and a.info["deck_a"] == "blue-eyes"
            and a.info["learner_deck"] == a.info["deck_b"]
        )
    assert {a.line.id for a in scripted} == {0, 1, 2}
    with pytest.raises(ValueError):
        SelfPlaySchedule(DeckPool(decks, "cross"), SnapshotPool(2), lines=library, line_share=1.5)


def test_the_trainer_loads_the_library_and_logs_line_stats(tmp_path):
    cfg = TrainConfig(decks=PAIR, num_envs=2, env_threads=1, steps=24, event_length=16, net=TINY, privileged_dim=8,
                      critic_hidden=16, max_decisions=20, eval_every=0, checkpoint_every=0, snapshot_every=0,
                      torch_threads=1, collect_threads=1, seed=4, selfplay_fraction=0.0,
                      pin_opponents=(), line_demos=(str(DEMOS),), line_share=1.0)  # fmt: skip
    logs = []
    trainer = Trainer(cfg, tmp_path, log=logs.append)
    assert len(trainer.lines) == 3 and any("scripted lines: 3" in m for m in logs)
    trainer.pool.add(trainer.model, update=0)  # an empty pool deals self-play; scripted games need a snapshot
    rec = trainer.step()
    assert rec["line_games"] >= 2 and "line_completed" in rec and rec["games"] == rec["line_games"]
    assert any("lines " in m and "completed" in m for m in logs[-1:])
    resumed = TrainConfig.from_dict(cfg.to_dict())
    assert resumed.line_demos == cfg.line_demos and resumed.line_share == 1.0
    with pytest.raises(ValueError):
        TrainConfig(decks=PAIR, line_share=0.5)


def test_library_leaves_out_records_bound_to_another_environment(db, vocab, tmp_path):
    import json

    rows = [json.loads(line) for line in DEMOS.read_text().splitlines()]
    rows[0]["environment"] = {"version": "md-1999-01", "fingerprint": "x"}
    path = tmp_path / "demos.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lib = LineLibrary.load([path], vocab, cards=db)
    assert len(lib) == 2 and len(lib.problems) == 1 and "md-1999-01" in lib.problems[0]
    assert [s.id for s in lib.scripts] == [0, 1] and isinstance(lib[0], LineScript)


def test_library_sample_can_be_restricted_to_dealable_ids():
    """LineLibrary.dealable (TrainConfig.line_labelled_only) limits sampling to the given ids; empty = error."""
    import numpy as np

    from ygorl.train.lines import LineLibrary, LineScript, LineStep
    from ygorl.cards.ydk import Deck

    scripts = [
        LineScript(i, Deck((1,), (), (), f"d{i}"), i, (0, 0, 0, 0), 0, {}, (LineStep((1,), False),)) for i in range(5)
    ]
    lib = LineLibrary(scripts)
    rng = np.random.default_rng(0)
    assert {lib.sample(rng).id for _ in range(50)} == set(range(5))
    lib.dealable = [1, 3]
    assert {lib.sample(rng).id for _ in range(50)} == {1, 3}
    lib.dealable = []
    with pytest.raises(ValueError, match="dealable"):
        lib.sample(rng)
