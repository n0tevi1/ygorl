"""Solver hand-trap labels on the learner's chain windows (ygorl.train.line_labels; docs/training.md §8.3).

``tests/data/line_demos_ash.jsonl`` holds two solved plain records of ``out/claude-scale/handtrap/full_fire8.jsonl``
whose lines give the responder Ash windows (branded hand 41: 3 windows, sky-striker hand 41: 2; the environment
binding removed), ``tests/data/line_labels_graded.jsonl`` their graded labels (``handtrap_labels_graded.py``):
sky-striker's first window is a bait (value 0 of 3), its second the starter; branded's first is a bait, the other
two take everything.
"""

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import Deck, load_ydk  # noqa: E402
from ygorl.engine.constants import MSG_SELECT_CHAIN  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.encoding import ACTION_KINDS  # noqa: E402
from ygorl.nets import NetConfig  # noqa: E402
from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.solver.batch import PASSIVE_OPPONENT  # noqa: E402
from ygorl.solver.demo import PASSIVE_KINDS, read_jsonl  # noqa: E402
from ygorl.train.line_labels import (  # noqa: E402
    FIRE,
    LineLabels,
    build_window_table,
    default_cache,
    read_graded,
    window_identities,
)
from ygorl.train.lines import LineLibrary, LinePlayer, chain_effect  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner  # noqa: E402
from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.selfplay import SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

HERE = Path(__file__).parent
DEMOS = HERE / "data" / "line_demos_ash.jsonl"
GRADED = HERE / "data" / "line_labels_graded.jsonl"
KASHTIRA = str(HERE / "decks" / "kashtira.ydk")
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}
PASSIVE = [ACTION_KINDS.index(k) + 1 for k in PASSIVE_KINDS]
# the solver's passive responder holding Ash: the deck the labels were graded against, in load order (Ash on top)
RESPONDER = Deck(PASSIVE_OPPONENT.main + (FIRE,), (), (), "responder")


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
    assert not lib.problems and [s.name for s in lib.scripts] == ["branded#41", "sky-striker#41"]
    return lib


@pytest.fixture
def graded(tmp_path):
    """A private copy of the graded file: the window cache is written next to it."""
    path = tmp_path / "graded.jsonl"
    shutil.copyfile(GRADED, path)
    return path


def passive(obs) -> int:
    legal = np.flatnonzero(obs["action_mask"])
    kinds = obs["actions"][:, 0]
    return int(next((i for k in PASSIVE for i in legal if kinds[i] == k), legal[0]))


def responder_spec(script, seed=1, max_decisions=400):
    """The scripted game against :data:`RESPONDER` in load order (``spec`` would shuffle the learner's deck)."""
    return replace(script.spec(RESPONDER, seed, DuelConfig(max_decisions=max_decisions)), deck_b=RESPONDER)


def tiny_model(vocab):
    torch.manual_seed(0)
    return ActorCritic(NetConfig(vocab_size=len(vocab), **TINY), privileged=True, privileged_dim=8, critic_hidden=16)


# ------------------------------------------------------------------------------------------ identity


def test_the_environment_decodes_the_window_identities_of_the_replay(library, db, vocab):
    """The C++ encoder's events at the learner's chain prompts name the same (effect, occurrence) as the Python
    replay the labels are built with, when the learner is the passive responder holding Ash."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    demos = {d.deck["name"]: d for d in read_jsonl(DEMOS)}
    ash = vocab.index(FIRE)
    for script in library.scripts:
        env.reset(0, responder_spec(script))
        player = LinePlayer(script, env.skip_forced)
        seen, counts = [], {}
        while True:
            (ev,) = env.recv(1)
            if ev.result is not None:
                break
            if ev.player == 0:
                row = player.act(ev.obs)
                if row is None:
                    break
                env.step(0, row)
                continue
            obs = ev.obs
            if int(obs["globals"][18]) == MSG_SELECT_CHAIN and int(np.count_nonzero(obs["action_mask"])) > 1:
                effect = chain_effect(obs)
                counts[effect] = counts.get(effect, 0) + 1
                assert effect >= CardVocab.FIRST_INDEX  # every window answers a chained effect
                assert ((obs["actions"][:, 2] == ash) & (obs["action_mask"] != 0)).any()
                seen.append((vocab.password(effect), counts[effect]))
            env.step(0, passive(obs))
        expected = window_identities(demos[script.deck.name], cards=db, event_length=16)
        assert seen == expected and len(seen) == read_graded(GRADED)[(script.deck.name, script.hand_index)]["windows"]


def test_the_window_table_is_built_once_and_cached_next_to_the_graded_file(library, db, vocab, graded):
    demos = {d.deck["name"]: d for d in read_jsonl(DEMOS)}
    labels = read_graded(graded)
    logs = []
    table = build_window_table(graded, [DEMOS], cards=db, event_length=16, log=logs.append)
    assert set(table) == {"branded#41", "sky-striker#41"} and "2 replayed" in logs[-1]
    for name, entry in table.items():
        deck, hand = name.split("#")
        assert entry["windows"] == len(entry["ids"]) == labels[(deck, int(hand))]["windows"]
        for password, occurrence in entry["ids"]:
            assert password in demos[deck].deck["main"] + demos[deck].deck["extra"] and occurrence >= 1
    cache = default_cache(graded)
    assert cache.exists() and json.loads(cache.read_text())["event_length"] == 16
    again = build_window_table(graded, [DEMOS], cards=db, event_length=16, log=logs.append)
    assert again == table and "0 replayed" in logs[-1]
    build_window_table(graded, [DEMOS], cards=db, event_length=32, log=logs.append)  # another window: rebuilt
    assert "2 replayed" in logs[-1]

    joined = LineLabels.load(graded, library, vocab, cards=db, event_length=16, log=logs.append)
    assert len(joined) == 5 and joined.hands == 2 and joined.fire_index == vocab.index(FIRE)
    by_line = {}
    for (line, effect, occurrence), label in joined.table.items():
        assert occurrence == 1 and effect >= CardVocab.FIRST_INDEX
        by_line.setdefault(line, []).append(label)
    sky = next(s.id for s in library.scripts if s.deck.name == "sky-striker")
    assert sorted((lb.u, lb.later) for lb in by_line[sky]) == [(0.0, 1.0), (1.0, 0.0)]
    assert sorted(lb.fire_bonus for lb in by_line[sky]) == [-1.0, 1.0]  # hold at the bait, fire at the starter
    assert sorted(lb.fire_bonus for lb in by_line[1 - sky]) == [-1.0, 0.0, 1.0]


# ------------------------------------------------------------------------------------------ PPO bonus


def scripted_collector(db, vocab, scripts, *, steps=24, max_decisions=80, num_envs=2):
    model = tiny_model(vocab)
    pool = SnapshotPool(2)
    sid = pool.add(model, update=0)
    it = iter(range(10**6))

    def next_game():
        k = next(it)
        script = scripts[k % len(scripts)]
        return Assignment(responder_spec(script, 100 + k, max_decisions), sid, 1, {"line": script.id}, script)

    env = EncodedVecEnv(num_envs, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    return model, RolloutCollector(env, model, next_game, steps, opponents=pool.get, seed=5)


def test_the_bonus_lands_on_labelled_ash_windows_with_the_sign_of_the_taken_action(library, db, vocab, graded):
    model, col = scripted_collector(db, vocab, library.scripts)
    ro = col.collect()
    T, B = ro.shape
    assert ro.line_effect.shape == ro.line_occurrence.shape == (T, B)
    assert ((ro.line_occurrence > 0) == (ro.line_effect >= 0)).all() and (
        (ro.line_occurrence > 0) <= (ro.line_window > 0)
    ).all()
    labels = LineLabels.load(graded, library, vocab, cards=db, event_length=16)
    bonus, stats = labels.bonus(ro)
    assert stats["line_labelled_rows"] > 0 and (bonus != 0).sum() <= stats["line_labelled_rows"]
    assert stats["line_fire_rows"] <= stats["line_labelled_rows"]
    codes = ro.obs["actions"][..., 2].reshape(T, B, -1)
    msg = ro.obs["globals"].reshape(T, B, -1)[:, :, 18]
    labelled = 0
    for t in range(T):
        for b in range(B):
            key = (int(ro.line_id[t, b]), int(ro.line_effect[t, b]), int(ro.line_occurrence[t, b]))
            label = labels.table.get(key) if key[2] > 0 else None
            ash = (codes[t, b] == labels.fire_index) & ro.action_mask[t, b]
            if label is None or not ash.any():
                assert float(bonus[t, b]) == 0.0
                continue
            labelled += 1
            assert int(msg[t, b]) == MSG_SELECT_CHAIN
            fired = bool(ash[int(ro.actions[t, b])])
            assert float(bonus[t, b]) == (label.fire_bonus if fired else -label.fire_bonus)
    assert labelled == stats["line_labelled_rows"]
    assert stats["line_bonus_mean"] == pytest.approx(float(bonus.sum()) / labelled)

    # the bonus is added to the raw advantages: with no normalization the difference is exactly coef * bonus ...
    plain = PPOLearner(model, PPOConfig(adv_norm="none")).targets(ro).advantages
    bumped = PPOLearner(model, PPOConfig(adv_norm="none", line_bonus_coef=0.5), line_labels=labels).targets(ro)
    assert torch.equal(bumped.advantages - plain, 0.5 * bonus)
    # ... and with the labels present but the coefficient 0 the targets are the estimator's, bit for bit
    off = PPOLearner(model, PPOConfig(), line_labels=labels).targets(ro).advantages
    assert torch.equal(off, PPOLearner(model, PPOConfig()).targets(ro).advantages)
    learner = PPOLearner(model, PPOConfig(line_bonus_coef=0.5, epochs=1), line_labels=labels)
    stats = learner.update(ro)
    assert stats["line_labelled_rows"] == labelled and "line_bonus_mean" in stats and "line_fire_rows" in stats
    assert "line_labelled_rows" not in PPOLearner(model, PPOConfig(epochs=1)).update(ro)


def test_the_trainer_joins_the_labels_and_logs_the_labelled_windows(tmp_path, graded):
    ash_deck = tmp_path / "ash.ydk"  # the learner always holds Ash: every window of a line is a labelled row
    ash_deck.write_text("#main\n" + f"{FIRE}\n" * 40 + "#extra\n!side\n")
    assert load_ydk(ash_deck).main == (FIRE,) * 40
    cfg = TrainConfig(decks=(str(ash_deck), KASHTIRA), num_envs=2, env_threads=1, steps=24, event_length=16, net=TINY,
                      privileged_dim=8, critic_hidden=16, max_decisions=60, eval_every=0, checkpoint_every=0,
                      snapshot_every=0, torch_threads=1, collect_threads=1, seed=4, selfplay_fraction=0.0,
                      line_demos=(str(DEMOS),), line_share=1.0, line_labels=str(graded),
                      ppo=PPOConfig(line_bonus_coef=0.5, line_ce_coef=0.5, epochs=1))  # fmt: skip
    logs = []
    trainer = Trainer(cfg, tmp_path / "run", log=logs.append)
    assert len(trainer.line_labels) == 5 and trainer.learner.line_labels is trainer.line_labels
    assert default_cache(graded).exists() and any("5 windows of 2 hands" in m for m in logs)
    trainer.pool.add(trainer.model, update=0)
    rec = trainer.step()
    assert rec["line_games"] >= 1 and rec["line_labelled_rows"] >= 1 and "labelled windows" in logs[-1]
    assert rec["line_ce_rows"] >= 1 and "line_ce_loss" in rec and "line CE rows" in logs[-1]
    assert TrainConfig.from_dict(cfg.to_dict()).line_labels == str(graded)
    with pytest.raises(ValueError):
        TrainConfig(decks=(KASHTIRA,), line_labels=str(graded))
    with pytest.raises(ValueError):
        PPOConfig(line_bonus_coef=-1.0)


# ------------------------------------------------------------------------------------------ PPO supervised term


@pytest.fixture(scope="module")
def labelled(library, db, vocab, tmp_path_factory):
    """A scripted rollout with labelled rows, its labels and the model that collected it."""
    path = tmp_path_factory.mktemp("ce") / "graded.jsonl"
    shutil.copyfile(GRADED, path)
    model, col = scripted_collector(db, vocab, library.scripts)
    ro = col.collect()
    labels = LineLabels.load(path, library, vocab, cards=db, event_length=16)
    return model, ro, labels


class OneRow:
    """``labels`` with only the first CE target of a rollout kept."""

    def __init__(self, labels):
        self.labels = labels

    def ce_targets(self, ro):
        target, weight, stats = self.labels.ce_targets(ro)
        first = (target >= 0).flatten().nonzero()[0, 0]
        keep = torch.zeros(target.numel(), dtype=torch.bool)
        keep[first] = True
        keep = keep.reshape(target.shape)
        return torch.where(keep, target, -1), torch.where(keep, weight, 0.0), {**stats, "line_ce_rows": 1}


def ce_learner(model, labels=None, **kw):
    base = {"epochs": 1, "minibatch_size": 4096, "target_kl": None}
    return PPOLearner(model, PPOConfig(**{**base, **kw}), line_labels=labels)


def logp_rows(model, ro):
    from ygorl.nets.policy import trim_padding

    model.eval()
    obs = trim_padding(ro.obs)
    return torch.log_softmax(PPOLearner(model, PPOConfig())._score(model, obs, ro.actions.numel()), -1)


def test_the_ce_targets_hold_at_the_bait_and_fire_at_the_starter(library, labelled):
    _, ro, labels = labelled
    T, B = ro.shape
    target, weight, stats = labels.ce_targets(ro)
    _, bonus_stats = labels.bonus(ro)
    assert target.shape == weight.shape == (T, B) and target.dtype == torch.long
    assert stats["line_ce_rows"] == int((target >= 0).sum()) > 0
    assert stats["line_ce_rows"] <= bonus_stats["line_labelled_rows"]
    assert ((target < 0) <= (weight == 0)).all()
    kinds = ro.obs["actions"][..., 0].reshape(T, B, -1)
    codes = ro.obs["actions"][..., 2].reshape(T, B, -1)
    sky = next(s.id for s in library.scripts if s.deck.name == "sky-striker")
    seen, fires = set(), 0
    for t, b in (target >= 0).nonzero().tolist():
        label = labels.table[(int(ro.line_id[t, b]), int(ro.line_effect[t, b]), int(ro.line_occurrence[t, b]))]
        k = int(target[t, b])
        assert bool(ro.action_mask[t, b, k]) and float(weight[t, b]) == abs(label.fire_bonus)
        if label.fire_bonus > 0:  # the starter: fire here
            assert int(codes[t, b, k]) == labels.fire_index
            fires += 1
        else:  # a bait (a later window hurts more) or a tie: hold
            assert int(kinds[t, b, k]) in PASSIVE and int(codes[t, b, k]) != labels.fire_index
            assert (float(weight[t, b]) > 0) == (label.later > label.u)
        if int(ro.line_id[t, b]) == sky:
            seen.add((label.u, label.later))
    assert fires == stats["line_ce_fire_targets"]
    # sky-striker's bait (u 0, a later window takes everything) holds with weight 1, its starter fires with weight 1
    for t, b in (target >= 0).nonzero().tolist():
        if int(ro.line_id[t, b]) == sky:
            label = labels.table[(sky, int(ro.line_effect[t, b]), int(ro.line_occurrence[t, b]))]
            assert float(weight[t, b]) == 1.0 and (int(codes[t, b, int(target[t, b])]) == labels.fire_index) == (
                label.u == 1.0
            )
    assert seen  # the rollout reached a sky-striker window


def test_the_ce_term_is_zero_without_labelled_rows_and_off_by_default(labelled, vocab):
    model, ro, labels = labelled
    logp = torch.log_softmax(torch.randn(5, 4), -1)
    none = PPOLearner.line_ce(logp, torch.full((5,), -1), torch.ones(5))
    assert float(none) == 0.0 and not torch.isnan(none)
    one = PPOLearner.line_ce(logp, torch.tensor([-1, 2, -1, -1, -1]), torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0]))
    assert float(one) == pytest.approx(-0.5 * float(logp[1, 2]))

    def run(labels, **kw):
        m = copy.deepcopy(model)
        learner = ce_learner(m, labels, **kw)
        torch.manual_seed(3)
        stats = learner.update(ro)
        return stats, [p.detach().clone() for p in m.parameters()]

    base_stats, base_params = run(None)
    for lbl, kw in ((labels, {}), (LineLabels({}, vocab.index(FIRE)), {"line_ce_coef": 1.0})):
        stats, params = run(lbl, **kw)
        assert all(torch.equal(a, b) for a, b in zip(base_params, params))
        for key in ("loss", "policy_loss", "entropy", "kl_ref", "q_loss", "v_loss", "grad_norm"):
            assert stats[key] == base_stats[key], key
        if kw:
            assert stats["line_ce_rows"] == 0 and stats["line_ce_loss"] == 0.0
        else:
            assert "line_ce_rows" not in stats and "line_ce_loss" not in stats


def test_one_step_of_the_ce_term_raises_the_target_probability(labelled):
    model, ro, labels = labelled
    one = OneRow(labels)
    target, _, _ = one.ce_targets(ro)
    r = int((target >= 0).flatten().nonzero()[0, 0])
    k = int(target.flatten()[r])
    T, B = ro.shape
    before = float(logp_rows(model, ro)[r, k])
    after = {}
    for coef in (0.0, 5.0):
        m = copy.deepcopy(model)
        learner = ce_learner(m, one, line_ce_coef=coef, lr=1e-2, entropy_coef=0.0, kl_ref_coef=0.0)
        torch.manual_seed(3)
        stats = learner.update(ro)
        after[coef] = float(logp_rows(m, ro)[r, k])
        if coef:
            assert stats["line_ce_rows"] == 1 and stats["line_ce_loss"] > 0
            assert stats["line_ce_fire_targets"] >= 0
    assert after[5.0] > after[0.0] and after[5.0] > before
    with pytest.raises(ValueError):
        PPOConfig(line_ce_coef=-1.0)


# ------------------------------------------------------------------------------------------ first seat


def first_seat_spec(script, learner=RESPONDER, seed=1, max_decisions=400):
    """The first-seat scripted game: ``learner`` in load order on seat 0, moving first; the line on seat 1."""
    spec = script.spec(learner, seed, DuelConfig(max_decisions=max_decisions), learner_seat=0)
    return replace(spec, deck_a=learner)


def drive_first_seat(env, spec, script, *, fire_at=None):
    """Seat 0 passes (or fires Ash at its ``fire_at``-th Ash window of turn 2), seat 1 passes until its turn, plays the
    line, then answers with the first legal row (the snapshot's stand-in). Returns the line player, the identities of
    the learner's turn-2 windows while the line was played, and the turns reached."""
    env.reset(0, spec)
    player = LinePlayer(script, env.skip_forced, first_seat=True)
    ash = env.vocab.index(FIRE) if hasattr(env, "vocab") else None
    seen, counts, ash_windows, after = [], {}, 0, 0
    while True:
        (ev,) = env.recv(1)
        if ev.result is not None:
            return player, seen, ev.result
        obs = ev.obs
        g = obs["globals"]
        if ev.player == 1:
            row = player.act(obs)
            if row is None:
                if player.status != "playing":
                    after += 1
                    if after > 30:
                        return player, seen, None
                env.step(0, passive(obs) if player.status == "playing" else int(np.flatnonzero(obs["action_mask"])[0]))
                continue
            env.step(0, row)
            continue
        action = passive(obs)
        if int(g[18]) == MSG_SELECT_CHAIN and int(np.count_nonzero(obs["action_mask"])) > 1 and int(g[3]) == 2:
            effect = chain_effect(obs)
            counts[effect] = counts.get(effect, 0) + 1
            if player.status == "playing":
                seen.append((effect, counts[effect]))
            fire = np.flatnonzero((obs["actions"][:, 2] == ash) & (obs["action_mask"] != 0))
            if len(fire):
                ash_windows += 1
                if fire_at is not None and ash_windows == fire_at:
                    action = int(fire[0])
        env.step(0, action)


def test_first_seat_lines_of_the_ash_records_and_their_window_identities(library, db, vocab):
    """With the responder deck moving first and passing, the ash records' lines are played on turn 2 and the learner's
    turn-2 windows have the identities the labels were built with, and both lines complete. Branded's step 28 picks a
    graveyard card whose sequence differs from the record's in either seat: the second seat (exact keys) breaks
    there, the first seat matches graveyard cards without their sequence."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=True)
    env.vocab = vocab
    demos = {d.deck["name"]: d for d in read_jsonl(DEMOS)}
    for script in library.scripts:
        player, seen, _ = drive_first_seat(env, first_seat_spec(script), script)
        expected = window_identities(demos[script.deck.name], cards=db, event_length=16)
        assert [(vocab.password(e), k) for e, k in seen] == expected
        assert player.status == "completed" and player.turn == 2
        if script.deck.name == "branded":
            second = LinePlayer(script, True)
            env.reset(0, responder_spec(script))
            while second.status == "playing":
                (ev,) = env.recv(1)
                row = second.act(ev.obs) if ev.player == 0 else passive(ev.obs)
                if row is not None:
                    env.step(0, row)
            assert second.status == "broken" and second.broken_at == 28


def test_ash_at_the_first_window_breaks_a_first_seat_line_and_the_seat_plays_on(library, db, vocab):
    """A learner of 40 Ash moving first fires at its first turn-2 Ash window: branded's line breaks there (the seat is
    played on by the stand-in, 30 more decisions); sky-striker's first window is the bait of its labels (value 0: the
    combo recovers) and its line still completes, while Ash at its second window (the starter) breaks it."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=True)
    env.vocab = vocab
    ashes = Deck((FIRE,) * 40, (), (), "ash")
    by_name = {s.deck.name: s for s in library.scripts}
    outcomes = {}
    for name, fire_at in (("branded", 1), ("sky-striker", 1), ("sky-striker", 2)):
        script = by_name[name]
        spec = script.spec(ashes, 3, DuelConfig(max_decisions=400), learner_seat=0)
        player, seen, result = drive_first_seat(env, spec, script, fire_at=fire_at)
        assert player.turn == 2 and len(seen) >= fire_at
        outcomes[name, fire_at] = (player.status, player.broken_at)
        if player.status == "broken":
            assert result is None  # the seat kept being asked after the break: the game played on
    assert outcomes == {("branded", 1): ("broken", 8), ("sky-striker", 1): ("completed", None),
                        ("sky-striker", 2): ("broken", 13)}  # fmt: skip


def test_the_label_join_holds_in_the_first_seat(library, db, vocab, graded):
    model = tiny_model(vocab)
    pool = SnapshotPool(2)
    sid = pool.add(model, update=0)
    it = iter(range(10**6))
    scripts = library.scripts

    def next_game():
        k = next(it)
        script = scripts[k % len(scripts)]
        # 40 Ash: the learner still holds one on turn 2 whatever it did on turn 1 (it may summon one)
        spec = first_seat_spec(script, Deck((FIRE,) * 40, (), (), "ash"), seed=100 + k, max_decisions=120)
        return Assignment(spec, sid, 0, {"line": script.id}, script)

    env = EncodedVecEnv(2, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    ro = RolloutCollector(env, model, next_game, 48, opponents=pool.get, seed=5).collect()
    T, B = ro.shape
    assert (ro.line_seat == 0).all() and (ro.players == 0).all()
    labels = LineLabels.load(graded, library, vocab, cards=db, event_length=16)
    bonus, stats = labels.bonus(ro)
    assert stats["line_labelled_rows"] >= 1 and (bonus != 0).any()
    codes = ro.obs["actions"][..., 2].reshape(T, B, -1)
    for t, b in (bonus != 0).nonzero().tolist():
        assert int(ro.obs["globals"].reshape(T, B, -1)[t, b, 3]) == 2  # the opponent's turn 2, where the line is
        label = labels.table[(int(ro.line_id[t, b]), int(ro.line_effect[t, b]), int(ro.line_occurrence[t, b]))]
        fired = bool(((codes[t, b] == labels.fire_index) & ro.action_mask[t, b])[int(ro.actions[t, b])])
        assert float(bonus[t, b]) == (label.fire_bonus if fired else -label.fire_bonus)
