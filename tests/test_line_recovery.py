"""Recovery lines of scripted-line opponents (ygorl.train.lines; docs/training.md §8.3 「恢复线」).

``tests/data/line_recovery.jsonl`` holds the plain and the Ash Blossom ``--fire`` records of three hands of
``out/claude-scale/handtrap/full_fire8.jsonl`` (the environment binding removed so the test does not depend on the
environment's fingerprint): blue-eyes hand 12 (one Ash window, on Maxx "C", the combo recovers: 3 fire lines), branded
hand 32 (two windows, both recover: 6 fire lines; at the first, Ash on a deck search, the recorded line itself breaks)
and branded hand 35 (two windows: the first recovers, 4 fire lines of which 2 hide a step behind the mask; Ash at the
second breaks the board, no fire line).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import Deck  # noqa: E402
from ygorl.engine.constants import MSG_SELECT_CHAIN  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.encoding import ACTION_KINDS, CARD_INDEX, CONTROLLER, LOCATION  # noqa: E402
from ygorl.nets import NetConfig  # noqa: E402
from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.solver.demo import PASSIVE_KINDS, read_jsonl  # noqa: E402
from ygorl.train.line_labels import FIRE, window_identities  # noqa: E402
from ygorl.train.lines import CHAIN_KIND, LineLibrary, LinePlayer, chain_effect, encoded_action_key  # noqa: E402
from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.selfplay import SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer, line_stats  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "line_recovery.jsonl"
PAIR = (str(HERE / "decks" / "snake_eye.ydk"), str(HERE / "decks" / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}
PASSIVE = [ACTION_KINDS.index(k) + 1 for k in PASSIVE_KINDS]
ASH_DECK = Deck((FIRE,) * 40, (), (), "ash")  # the learner holds Ash at every window


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
def plain(db, vocab):
    lib = LineLibrary.load([DATA], vocab, cards=db)
    assert not lib.problems and [s.name for s in lib.scripts] == ["blue-eyes#12", "branded#32", "branded#35"]
    return lib


@pytest.fixture(scope="module")
def library(db, vocab):
    return LineLibrary.load([DATA], vocab, cards=db, recovery=[DATA])


@pytest.fixture(scope="module")
def windows(db):
    """Window identities (effect password, occurrence) of every line, as the labels identify them."""
    demos = {f"{d.deck['name']}#{d.hand_index}": d for d in read_jsonl(DATA) if d.variant == "plain"}
    return {name: window_identities(demo, cards=db) for name, demo in demos.items()}


def passive(obs) -> int:
    legal = np.flatnonzero(obs["action_mask"])
    kinds = obs["actions"][:, 0]
    return int(next((i for k in PASSIVE for i in legal if kinds[i] == k), legal[0]))


def field(obs) -> list[int]:
    """Card vocab indices on the decider's own monster and spell / trap zones."""
    c = obs["cards"]
    return sorted(int(r[CARD_INDEX]) for r in c if r[CONTROLLER] == 0 and r[LOCATION] in (3, 4) and r[CARD_INDEX])


def play(env, script, vocab, fire_at=None, *, first_seat=False, seed=123):
    """One scripted game on ``script``: the line seat plays the line (passing once it is over), the learner (40 Ash)
    passes everywhere except at the window ``fire_at`` = (effect password, occurrence), where it activates Ash and
    tells the line player, as the collector does. Returns the player, the keys it played, whether it switched, and
    the line seat's observation at which the line was over (None: the game ended first)."""
    seat = 0 if first_seat else 1
    env.reset(0, script.spec(ASH_DECK, seed, DuelConfig(max_decisions=600), learner_seat=seat))
    player = LinePlayer(script, env.skip_forced, first_seat=first_seat)
    counts, chosen, switched = {}, [], None
    ash = vocab.index(FIRE)
    while True:
        (ev,) = env.recv(1)
        if ev.result is not None:
            return player, chosen, switched, None
        obs = ev.obs
        if ev.player != seat:
            row = player.act(obs)
            if row is None:
                if player.status not in ("playing", "recovering"):
                    return player, chosen, switched, obs
                env.step(0, passive(obs))
            else:
                chosen.append(encoded_action_key(obs, row))
                env.step(0, row)
            continue
        g, a = obs["globals"], passive(obs)
        if int(g[18]) == MSG_SELECT_CHAIN and int((obs["action_mask"] != 0).sum()) > 1:
            if not first_seat or (int(g[3]) == 2 and not int(g[2])):
                effect = chain_effect(obs)
                counts[effect] = counts.get(effect, 0) + 1
                ident = (vocab.password(effect) if effect else 0, counts[effect])
                rows = [i for i in np.flatnonzero(obs["action_mask"])
                        if obs["actions"][i, 2] == ash and obs["actions"][i, 0] == CHAIN_KIND]  # fmt: skip
                if rows and ident == fire_at and switched is None:
                    a = int(rows[0])
                    switched = player.interrupt(effect, counts[effect], ash)
        env.step(0, a)


# ------------------------------------------------------------------------------------------ library

MAXX_C = 23434538


def test_fire_lines_join_their_lines_by_window_and_hand_trap(library, plain, vocab, windows):
    ash = vocab.index(FIRE)
    recovered = {"blue-eyes#12": [0], "branded#32": [0, 1], "branded#35": [0]}  # windows the combo recovers from
    for script in library.scripts:
        ids = windows[script.name]
        want = [(vocab.index(ids[k][0]), ids[k][1], ash) for k in recovered[script.name]]
        assert sorted(r.key for r in script.recovery) == sorted(want)  # the first fire line of a window is kept
        assert all(r.password == FIRE and r.steps and r.source == str(DATA) for r in script.recovery)
    assert len(windows["branded#35"]) == 2  # Ash at its second window breaks the board: no recovery line
    assert library.recovery_windows == 4
    assert library.recovery_problems and all("mask" in p for p in library.recovery_problems)
    # the lines themselves are the plain library's, and without recovery files nothing is attached
    assert [replace(s, recovery=()) for s in library.scripts] == plain.scripts
    assert all(s.recovery == () for s in plain.scripts) and plain.recovery_windows == 0


def test_a_fire_line_must_leave_its_line_only_at_the_hand_trap(db, vocab, tmp_path):
    import json

    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    for r in rows:
        if r["variant"] == "fire" and r["deck"]["name"] == "blue-eyes":
            for ln in r["lines"]:
                k = ln["players"].index(0)
                ln["actions"][k] += 1  # the combo's first step differs from the plain line's
    path = tmp_path / "fire.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lib = LineLibrary.load([DATA], vocab, cards=db, recovery=[path])
    assert not lib[0].recovery and lib[1].recovery
    assert sum("blue-eyes#12" in p for p in lib.recovery_problems) == 3


# ------------------------------------------------------------------------------------------ games


@pytest.mark.parametrize("first_seat", [False, True])
@pytest.mark.parametrize("skip_forced", [True, False])
def test_ash_at_a_recoverable_window_switches_to_the_recovery_line_and_reaches_the_board(
    library, plain, db, vocab, first_seat, skip_forced
):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=skip_forced)
    for k, script in enumerate(library.scripts):
        _, _, _, untouched = play(env, script, vocab, first_seat=first_seat)
        for rec in script.recovery:
            w = (vocab.password(rec.effect), rec.occurrence)
            player, chosen, switched, obs = play(env, script, vocab, w, first_seat=first_seat)
            assert switched and player.status == "completed" and player.recoveries == 1
            out = player.finish()
            assert out["recoveries"] == 1 and out["recovered_at"] == rec.at and out["recovery_window"] == list(rec.key)
            assert out["seat"] == (0 if first_seat else 1) and out["turn"] == (2 if first_seat else 1)
            expected = [st.key for st in script.steps[: rec.at] + rec.steps if not (skip_forced and st.forced)]
            assert chosen == expected
            # the recovery line rebuilds the board the uninterrupted line ends on (the solver's targets)
            assert obs is not None and untouched is not None and field(obs) == field(untouched)
            # without its recovery line the line plays on, as before: Ash on branded#32's deck search breaks it (the
            # snapshot would take over); after Ash on Maxx "C" the recorded line mostly still matches
            player, _, switched, _ = play(env, plain[k], vocab, w, first_seat=first_seat)
            assert switched is False and "recoveries" not in player.finish()
            if script.name == "branded#32" and w[0] != MAXX_C:
                assert player.status == "broken"


@pytest.mark.parametrize("first_seat", [False, True])
def test_ash_at_a_window_without_recovery_breaks_the_line_as_before(library, plain, db, vocab, windows, first_seat):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=True)
    w = windows["branded#35"][1]
    player, chosen, switched, _ = play(env, library[2], vocab, w, first_seat=first_seat)
    before, chosen_before, _, _ = play(env, plain[2], vocab, w, first_seat=first_seat)
    assert switched is False and player.status == "broken" and player.recoveries == 0
    out = player.finish()
    assert out["recoveries"] == 0 and out["recovered_at"] is None and out["recovery_window"] is None
    assert {k: v for k, v in out.items() if not k.startswith("recover")} == before.finish() and chosen == chosen_before


def test_a_recovering_line_does_not_switch_twice_and_a_finished_one_not_at_all(library, vocab):
    script = library[0]
    (rec,) = script.recovery
    p = LinePlayer(script, True)
    assert not p.interrupt(rec.effect, rec.occurrence, rec.card + 1)  # another card
    assert not p.interrupt(rec.effect, rec.occurrence + 1, rec.card)  # another window
    assert p.status == "playing" and p.interrupt(*rec.key) and p.status == "recovering" and p.steps == rec.steps
    assert not p.interrupt(*rec.key) and p.recoveries == 1
    q = LinePlayer(script, True)
    q.status = "broken"
    assert not q.interrupt(*rec.key) and q.status == "broken"


# ------------------------------------------------------------------------------------------ collector


class FiresAsh(ActorCritic):
    """The tiny actor-critic, except that the learner activates Ash wherever it can."""

    ash = 0

    def forward(self, obs, privileged=None):
        out = super().forward(obs, privileged)
        actions = obs["actions"][:, : out.logits.shape[1]]
        hit = (actions[..., 0] == CHAIN_KIND) & (actions[..., 2] == self.ash) & obs["action_mask"].bool()
        return out._replace(logits=torch.where(hit, out.logits + 1e3, out.logits))


def fire_collector(db, vocab, scripts, *, first_seat=False, steps=64, num_envs=2):
    torch.manual_seed(0)
    model = FiresAsh(NetConfig(vocab_size=len(vocab), **TINY), privileged=True, privileged_dim=8, critic_hidden=16)
    model.ash = vocab.index(FIRE)
    pool = SnapshotPool(2)
    sid = pool.add(model, update=0)
    seat = 0 if first_seat else 1
    it = iter(range(10**6))

    def next_game():
        k = next(it)
        script = scripts[k % len(scripts)]
        spec = script.spec(ASH_DECK, 100 + k, DuelConfig(max_decisions=60), learner_seat=seat)
        return Assignment(spec, sid, seat, {"line": script.id, "learner_seat": seat}, script)

    env = EncodedVecEnv(num_envs, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    return RolloutCollector(env, model, next_game, steps, opponents=pool.get, seed=5)


@pytest.mark.parametrize("first_seat", [False, True])
def test_the_collector_switches_a_line_the_learner_interrupts_and_records_it(library, plain, db, vocab, first_seat):
    """The learner fires Ash wherever it can; branded#32 recovers from its first window: with recovery lines the line
    seat switches to its recovery line, without them the line breaks there; the games' records count the switch. (In
    first-seat games the snapshot plays the line seat during the learner's turn 1, so not every game reaches the
    window with the line intact; a second Ash later in the recovery line can still break it.)"""
    script = library[1]
    first = next(r for r in script.recovery if vocab.password(r.effect) != MAXX_C)
    ro = fire_collector(db, vocab, [script], first_seat=first_seat).collect()
    assert ro.games and all("recoveries" in g.line for g in ro.games)
    switched = [g for g in ro.games if g.line["recoveries"]]
    assert switched if first_seat else len(switched) == len(ro.games)
    assert all(g.line["recovery_window"] == list(first.key) for g in switched)
    stats = line_stats(ro.games, recovery=True)
    assert stats["line_recovered"] == len(switched)
    # second seat: the recovery line completes; first seat: the snapshot that played the line seat on turn 1 may have
    # spent a card the recovery line uses (here Maxx "C", chained to Ash), and it breaks
    assert first_seat or stats["line_recovered_completed"] == len(switched)
    if first_seat:
        assert stats["line_first_recovered"] == len(switched)
    ro = fire_collector(db, vocab, [plain[1]], first_seat=first_seat).collect()
    assert ro.games and all("recoveries" not in g.line for g in ro.games)
    assert first_seat or all(g.line["status"] == "broken" for g in ro.games)
    assert "line_recovered" not in line_stats(ro.games)


def test_the_trainer_loads_recovery_lines_and_logs_them(tmp_path):
    cfg = TrainConfig(decks=PAIR, num_envs=2, env_threads=1, steps=24, event_length=16, net=TINY, privileged_dim=8,
                      critic_hidden=16, max_decisions=20, eval_every=0, checkpoint_every=0, snapshot_every=0,
                      torch_threads=1, collect_threads=1, seed=4, selfplay_fraction=0.0, pin_opponents=(),
                      line_demos=(str(DATA),), line_share=1.0, line_recovery=(str(DATA),))  # fmt: skip
    logs = []
    trainer = Trainer(cfg, tmp_path, log=logs.append)
    assert trainer.lines.recovery_windows == 4 and any("recovery lines: 4 windows on 3 lines" in m for m in logs)
    trainer.pool.add(trainer.model, update=0)
    rec = trainer.step()
    assert "line_recovered" in rec and "recovered" in logs[-1]
    assert TrainConfig.from_dict(cfg.to_dict()).line_recovery == (str(DATA),)
    with pytest.raises(ValueError, match="line_recovery"):
        TrainConfig(decks=PAIR, line_recovery=(str(DATA),))
