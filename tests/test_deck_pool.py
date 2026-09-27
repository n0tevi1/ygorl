"""The evolving deck pool (#108): a deck-pool manifest that the trainer re-reads while it runs; evolved decks take a
configured share of the deals, weighted toward the decks the policy pilots badly; no live evolved deck = the fixed
pool, bit for bit; the pool state survives checkpoints."""

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.selfplay import DeckPool, EvolvedDecks, SelfPlaySchedule, SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

DECK_DIR = Path(__file__).parent / "decks"
CORPUS = ("snake_eye", "kashtira")
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


def manifest(tmp_path, decks, name="pool.json"):
    """Write a manifest of ``{id: (deck file stem, status)}`` with the .ydk files next to it."""
    (tmp_path / "decks").mkdir(exist_ok=True)
    entries = []
    for did, (stem, status, *weight) in decks.items():
        shutil.copy(DECK_DIR / f"{stem}.ydk", tmp_path / "decks" / f"{did}.ydk")
        entries.append(
            {"id": did, "file": f"decks/{did}.ydk", "status": status, **({"weight": weight[0]} if weight else {})}
        )
    path = tmp_path / name
    path.write_text(json.dumps({"format": "ygorl-deck-pool", "version": 1, "decks": entries}))
    return path


def schedule(evolved=None, seed=3):
    decks = [load_ydk(DECK_DIR / f"{s}.ydk") for s in CORPUS]
    return SelfPlaySchedule(DeckPool(decks, "cross"), SnapshotPool(2), DuelConfig(max_decisions=40),
                            selfplay_fraction=1.0, seed=seed, evolved=evolved)  # fmt: skip


def deals(s, n):
    """The (deck_a, deck_b, seed, first, evolved) of the next ``n`` games."""
    out = []
    for _ in range(n):
        a = s()
        out.append((a.spec.deck_a.name, a.spec.deck_b.name, a.spec.seed, a.spec.first, a.info.get("evolved")))
    return out


def test_without_live_evolved_decks_the_deals_are_those_of_the_fixed_pool(tmp_path):
    fixed = deals(schedule(), 400)
    missing = EvolvedDecks(tmp_path / "none.json")
    missing.reload()
    assert deals(schedule(missing), 400) == fixed
    only_history = EvolvedDecks(manifest(tmp_path, {"old": ("labrynth", "history")}))
    only_history.reload()
    assert only_history.ids("history") == ["old"]
    assert deals(schedule(only_history), 400) == fixed


def test_evolved_decks_take_their_share_and_the_rest_comes_from_corpus_or_history(tmp_path):
    ev = EvolvedDecks(manifest(tmp_path, {"e1": ("labrynth", "probation"), "e2": ("purrely", "active"),
                                          "old": ("yubel", "history")}), share=0.3)  # fmt: skip
    ev.reload()
    games = deals(schedule(ev), 4000)
    pairs = games[::2]  # a deal is two games (first / second) of one pairing
    assert all(g[:3] == h[:3] and g[4] == h[4] for g, h in zip(games[::2], games[1::2]))
    evolved = [p for p in pairs if p[4] is not None]
    assert 0.26 < len(evolved) / len(pairs) < 0.34
    assert {p[4] for p in evolved} == {"e1", "e2"}  # history decks are never the evolved side
    names = {load_ydk(DECK_DIR / f"{s}.ydk").name for s in CORPUS}
    for a, b, _, _, did in evolved:
        other = b if a == did else a
        assert did in (a, b) and other in names | {"old"}
    assert any("old" in (a, b) for a, b, *_ in evolved)
    assert {a == did for a, _, _, _, did in evolved} == {True, False}  # the evolved deck sits on either side
    assert all({a, b} <= names for a, b, _, _, did in pairs if did is None)


def test_the_policys_weak_decks_are_dealt_more_often(tmp_path):
    ev = EvolvedDecks(manifest(tmp_path, {"won": ("labrynth", "active"), "lost": ("purrely", "active")}),
                      share=1.0, power=1.0)  # fmt: skip
    ev.reload()
    for _ in range(40):
        ev.record("won", 1.0)
        ev.record("lost", 0.0)
    assert ev.win_rate("won") > 0.95 and ev.win_rate("lost") < 0.05
    n = Counter(p[4] for p in deals(schedule(ev), 2000)[::2])
    assert n["lost"] > 10 * n["won"]
    heavy = EvolvedDecks(manifest(tmp_path, {"a": ("labrynth", "active", 3.0), "b": ("purrely", "active")}, "w.json"))
    heavy.reload()
    ids, p = heavy.weights()
    assert dict(zip(ids, p))["a"] == pytest.approx(0.75)  # no games yet: the manifest weight alone


def test_after_a_reload_only_the_current_decks_are_dealt(tmp_path):
    path = manifest(tmp_path, {"e1": ("labrynth", "probation"), "e2": ("purrely", "probation")})
    ev = EvolvedDecks(path, share=1.0)
    assert ev.reload() and not ev.reload()  # unchanged manifest: nothing to do
    s = schedule(ev)
    assert {g[4] for g in deals(s, 200)} == {"e1", "e2"}
    manifest(tmp_path, {"e2": ("purrely", "history"), "e3": ("yubel", "active")})
    assert ev.reload()
    assert {g[4] for g in deals(s, 200)} == {"e3"}


def test_an_illegal_deck_is_left_out_with_a_note(tmp_path):
    path = manifest(tmp_path, {"ok": ("labrynth", "active"), "bad": ("purrely", "active")})
    ev = EvolvedDecks(path, validate=lambda d: ["too many copies"] if d.name == "bad" else [])
    ev.reload()
    assert ev.ids("active") == ["ok"] and ev.problems == ["bad: too many copies"]
    with pytest.raises(ValueError, match="status"):
        broken = json.loads(path.read_text())
        broken["decks"][0]["status"] = "retired"
        path.write_text(json.dumps(broken))
        ev.reload()


def test_the_pool_state_round_trips(tmp_path):
    ev = EvolvedDecks(manifest(tmp_path, {"e1": ("labrynth", "active"), "e2": ("purrely", "probation")}), share=0.5)
    ev.reload()
    ev.record("e1", 1.0)
    ev.record("e2", 0.5)
    s = schedule(ev)
    deals(s, 50)
    saved_ev, saved_s = json.loads(json.dumps(ev.state_dict())), s.state_dict()
    rest = deals(s, 200)
    ev2 = EvolvedDecks(tmp_path / "moved-away.json", share=0.5)  # the state, not the file, restores the pool
    ev2.load_state_dict(saved_ev)
    s2 = schedule(ev2)
    s2.load_state_dict(saved_s)
    assert ev2.ids("active", "probation") == ["e1", "e2"] and ev2.stats == {"e1": [1, 1.0], "e2": [1, 0.5]}
    assert deals(s2, 200) == rest


def small_cfg(**kw):
    return TrainConfig(decks=tuple(str(DECK_DIR / f"{s}.ydk") for s in CORPUS), num_envs=2, env_threads=1, steps=256,
                       event_length=16, net=TINY, privileged_dim=8, critic_hidden=16,
                       ppo=PPOConfig(epochs=1, minibatch_size=64), eval_every=0, checkpoint_every=0, snapshot_every=0,
                       torch_threads=1, collect_threads=1, seed=4, **kw)  # fmt: skip


def test_the_trainer_deals_records_and_checkpoints_the_evolved_pool(tmp_path):
    path = manifest(tmp_path, {"e1": ("labrynth", "active")})
    cfg = small_cfg(deck_pool=str(path), evolved_share=1.0, deck_pool_every=1)
    trainer = Trainer(cfg, tmp_path / "run", log=None)
    trainer.train(max_updates=2)
    games, _ = trainer.evolved.stats["e1"]
    metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert games > 0 and sum(m["evolved_games"] for m in metrics) == games  # self-play: the learner pilots it
    ckpt = trainer.save()
    resumed = Trainer.resume(ckpt, log=None)
    assert resumed.evolved.state_dict() == trainer.evolved.state_dict()
    assert resumed.schedule.state_dict() == trainer.schedule.state_dict()
    manifest(tmp_path, {"e1": ("labrynth", "history"), "e2": ("purrely", "probation")})
    resumed.train(max_updates=1)  # re-read after the update
    assert resumed.evolved.ids("probation", "active") == ["e2"]
    assert resumed.schedule.evolved is resumed.evolved


def test_the_evolved_pool_defaults_follow_the_spec():
    cfg = replace(small_cfg(), deck_pool="x.json")
    assert cfg.evolved_share == 0.3 and cfg.deck_pool_every > 0
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
