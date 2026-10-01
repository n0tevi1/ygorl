"""Seat-split training (--seat-split): one actor-critic per seat, rows routed by the first-player flag."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.nets.seat_split import SeatSplit, first_player_rows  # noqa: E402
from ygorl.train.checkpoint import load_actor, load_actor_critic, load_checkpoint  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner, SeatSplitLearner  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

DECK_DIR = Path(__file__).parent / "decks"
PAIR = (str(DECK_DIR / "snake_eye.ydk"), str(DECK_DIR / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _cfg(**kw):
    base = dict(decks=PAIR, num_envs=2, env_threads=1, steps=12, event_length=16, net=TINY, privileged_dim=8,
                critic_hidden=16, max_decisions=40, ppo=PPOConfig(epochs=1, minibatch_size=16), eval_every=0,
                checkpoint_every=0, snapshot_every=0, torch_threads=1, collect_threads=1, seed=4, seat_split=True)  # fmt: skip
    return TrainConfig(**{**base, **kw})


def _params(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _moved(before, module):
    return any(not torch.equal(v, before[k]) for k, v in module.state_dict().items())


def test_rows_are_routed_per_seat_and_both_nets_update(tmp_path, monkeypatch):
    t = Trainer(_cfg(), tmp_path, log=None)
    assert isinstance(t.model, SeatSplit) and isinstance(t.learner, SeatSplitLearner)
    a0, b0 = _params(t.model.nets[0]), _params(t.model.nets[1])
    ro = t.collector.collect()
    first = first_player_rows(ro.obs).reshape(ro.shape)
    assert first.any() and (~first).any()  # self-play rows of both seats in one rollout

    # acting: every row's logits / Q / V came from its own seat's net
    with torch.no_grad():
        nets = [n.eval() for n in t.model.nets]
        flat = first.reshape(-1)
        for seat, sel in ((0, flat), (1, ~flat)):
            idx = sel.nonzero().squeeze(1)
            out = nets[seat]({k: v[idx] for k, v in ro.obs.items()}, {k: v[idx] for k, v in ro.privileged.items()})
            torch.testing.assert_close(out.v.float(), ro.values.reshape(-1)[idx])
            whole = t.model(ro.obs, ro.privileged)
            torch.testing.assert_close(whole.logits[idx], out.logits)

    # the update: each learner sees its own seat's rows only, with targets of the whole rollout
    seen = []
    original = PPOLearner.update

    def spy(self, ro_, policy=True, *, rows=None, est=None):
        seen.append((self, rows.clone(), est))
        return original(self, ro_, policy, rows=rows, est=est)

    monkeypatch.setattr(PPOLearner, "update", spy)
    stats = t.learner.update(ro)
    assert [s[0] for s in seen] == t.learner.learners
    assert torch.equal(seen[0][1], first) and torch.equal(seen[1][1], ~first)
    assert seen[0][2] is seen[1][2]
    assert stats["seat0/rows"] == int(first.sum()) and stats["seat1/rows"] == int((~first).sum())
    assert stats["rows"] == ro.players.numel()
    assert _moved(a0, t.model.nets[0]) and _moved(b0, t.model.nets[1])
    opt = [lr.optimizer for lr in t.learner.learners]
    assert opt[0] is not opt[1]
    own = {id(p) for p in t.model.nets[0].parameters()}
    assert all(id(p) in own for g in opt[0].param_groups for p in g["params"])


def test_seat_split_run_checkpoints_snapshots_and_plays(tmp_path):
    cfg = _cfg(selfplay_fraction=0.5, pool_size=2, snapshot_every=1, checkpoint_every=1, eval_every=2, eval_pairs=1,
               eval_opponents=("random",), keep_best_by="random", eval_workers=1)  # fmt: skip
    t = Trainer(cfg, tmp_path, log=None)
    t.train(max_updates=2)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["update"] for m in metrics] == [1, 2]
    assert all(m["seat0/rows"] + m["seat1/rows"] == m["rows"] for m in metrics)
    assert all(isinstance(t.pool.get(i), SeatSplit) for i in t.pool.ids())  # snapshots hold the pair
    assert json.loads((tmp_path / "eval.jsonl").read_text().splitlines()[0])["games"] == 4  # policy: agent played

    ckpt = tmp_path / "checkpoints" / "latest.pt"
    policy = load_actor(ckpt)
    assert isinstance(policy.net, SeatSplit)
    obs = t.collector.collect().obs
    first = first_player_rows(obs)
    assert first.any() and (~first).any()
    with torch.no_grad():
        logits = policy.net(obs).logits
        for seat, sel in ((0, first), (1, ~first)):
            own = policy.net.nets[seat]({k: v[sel] for k, v in obs.items()}).logits
            torch.testing.assert_close(logits[sel], own)
    assert isinstance(load_actor_critic(ckpt).model, SeatSplit)

    resumed = Trainer.resume(ckpt, log=None)
    for a, b in zip(t.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.equal(a, b)
    assert load_checkpoint(ckpt)["learner"]["updates"] == 2 and resumed.learner.updates == 2


def test_seat_split_warm_starts_both_nets_from_one_actor(tmp_path):
    single = Trainer(_cfg(seat_split=False), tmp_path / "single", log=None)
    path = single.save()
    t = Trainer(_cfg(init_from=str(path)), tmp_path / "split", log=None)
    want = single.model.actor.state_dict()
    for net in t.model.nets:
        assert all(torch.equal(v, want[k]) for k, v in net.actor.state_dict().items())
