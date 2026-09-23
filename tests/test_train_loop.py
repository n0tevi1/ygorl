"""PPO self-play on real duels (T4b.4): rollout layout on EncodedVecEnv, trainer checkpoints and resume,
and the ``policy:<checkpoint>`` agent (lockstep C++ host) on the arena / duel path."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.agents import RandomAgent, make_agent  # noqa: E402
from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.cli import main  # noqa: E402
from ygorl.engine.duel import Duel, DuelConfig  # noqa: E402
from ygorl.env import GameSpec  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.nets import NetConfig, collate  # noqa: E402
from ygorl.nets.actor_critic import PRIVILEGED_LISTS, ActorCritic  # noqa: E402
from ygorl.train.checkpoint import load_checkpoint, load_policy  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.rollout import RolloutCollector  # noqa: E402
from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer, summarize_metrics  # noqa: E402

DECK_DIR = Path(__file__).parent / "decks"
PAIR = (str(DECK_DIR / "snake_eye.ydk"), str(DECK_DIR / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


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


def tiny_model(vocab):
    torch.manual_seed(0)
    return ActorCritic(NetConfig(vocab_size=len(vocab), **TINY), privileged=True, privileged_dim=8, critic_hidden=16)


def collector(db, vocab, *, max_decisions, steps, selfplay=1.0, snapshot=False, num_envs=2):
    decks = [load_ydk(p) for p in PAIR]
    pool = SnapshotPool(2)
    model = tiny_model(vocab)
    if snapshot:
        pool.add(model, update=0)
    schedule = SelfPlaySchedule(DeckPool(decks, "cross"), pool, DuelConfig(max_decisions=max_decisions),
                                selfplay_fraction=selfplay, seed=5)  # fmt: skip
    env = EncodedVecEnv(num_envs, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    return RolloutCollector(env, model, schedule, steps, opponents=pool.get, seed=5)


def test_self_play_layout_on_real_duels(db, vocab):
    """Both seats are rows in time order; a decision-limit ending is a truncation (done, no reward)."""
    col = collector(db, vocab, max_decisions=30, steps=45)
    ro = col.collect()
    T, B = 45, 2
    assert ro.players.shape == (T, B) and ro.action_mask.shape == (T, B, 128) and ro.q.shape == (T, B, 128)
    assert set(ro.players.unique().tolist()) <= {0, 1}
    assert ro.action_mask.gather(-1, ro.actions.unsqueeze(-1)).all()
    assert (ro.probs[~ro.action_mask] == 0).all()
    assert (ro.q[~ro.action_mask] == 0).all()
    # every column starts a game at row 0; 30 decisions (all rows in self-play) later the limit cuts it
    assert ro.dones[29].all() and ro.truncated[29].all()
    assert ro.dones.sum() == ro.truncated.sum() == B and (ro.rewards == 0).all()
    assert len(ro.games) == B and all(g.reason == "decision_limit" and g.truncated for g in ro.games)
    assert all(g.learner_rows == 30 == g.result["decisions"] for g in ro.games)
    # the privileged ground truth reaches the critic batch, never the observation batch
    assert set(PRIVILEGED_LISTS) <= set(ro.privileged) and ro.privileged["op_hand"].shape[:2] == (T * B, 32)
    assert not set(PRIVILEGED_LISTS) & set(ro.obs)
    assert ro.bootstrap_mask.any(-1).all() and ro.opponent_decisions == 0
    # the next segment starts from the held bootstrap decisions
    ro2 = col.collect()
    assert torch.equal(ro2.players[0], ro.bootstrap_player)
    assert torch.equal(ro2.action_mask[0], ro.bootstrap_mask)


def test_snapshot_games_keep_only_the_learner_rows(db, vocab):
    col = collector(db, vocab, max_decisions=40, steps=30, selfplay=0.0, snapshot=True)
    ro = col.collect()
    assert ro.opponent_decisions > 0
    assert all(g.assignment.opponent is not None for g in ro.games)
    for b in range(2):
        game_start = 0
        for t in range(30):
            assert ro.players[t, b] == ro.players[game_start, b]  # one seat per game
            if ro.dones[t, b]:
                game_start = t + 1
    for g in ro.games:  # the learner's rows of a game = its decisions in that game
        assert 0 < g.learner_rows < g.result["decisions"]


# ------------------------------------------------------------------------------------------ trainer


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = TrainConfig(decks=PAIR, num_envs=2, env_threads=1, steps=12, event_length=16, net=TINY, privileged_dim=8,
                      critic_hidden=16, max_decisions=40, ppo=PPOConfig(epochs=1, minibatch_size=16),
                      selfplay_fraction=0.5, pool_size=2, snapshot_every=1, checkpoint_every=1, eval_every=2,
                      eval_pairs=1, eval_opponents=("random",), keep_best_by="random", eval_workers=1,
                      torch_threads=1, collect_threads=1, seed=3)  # fmt: skip
    out = tmp_path_factory.mktemp("run")
    trainer = Trainer(cfg, out, log=None)
    trainer.train(max_updates=2)
    return trainer, out


def test_trainer_writes_metrics_checkpoints_and_keeps_the_best(run):
    trainer, out = run
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [m["update"] for m in metrics] == [1, 2]
    for m in metrics:
        for key in ("entropy", "kl_ref", "q_loss", "v_loss", "policy_loss", "decisions_per_s", "rows_per_s"):
            assert np.isfinite(m[key]), key
        assert m["rows"] == 24 and m["total"]["rows"] == 24 * m["update"]
    evals = [json.loads(line) for line in (out / "eval.jsonl").read_text().splitlines()]
    assert len(evals) == 1 and evals[0]["vs"] == "random" and evals[0]["games"] == 4 and evals[0]["best"]
    assert (out / "best.pt").is_file() and (out / "checkpoints" / "latest.pt").is_file()
    assert (out / "checkpoints" / "update_000002.pt").is_file()
    assert trainer.pool.best_id is not None and trainer.best["update"] == 2
    # vocab indices travel with the model (CardVocab.save next to the run and inside every checkpoint)
    saved = CardVocab.load(out / "vocab.json")
    state = load_checkpoint(out / "best.pt")
    assert json.loads(state["vocab"])["passwords"] == [saved.password(i) for i in range(2, len(saved))]
    assert state["config"]["decks"] == list(PAIR) and state["counters"]["updates"] == 2
    summary = summarize_metrics(out / "metrics.jsonl")
    assert summary["updates"] == 2 and summary["last"]["entropy"] is not None


def test_checkpoint_round_trip_and_resume(run, db):
    trainer, out = run
    ckpt = out / "checkpoints" / "latest.pt"
    resumed = Trainer.resume(ckpt, log=None)
    assert resumed.run_dir == out.resolve()  # RUN/checkpoints/latest.pt -> RUN
    for a, b in zip(trainer.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(trainer.learner.reference.state_dict().values(), resumed.learner.reference.state_dict().values()):
        assert torch.equal(a, b)
    assert resumed.pool.ids() == trainer.pool.ids() and resumed.pool.best_id == trainer.pool.best_id
    assert resumed.counters == trainer.counters and resumed.best == trainer.best
    opt_a, opt_b = trainer.learner.optimizer.state_dict()["state"], resumed.learner.optimizer.state_dict()["state"]
    assert opt_a.keys() == opt_b.keys()
    assert all(torch.equal(opt_a[k]["exp_avg"], opt_b[k]["exp_avg"]) for k in opt_a)
    resumed.train(max_updates=1)
    assert resumed.counters["updates"] == 3
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["update"] == 3
    # the actor alone, as loaded for play, scores observations like the trained model
    policy = load_policy(out / "best.pt")
    env = EncodedVecEnv(1, 1, cards=db, vocab=policy.vocab, event_length=policy.event_length)
    decks = [load_ydk(p) for p in PAIR]
    env.reset(0, GameSpec(seed=9, deck_a=decks[0], deck_b=decks[1]))
    (ev,) = env.recv(1)
    best = load_checkpoint(out / "best.pt")["learner"]["model"]
    model = trainer._new_model()
    model.load_state_dict(best)
    batch = collate([ev.obs])
    torch.testing.assert_close(policy.net(batch).logits, model.policy_logits(batch))


def test_policy_agent_plays_legal_moves_in_lockstep(run, db):
    _, out = run
    ckpt = str(out / "best.pt")
    decks = [load_ydk(p) for p in PAIR]
    cfg = DuelConfig(max_turns=3)
    agent, rival = make_agent(f"policy:{ckpt}", seed=1), RandomAgent(2)
    order = []  # every decision of the duel in time order: (seat, index, the policy's observation or None)

    def spy(who, is_policy):
        act = who.act

        def wrapped(point):
            obs = who.host.observe() if is_policy else None
            index = act(point)
            order.append((point.player, index, obs))
            return index

        return wrapped

    agent.act, rival.act = spy(agent, True), spy(rival, False)
    result = Duel(7, None, decks[0], decks[1], config=cfg, first=1).run(agent, rival)
    assert result.reason in ("win", "turn_limit")  # Duel.run rejects illegal indices
    assert agent.last_probs is not None and abs(sum(agent.last_probs) - 1) < 1e-4
    # replayed in EncodedVecEnv, every decision of the policy shows exactly the lockstep host's observation
    policy = load_policy(ckpt)
    env = EncodedVecEnv(1, 1, cards=db, vocab=policy.vocab, event_length=policy.event_length)
    env.reset(0, GameSpec(seed=7, deck_a=decks[0], deck_b=decks[1], first=1, config=cfg))
    checked = 0
    for seat, index, obs in order:
        (ev,) = env.recv(1)
        assert ev.player == seat
        if obs is not None:
            assert all(np.array_equal(ev.obs[k], obs[k]) for k in ev.obs)
            checked += 1
        env.step(0, index)
    assert checked > 5
    # both seats as policies, and the argmax variant
    both = Duel(8, None, decks[0], decks[1], config=cfg).run(make_agent(f"policy:{ckpt}", 3),
                                                              make_agent(f"policy-greedy:{ckpt}", 4))  # fmt: skip
    assert both.reason in ("win", "turn_limit")


def test_parallel_arena_plays_a_checkpoint_like_one_worker(run):
    """Workers are spawned when PyTorch is loaded (forking it can deadlock); results do not depend on workers."""
    from ygorl.agents.registry import agent_factory
    from ygorl.eval.arena import Arena

    _, out = run
    decks = [load_ydk(p) for p in PAIR]
    reports = []
    for workers in (1, 2):
        arena = Arena(agent_factory(f"policy:{out / 'best.pt'}"), agent_factory("random"),
                      config=DuelConfig(max_turns=2), workers=workers)  # fmt: skip
        reports.append(arena.run(decks[0], decks[1], pairs=1, seed=5))
    assert reports[0].errors == 0 and reports[0].records == reports[1].records


def test_policy_agent_needs_duel_run_and_a_checkpoint(run):
    _, out = run
    agent = make_agent(f"policy:{out / 'best.pt'}", seed=0)
    with pytest.raises(RuntimeError, match="Duel.run"):
        agent.act(object())
    with pytest.raises(ValueError, match="checkpoint"):
        make_agent("policy:no/such/file.pt")
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_agent("policy")
    # one spec syntax for PPO and BC checkpoints (the format field picks the loader)
    from ygorl.agents.checkpoint import CheckpointAgent

    copied = out / "with@sign" / "best.pt"  # a checkpoint outside checkpoints/, with '@' in its path
    copied.parent.mkdir(exist_ok=True)
    copied.write_bytes((out / "best.pt").read_bytes())
    assert make_agent(f"policy:{copied}@greedy", seed=0).greedy
    assert Trainer.resume(copied, log=None, eval_every=0).run_dir == copied.parent.resolve()
    warm = make_agent(f"policy:{out / 'best.pt'}@t=0.5", seed=0)
    cold = make_agent(f"policy:{out / 'best.pt'}@greedy", seed=0)
    short = make_agent(f"policy-greedy:{out / 'best.pt'}", seed=0)
    assert isinstance(warm, CheckpointAgent) and warm.temperature == 0.5 and not warm.greedy
    assert cold.greedy and short.greedy
    for bad in ("@t=0", "@hot"):
        with pytest.raises(ValueError, match="temperature|option"):
            make_agent(f"policy:{out / 'best.pt'}{bad}")
    junk = out / "junk.pt"
    torch.save({"format": "something-else"}, junk)
    with pytest.raises(ValueError, match="not a ygorl policy checkpoint"):
        make_agent(f"policy:{junk}")


def test_duel_command_plays_a_checkpoint(run, capsys):
    _, out = run
    code = main(["duel", PAIR[0], PAIR[1], "--agent-a", f"policy:{out / 'best.pt'}", "--seed", "4",
                 "--max-turns", "2"])  # fmt: skip
    assert code == 0, capsys.readouterr().err
    assert "policy:" in capsys.readouterr().out


def test_train_tool_runs_resumes_and_summarizes(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("train_ppo", Path(__file__).parents[1] / "tools" / "train_ppo.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = tmp_path / "run"
    small = ["--envs", "2", "--env-threads", "1", "--steps", "8", "--event-length", "8", "--d-model", "16",
             "--max-decisions", "30", "--eval-every", "0", "--torch-threads", "1", "--collect-threads", "1"]  # fmt: skip
    assert tool.main([*PAIR, "--updates", "1", "--out", str(out), *small]) == 0
    assert tool.main(["--resume", str(out / "checkpoints" / "latest.pt"), "--updates", "1"]) == 0
    assert [json.loads(line)["update"] for line in (out / "metrics.jsonl").read_text().splitlines()] == [1, 2]
    capsys.readouterr()
    assert tool.main(["--summary", str(out / "metrics.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["updates"] == 2
    assert tool.main([str(tmp_path / "missing.ydk"), "--updates", "1"]) == 2  # decks are checked like the CLI


def test_padding_trimming_leaves_the_ppo_update_unchanged(db, vocab, monkeypatch):
    """The learner cuts the rollout's padding once and the net cuts each batch's: the same update either way."""
    import copy

    from ygorl.nets import actor_critic as ac_mod
    from ygorl.nets.policy import PolicyNet, trim_padding
    from ygorl.train import ppo as ppo_mod
    from ygorl.train.ppo import PPOLearner

    col = collector(db, vocab, max_decisions=120, steps=24)
    ro = col.collect()
    base = col.model
    results = []
    for trim in (False, True):
        model = copy.deepcopy(base)
        monkeypatch.setattr(PolicyNet, "trim_padding", trim)
        for mod in (ppo_mod, ac_mod):
            monkeypatch.setattr(mod, "trim_padding", trim_padding if trim else (lambda obs: obs))
        learner = PPOLearner(model, PPOConfig(minibatch_size=16, epochs=2))
        torch.manual_seed(3)
        stats = learner.update(ro)
        results.append((stats, {k: v.detach().clone() for k, v in model.state_dict().items()}))
    (s0, p0), (s1, p1) = results
    for key in ("loss", "policy_loss", "entropy", "kl_ref", "q_loss", "v_loss", "approx_kl"):
        assert abs(s0[key] - s1[key]) < 1e-4 * max(1.0, abs(s0[key])), key
    for name in p0:
        assert torch.allclose(p0[name].float(), p1[name].float(), atol=1e-5, rtol=1e-4), name


def test_actor_critic_heads_on_the_trimmed_batch_match_the_padded_one(db, vocab, monkeypatch):
    """ActorCritic scores only the batch's used action rows and pads logits / Q back to the input width."""
    from ygorl.nets import actor_critic as ac_mod
    from ygorl.nets.heads import MASKED_LOGIT
    from ygorl.nets.policy import PolicyNet

    ro = collector(db, vocab, max_decisions=120, steps=24).collect()
    model = tiny_model(vocab)
    obs, priv = ro.obs, ro.privileged
    with torch.no_grad():
        trimmed = model(obs, priv)
        monkeypatch.setattr(PolicyNet, "trim_padding", False)
        monkeypatch.setattr(ac_mod, "trim_padding", lambda o: o)
        padded = model(obs, priv)
    mask = obs["action_mask"]
    assert trimmed.logits.shape == padded.logits.shape == trimmed.q.shape == mask.shape
    assert mask.sum(1).max() < mask.shape[1]  # the trimmed run really dropped columns
    assert torch.allclose(trimmed.logits[mask], padded.logits[mask], atol=1e-5)
    assert (trimmed.logits[~mask] == MASKED_LOGIT).all() and (trimmed.q[~mask] == 0).all()
    assert torch.allclose(trimmed.q[mask], padded.q[mask], atol=1e-5)
    assert torch.allclose(trimmed.v, padded.v, atol=1e-5)


def _small_cfg(**kw):
    return TrainConfig(decks=PAIR, num_envs=2, env_threads=1, steps=8, event_length=16, net=TINY, privileged_dim=8,
                       critic_hidden=16, max_decisions=40, ppo=PPOConfig(epochs=1, minibatch_size=16),
                       eval_every=0, checkpoint_every=0, snapshot_every=0, torch_threads=1, collect_threads=1,
                       seed=4, **kw)  # fmt: skip


def test_init_from_and_bc_prior_accept_ppo_and_bc_checkpoints(run, vocab, tmp_path):
    """--init-from / --bc-prior take a PPO training checkpoint or a BC policy checkpoint (by its format field);
    the actor's weights are copied exactly and the prior is the checkpoint's network."""
    from ygorl.nets import PolicyNet
    from ygorl.nets.agent import save_checkpoint as save_policy

    trainer, out = run
    ppo_ckpt = out / "checkpoints" / "latest.pt"
    torch.manual_seed(7)
    bc_net = PolicyNet(trainer.net_config)
    bc_ckpt = save_policy(tmp_path / "bc.pt", bc_net, trainer.vocab, event_length=16)
    from ygorl.train.checkpoint import load_policy

    sources = {ppo_ckpt: load_policy(ppo_ckpt).net.state_dict(), bc_ckpt: bc_net.state_dict()}
    for path, expect in sources.items():
        t = Trainer(_small_cfg(init_from=str(path), bc_prior=str(path)), tmp_path / f"run-{path.stem}", log=None)
        got = t.model.actor.state_dict()
        assert got.keys() == expect.keys()
        for name in expect:
            assert torch.equal(got[name], expect[name]), name
        assert t.learner.prior is not None
        for name, p in t.learner.prior.state_dict().items():
            assert torch.equal(p, expect[name]), name
        assert all(p.requires_grad for p in t.model.actor.parameters())  # the copy trains


def test_init_from_rejects_a_different_network_or_vocab(run, vocab, tmp_path):
    from dataclasses import replace as dc_replace

    from ygorl.cards.cdb import CardVocab
    from ygorl.nets import PolicyNet
    from ygorl.nets.agent import save_checkpoint as save_policy

    trainer, _ = run
    wide = PolicyNet(dc_replace(trainer.net_config, d_model=32))
    save_policy(tmp_path / "wide.pt", wide, trainer.vocab, event_length=16)
    with pytest.raises(ValueError, match="network"):
        Trainer(_small_cfg(init_from=str(tmp_path / "wide.pt")), tmp_path / "a", log=None)
    passwords = [trainer.vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(trainer.vocab))]
    shuffled = CardVocab(passwords[1:] + passwords[:1])  # same size, other card order
    save_policy(tmp_path / "other.pt", PolicyNet(trainer.net_config), shuffled, event_length=16)
    with pytest.raises(ValueError, match="vocab"):
        Trainer(_small_cfg(bc_prior=str(tmp_path / "other.pt")), tmp_path / "b", log=None)
