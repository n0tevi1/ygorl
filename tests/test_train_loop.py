"""PPO self-play on real duels (T4b.4): rollout layout on EncodedVecEnv, trainer checkpoints and resume,
and the ``policy:<checkpoint>`` agent (lockstep C++ host) on the arena / duel path."""

import importlib.util
import json
from dataclasses import replace
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
from ygorl.env.privileged import P_WIDTHS  # noqa: E402
from ygorl.nets import NetConfig, collate  # noqa: E402
from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.train.checkpoint import load_checkpoint, load_policy  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.rollout import RolloutCollector  # noqa: E402
from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer, summarize_metrics  # noqa: E402

DECK_DIR = Path(__file__).parent / "decks"
PAIR = (str(DECK_DIR / "snake_eye.ydk"), str(DECK_DIR / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")
def test_gpu_resume_restores_minibatch_and_collector_rng(tmp_path):
    cfg = TrainConfig(decks=PAIR, net=TINY, device="cuda", num_envs=2, steps=4, event_length=8,
                      checkpoint_every=0, eval_every=0, ppo=PPOConfig(epochs=1, minibatch_size=8))  # fmt: skip
    trainer = Trainer(cfg, tmp_path / "original")
    trainer.step()
    trainer.pool.add(trainer.model, update=1)  # restoring a pool must happen before restoring RNG
    path = trainer.save()
    expected_perm = torch.randperm(257, device=trainer.device)
    expected_cpu = torch.rand(32)
    expected_acting = torch.rand(32, generator=trainer.collector.generator)
    torch.manual_seed(234567)  # simulate unrelated activity/new process before resume
    resumed = Trainer.resume(path, tmp_path / "resumed", log=None)
    assert torch.equal(torch.randperm(257, device=resumed.device), expected_perm)
    assert torch.equal(torch.rand(32), expected_cpu)
    assert torch.equal(torch.rand(32, generator=resumed.collector.generator), expected_acting)


def test_cpu_checkpoint_does_not_access_cuda_rng(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU checkpoints must not read or write CUDA RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected)
    monkeypatch.setattr(torch.cuda, "set_rng_state", unexpected)
    cfg = TrainConfig(decks=PAIR, net=TINY, num_envs=2, steps=4, event_length=8, eval_every=0)
    trainer = Trainer(cfg, tmp_path / "original")
    assert "cuda" not in trainer.state_dict()["rng"]
    path = trainer.save()
    expected = torch.randperm(257)
    Trainer.resume(path, tmp_path / "resumed", log=None)
    assert torch.equal(torch.randperm(257), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")
def test_legacy_gpu_checkpoint_without_cuda_rng_still_resumes(tmp_path):
    cfg = TrainConfig(decks=PAIR, net=TINY, device="cuda", num_envs=2, steps=4, event_length=8,
                      checkpoint_every=0, eval_every=0, ppo=PPOConfig(epochs=1, minibatch_size=8))  # fmt: skip
    trainer = Trainer(cfg, tmp_path / "original")
    state = trainer.state_dict()
    del state["rng"]["cuda"]
    resumed = Trainer(cfg, tmp_path / "legacy", state=state)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    assert resumed.step()["update"] == 1


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


def test_a_dealt_pair_keeps_its_snapshot_after_eviction(vocab):
    # The second game of a pool pair starts later; the snapshot may have left the pool by then.
    pool = SnapshotPool(1)
    model = tiny_model(vocab)
    sid = pool.add(model, update=0)
    schedule = SelfPlaySchedule(DeckPool([load_ydk(p) for p in PAIR], "cross"), pool, selfplay_fraction=0.0, seed=5)
    first = schedule()
    assert first.opponent == sid
    held = schedule.opponent(sid)
    pool.add(model, update=1)  # evicts sid
    assert sid not in pool.ids()
    second = schedule()
    assert second.opponent == sid and schedule.opponent(sid) is held
    schedule()  # the next deal releases it
    with pytest.raises(KeyError):
        schedule.opponent(sid)


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
    assert set(P_WIDTHS) <= set(ro.privileged) and ro.privileged["op_hand"].shape[:2] == (T * B, 32)
    assert not set(P_WIDTHS) & set(ro.obs)
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


def test_update_reports_the_mean_advantage_per_action_kind(db, vocab):
    """Diagnostics: ``adv/<kind>`` is the mean advantage of the rows whose chosen action has that kind."""
    from ygorl.env.encoding import ACTION_KINDS
    from ygorl.train.ppo import PPOLearner

    col = collector(db, vocab, max_decisions=120, steps=24)
    ro = col.collect()
    stats = PPOLearner(col.model, PPOConfig(minibatch_size=16, epochs=1)).update(ro)
    kinds = {k[4:] for k in stats if k.startswith("adv/")}
    assert kinds and kinds <= set(ACTION_KINDS)
    assert all(np.isfinite(stats[f"adv/{k}"]) and stats[f"rows/{k}"] > 0 for k in kinds)
    assert sum(stats[f"rows/{k}"] for k in kinds) == stats["rows"]


def test_pinned_opponents_join_the_pool_and_must_match_the_run(run, tmp_path):
    from dataclasses import replace as dc_replace

    from ygorl.cards.cdb import CardVocab
    from ygorl.nets import PolicyNet
    from ygorl.nets.agent import save_checkpoint as save_policy

    trainer, _ = run
    bc = save_policy(tmp_path / "bc.pt", PolicyNet(trainer.net_config), trainer.vocab, event_length=16)
    t = Trainer(_small_cfg(pin_opponents=(str(bc),), pinned_share=1.0, selfplay_fraction=0.0), tmp_path / "a",
                log=None)  # fmt: skip
    assert t.pool.ids() == [-1] and t.pool.info()[0]["tag"] == "pinned:bc.pt"
    ro = t.collector.collect()  # every game against the pinned opponent
    assert ro.opponent_decisions > 0
    assert TrainConfig.from_dict(t.cfg.to_dict()).pin_opponents == (str(bc),)
    long_events = save_policy(tmp_path / "e64.pt", PolicyNet(trainer.net_config), trainer.vocab, event_length=64)
    with pytest.raises(ValueError, match="event tokens"):
        Trainer(_small_cfg(pin_opponents=(str(long_events),)), tmp_path / "b", log=None)
    passwords = [trainer.vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(trainer.vocab))]
    other = save_policy(tmp_path / "other.pt", PolicyNet(dc_replace(trainer.net_config)),
                        CardVocab(passwords[1:] + passwords[:1]), event_length=16)  # fmt: skip
    with pytest.raises(ValueError, match="vocab"):
        Trainer(_small_cfg(pin_opponents=(str(other),)), tmp_path / "c", log=None)


def _params(module):
    return [p.detach().clone() for p in module.parameters()]


def test_overlapped_collection_acts_with_the_weights_the_update_started_from(tmp_path):
    """overlap_collect: while update k runs, the next rollout is collected with a copy of the weights before update k;
    league bookkeeping (snapshots) between steps still works and every step trains on a full rollout."""
    cfg = replace(_small_cfg(overlap_collect=True, selfplay_fraction=0.5), snapshot_every=1)
    trainer = Trainer(cfg, tmp_path, log=None)
    assert trainer.acting is not trainer.model and trainer.collector.model is trainer.acting
    before = _params(trainer.model)
    trainer.step()
    after_first = _params(trainer.model)
    assert all(torch.equal(a, b) for a, b in zip(_params(trainer.acting), before))
    assert not all(torch.equal(a, b) for a, b in zip(after_first, before))
    trainer.train(max_updates=1)
    start = _params(trainer.model)
    trainer.step()
    assert all(torch.equal(a, b) for a, b in zip(_params(trainer.acting), start))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["update"] for m in metrics] == [1, 2, 3] and all(m["rows"] == 16 for m in metrics)
    assert len(trainer.pool) > 0


def test_snapshot_admission_threshold_and_pool_results(tmp_path):
    """snapshot_min_win_rate: a due snapshot joins only after the learner scored above the threshold in enough pool
    games since the last one joined; an empty pool always takes one; pool results reach the snapshots."""
    from ygorl.train.rollout import Assignment, FinishedGame

    cfg = replace(_small_cfg(), snapshot_min_win_rate=0.55, snapshot_min_games=4)
    trainer = Trainer(cfg, tmp_path, log=None)
    assert trainer.maybe_snapshot(1) and len(trainer.pool) == 1  # empty pool
    assert not trainer.maybe_snapshot(2)  # no pool games yet
    sid = trainer.pool.ids()[0]

    def played(scores):
        for s in scores:
            g = FinishedGame(Assignment(None, sid, 0), None if s == 0.5 else (0 if s == 1 else 1), "win", False, 1, {})
            trainer.pool.record(g.assignment.opponent, g.learner_score)
            trainer.league["games"] += 1
            trainer.league["points"] += g.learner_score

    played([1, 0, 1, 0])  # 0.5: not above 0.55
    assert not trainer.maybe_snapshot(3)
    played([1, 1, 1, 1])  # 6 / 8 = 0.75
    assert trainer.maybe_snapshot(4) and len(trainer.pool) == 2 and trainer.league == {"games": 0, "points": 0.0}
    assert trainer.counters["snapshots"] == 2 and trainer.counters["snapshots_skipped"] == 2
    info = {i["id"]: i for i in trainer.pool.info()}
    assert info[sid]["games"] == 8 and info[sid]["learner_win_rate"] == round(6.5 / 9, 3)
    state = trainer.state_dict()
    assert state["league"] == {"games": 0, "points": 0.0}


def test_pool_games_are_recorded_during_training(tmp_path):
    """Finished (not truncated) pool games reach the snapshots' records and the admission window."""
    cfg = replace(_small_cfg(selfplay_fraction=0.0, pool_sampling="pfsp"), snapshot_every=1, max_decisions=None,
                  steps=256)  # fmt: skip
    trainer = Trainer(cfg, tmp_path, log=None)
    trainer.train(max_updates=4)
    recorded = sum(i["games"] for i in trainer.pool.info())
    assert trainer.counters["snapshots"] == 4 and trainer.pool.sampling == "pfsp"
    assert recorded > 0 and trainer.league["games"] <= recorded


def test_league_defaults_follow_design_i1():
    """Design I1: PFSP over the snapshot pool; the admission threshold stays off (it did not help, #61)."""
    cfg = TrainConfig(decks=PAIR)
    assert cfg.pool_sampling == "pfsp" and cfg.snapshot_min_win_rate is None


def test_eval_pairings_is_a_fixed_sample_of_the_training_pairings(tmp_path):
    cfg = replace(_small_cfg(pairings="all"), eval_pairings=2)
    a = Trainer(cfg, tmp_path / "a", log=None).eval_pairings()
    b = Trainer(cfg, tmp_path / "b", log=None).eval_pairings()
    everything = Trainer(_small_cfg(pairings="all"), tmp_path / "c", log=None)
    assert a == b and len(a) == 2 and set(a) <= set(everything.schedule.decks.pairs)
    assert everything.eval_pairings() == list(everything.schedule.decks.pairs)


def _facts_dir(tmp_path, vocab):
    """A card-feature directory with only card facts (every card in archetype 0x1)."""
    feat = tmp_path / "features"
    feat.mkdir()
    pws = [vocab.password(i) for i in range(2, len(vocab))]
    n = len(pws)
    np.savez_compressed(feat / "card_facts.npz", passwords=np.asarray(pws, np.int64),
                        setcodes=np.tile([[0x1, 0, 0, 0]], (n, 1)), references=np.zeros((n, 2), np.int64),
                        categories=np.zeros(n, np.uint64), queries=np.zeros((n, 3), np.uint8),
                        query_names=np.array(["a", "b", "c"]))  # fmt: skip
    return feat


def test_a_bc_checkpoint_with_card_views_warm_starts_ppo(tmp_path, vocab):
    """tools/train_bc.py --text-dir --card-facts --id-dropout builds (from its own flags) the network PPO builds for
    the same switches, so init_from with the same text_dir picks up every weight; loading the checkpoint without the
    features is an error, not silently different weights."""
    from ygorl.nets import PolicyNet
    from ygorl.nets.agent import load_checkpoint, save_checkpoint
    from ygorl.nets.text import TextFeatures

    spec = importlib.util.spec_from_file_location("train_bc", Path(__file__).parents[1] / "tools" / "train_bc.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    feat = _facts_dir(tmp_path, vocab)
    text = TextFeatures.load(feat, vocab)
    args = tool.build_parser().parse_args(["--train", "x.jsonl", "--d-model", "16", "--layers", "1", "--text-dir",
                                           str(feat), "--card-facts", "--id-dropout", "0.3"])  # fmt: skip
    cfg = tool.net_config(args, vocab, text)
    net = PolicyNet(cfg, text)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01)  # the view modules start at zero: make them non-zero so the copy is visible
    ckpt = save_checkpoint(tmp_path / "bc.pt", net, vocab, event_length=_small_cfg().event_length)
    with pytest.raises(ValueError, match="card_facts"):
        load_checkpoint(ckpt)
    ppo_net = {"d_model": 16, "board_layers": 1, "history_layers": 1, "card_facts": True, "id_dropout": 0.3}
    trainer = Trainer(replace(_small_cfg(), init_from=str(ckpt), text_dir=str(feat), net=ppo_net), tmp_path / "run",
                      log=None)  # fmt: skip
    assert trainer.net_config == cfg and cfg.n_archetypes == 1 and cfg.id_dropout == 0.3
    bc = dict(net.named_parameters())
    for name, p in trainer.model.actor.named_parameters():
        assert torch.equal(p.cpu(), bc[name]), name


def test_warm_start_may_add_card_views(tmp_path, vocab):
    """init_from a checkpoint without card facts / ID dropout: the old weights load, the new view modules start
    fresh; any other network difference is still an error."""
    from ygorl.nets.text import TextFeatures

    base = Trainer(_small_cfg(), tmp_path / "base", log=None)
    base.step()
    ckpt = base.save()
    feat = _facts_dir(tmp_path, vocab)
    assert TextFeatures.load(feat, vocab).has_facts
    grown = replace(
        _small_cfg(), init_from=str(ckpt), text_dir=str(feat), net={**TINY, "id_dropout": 0.2, "card_facts": True}
    )
    trainer = Trainer(grown, tmp_path / "grown", log=None)
    assert trainer.net_config.n_archetypes == 1 and trainer.net_config.id_dropout == 0.2
    old = dict(base.model.actor.named_parameters())
    for name, p in trainer.model.actor.named_parameters():
        if name in old:
            assert torch.equal(p, old[name]), name
    # the new views start at zero: the grown actor plays exactly like the checkpoint (eval mode, no dropout)
    ro = base.collector.collect()
    trainer.model.actor.eval()
    base.model.actor.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer.model.actor(ro.obs).logits, base.model.actor(ro.obs).logits)
    trainer.model.actor.train()
    assert trainer.step()["rows"] == 16
    with pytest.raises(ValueError):
        Trainer(replace(grown, net={**TINY, "d_model": 32}), tmp_path / "bad", log=None)


def test_a_checkpoint_with_card_facts_reloads_them(tmp_path, vocab):
    from ygorl.train.checkpoint import load_actor

    feat = tmp_path / "features"
    feat.mkdir()
    pws = [vocab.password(i) for i in range(2, len(vocab))]
    n = len(pws)
    np.savez_compressed(feat / "card_facts.npz", passwords=np.asarray(pws, np.int64),
                        setcodes=np.tile([[0x1, 0, 0, 0]], (n, 1)), references=np.zeros((n, 2), np.int64),
                        categories=np.zeros(n, np.uint64), queries=np.zeros((n, 3), np.uint8),
                        query_names=np.array(["a", "b", "c"]))  # fmt: skip
    trainer = Trainer(replace(_small_cfg(), text_dir=str(feat), net={**TINY, "card_facts": True}), tmp_path / "run",
                      log=None)  # fmt: skip
    path = trainer.save()
    pol = load_actor(path)  # the run's text_dir comes from its config
    assert pol.net_config.n_archetypes == 1 and pol.net.identity.archetype is not None


def test_engine_errors_are_logged_for_replay(tmp_path):
    from ygorl.train.rollout import Assignment, FinishedGame

    trainer = Trainer(_small_cfg(), tmp_path, log=None)
    spec = GameSpec(seed=42, deck_a=load_ydk(PAIR[0]), deck_b=load_ydk(PAIR[1]), first=1)
    bad = FinishedGame(Assignment(spec, None, 0, {"deck_a": "snake_eye"}), None, "error", True, 3,
                       {"error": "engine loop: no decision after 100000 engine steps", "decisions": 7,
                        "responses": [b"\x01\x00", b"\x02"]})  # fmt: skip
    ok = FinishedGame(Assignment(spec, None, 0, {}), 0, "win", False, 3, {})
    trainer._log_errors([ok, bad])
    rows = [json.loads(line) for line in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["seed"] == 42 and rows[0]["first"] == 1
    assert rows[0]["responses"] == ["0100", "02"] and "engine loop" in rows[0]["error"]


def test_real_lua_error_truncates_rollout_without_a_win_reward(db, vocab, tmp_path):
    from ygorl import _core, paths
    from ygorl.engine.duel import default_scripts
    from ygorl.train.rollout import Assignment

    base = default_scripts()
    suffix = b"""\nlocal old_initial_effect=s.initial_effect
        function s.initial_effect(c)
            old_initial_effect(c)
            local e=Effect.CreateEffect(c)
            e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
            e:SetCode(EVENT_PHASE_START+PHASE_DRAW)
            e:SetOperation(function() if Duel.GetTurnCount()==2 then error("rollout Lua regression") end end)
            Duel.RegisterEffect(e,0)
        end"""
    scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()],
                                   {"c14558127.lua": base.read("c14558127.lua") + suffix,
                                    **{n: base.read(n) for n in ("proc_fusion.lua", "proc_synchro.lua")}})  # fmt: skip
    deck = load_ydk(DECK_DIR / "branded_despia.ydk")
    spec = GameSpec(0, deck, deck)
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, scripts=scripts, privileged=True, event_length=16)
    col = RolloutCollector(env, tiny_model(vocab), lambda: Assignment(spec), 256, seed=5)
    ro = col.collect()
    errors = [g for g in ro.games if g.result.get("script_errors")]
    assert errors and all(g.reason == "error" and g.truncated and g.winner is None for g in errors)
    assert (ro.rewards == 0).all() and (ro.truncated == ro.dones).all()
    trainer = Trainer(_small_cfg(), tmp_path, log=None)
    trainer._log_errors(errors)
    rows = [json.loads(line) for line in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert len(rows) == len(errors) and all("rollout Lua regression" in r["script_errors"][0] for r in rows)
    assert all(r["retries"] == r["unknown_messages"] == 0 for r in rows)


def test_a_stalled_rollout_saves_a_checkpoint_and_propagates(tmp_path):
    """Training cannot unwind past a stuck engine thread, so train() saves latest.pt and re-raises for the caller
    to end the process (tools/train_ppo.py exits with os._exit(3))."""
    from ygorl.train.rollout import RolloutStalled

    trainer = Trainer(_small_cfg(), tmp_path, log=None)
    trainer.step()

    def stuck():
        raise RolloutStalled("no environment event for 900s")

    trainer.collector.collect = stuck
    with pytest.raises(RolloutStalled):
        trainer.train(max_updates=3)
    state = load_checkpoint(tmp_path / "checkpoints" / "latest.pt")
    assert state["counters"]["updates"] == 1


def test_critic_warmup_freezes_the_policy_then_lets_it_train(tmp_path):
    """--critic-warmup: the first updates train the critic only; the actor, shared trunk included, does not move."""
    t = Trainer(replace(_small_cfg(), critic_warmup=2, critic_warmup_ev=2.0), tmp_path / "w", log=None)  # EV never met
    actor0 = {k: v.clone() for k, v in t.model.actor.state_dict().items()}
    critic0 = {k: v.clone() for k, v in t.model.critic.state_dict().items()}
    t.step()
    assert all(torch.equal(v, actor0[k]) for k, v in t.model.actor.state_dict().items())
    assert any(not torch.equal(v, critic0[k]) for k, v in t.model.critic.state_dict().items())
    assert t.counters["critic_warmup_done"] == 0
    t.step()
    assert t.counters["critic_warmup_done"] == 2  # the cap
    assert all(torch.equal(v, actor0[k]) for k, v in t.model.actor.state_dict().items())
    t.step()  # the policy trains now
    assert any(not torch.equal(v, actor0[k]) for k, v in t.model.actor.state_dict().items())


def test_critic_warmup_ends_early_once_the_critic_explains_enough(tmp_path):
    t = Trainer(replace(_small_cfg(), critic_warmup=50, critic_warmup_ev=-1e9), tmp_path / "w", log=None)
    for _ in range(5):
        t.step()
    assert t.counters["critic_warmup_done"] == 5  # the 5-update average is reached at the 5th update


def test_turn_discount_scales_the_terminal_reward_by_game_length(tmp_path):
    """--turn-discount G (design T6, a diagnostic arm): a decided game ends in +/- G ** turns instead of +/- 1."""
    t = Trainer(replace(_small_cfg(), turn_discount=0.9, steps=400, max_decisions=3000), tmp_path / "d", log=None)
    seen = []
    for _ in range(20):
        ro = t._collect()
        for g in ro.games:
            if g.winner is not None and not g.truncated:
                seen.append(0.9 ** int(g.result["turns"]))
        rewards = ro.rewards[ro.rewards != 0].abs().tolist()
        assert all(any(abs(r - v) < 1e-6 for v in (0.9**k for k in range(1, 200))) for r in rewards)
        if len(seen) >= 2:
            break
    assert seen and all(v < 1 for v in seen)
    with pytest.raises(ValueError, match="turn_discount"):
        Trainer(replace(_small_cfg(), turn_discount=0.0), tmp_path / "bad", log=None)


def test_unhealthy_evaluation_cannot_create_or_replace_best(tmp_path, monkeypatch):
    from ygorl.eval.arena import GameRecord, summarize
    from ygorl.train import trainer as module

    trainer = Trainer(_small_cfg(eval_opponents=("random",), keep_best_by="random"), tmp_path, log=None)
    good = GameRecord(pair=0, first=0, seed=0, winner=0, reason="win")
    bad = replace(good, pair=1, script_errors=1)

    def report(records):
        return summarize(records, agent_a="a", agent_b="b", deck_a="x", deck_b="y", seed=0), 0.1

    monkeypatch.setattr(module, "evaluate_checkpoint", lambda *a, **kw: report([good, bad]))
    result = trainer.evaluate()
    assert trainer.best["score"] is None and trainer.pool.best_id is None and not (tmp_path / "best.pt").exists()
    assert not result["random"]["valid"] and result["random"]["attempted_games"] == 2
    failure = json.loads((tmp_path / "eval-errors.jsonl").read_text().splitlines()[0])
    assert failure["report"]["records"][1]["script_errors"] == 1
    monkeypatch.setattr(module, "evaluate_checkpoint", lambda *a, **kw: report([replace(good, winner=1)]))
    trainer.evaluate()
    before = (tmp_path / "best.pt").read_bytes()
    state = dict(trainer.best)
    pinned = trainer.pool.best_id
    monkeypatch.setattr(module, "evaluate_checkpoint", lambda *a, **kw: report([good, bad]))
    trainer.evaluate()
    assert (tmp_path / "best.pt").read_bytes() == before and trainer.best == state and trainer.pool.best_id == pinned
    assert json.loads((tmp_path / "eval.jsonl").read_text().splitlines()[-1])["best"] is False


def test_another_invalid_baseline_blocks_an_otherwise_best_score(tmp_path, monkeypatch):
    from ygorl.eval.arena import GameRecord, summarize
    from ygorl.train import trainer as module

    trainer = Trainer(_small_cfg(), tmp_path, log=None)

    def evaluate(path, decks, opponent, **kw):
        rows = [GameRecord(pair=0, first=0, seed=0, winner=0, reason="win")] if opponent == "greedy" else []
        return summarize(rows, agent_a="a", agent_b=opponent, deck_a="x", deck_b="y", seed=0), 0.1

    monkeypatch.setattr(module, "evaluate_checkpoint", evaluate)
    result = trainer.evaluate()
    assert result["greedy"]["valid"] and result["greedy"]["win_rate"] == 1
    assert not result["random"]["valid"] and result["random"]["attempted_games"] == 0
    assert trainer.best["score"] is None and trainer.pool.best_id is None and not (tmp_path / "best.pt").exists()


def test_policy_kl_guard_does_not_block_a_frozen_actor_critic_warmup(tmp_path):
    trainer = Trainer(_small_cfg(), tmp_path, log=None)
    ro = trainer.collector.collect()
    ro.log_probs -= 1  # stale behavior likelihoods do not prevent fitting the critic with a frozen actor
    before = {k: v.clone() for k, v in trainer.model.actor.state_dict().items()}
    stats = trainer.learner.update(ro, policy=False)
    assert stats["minibatches"] == stats["evaluated_minibatches"] == 1 and stats["early_stop"] == 0
    assert stats["approx_kl"] > 1.5 * trainer.cfg.ppo.target_kl and stats["stop_approx_kl"] == 0
    assert all(torch.equal(v, trainer.model.actor.state_dict()[k]) for k, v in before.items())
