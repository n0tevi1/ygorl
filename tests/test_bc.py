"""Behaviour cloning from solver demonstrations (T4a.2): observations, training, openings, checkpoints.

``tests/data/bc_demo.jsonl`` is one real record of ``tools/solve_openings.py`` (branded_despia, hand 10,
one verified 14-step line), so everything runs without the solver binary.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.agents import PolicyAgent, RandomAgent, make_agent  # noqa: E402
from ygorl.agents.registry import AgentSpec  # noqa: E402
from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import Deck, load_ydk  # noqa: E402
from ygorl.engine.duel import Duel, DuelConfig  # noqa: E402
from ygorl.env import GameSpec  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv, chooser  # noqa: E402
from ygorl.env.observer import PointObserver  # noqa: E402
from ygorl.eval.arena import Arena  # noqa: E402
from ygorl.nets import NetConfig, PolicyNet  # noqa: E402
from ygorl.nets.agent import NetPolicy, load_checkpoint, save_checkpoint  # noqa: E402
from ygorl.nets.batch import to_tensors  # noqa: E402
from ygorl.solver import read_jsonl  # noqa: E402
from ygorl.env.encoding import canonical_action  # noqa: E402
from ygorl.train.bc import (  # noqa: E402
    action_key,
    BCConfig,
    build_dataset,
    demo_player_actions,
    demo_player_keys,
    hand_overlap,
    opening_report,
    play_opening,
    start_replay,
    step_accuracy,
    train_bc,
    trim_padding,
    undone_steps,
)
from ygorl.train.bc import DemoStep  # noqa: E402

HERE = Path(__file__).parent
DECKS = {p.stem: load_ydk(p) for p in sorted((HERE / "decks").glob("*.ydk"))}
DEMO_FILE = HERE / "data" / "bc_demo.jsonl"


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


@pytest.fixture(scope="module")
def demo():
    (d,) = read_jsonl(DEMO_FILE)
    return d


@pytest.fixture(scope="module")
def data(demo, vocab, db):
    return build_dataset([demo], vocab, cards=db, event_length=32)


def tiny_net(vocab, **kw) -> PolicyNet:
    torch.manual_seed(0)
    cfg = NetConfig(vocab_size=len(vocab), d_model=32, n_heads=2, board_layers=1, history_layers=1, **kw)
    return PolicyNet(cfg)


class Replayer:
    """An agent that plays a fixed action list."""

    def __init__(self, actions):
        self.actions = list(actions)

    def act(self, point):
        return self.actions.pop(0)


# ------------------------------------------------------------------ observations


def test_duel_run_shows_every_point_to_observing_agents():
    seen, acted = [], []

    class Watcher(RandomAgent):
        def __init__(self, seed, tag):
            super().__init__(seed)
            self.tag = tag

        def observe(self, point, core):
            assert core is not None
            seen.append((self.tag, point.index))

        def act(self, point):
            acted.append(point.index)
            assert (self.tag, point.index) in seen  # observed before acting
            return super().act(point)

    cfg = DuelConfig(max_turns=3)
    Duel(5, None, DECKS["snake_eye"], DECKS["yubel"], config=cfg).run(Watcher(1, "a"), Watcher(2, "b"))
    for tag in "ab":  # both seats see every point, once, in order
        assert [i for t, i in seen if t == tag] == acted
    seen.clear(), acted.clear()
    same = Watcher(3, "x")
    Duel(5, None, DECKS["snake_eye"], DECKS["yubel"], config=cfg).run(same, same)
    assert [i for _, i in seen] == acted  # an agent in both seats is shown each point once


def test_point_observer_matches_the_cpp_environment(db, vocab):
    """PointObserver on Duel.run == EncodedVecEnv (C++ tracker, encoder and event stream), step by step."""
    spec = GameSpec(seed=17, deck_a=DECKS["snake_eye"], deck_b=DECKS["tenpai"], config=DuelConfig(max_turns=4))
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, event_length=48)
    env.reset(0, spec)
    observer = PointObserver(db, vocab, 48)
    steps = []

    class Lockstep:
        def observe(self, point, core):
            self.core = core

        def act(self, point):
            (ev,) = env.recv(1)
            mine = observer.encode(point, self.core)
            assert set(mine) == set(ev.obs)
            for key, value in ev.obs.items():
                np.testing.assert_array_equal(mine[key], value, err_msg=f"{key} at decision {point.index}")
            idx = chooser(17, len(steps), len(point.actions))
            env.step(0, idx)
            steps.append(idx)
            return idx

    agent = Lockstep()
    Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, config=spec.config).run(agent, agent)
    assert len(steps) > 30


# ------------------------------------------------------------------ dataset and start positions


def test_dataset_holds_the_decisions_of_the_deck_under_study(demo, data):
    line = demo.lines[0]
    ours = [(s, a) for s, (a, p) in enumerate(zip(line.actions, line.players)) if p == 0]
    assert len(data) + sum(data.skipped.values()) == len(ours)
    assert all(m["solver"] for m in data.meta)
    kept = {m["step"] for m in data.meta}
    raw = [a for s, a in ours if s in kept]
    # a demonstrated copy other than the first maps to the row the policy can choose (docs/encoding.md)
    labels = [canonical_action({k: v[i] for k, v in data.obs.items()}, a) for i, a in enumerate(raw)]
    assert [int(a) for a in data.actions] == labels and labels != raw  # this line does pick later copies
    assert all(data.obs["action_mask"][i, a] for i, a in enumerate(data.actions))
    assert all(m["n_choices"] > 1 and m["n_legal"] >= m["n_choices"] for m in data.meta)
    assert data.obs["events"].shape[1:] == (32, 20) and data.obs["cards"].shape[1:] == (160, 23)


def test_select_unselect_toggles_are_left_out():
    def st(i, kind, card, player=0):
        return DemoStep(i, player, 0, 2, kind, card)

    x, y = (1, "hand"), (2, "hand")
    steps = [st(0, "activate", x), st(1, "select", x), st(2, "unselect", x), st(3, "select", x), st(4, "unselect", x),
             st(5, "select", y), st(6, "unselect", x), st(7, "select", x), st(8, "unselect", x, player=1),
             st(9, "select", y), st(10, "finish", None)]  # fmt: skip
    assert undone_steps(steps) == {1, 2, 3, 4}
    assert undone_steps([st(0, "select", x), st(1, "unselect", x), st(2, "unselect", x)]) == {0, 1}


def test_start_replay_rebuilds_the_solver_start(demo):
    recorded = start_replay(demo)
    rebuilt = start_replay(dataclasses.replace(demo, start=None, lines=[], status="unsolved"))
    for key in ("decks", "seed_words", "rule_flags", "player", "shuffle_decks", "first"):
        assert getattr(rebuilt, key) == getattr(recorded, key), key
    assert recorded.responses == [] and demo.lines[0].responses  # the record is not modified
    with pytest.raises(ValueError, match="does not give the recorded hand"):
        start_replay(dataclasses.replace(demo, start=None, lines=[], hand=list(reversed(demo.hand))))


def test_play_opening_scores_the_demonstrated_line(demo, db):
    (line,) = demo_player_actions(demo)
    res = play_opening(start_replay(demo), Replayer(line), demo.targets, cards=db)
    assert res.actions == line and res.reached and res.placed == res.targets == 1 and not res.capped
    assert res.keys == demo_player_keys(demo)[0]
    assert res.board == demo.lines[0].board
    capped = play_opening(start_replay(demo), RandomAgent(0), demo.targets, cards=db, max_steps=2)
    assert len(capped.actions) == 2 and capped.capped and capped.board["turn"] == 2
    report = opening_report([demo], lambda i: Replayer(line), cards=db)
    total = report["totals"]["total"]
    assert total["hands"] == 1 and total["reproduction_rate"] == 1.0 and total["reach_rate"] == 1.0
    assert total["mean_prefix"] == 1.0 and report["totals"]["branded_despia"]["solver_solved"] == 1
    assert hand_overlap([demo], [demo]) == 1


# ------------------------------------------------------------------ training


def test_trim_padding_does_not_change_the_logits(data, vocab):
    for history in ("transformer", "lstm"):
        net = tiny_net(vocab, history=history).eval()
        idx = np.arange(len(data))
        full = to_tensors({k: v[idx] for k, v in data.obs.items()})
        trimmed, _ = data.batch(idx)
        assert trimmed["cards"].shape[1] < 160 and trimmed["actions"].shape[1] < 128
        with torch.no_grad():
            a, b = net(full).logits, net(trimmed).logits
        k = b.shape[1]
        legal = full["action_mask"][:, :k]
        torch.testing.assert_close(a[:, :k][legal], b[legal])
        assert not full["action_mask"][:, k:].any()
    small = trim_padding({k: v[:1] for k, v in data.obs.items()})
    assert small["action_mask"].shape[1] == int(data.obs["action_mask"][0].sum())


def test_bc_fits_a_line_and_the_checkpoint_replays_it(demo, data, vocab, db, tmp_path):
    net = tiny_net(vocab)
    before = step_accuracy(net, data)
    history = train_bc(net, data, BCConfig(epochs=60, batch_size=16, lr=3e-3, warmup_steps=5, seed=0))
    after = step_accuracy(net, data)
    assert history[-1]["loss"] < history[0]["loss"] and after["nll"] < before["nll"]
    assert after["accuracy"] == 1.0 and 0 < after["uniform_accuracy"] < 1

    # greedy free-running play reproduces the demonstrated line and reaches the target
    agent = PolicyAgent(NetPolicy(net, vocab, event_length=32, cards=db), greedy=True)
    res = play_opening(start_replay(demo), agent, demo.targets, cards=db)
    assert res.keys == demo_player_keys(demo)[0] and res.reached  # the same choices (copies may differ by index)

    # the checkpoint round-trips and plays through the agent registry and the arena
    path = save_checkpoint(tmp_path / "policy.pt", net, vocab, event_length=32, meta={"trainer": "bc"})
    ckpt = load_checkpoint(path)
    assert ckpt.event_length == 32 and ckpt.meta["trainer"] == "bc" and ckpt.environment is None
    idx = np.arange(len(data))
    obs, _ = data.batch(idx)
    with torch.no_grad():
        torch.testing.assert_close(ckpt.net(obs).logits, net.eval()(obs).logits)
    agent = make_agent(f"policy:{path}@greedy", seed=0)
    res = play_opening(start_replay(demo), agent, demo.targets, cards=db)
    assert res.keys == demo_player_keys(demo)[0]
    cfg = DuelConfig(max_turns=4)
    result = Duel(3, None, DECKS["branded_despia"], DECKS["yubel"], config=cfg).run(make_agent(f"policy:{path}", 1),
                                                                                   RandomAgent(2))  # fmt: skip
    assert result.reason in ("win", "turn_limit") and result.decisions > 10
    report = Arena(AgentSpec(f"policy:{path}"), AgentSpec("random"), config=cfg).run(
        DECKS["branded_despia"], DECKS["branded_despia"], pairs=1)  # fmt: skip
    assert report.games == 2 and report.errors == 0


def test_policy_specs_are_checked(tmp_path):
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_agent("policy")
    with pytest.raises(ValueError, match="no policy checkpoint"):
        make_agent(f"policy:{tmp_path / 'missing.pt'}")
    (tmp_path / "x.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="unknown policy option"):
        make_agent(f"policy:{tmp_path / 'x.pt'}@fast")
    torch.save({"format": "other"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a ygorl policy checkpoint"):
        load_checkpoint(tmp_path / "other.pt")


def test_action_key_is_the_same_for_equivalent_copies(db):
    """Five Maxx "C" in the hand: their chain rows share a key (one choice), passing has another."""
    maxx, celtic = 23434538, 91152256
    seen = []

    class Passive:
        def act(self, point):
            seen.append(point)
            kinds = [a.kind for a in point.actions]
            return next((kinds.index(k) for k in ("pass", "end_phase", "no") if k in kinds), 0)

    duel = Duel(1, None, Deck(main=(celtic,) * 40), Deck(main=(maxx,) * 40), cards=db,
                config=DuelConfig(max_turns=1, shuffle_decks=False))  # fmt: skip
    duel.run(Passive(), Passive())
    point = next(p for p in seen if p.player == 1 and sum(a.kind == "chain" for a in p.actions) == 5)
    keys = [action_key(point, i) for i in range(len(point.actions))]
    assert len(set(keys[:5])) == 1 and keys[5] != keys[0] and point.actions[5].kind == "pass"


# ------------------------------------------------------------------ heuristic demonstrations beyond turn 1


def test_greedy_demonstrations_record_later_turns(db, vocab, data, tmp_path):
    """DemoRecorder: Greedy's non-forced decisions from turn 2 on, as the observations NetPolicy would see."""
    from ygorl.agents import GreedyAgent
    from ygorl.train.heuristic_demos import (
        DemoRecorder,
        concat,
        is_battle_decision,
        load_data,
        record_games,
        save_data,
        select,
    )

    cfg = DuelConfig(max_turns=5)
    arena = Arena(AgentSpec("greedy"), AgentSpec("random"), config=cfg)
    specs = arena.game_specs(DECKS["snake_eye"], DECKS["kashtira"], pairs=1, seed=3)  # both seats
    got, games = record_games(specs, GreedyAgent, vocab, event_length=32)
    assert len(games) == 2 and not any("error" in g for g in games) and {g["first"] for g in games} == {0, 1}
    assert len(got) > 20 and got.skipped["early_turn"] > 0 and got.skipped["forced"] > 0
    assert all(m["turn"] >= 2 and m["n_choices"] >= 2 for m in got.meta)
    assert all(got.obs["action_mask"][i, a] for i, a in enumerate(got.actions))  # labels are choosable rows
    assert (got.obs["globals"][:, 3] >= 2).all() and got.obs["events"].shape[1:] == (32, 20)
    assert any(m["kind"] == "attack" for m in got.meta) and any(m["kind"] == "battle_phase" for m in got.meta)

    # the recorder plays exactly like the wrapped agent
    spec = specs[0]
    plain = spec.duel().run(GreedyAgent(spec.agent_seeds[0]), spec.agent_b(spec.agent_seeds[1]))
    rec = DemoRecorder(GreedyAgent(spec.agent_seeds[0]), db, vocab, event_length=32)
    recorded = spec.duel().run(rec, spec.agent_b(spec.agent_seeds[1]))
    assert (recorded.winner, recorded.turns, recorded.decisions) == (plain.winner, plain.turns, plain.decisions)
    assert len(rec.actions) == games[0]["samples"]

    battle = select(got, "battle")
    assert 0 < len(battle) < len(got) and all(is_battle_decision(m) for m in battle.meta)
    assert all(m["own_turn"] for m in battle.meta)
    assert len(select(got, "all", max_samples=5, seed=1)) == 5
    with pytest.raises(ValueError):
        select(got, "chains")

    both = concat([data, got])  # solver turn-1 samples (event length 32) + heuristic samples
    assert len(both) == len(data) + len(got) and both.meta[len(data)]["source"] == "heuristic"
    back, info = load_data(save_data(tmp_path / "g.npz", got, note="x"))
    assert info == {"note": "x"} and back.meta == got.meta and np.array_equal(back.actions, got.actions)
    for k, v in got.obs.items():
        np.testing.assert_array_equal(back.obs[k], v)
    longer, _ = record_games(specs[:1], GreedyAgent, vocab, event_length=48)
    with pytest.raises(ValueError, match="event length"):
        concat([got, longer])


@pytest.mark.parametrize("include_closing", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")),
    ],
)
def test_train_bc_fine_tunes_an_existing_actor(vocab, tmp_path, include_closing, device):
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_bc", HERE.parent / "tools" / "train_bc.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    import json

    base = save_checkpoint(tmp_path / "base.pt", tiny_net(vocab), vocab, event_length=24)
    out = tmp_path / "ft"
    flags = ["--include-synthetic-closing"] if include_closing else []
    assert tool.main(["--train", str(DEMO_FILE), "--heldout", str(DEMO_FILE), "--init-from", str(base),
                      "--epochs", "1", "--openings", "heldout", "--out", str(out), "--threads", "1", "--device", device, *flags]) == 0  # fmt: skip
    ckpt = load_checkpoint(out / "policy.pt")
    # the base network, vocab and event window carry over (--d-model etc. are ignored), and the weights moved
    assert ckpt.event_length == 24 and ckpt.net.cfg == load_checkpoint(base).net.cfg
    policy = {"include_synthetic_closing": include_closing, "full_replay_validation": True}
    report = json.loads((out / "report.json").read_text())
    assert ckpt.meta["data_policy"] == report["data_policy"] == policy
    assert ckpt.meta["training_device"] == report["device"]
    assert report["device"]["actual"].startswith(device) and report["opening_device"] == "cpu"
    assert next(ckpt.net.parameters()).device.type == "cpu"
    assert report["openings"]["bc"]["heldout"]["totals"]["total"]["hands"] == 1
    assert report["data"]["train"]["samples"] == report["data"]["heldout"]["samples"]
    assert bool(report["data"]["train"]["skipped"].get("synthetic_closing")) != include_closing
    before = load_checkpoint(base).net.state_dict()
    assert any(not torch.equal(v, before[k]) for k, v in ckpt.net.state_dict().items())


def test_extra_bc_data_checks_environment_vocab_and_window(data, vocab, tmp_path):
    from ygorl.nets.agent import vocab_passwords
    from ygorl.train.heuristic_demos import data_identity, load_compatible_data, save_data

    stamp = {"version": "test", "fingerprint": "revision-a"}
    path = save_data(tmp_path / "extra.npz", data, identity=data_identity(vocab, 32, stamp))
    got, _ = load_compatible_data(path, vocab=vocab, event_length=32, environment=stamp)
    np.testing.assert_array_equal(got.actions, data.actions)
    # Same shape and vocabulary size, different card indices: silently mixing these corrupts observations.
    passwords = vocab_passwords(vocab)
    passwords[0], passwords[1] = passwords[1], passwords[0]
    with pytest.raises(ValueError, match="vocab_sha256 mismatch"):
        load_compatible_data(path, vocab=CardVocab(passwords), event_length=32, environment=stamp)
    for wrong in [None, {"version": "test", "fingerprint": "revision-b"}]:
        with pytest.raises(ValueError, match="environment mismatch"):
            load_compatible_data(path, vocab=vocab, event_length=32, environment=wrong)
    with pytest.raises(ValueError, match="event_length mismatch"):
        load_compatible_data(path, vocab=vocab, event_length=64, environment=stamp)
    stale = save_data(tmp_path / "stale.npz", data, identity=data_identity(vocab, 64, stamp))
    with pytest.raises(ValueError, match="encoded observations"):
        load_compatible_data(stale, vocab=vocab, event_length=64, environment=stamp)
    legacy = save_data(tmp_path / "legacy.npz", data)
    with pytest.raises(ValueError, match="missing BC data identity"):
        load_compatible_data(legacy, vocab=vocab, event_length=32)


def test_heuristic_samples_from_engine_error_games_are_discarded(vocab, monkeypatch):
    from ygorl.agents import GreedyAgent
    from ygorl.train.heuristic_demos import record_games

    arena = Arena(AgentSpec("greedy"), AgentSpec("random"), config=DuelConfig(max_turns=5))
    specs = arena.game_specs(DECKS["snake_eye"], DECKS["kashtira"], pairs=1, seed=3)
    run = Duel.run

    def failed_result(self, a, b):
        # The engine reports budget errors as results, without throwing a Python exception.
        result = run(self, a, b)
        assert a.actions  # partial recorded samples must be dropped along with the failed game
        return dataclasses.replace(result, winner=None, reason="error", error="script budget exceeded")

    monkeypatch.setattr(Duel, "run", failed_result)
    data, games = record_games(specs[:1], GreedyAgent, vocab, event_length=32)
    assert len(data) == 0 and data.skipped["error_games"] == 1
    assert games[0]["samples"] == 0 and games[0]["error"] == "script budget exceeded"


@pytest.mark.parametrize("flag", ["--extra", "--extra-heldout"])
def test_train_bc_rejects_unidentified_extra_data(data, tmp_path, flag):
    import importlib.util

    from ygorl.train.heuristic_demos import save_data

    spec = importlib.util.spec_from_file_location("train_bc", HERE.parent / "tools" / "train_bc.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    path = save_data(tmp_path / "legacy.npz", data)
    with pytest.raises(ValueError, match="missing BC data identity"):
        tool.main(["--train", str(DEMO_FILE), flag, str(path), "--event-length", "32",
                   "--epochs", "1", "--openings", "none", "--out", str(tmp_path / "run")])  # fmt: skip
    assert not (tmp_path / "run" / "policy.pt").exists()


def test_train_bc_seed_controls_initialization_and_training(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_bc", HERE.parent / "tools" / "train_bc.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    states = []
    for run, seed in enumerate([17, 17, 18]):
        torch.manual_seed(900 + run)  # unrelated caller RNG must not change this command's initialization
        out = tmp_path / str(run)
        assert tool.main(["--train", str(DEMO_FILE), "--seed", str(seed), "--epochs", "1",
                          "--d-model", "32", "--layers", "1", "--event-length", "24", "--no-text",
                          "--openings", "none", "--out", str(out), "--threads", "1"]) == 0  # fmt: skip
        states.append(load_checkpoint(out / "policy.pt").net.state_dict())
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())
    assert any(not torch.equal(v, states[2][k]) for k, v in states[0].items())


@pytest.mark.parametrize("wrong_fingerprint", [False, True])
def test_train_bc_checks_bound_demo_revision_and_legality(demo, tmp_path, wrong_fingerprint):
    import importlib.util

    from ygorl.data.environment import load_environment

    spec = importlib.util.spec_from_file_location("train_bc", HERE.parent / "tools" / "train_bc.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    env_path = HERE.parent / "environments" / "md-2026-09"
    env = load_environment(env_path)
    stamp = {"version": env.version, "fingerprint": "wrong" if wrong_fingerprint else env.fingerprint}
    path = tmp_path / "relabeled.jsonl"
    dataclasses.replace(demo, environment=stamp).append_to(path)
    message = "pass the same --env" if wrong_fingerprint else "illegal demonstration deck"
    with pytest.raises(SystemExit, match=message):
        tool.main(["--train", str(path), "--env", str(env_path), "--out", str(tmp_path / "run"),
                   "--epochs", "1", "--openings", "none"])  # fmt: skip
    assert not (tmp_path / "run" / "policy.pt").exists()


def test_unsolved_opening_retains_recorded_shuffle_and_start(demo):
    from copy import deepcopy
    from ygorl.engine import constants as C

    start = deepcopy(demo.start)
    start["rule_flags"] &= ~C.DUEL_PSEUDO_SHUFFLE
    start["core_seed"] = [4, 3, 2, 1]
    start["decks"]["a"]["main"].reverse()
    unsolved = dataclasses.replace(demo, start=start, lines=[], status="unsolved")
    rep = start_replay(unsolved)
    assert rep.rule_flags == start["rule_flags"]
    assert not rep.rule_flags & C.DUEL_PSEUDO_SHUFFLE
    assert rep.core_seed == start["core_seed"] and not rep.responses
    assert rep.decks["a"]["main"] == start["decks"]["a"]["main"]
    assert demo.lines[0].responses  # no mutation of the solved source


def test_synthetic_closing_is_not_an_expert_label_but_is_still_verified(vocab, db):
    import copy

    from ygorl.data import load_environment
    from ygorl.solver import DemoError
    from ygorl.train.bc import line_steps

    (demo,) = read_jsonl(HERE / "data/bc_synthetic_closing.jsonl")
    env = load_environment("md-2026-09")
    original = copy.deepcopy(demo.to_json())
    old = build_dataset([demo], vocab, cards=db, env=env, event_length=32, include_synthetic_closing=True)
    data = build_dataset([demo], vocab, cards=db, env=env, event_length=32)
    steps = line_steps(demo, 0, cards=db, env=env)
    # The actual bad label declined The Man with the Mark's free search.
    declined = next(s for s in steps if s.kind == "no" and s.card and s.card[0] == 97522863)
    assert any(m["step"] == declined.step for m in old.meta)
    assert all(m["step"] != declined.step and m["solver"] for m in data.meta)
    idx = np.asarray([i for i, m in enumerate(old.meta) if m["solver"]])
    np.testing.assert_array_equal(data.actions, old.actions[idx])
    for key, value in data.obs.items():
        np.testing.assert_array_equal(value, old.obs[key][idx])
    assert data.skipped["synthetic_closing"] == len(old) - len(data) > 0
    assert demo.to_json() == original

    # Damage only a discarded closing label. It must still fail full validation.
    broken = copy.deepcopy(demo)
    broken.lines[0].responses[-1] = b"wrong response"
    with pytest.raises(DemoError, match="responses"):
        build_dataset([broken], vocab, cards=db, env=env, event_length=32)
    broken = copy.deepcopy(demo)
    broken.lines[0].solver_steps = len(broken.lines[0].actions) + 1
    with pytest.raises(DemoError, match="solver_steps"):
        build_dataset([broken], vocab, cards=db, env=env, event_length=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")
def test_bc_gpu_updates_and_cpu_checkpoint_agree(data, vocab, tmp_path):
    net = tiny_net(vocab).cuda()
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    history = train_bc(
        net, data, BCConfig(epochs=2, batch_size=16, lr=1e-3, warmup_steps=0), eval_sets={"heldout": data}
    )
    assert all(np.isfinite(h["loss"]) and np.isfinite(h["heldout_loss"]) for h in history)
    assert any(not torch.equal(v, before[k]) for k, v in net.state_dict().items())
    gpu = step_accuracy(net, data)
    assert next(net.parameters()).is_cuda
    path = save_checkpoint(tmp_path / "gpu.pt", net, vocab, event_length=32)
    cpu = load_checkpoint(path)
    assert next(cpu.net.parameters()).device.type == "cpu"
    got = step_accuracy(cpu.net, data)
    assert got["accuracy"] == gpu["accuracy"]
    assert got["nll"] == pytest.approx(gpu["nll"], rel=1e-4, abs=1e-5)
