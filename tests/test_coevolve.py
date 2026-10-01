"""Co-evolution driver (ygorl.build.coevolve, tools/coevolve.py, #151): stage order, resume after interruption, and
the real stages' plumbing (an evolution state's accepted children; a training segment that logs its games)."""

import importlib.util
import json
from pathlib import Path

import pytest

from ygorl.build.coevolve import CoEvolution, CoevoConfig, EvolveResult, accepted_children, check_round

TOOL = Path(__file__).parents[1] / "tools" / "coevolve.py"


class FakeStages:
    """Stand-in stages: cycle 1 accepts a child of deck slot ``a``; ``fail`` = {stage key: times to fail first}."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = dict(fail or {})

    def _maybe_fail(self, key):
        if self.fail.get(key, 0) > 0:
            self.fail[key] -= 1
            raise RuntimeError(f"machine reclaimed during {key}")

    def evolve(self, cycle, checkpoint, value_model, parents):
        self.calls.append(("evolve", cycle, checkpoint, value_model, dict(parents)))
        self._maybe_fail(f"evolve-{cycle}")
        acc = [{"slot": "a", "parent": "corpus:a", "id": "evo-1", "file": "evo/decks/evo-1.ydk", "diff": 0.02}]
        return EvolveResult(cycle, 1000, acc if cycle == 1 else [])

    def train(self, cycle, checkpoint, until):
        self.calls.append(("train", cycle, checkpoint, until))
        self._maybe_fail(f"train-{cycle}")
        return f"run/checkpoints/update_{until:06d}.pt"

    def fit(self, cycle, checkpoint):
        self.calls.append(("fit", cycle, checkpoint))
        self._maybe_fail(f"fit-{cycle}")
        return f"vm/cycle_{cycle:02d}.pt"


CFG = CoevoConfig(cycles=3, updates=50, start_update=400, decks={"a": "a.ydk", "b": "b.ydk"})


def test_cycles_run_evolve_train_fit_and_pass_the_latest_policy_model_and_decks(tmp_path):
    st = FakeStages()
    summary = CoEvolution(tmp_path, st, CFG, "base/checkpoints/update_000400.pt").run()
    kinds = [c[:2] for c in st.calls]
    assert kinds == [("fit", 0), ("evolve", 1), ("train", 1), ("fit", 1), ("evolve", 2), ("train", 2), ("fit", 2),
                     ("evolve", 3), ("train", 3), ("fit", 3)]  # fmt: skip
    ev1, tr1, ev2 = st.calls[1], st.calls[2], st.calls[4]
    assert ev1[2] == "base/checkpoints/update_000400.pt" and ev1[3] == "vm/cycle_00.pt"
    assert tr1[3] == 450 and st.calls[5][3] == 500 and st.calls[8][3] == 550
    assert ev2[2] == "run/checkpoints/update_000450.pt" and ev2[3] == "vm/cycle_01.pt"
    assert ev2[4] == {"a": "evo/decks/evo-1.ydk", "b": "b.ydk"}  # the accepted child replaced its parent
    assert summary["games"] == 3000 and summary["accepted"] == 1
    assert summary["games_per_pp"] == pytest.approx(1500)  # 3,000 games for +2 pp


def test_an_interrupted_run_resumes_without_repeating_finished_stages(tmp_path):
    st = FakeStages(fail={"train-2": 1})
    with pytest.raises(RuntimeError, match="train-2"):
        CoEvolution(tmp_path, st, CFG, "base/checkpoints/update_000400.pt").run()
    again = FakeStages()
    loop = CoEvolution(tmp_path, again, CFG, "base/checkpoints/update_000400.pt")
    summary = loop.run()
    # the rerun starts at the unfinished stage, with the state the finished ones left
    assert [c[:2] for c in again.calls] == [("train", 2), ("fit", 2), ("evolve", 3), ("train", 3), ("fit", 3)]
    assert again.calls[0][2] == "run/checkpoints/update_000450.pt" and again.calls[0][3] == 500
    assert again.calls[2][4]["a"] == "evo/decks/evo-1.ydk"
    assert summary["games"] == 3000 and len(summary["cycles"]) == 3
    assert CoEvolution(tmp_path, FakeStages(), CFG, "x").run() == summary  # finished: nothing to do
    with pytest.raises(ValueError, match="deck"):
        CoEvolution(tmp_path / "new", FakeStages(), CoevoConfig(), "x")


def test_the_evolution_stage_reads_an_evolution_states_accepted_children(tmp_path):
    from tests.test_evolve import run

    ev, report = run(tmp_path / "evo")
    games, acc = accepted_children(tmp_path / "evo", 1, {"main": "decks/base.ydk"})
    assert games == report["games"] and len(acc) == report["accepted"] == 1
    a = acc[0]
    assert a["slot"] == "main" and a["parent"] == "corpus:base" and Path(a["file"]).is_file()
    assert a["diff"] == next(r for r in ev.lineage() if r["accepted"])["validation"]["diff"]
    check_round(tmp_path / "evo", 1)  # resumable / readable as cycle 1
    check_round(tmp_path / "evo", 2)  # the next one
    with pytest.raises(ValueError, match="rounds"):
        check_round(tmp_path / "evo", 3)


def _tool():
    spec = importlib.util.spec_from_file_location("coevolve_tool", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_the_command_stages_build_the_tool_commands(tmp_path, monkeypatch):
    from tests.test_evolve import run

    tool = _tool()
    run(tmp_path / "co" / "evo")  # round 1 already finished: read, not rerun
    calls = []
    monkeypatch.setattr("ygorl.build.coevolve.run_command", lambda cmd, log=None, env=None: calls.append(cmd))
    args = tool.argparse.Namespace(state=tmp_path / "co", checkpoint="out/why/x/checkpoints/update_000400.pt",
                                   env="md-test", device="cpu", envs=4, seed=0, deck_model=Path("dm.pt"),
                                   learned_generator=8, rules=0, value_model_weight=0.5, warm_start=["w.json"],
                                   evolve_arg=["--budget=100"], fit_arg=[], eval_every=None, paired=["m2.json@c.pt"])  # fmt: skip
    stages = tool.CommandStages(args)
    res = stages.evolve(1, "ckpt.pt", "vm.pt", {"main": "decks/base.ydk"})
    assert calls == [] and res.accepted and res.round == 1
    with pytest.raises(RuntimeError, match="did not write"):
        stages.train(1, "ckpt.pt", 450)
    assert calls[-1][1:3] == [str(TOOL), "train-segment"] and calls[-1][-1].endswith("manifest.json")
    stages.fit(1, "run/checkpoints/update_000450.pt")
    fit = calls[-1]
    assert fit[fit.index("--same-run") + 1].endswith("out/why/x")
    assert "m2.json@c.pt" in fit and str(tmp_path / "co" / "evo") in fit
    calls.clear()
    with pytest.raises(FileNotFoundError):  # round 2 never ran here (the stand-in command did nothing)
        stages.evolve(2, "c2.pt", "vm.pt", {"main": "decks/base.ydk"})
    cmd = calls[0]
    assert cmd[cmd.index("--value-model") + 1] == "vm.pt" and cmd[cmd.index("--parent") + 1] == "decks/base.ydk"
    assert "--budget=100" in cmd and cmd[cmd.index("--warm-start") + 1] == "w.json"


def test_a_training_segment_continues_a_run_and_logs_its_games(tmp_path):
    pytest.importorskip("torch")
    from ygorl.build.edit_labels import read_game_log

    spec = importlib.util.spec_from_file_location("train_ppo", Path(__file__).parents[1] / "tools" / "train_ppo.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    decks = [str(Path(__file__).parent / "decks" / f) for f in ("snake_eye.ydk", "kashtira.ydk")]
    small = ["--envs", "2", "--env-threads", "1", "--steps", "8", "--event-length", "8", "--d-model", "16",
             "--max-decisions", "30", "--eval-every", "0", "--torch-threads", "1", "--collect-threads", "1"]  # fmt: skip
    assert train.main([*decks, "--updates", "1", "--out", str(tmp_path / "base"), *small]) == 0
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "ygorl-deck-pool", "version": 1, "decks": []}))
    tool = _tool()
    seg = ["train-segment", "--from", str(tmp_path / "base" / "checkpoints" / "latest.pt"), "--run",
           str(tmp_path / "seg"), "--until", "3", "--deck-pool", str(manifest)]  # fmt: skip
    assert tool.main(seg) == 0
    assert (tmp_path / "seg" / "checkpoints" / "update_000003.pt").is_file()
    cfg = json.loads((tmp_path / "seg" / "config.json").read_text())
    assert cfg["log_games"] and cfg["deck_pool"] == str(manifest.resolve())
    import gzip

    with gzip.open(tmp_path / "seg" / "games.jsonl.gz", "rt") as f:
        raw = [json.loads(line) for line in f.read().splitlines()]
    assert raw and {r["update"] for r in raw} <= {2, 3}
    g = read_game_log(tmp_path / "seg")  # a 30-decision limit truncates most games: those are no result
    assert {n for r in raw for n in r["decks"]} <= set(g.decks) == {"snake_eye", "kashtira"}
    assert len(g.games) + g.skipped == len(raw) and all(x.checkpoint.update in (1, 2) for x in g.games)
    assert tool.main(seg) == 0  # rerun after the segment finished: resumes from latest.pt, nothing to train
    metrics = [json.loads(line)["update"] for line in (tmp_path / "seg" / "metrics.jsonl").read_text().splitlines()]
    assert metrics == [2, 3]
