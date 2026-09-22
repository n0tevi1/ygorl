"""Command line: `ygorl` subcommands (T2.9 adds `branch`)."""

import json
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.cli import main
from ygorl.data import load_environment
from ygorl.engine.branch import fork
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.engine.replay import Replay

DECKS = Path(__file__).parent / "decks"
A, B = load_ydk(DECKS / "snake_eye.ydk"), load_ydk(DECKS / "tearlaments.ydk")


def save_replay(path, env=None, seed=3):
    duel = Duel(seed, env, A, B, first=1, config=DuelConfig(max_turns=3))
    result = duel.run(RandomAgent(seed), RandomAgent(seed + 1))
    rep = Replay.from_duel(duel, result)
    rep.save(path)
    return rep, result


@pytest.fixture(scope="module")
def replay_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("replays") / "game.json.gz"
    rep, result = save_replay(path)
    t = next(t for t in range(len(result.actions)) if 3 <= len(fork(rep, t).point.actions) <= 12)
    return path, rep, result, t


def table_rows(out):
    lines = out.splitlines()
    head = next(i for i, line in enumerate(lines) if line.split()[:2] == ["cand", "action"])
    return lines[head], [line for line in lines[head + 1 :] if line.strip()]


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: ygorl" in out and "branch" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("ygorl ")


def test_branch_prints_one_row_per_candidate(replay_file, capsys):
    path, rep, result, t = replay_file
    code = main(["branch", str(path), "--at", str(t), "--try", "all", "--policy", "random", "--seed", "3"])
    out = capsys.readouterr().out
    assert code == 0, out
    header, rows = table_rows(out)
    for col in ("winner", "reason", "turns", "lp_a", "lp_b"):
        assert col in header.split()
    branch = fork(rep, t)
    outcomes = branch.try_all(RandomAgent, seed=3)
    assert len(rows) == len(branch.point.actions) == len(outcomes)
    for row, o in zip(rows, outcomes, strict=True):
        r = o.results[0]
        cells = row.split()
        assert cells[0].lstrip("*") == str(o.index) or cells[1] == str(o.index)
        assert row.startswith("*") == o.recorded
        assert [cells[-5], cells[-4], cells[-3], cells[-2], cells[-1]] == [
            {0: "a", 1: "b", None: "draw"}[r.winner], r.reason, str(r.turns), str(r.lp[0]), str(r.lp[1])
        ]
    assert f"t={t}" in out
    assert f"recorded: action {branch.recorded_action}" in out
    assert result.reason in out  # the recorded game's end is shown for reference


def test_branch_subset_with_several_rollouts(replay_file, capsys):
    path, _, _, t = replay_file
    assert main(["branch", str(path), "--at", str(t), "--try", "0,2", "--rollouts", "3"]) == 0
    header, rows = table_rows(capsys.readouterr().out)
    assert len(rows) == 2
    assert {"wins", "draws", "losses", "win%"} <= set(header.split())
    for row in rows:
        wins, draws, losses = (int(x) for x in row.split()[-7:-4])
        assert wins + draws + losses == 3


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--at", "100000"], "out of range"),
        (["--at", "0", "--policy", "nope"], "unknown agent 'nope'"),
        (["--at", "0", "--try", "99"], "candidate 99 out of range"),
    ],
)
def test_branch_errors_are_reported(replay_file, capsys, args, message):
    path = replay_file[0]
    assert main(["branch", str(path), *args]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl branch: error:") and message in err


def test_branch_rejects_a_malformed_try_list(replay_file, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["branch", str(replay_file[0]), "--at", "0", "--try", "1,x"])
    assert exc.value.code == 2
    assert "--try" in capsys.readouterr().err


def test_branch_missing_replay_file(tmp_path, capsys):
    assert main(["branch", str(tmp_path / "none.json"), "--at", "0"]) == 2
    assert "none.json" in capsys.readouterr().err


def test_branch_with_environment(tmp_path, capsys, monkeypatch):
    root = tmp_path / "envs"
    d = root / "env-2026-09"
    (d / "meta").mkdir(parents=True)
    (d / "environment.json").write_text(json.dumps({"version": "env-2026-09", "format": "md", "rules": {"mode": "MR5"}}))
    (d / "pool.json").write_text(json.dumps({"cards": sorted(set(A.main + A.extra + B.main + B.extra))}))
    (d / "banlist.lflist.conf").write_text("!none\n")
    (d / "meta.json").write_text(json.dumps({"decks": []}))
    path = tmp_path / "game.json"
    save_replay(path, env=load_environment(d))
    assert main(["branch", str(path), "--at", "1", "--env", str(d)]) == 0
    assert "env-2026-09" in capsys.readouterr().out
    # without --env the recorded version is looked up in the environments root
    monkeypatch.setenv("YGORL_ENVIRONMENTS", str(tmp_path / "elsewhere"))
    assert main(["branch", str(path), "--at", "1"]) == 2
    assert "--env" in capsys.readouterr().err
    monkeypatch.setenv("YGORL_ENVIRONMENTS", str(root))
    assert main(["branch", str(path), "--at", "1"]) == 0
