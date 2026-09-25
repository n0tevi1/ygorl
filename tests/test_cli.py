"""Command line: `ygorl` subcommands (T2.9 `branch`; T3.4 `duel`, `arena`, `matrix`, `replay`)."""

import gzip
import json
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent, agent_factory
from ygorl.cards.legality import IllegalDeck
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.cli import main
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.branch import fork
from ygorl.engine.duel import Duel, DuelConfig, default_cards
from ygorl.engine.replay import REPLAY_COMPRESSED, Replay, load_yrp
from ygorl.eval.arena import Arena, derive_seed, merge

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


def make_env(tmp_path, meta=()):
    """Environment ``env-2026-09`` under ``tmp_path/envs`` whose pool holds the test decks A and B;
    ``meta`` names test decks to copy in as meta decks."""
    d = tmp_path / "envs" / "env-2026-09"
    (d / "meta").mkdir(parents=True)
    (d / "environment.json").write_text(
        json.dumps({"version": "env-2026-09", "format": "md", "rules": {"mode": "MR5"}})
    )
    decks = [A, B] + [load_ydk(DECKS / f"{n}.ydk") for n in meta]
    (d / "pool.json").write_text(json.dumps({"cards": sorted({c for x in decks for c in x.main + x.extra})}))
    (d / "banlist.lflist.conf").write_text("!none\n")
    for n in meta:
        (d / "meta" / f"{n}.ydk").write_text((DECKS / f"{n}.ydk").read_text())
    entries = [{"name": n, "file": f"meta/{n}.ydk", "share": round(1 / len(meta), 3)} for n in meta]
    (d / "meta.json").write_text(json.dumps({"decks": entries}))
    return d


def fields(out):
    """``key  value`` lines of a report as a dict (the first word is the key)."""
    rows = [line.split(None, 1) for line in out.splitlines() if line.strip()]
    return {row[0]: row[1] if len(row) > 1 else "" for row in rows}


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
            {0: "a", 1: "b", None: "draw"}[r.winner],
            r.reason,
            str(r.turns),
            str(r.lp[0]),
            str(r.lp[1]),
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
    d = make_env(tmp_path)
    root = d.parent
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


# ------------------------------------------------------------------ help


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("branch", "duel", "arena", "matrix", "replay"):
        assert command in out


def test_help_does_not_load_the_engine():
    import subprocess
    import sys

    code = "import sys; from ygorl.cli import build_parser; build_parser(); print(sorted(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    for heavy in ("ygorl._core", "ygorl.engine.duel", "ygorl.eval.matchup", "ygorl.eval.arena", "numpy"):
        assert f"'{heavy}'" not in out


# ------------------------------------------------------------------ duel


def test_duel_prints_the_same_result_as_the_api(tmp_path, capsys):
    rep_path, yrpx = tmp_path / "g.json.gz", tmp_path / "g.yrpX"
    code = main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--seed", "3", "--first", "b",
                 "--max-turns", "3", "--save-replay", str(rep_path), "--yrpx", str(yrpx)])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    duel = Duel(3, None, A, B, first=1, config=DuelConfig(max_turns=3))
    want = duel.run(RandomAgent(3), RandomAgent(4))
    f = fields(out)
    assert f["winner"].split()[0] == {0: "a", 1: "b", None: "draw"}[want.winner]
    assert f["reason"].split()[0] == want.reason
    assert f["turns"] == str(want.turns)
    assert f["lp"] == f"a {want.lp[0]}, b {want.lp[1]}"
    assert f["decisions"].split()[0] == str(want.decisions)
    assert "snake_eye" in f["duel"] and "tearlaments" in f["duel"] and "b moves first" in f["duel"]
    rep = Replay.load(rep_path)
    assert rep.responses == want.responses and rep.first == 1 and rep.max_turns == 3
    assert yrpx.read_bytes()[:4] == b"yrpX" and load_yrp(yrpx).flag & REPLAY_COMPRESSED
    assert str(rep_path) in out and str(yrpx) in out


def test_duel_agents_and_first_player(capsys):
    args = ["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--seed", "5", "--max-turns", "2",
            "--agent-a", "greedy", "--agent-b", "random"]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    want = Duel(5, None, A, B, first=0, config=DuelConfig(max_turns=2)).run(agent_factory("greedy")(5), RandomAgent(6))
    assert fields(out)["decisions"].split()[0] == str(want.decisions)
    assert "(greedy)" in out and "a moves first" in out


def test_duel_under_an_environment(tmp_path, capsys, monkeypatch):
    d = make_env(tmp_path)
    monkeypatch.setenv("YGORL_ENVIRONMENTS", str(d.parent))
    path = tmp_path / "g.json"
    args = ["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--max-turns", "2", "--save-replay",
            str(path), "--env", "env-2026-09"]  # fmt: skip
    assert main(args) == 0
    assert "environment env-2026-09" in capsys.readouterr().out
    assert Replay.load(path).environment["version"] == "env-2026-09"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["missing.ydk", str(DECKS / "kashtira.ydk")], "missing.ydk"),
        ([str(DECKS / "kashtira.ydk"), str(DECKS / "kashtira.ydk"), "--agent-a", "nope"], "unknown agent 'nope'"),
        ([str(DECKS / "kashtira.ydk"), str(DECKS / "kashtira.ydk"), "--env", "no-such-env"], "no-such-env"),
        ([str(DECKS / "kashtira.ydk"), str(DECKS / "kashtira.ydk"), "--yrpx-uncompressed"], "needs --yrpx PATH"),
    ],
)
def test_duel_errors_are_reported(capsys, args, message):
    assert main(["duel", *args]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl duel: error:") and message in err


def test_duel_rejects_a_bad_first_player(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["duel", str(DECKS / "kashtira.ydk"), str(DECKS / "kashtira.ydk"), "--first", "c"])
    assert exc.value.code == 2
    assert "--first" in capsys.readouterr().err


# ------------------------------------------------------------------ replay


def test_replay_shows_metadata(replay_file, capsys):
    path, rep, result, _ = replay_file
    assert main(["replay", str(path)]) == 0
    out = capsys.readouterr().out
    f = fields(out)
    assert f["seed"].split()[0] == str(rep.seed)
    assert "b moves first" in out
    assert "snake_eye" in f["deck_a"] and "tearlaments" in f["deck_b"]
    assert f["responses"].split()[0] == str(len(rep.responses))
    assert f["environment"] == "none"
    assert result.reason in f["recorded"] and f"turns={result.turns}" in f["recorded"]


def test_replay_verify_and_export(replay_file, tmp_path, capsys):
    path = replay_file[0]
    out_path = tmp_path / "x.yrpX"
    assert main(["replay", str(path), "--verify", "--export-yrpx", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert fields(out)["verify"].startswith("ok")
    assert out_path.read_bytes()[:4] == b"yrpX" and str(out_path) in out
    packed = load_yrp(out_path)
    assert packed.flag & REPLAY_COMPRESSED
    assert main(["replay", str(path), "--export-yrpx", str(out_path), "--yrpx-uncompressed"]) == 0
    raw = load_yrp(out_path)
    assert not raw.flag & REPLAY_COMPRESSED and not raw.embedded.flag & REPLAY_COMPRESSED
    assert raw.packets[:-1] == packed.packets[:-1] and raw.embedded.responses == packed.embedded.responses
    assert main(["replay", str(path), "--yrpx-uncompressed"]) == 2
    assert "needs --export-yrpx OUT" in capsys.readouterr().err


def test_replay_verify_detects_a_different_end(replay_file, tmp_path, capsys):
    rep = Replay.load(replay_file[0])
    rep.responses = rep.responses[: len(rep.responses) // 2]  # the log runs out before the recorded end
    path = tmp_path / "cut.json"
    rep.save(path)
    assert main(["replay", str(path), "--verify"]) == 1
    out = capsys.readouterr().out
    assert fields(out)["verify"].startswith("MISMATCH") and "log_exhausted" in out


def test_replay_needs_its_environment(tmp_path, capsys, monkeypatch):
    d = make_env(tmp_path)
    path = tmp_path / "g.json"
    save_replay(path, env=load_environment(d))
    monkeypatch.setenv("YGORL_ENVIRONMENTS", str(tmp_path / "elsewhere"))
    assert main(["replay", str(path), "--verify"]) == 2
    assert "--env" in capsys.readouterr().err
    assert main(["replay", str(path), "--verify", "--env", str(d)]) == 0
    assert "env-2026-09" in capsys.readouterr().out
    assert main(["replay", str(path)]) == 0  # metadata alone needs no environment


def test_replay_missing_or_bad_file(tmp_path, capsys):
    assert main(["replay", str(tmp_path / "none.json")]) == 2
    assert "none.json" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["replay", str(bad)]) == 2
    assert "not a ygorl replay" in capsys.readouterr().err


# ------------------------------------------------------------------ arena


def test_arena_matches_the_library(tmp_path, capsys):
    out_path = tmp_path / "arena.json"
    decks = [str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk")]
    code = main(["arena", *decks, "--agent-a", "greedy", "--agent-b", "random", "--games", "8", "--max-turns", "3",
                 "--seed", "1", "--out", str(out_path)])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    arena = Arena(agent_factory("greedy"), agent_factory("random"), config=DuelConfig(max_turns=3))
    ds = [A, B]
    cells = [(x, y, derive_seed(1, i, j)) for i, x in enumerate(ds) for j, y in enumerate(ds)]
    want = merge(arena.run_many(cells, pairs=1))
    got = json.loads(out_path.read_text())
    assert (got["games"], got["wins"], got["losses"], got["draws"]) == (want.games, want.wins, want.losses, want.draws)
    assert got["agent_a"] == "greedy" and got["agent_b"] == "random"
    assert want.summary() in out
    assert "4 deck pairings" in out and "snake_eye" in out


def test_arena_cross_decks_from_a_directory(tmp_path, capsys):
    deck_dir = tmp_path / "decks"
    deck_dir.mkdir()
    for n in ("snake_eye", "kashtira"):
        (deck_dir / f"{n}.ydk").write_text((DECKS / f"{n}.ydk").read_text())
    args = ["arena", str(deck_dir), "--vs", str(DECKS / "tearlaments.ydk"), "--games", "2", "--max-turns", "2",
            "--workers", "2"]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "2 deck pairings" in out
    assert "kashtira" in out and "snake_eye" in out and "tearlaments" in out


def test_arena_errors(tmp_path, capsys):
    assert main(["arena", str(tmp_path)]) == 2
    assert "no .ydk files" in capsys.readouterr().err
    assert main(["arena", str(DECKS / "kashtira.ydk"), "--agent-b", "nope"]) == 2
    assert "unknown agent 'nope'" in capsys.readouterr().err


# ------------------------------------------------------------------ matrix


def test_matrix_matches_the_library(tmp_path, capsys):
    from ygorl.eval.matchup import MetaGame, analyze, build_matrix

    names = ("snake_eye", "kashtira", "yubel")
    out_path = tmp_path / "m.json"
    code = main(["matrix", *(str(DECKS / f"{n}.ydk") for n in names), "--agent", "random", "--games", "2",
                 "--max-turns", "4", "--seed", "0", "--out", str(out_path)])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    decks = [load_ydk(DECKS / f"{n}.ydk") for n in names]
    want = analyze(build_matrix(decks, agent_factory("random"), pairs=1, config=DuelConfig(max_turns=4)))
    got = MetaGame.load(out_path)
    assert got.matrix.win_rate == want.matrix.win_rate
    assert got.nash == pytest.approx(want.nash) and got.alpha_rank == pytest.approx(want.alpha_rank)
    assert got.matrix.agent == "random"
    for n in names:
        row = next(line for line in out.splitlines() if line.split()[:1] == [n])
        assert len(row.split()) == 1 + len(names) + 2  # name, one win rate per deck, nash, alpha-rank
    assert str(out_path) in out


def test_matrix_of_environment_meta_decks(tmp_path, capsys):
    from ygorl.eval.matchup import MetaGame

    d = make_env(tmp_path, meta=("snake_eye", "kashtira"))
    assert main(["matrix", "--env", str(d), "--games", "2", "--max-turns", "2", "--name", "smoke"]) == 0
    out = capsys.readouterr().out
    path = d / "artifacts" / "matrix" / "smoke.json"
    meta = MetaGame.load(path, env=load_environment(d))
    assert meta.matrix.decks == ("snake_eye", "kashtira")
    assert str(path) in out


def test_matrix_errors(capsys):
    assert main(["matrix"]) == 2
    assert "no decks" in capsys.readouterr().err
    assert main(["matrix", str(DECKS / "kashtira.ydk"), str(DECKS / "kashtira.ydk")]) == 2
    assert "duplicate deck name" in capsys.readouterr().err


# ------------------------------------------------------------------ strength


def test_strength_matches_the_library(tmp_path, capsys):
    from ygorl.eval.agent_matrix import AgentMatrix, build_agent_matrix

    names = ("snake_eye", "kashtira", "yubel")
    out_path = tmp_path / "s.json"
    code = main(["strength", "random", "g=greedy", "--decks", *(str(DECKS / f"{n}.ydk") for n in names),
                 "--pairings", "2", "--max-turns", "4", "--seed", "5", "--out", str(out_path)])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    decks = [load_ydk(DECKS / f"{n}.ydk") for n in names]
    want = build_agent_matrix({"random": agent_factory("random"), "g": agent_factory("greedy")}, decks, pairings=2,
                              seed=5, config=DuelConfig(max_turns=4))  # fmt: skip
    got = AgentMatrix.load(out_path)
    assert got == want and got.specs == ("greedy", "random")
    assert "ranking: " in out and str(out_path) in out
    for n in ("g", "random"):
        row = next(line for line in out.splitlines() if line.split()[:1] == [n])
        assert len(row.split()) == 1 + 2 + 2  # name, one win rate per agent, nash, alpha-rank


def test_strength_of_environment_meta_decks(tmp_path, capsys):
    from ygorl.eval.agent_matrix import AgentMatrix

    d = make_env(tmp_path, meta=("snake_eye", "kashtira"))
    assert main(["strength", "random", "greedy", "--env", str(d), "--pairings", "1", "--max-turns", "2",
                 "--name", "smoke"]) == 0  # fmt: skip
    path = d / "artifacts" / "agent-matrix" / "smoke.json"
    assert AgentMatrix.load(path, env=load_environment(d)).decks == ("snake_eye", "kashtira")
    assert str(path) in capsys.readouterr().out


def test_strength_errors(capsys):
    deck = str(DECKS / "kashtira.ydk")
    assert main(["strength", "random"]) == 2
    assert "at least two agents" in capsys.readouterr().err
    assert main(["strength", "greedy", "greedy", "--decks", deck, str(DECKS / "yubel.ydk")]) == 2
    assert "duplicate agent name" in capsys.readouterr().err
    assert main(["strength", "random", "greedy"]) == 2
    assert "no deck pool" in capsys.readouterr().err
    assert main(["strength", "random", "nope", "--decks", deck, str(DECKS / "yubel.ydk")]) == 2
    assert "unknown agent 'nope'" in capsys.readouterr().err


def test_strength_agent_names_and_specs():
    from ygorl.commands.strength import parse_agents

    assert parse_agents(["greedy", "g2=greedy", "policy:out/p.pt@t=0.5"]) == {
        "greedy": "greedy", "g2": "greedy", "policy:out/p.pt@t=0.5": "policy:out/p.pt@t=0.5"}  # fmt: skip
    assert parse_agents(["best=policy:out/p.pt@t=1", "random"])["best"] == "policy:out/p.pt@t=1"


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", ".hidden", "a\\b", ""])
def test_matrix_rejects_an_artifact_name_with_a_path(tmp_path, capsys, name):
    d = make_env(tmp_path, meta=("snake_eye", "kashtira"))
    assert main(["matrix", "--env", str(d), "--games", "2", "--max-turns", "1", "--name", name]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl matrix: error:") and "--name" in err
    assert not (d / "artifacts").exists() or not any((d / "artifacts").rglob("*.json"))
    assert not (tmp_path / "envs" / "escape.json").exists()


# ------------------------------------------------------------------ deck legality


def write_deck(path, main=(), extra=(), side=()):
    path.write_text(Deck(tuple(main), tuple(extra), tuple(side)).to_ydk())
    return path


def illegal_decks(tmp_path):
    """(file, expected message) for decks that break a structural rule (no environment needed)."""
    most = max(set(A.main), key=A.main.count)
    four = (most,) * (4 - A.main.count(most))
    spell = next(p for p in A.main if not default_cards()[p].type & C.TYPE_MONSTER)
    return [
        (write_deck(tmp_path / "unknown.ydk", A.main[:-1] + (12345678,), A.extra), "12345678 does not exist"),
        (write_deck(tmp_path / "empty.ydk"), "Main Deck has 0 cards (minimum 40)"),
        (write_deck(tmp_path / "four.ydk", A.main + four, A.extra), "4 copies (maximum 3)"),
        (write_deck(tmp_path / "extra_in_main.ydk", A.main[:-1] + A.extra[:1], A.extra[1:]),
         "is an Extra Deck monster but is in the Main Deck"),
        (write_deck(tmp_path / "main_in_extra.ydk", A.main, A.extra[:-1] + (spell,)),
         "is not an Extra Deck monster but is in the Extra Deck"),
        (write_deck(tmp_path / "big.ydk", A.main[:39] + (60 * (A.main[39],)), A.extra), "maximum 60"),
    ]  # fmt: skip


def test_illegal_decks_are_rejected_before_playing(tmp_path, capsys):
    for path, message in illegal_decks(tmp_path):
        for command in (["duel", str(path), str(DECKS / "snake_eye.ydk")],
                        ["duel", str(DECKS / "snake_eye.ydk"), str(path)],
                        ["arena", str(DECKS / "snake_eye.ydk"), "--vs", str(path), "--games", "2"],
                        ["matrix", str(DECKS / "snake_eye.ydk"), str(path), "--games", "2"]):  # fmt: skip
            assert main(command) == 2, (command, capsys.readouterr())
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err.startswith(f"ygorl {command[0]}: error: {path}: deck '{path.stem}' is illegal")
            assert message in captured.err


def test_empty_ydk_file_is_rejected(tmp_path, capsys):
    path = tmp_path / "nothing.ydk"
    path.write_text("")
    assert main(["duel", str(path), str(DECKS / "snake_eye.ydk")]) == 2
    assert "Main Deck has 0 cards" in capsys.readouterr().err


def test_decks_are_checked_against_the_environment(tmp_path, capsys):
    d = make_env(tmp_path)
    banned = A.main[0]
    (d / "banlist.lflist.conf").write_text(f"!ban\n{banned} 0\n")
    args = ["--max-turns", "1", "--env", str(d)]
    assert main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), *args]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"ygorl duel: error: {DECKS / 'snake_eye.ydk'}: deck 'snake_eye' is illegal in environment "
                          "env-2026-09")  # fmt: skip
    assert f"({banned}) is forbidden" in err
    # the same deck is legal without the environment
    assert main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--max-turns", "1"]) == 0
    capsys.readouterr()
    # a deck outside the environment's card pool
    (d / "banlist.lflist.conf").write_text("!none\n")
    for command in (["duel", str(DECKS / "kashtira.ydk"), str(DECKS / "snake_eye.ydk")],
                    ["arena", str(DECKS / "snake_eye.ydk"), "--vs", str(DECKS / "kashtira.ydk"), "--games", "2"],
                    ["matrix", str(DECKS / "snake_eye.ydk"), str(DECKS / "kashtira.ydk"), "--games", "2"]):  # fmt: skip
        assert main([*command, *args]) == 2
        err = capsys.readouterr().err
        assert "deck 'kashtira' is illegal in environment env-2026-09" in err
        assert "is not in this format's card pool" in err


def test_environment_meta_decks_are_checked(tmp_path, capsys):
    d = make_env(tmp_path, meta=("snake_eye", "tearlaments"))
    meta_file = d / "meta" / "tearlaments.ydk"
    most_b = max(set(B.main), key=B.main.count)
    write_deck(meta_file, B.main + (most_b,) * (4 - B.main.count(most_b)), B.extra)
    assert main(["matrix", "--env", str(d), "--games", "2", "--max-turns", "1"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl matrix: error:")
    assert str(meta_file) in err and "'tearlaments'" in err and "4 copies" in err
    assert not (d / "artifacts" / "matrix").exists()
    # the environment is refused wherever it is loaded, not only by matrix
    assert main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--env", str(d)]) == 2
    assert str(meta_file) in capsys.readouterr().err


def test_bad_input_files_are_clean_errors(replay_file, tmp_path, capsys):
    good = replay_file[0].read_bytes()
    data = json.loads(gzip.decompress(good))
    (tmp_path / "trunc.json.gz").write_bytes(good[: len(good) // 2])
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "seed.json").write_text(json.dumps({**data, "seed": "abc"}))
    (tmp_path / "hex.json").write_text(json.dumps({**data, "responses": ["zz"]}))
    (tmp_path / "decks.json").write_text(json.dumps({**data, "decks": {"a": [1]}}))
    (tmp_path / "big.ydk").write_text("#main\n" + "\n".join(map(str, A.main[:39])) + "\n99999999999\n")
    cases = [
        (["replay", str(tmp_path / "trunc.json.gz")], "truncated or corrupt gzip"),
        (["replay", str(tmp_path / "list.json"), "--verify"], "expected a JSON object, not list"),
        (["replay", str(tmp_path / "seed.json"), "--verify"], "'seed' must be an integer, not str"),
        (["replay", str(tmp_path / "hex.json")], "'responses[0]' is not a hex string"),
        (["branch", str(tmp_path / "decks.json"), "--at", "0"], "'decks' must be an object"),
        (["duel", str(tmp_path / "big.ydk"), str(DECKS / "snake_eye.ydk")],
         "big.ydk:41: invalid card password 99999999999"),
    ]  # fmt: skip
    for args, message in cases:
        assert main(args) == 2, args
        err = capsys.readouterr().err
        assert err.startswith(f"ygorl {args[0]}: error:") and message in err, err
        assert "Traceback" not in err


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"player": [1, 2]}, "player must be an object"),
        ({"player": {"starting_lp": 0}}, "starting_lp must be at least 1"),
        ({"player": {"starting_hand": 70}}, "starting_hand 70 is larger than deck.main_min 40"),
        ({"rules": {"mode": "MR5", "extra_flags": "TCG_SEGOC_NONPUBLIC"}}, "rules.extra_flags must be a list"),
    ],
)
def test_degenerate_environments_are_rejected(tmp_path, capsys, manifest, message):
    d = make_env(tmp_path)
    (d / "environment.json").write_text(json.dumps({"version": "env-2026-09", "format": "md", **manifest}))
    assert main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--env", str(d)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl duel: error:") and message in err


def test_non_utf8_banlist_is_a_clean_error(tmp_path, capsys):
    d = make_env(tmp_path)
    (d / "banlist.lflist.conf").write_bytes(b"!caf\xe9\n")
    assert main(["duel", str(DECKS / "snake_eye.ydk"), str(DECKS / "tearlaments.ydk"), "--env", str(d)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ygorl duel: error:") and f"{d / 'banlist.lflist.conf'}: not valid UTF-8" in err


def test_illegal_decks_still_play_through_the_library():
    deck = Deck(A.main[:-1] + (12345678,), A.extra)
    with pytest.raises(IllegalDeck, match="12345678"):
        Duel(0, None, deck, B, validate=True)
    result = Duel(0, None, Deck(A.main + (A.main[0],), A.extra), B, config=DuelConfig(max_turns=1)).run(
        RandomAgent(0), RandomAgent(1)
    )
    assert result.reason in ("turn_limit", "win")
