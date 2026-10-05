"""ygorl.solver: combo-solver wrapper, demonstration conversion and verification (T4a.1).

Everything except the last tests runs without the solver binary: argument assembly, output
parsing and the conversion of lines into our action indices are checked on replays produced
by our own engine. The end-to-end tests run the real binary when ``tools/build_combo_solver.sh``
has built it (or ``$YGORL_COMBO_SOLVER`` points at one) and are skipped otherwise.
"""

import json
import struct
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.engine.replay import Replay, load_yrp
from ygorl.solver import (
    DEMO_FORMAT,
    DemoError,
    Demonstration,
    SolveRequest,
    SolverNotFound,
    TargetCard,
    Workdir,
    board_summary_missing,
    canonical_response,
    convert_line,
    find_solver,
    iter_steps,
    make_template,
    parse_events,
    parse_solution_name,
    read_jsonl,
    sample_hand,
    verify_line,
)

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
SNAKE = DECKS["snake_eye"]
ASH, OAK, SPOILS, ASH_BLOSSOM, IMPERM = 9674034, 45663742, 89023486, 14558127, 10045474


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


# ------------------------------------------------------------------ targets


@pytest.mark.parametrize(
    ("spec", "code", "zone", "facedown", "arg"),
    [
        ("48452496", 48452496, "atk", False, "48452496@atk"),
        ("48452496@def", 48452496, "def", False, "48452496@def"),
        ("2511@szone:fd", 2511, "szone", True, "2511@szone:fd"),
        ("9674034@grave", 9674034, "grave", False, "9674034@grave"),
    ],
)
def test_target_card_parsing(spec, code, zone, facedown, arg):
    t = TargetCard.parse(spec)
    assert (t.code, t.zone, t.facedown) == (code, zone, facedown)
    assert t.to_arg() == arg and TargetCard.parse(t.to_arg()) == t


@pytest.mark.parametrize(
    ("spec", "message"),
    [("Snake-Eyes Flamberge Dragon", "password"), ("48452496@field", "zone"), ("48452496@extra", "zone"),
     ("48452496@grave:fd", "face-down"), ("", "password")],
)  # fmt: skip
def test_target_card_rejects_names_and_ambiguous_zones(spec, message):
    with pytest.raises(ValueError, match=message):
        TargetCard.parse(spec)


def test_board_check_counts_copies_and_zones():
    # positions: 1 = face-up attack, 4 = face-up defense, 10 = face-down (set)
    board = {"players": [{"mzone": [{"code": 1, "position": 1}, {"code": 1, "position": 4}], "szone": [{"code": 3, "position": 10}],
                          "hand": [7], "grave": [5, 5], "banished": [], "lp": 8000}, {}]}  # fmt: skip
    ok = [
        TargetCard(1, "atk"),
        TargetCard(1, "mzone"),
        TargetCard(3, "szone", True),
        TargetCard(5, "grave"),
        TargetCard(5, "grave"),
    ]
    assert board_summary_missing(board, ok) == []
    # like the solver's judge, atk/def only say "face-up in a monster zone": two face-up copies satisfy 1@def twice
    assert board_summary_missing(board, [TargetCard(1, "def"), TargetCard(1, "def")]) == []
    missing = board_summary_missing(board, [TargetCard(1, "def"), TargetCard(1, "mzone", True), TargetCard(1, "atk"),
                                            TargetCard(1, "atk"), TargetCard(3, "szone"), TargetCard(7, "banished")])  # fmt: skip
    assert [t.to_arg() for t in missing] == ["1@mzone:fd", "1@atk", "3@szone", "7@banished"]


# ------------------------------------------------------------------ hands, template, arguments


def test_sample_hand_is_deterministic_and_drawn_from_the_deck():
    hand, order = sample_hand(SNAKE, seed=5, size=5)
    assert (hand, order) == sample_hand(SNAKE, seed=5, size=5)
    assert len(hand) == 5 and sorted(order) == sorted(SNAKE.main) and order[-5:] == hand
    assert sample_hand(SNAKE, seed=6, size=5)[1] != order


def test_template_carries_rules_seed_and_a_passive_opponent(tmp_path, db):
    path = make_template(tmp_path / "t.yrpX", SNAKE, seed=9, config=DuelConfig(), cards=db)
    yrp = load_yrp(path).replayable()
    assert yrp.player == (8000, 5, 1) and len(yrp.decks) == 2
    assert yrp.decks[0] == (tuple(SNAKE.main), tuple(SNAKE.extra))
    opponent = set(yrp.decks[1][0])
    assert len(yrp.decks[1][0]) == 40 and all(db[c].is_monster and not db[c].type & C.TYPE_EFFECT for c in opponent)


def test_solve_request_arguments(tmp_path):
    wd = Workdir(tmp_path, (tmp_path / "script-root", tmp_path / "official"))
    req = SolveRequest(template=Path("t.yrpX"), deck=Path("d.ydk"), hand=(ASH, OAK), targets=(TargetCard.parse("48452496"),),
                       solve_ms=5000, threads=1, seed=3, max_written=2)  # fmt: skip
    args = req.args(wd, tmp_path / "out")
    assert args[:1] == ["t.yrpX"]
    joined = " ".join(args)
    assert (
        f"--workdir {tmp_path}" in joined
        and f"--scriptdir {tmp_path / 'script-root'} --scriptdir {tmp_path / 'official'}" in joined
    )
    assert "--no-ref --deck d.ydk --hand 9674034|45663742 --target 48452496@atk" in joined
    assert "--solve-ms 5000 --threads 1 --seed 3 --max-written 2 --json" in joined and args[-2:] == [
        "--outdir",
        str(tmp_path / "out"),
    ]
    fire = SolveRequest(template=Path("line.yrpX"), fire=ASH_BLOSSOM, fire_ms=4000, solve_ms=1000)
    fargs = fire.args(wd, tmp_path / "o")
    assert (
        "--fire 14558127 --fire-bake --fire-ms 4000" in " ".join(fargs)
        and "--no-ref" not in fargs
        and "--deck" not in fargs
    )
    assert "--max-rollouts" not in fargs
    counted = SolveRequest(template=Path("line.yrpX"), fire=ASH_BLOSSOM, max_rollouts=500).args(wd, tmp_path / "o")
    assert "--max-rollouts 500" in " ".join(counted)
    with pytest.raises(ValueError, match="target"):
        SolveRequest(template=Path("t"), deck=Path("d.ydk"), hand=(ASH,)).args(wd, tmp_path)


def test_workdir_mirrors_our_script_priority(tmp_path, db):
    wd = Workdir.create(tmp_path / "wd")
    assert (wd.path / "cards.cdb").resolve().is_file()
    root = wd.scriptdirs[0]
    assert (root / "constant.lua").is_file() and not any(p.is_dir() for p in root.iterdir())
    assert [p.name for p in wd.scriptdirs[1:]] == [
        "official",
        "pre-release",
        "pre-errata",
        "goat",
        "rush",
        "skill",
        "unofficial",
    ]
    assert Workdir.create(tmp_path / "wd").scriptdirs == wd.scriptdirs  # idempotent


def test_find_solver_errors_are_actionable(tmp_path, monkeypatch):
    with pytest.raises(SolverNotFound, match="build_combo_solver.sh"):
        find_solver(tmp_path / "missing")
    fake = tmp_path / "combosolver"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("YGORL_COMBO_SOLVER", str(fake))
    assert find_solver() == fake


STDOUT = """loading
@event {"type":"phase","phase":"loading"}
  cards : 14755
@event {"type":"health","msgRetry":0}
@event {"type":"seed","seed":13016568724627090460}
@event {"type":"broken json
@event {"type":"fireVerdict","converted":1,"windows":2}
@event {"type":"written","written":2,"candidates":48,"outdir":"out1"}
"""


def test_parse_events_and_solution_names():
    events = parse_events(STDOUT)
    assert [e["type"] for e in events] == ["phase", "health", "seed", "fireVerdict", "written"]
    assert events[2]["seed"] == 13016568724627090460
    sol = parse_solution_name("solution_03_b2_a5.yrp")
    assert (sol.index, sol.burned, sol.actions, sol.alt) == (3, 2, 5, False)
    assert parse_solution_name("solution_10_b0_a12_alt.yrp").alt is True
    assert parse_solution_name("but_rejete_00_board.yrp") is None


# ------------------------------------------------------------------ conversion (no solver needed)


def test_canonical_response_rewrites_compact_card_lists():
    from ygorl.engine import messages as M

    cards = tuple(M.CardInfo(100 + i, M.Location(0, 1, i, 8)) for i in range(4))
    decision = M.SelectCard(player=0, cancelable=False, min=1, max=2, cards=cards)
    mode0 = struct.pack("<iI2I", 0, 2, 3, 1)
    assert canonical_response(decision, mode0) == mode0
    assert canonical_response(decision, struct.pack("<iI2B", 2, 2, 3, 1)) == mode0  # u8 indices
    assert canonical_response(decision, struct.pack("<iI2H", 1, 2, 3, 1)) == mode0  # u16 indices
    assert canonical_response(decision, struct.pack("<iB", 3, 0b1010)) == struct.pack("<iI2I", 0, 2, 1, 3)  # bit field
    assert canonical_response(decision, struct.pack("<i", -1)) == struct.pack("<i", -1)
    other = M.SelectYesNo(player=0, description=0)
    assert canonical_response(other, b"\x01\x00\x00\x00") == b"\x01\x00\x00\x00"


def _short_game(db, tmp_path, seed=41):
    duel = Duel(seed, None, SNAKE, DECKS["kashtira"], cards=db, config=DuelConfig(max_turns=1))
    result = duel.run(RandomAgent(seed), RandomAgent(seed + 1))
    path = tmp_path / "g.yrpX"
    Replay.from_duel(duel, result).to_yrpx(path)
    return result, load_yrp(path)


def test_convert_line_maps_every_response_to_action_indices(db, tmp_path):
    result, yrp = _short_game(db, tmp_path)
    solver_part = yrp.replayable().responses[:12]
    line = convert_line(yrp, [], responses=solver_part, cards=db, close_turn=False)
    assert line.responses == list(solver_part) and line.solver_responses == 12
    assert line.actions == result.actions[: len(line.actions)] and len(line.players) == len(line.actions)
    assert line.board["players"][0]["lp"] == 8000


def test_convert_line_closes_the_turn_and_checks_the_board(db, tmp_path):
    _, yrp = _short_game(db, tmp_path)
    solver_part = yrp.replayable().responses[:6]
    line = convert_line(yrp, [], responses=solver_part, cards=db)
    assert line.solver_responses == 6 and len(line.responses) > 6 and line.board["turn"] == 2
    with pytest.raises(DemoError, match="final board lacks 48452496@atk"):
        convert_line(yrp, [TargetCard(48452496)], responses=solver_part, cards=db)


def test_convert_line_rejects_a_response_the_engine_refuses(db, tmp_path):
    _, yrp = _short_game(db, tmp_path)
    bad = list(yrp.replayable().responses[:3]) + [struct.pack("<i", 999)]
    with pytest.raises(DemoError, match="response 3"):
        convert_line(yrp, [], responses=bad, cards=db, close_turn=False)


def test_demonstration_roundtrip_and_iter_steps(db, tmp_path):
    _, yrp = _short_game(db, tmp_path)
    line = convert_line(yrp, [], responses=yrp.replayable().responses[:10], cards=db)
    demo = Demonstration.start_from(
        yrp, deck=SNAKE, hand=[1, 2, 3], hand_index=0, hand_seed=1, variant="plain", targets=[], environment=None
    )
    demo.lines.append(line)
    demo.status = "solved"
    path = tmp_path / "demos.jsonl"
    demo.append_to(path)
    demo.append_to(path)
    loaded = list(read_jsonl(path))
    assert len(loaded) == 2 and loaded[0] == demo and loaded[0].to_json()["format"] == DEMO_FORMAT
    json.dumps(loaded[0].to_json())  # plain JSON
    steps = list(iter_steps(loaded[0], 0, cards=db))
    assert [a for _, a in steps] == line.actions
    assert [p.player for p, _ in steps] == line.players
    assert all(0 <= a < len(p.actions) for p, a in steps)
    rep = loaded[0].replay(0)
    assert rep.play(cards=db).responses == line.responses
    verify_line(loaded[0], 0, cards=db)
    tampered = Demonstration.from_json(loaded[0].to_json())
    tampered.lines[0].board["players"][0]["lp"] = 1
    with pytest.raises(DemoError, match="board differs"):
        verify_line(tampered, 0, cards=db)
    first_multi = next(i for i, (p, _) in enumerate(steps) if len(p.actions) > 1)
    tampered.lines[0].actions[first_multi] = (tampered.lines[0].actions[first_multi] + 1) % len(
        steps[first_multi][0].actions
    )
    with pytest.raises(DemoError):
        verify_line(tampered, 0, cards=db)


# ------------------------------------------------------------------ with the real solver binary


def _solver_or_skip():
    try:
        return find_solver()
    except SolverNotFound as exc:
        pytest.skip(f"combo solver binary not available ({exc}); build it with tools/build_combo_solver.sh")


def test_end_to_end_solve_convert_and_verify(db, tmp_path):
    from ygorl.solver import solve_hand, HandJob

    binary = _solver_or_skip()
    deck_path = Path(__file__).parent / "decks" / "snake_eye.ydk"
    job = HandJob(deck_path=deck_path, hand_index=0, hand_seed=11, targets=("9674034",), solve_ms=3000, threads=1,
                  hand=(ASH, OAK, SPOILS, ASH_BLOSSOM, IMPERM), lines=2, workdir=tmp_path / "wd", scratch=tmp_path / "s",
                  binary=binary, solver_seed=1)  # fmt: skip
    demo = solve_hand(job, cards=db)
    assert demo.status == "solved", demo.error
    assert demo.hand == [ASH, OAK, SPOILS, ASH_BLOSSOM, IMPERM]
    assert 1 <= len(demo.lines) <= 2
    for line in demo.lines:
        assert line.score["burned"] >= 0 and line.solver_responses <= len(line.responses)
        assert board_summary_missing(line.board, [TargetCard(ASH)]) == []
        steps = list(iter_steps(demo, demo.lines.index(line), cards=db))
        assert [a for _, a in steps] == line.actions


def test_end_to_end_fire_variant(db, tmp_path):
    """Labrynth: Arianna + Welcome Labrynth set; the opponent's Ash Blossom hits the search and the solver recovers."""
    from ygorl.solver import HandJob, solve_fire, solve_hand

    binary = _solver_or_skip()
    hand = (6351147, 23434538, 5380979, 1225009, 23434538)
    job = HandJob(deck_path=Path(__file__).parent / "decks" / "labrynth.ydk", hand_index=0, hand_seed=10238517081721291411,
                  targets=("1225009", "5380979@szone:fd"), solve_ms=4000, threads=1, hand=hand, fire_ms=3000,
                  workdir=tmp_path / "wd", scratch=tmp_path / "s", binary=binary, solver_seed=1)  # fmt: skip
    base = solve_hand(job, cards=db)
    assert base.status == "solved", base.error
    fire = solve_fire(job, base, ASH_BLOSSOM, cards=db)
    assert fire.variant == "fire" and fire.fire == ASH_BLOSSOM and fire.solver.get("windows", 0) >= 1, fire.error
    assert fire.status in ("solved", "unsolved"), fire.error
    assert ASH_BLOSSOM in fire.start["decks"]["b"]["main"]  # baked into the opponent's deck
    for i, line in enumerate(fire.lines):
        steps = list(iter_steps(fire, i, cards=db))
        assert any(
            p.player == 1 and p.actions[a].kind == "chain" and p.actions[a].card.code == ASH_BLOSSOM for p, a in steps
        )


FAKE_SOLVER = """#!/bin/sh
echo '@event {"type":"phase","phase":"loading"}'
echo '!! decklist unreadable: nothing to solve'
exit 3
"""


def _fake_solver(tmp_path, text=FAKE_SOLVER):
    fake = tmp_path / "combosolver"
    fake.write_text(text)
    fake.chmod(0o755)
    return fake


def test_solver_failures_become_error_records(db, tmp_path):
    from ygorl.solver import HandJob, solve_hand

    deck_path = Path(__file__).parent / "decks" / "snake_eye.ydk"
    common = dict(deck_path=deck_path, hand_index=0, hand_seed=11, solve_ms=1000, workdir=tmp_path / "wd", scratch=tmp_path / "s",
                  binary=_fake_solver(tmp_path))  # fmt: skip
    demo = solve_hand(HandJob(targets=("48452496",), **common), cards=db)
    assert demo.status == "error" and "exited with 3" in demo.error and "decklist unreadable" in demo.error
    assert demo.solver["returncode"] == 3 and not demo.lines
    missing = solve_hand(HandJob(targets=("99999999",), **common), cards=db)  # checked before the solver runs
    assert missing.status == "error" and "99999999" in missing.error


def test_batch_planning_and_resume(tmp_path, monkeypatch):
    import importlib.util

    from ygorl.solver.batch import done_keys, hand_seed, load_targets, repair_jsonl, summarize

    assert hand_seed(0, "snake_eye", 3) == hand_seed(0, "snake_eye", 3) != hand_seed(0, "snake_eye", 4)
    targets = load_targets(Path(__file__).parent / "decks" / "solver_targets.json")
    assert set(targets) == set(DECKS) and all(t["targets"] for t in targets.values())
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"snake_eye": ["Flamberge"]}))
    with pytest.raises(ValueError, match="snake_eye"):
        load_targets(bad)

    spec = importlib.util.spec_from_file_location(
        "solve_openings", Path(__file__).parents[1] / "tools" / "solve_openings.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    unsolved = _fake_solver(
        tmp_path, '#!/bin/sh\necho \'@event {"type":"written","written":0,"candidates":0}\'\nexit 0\n'
    )
    out = tmp_path / "demos.jsonl"
    argv = [str(Path(__file__).parent / "decks" / "snake_eye.ydk"), "--hands", "2", "--solve-ms", "100", "--workers", "1",
            "--binary", str(unsolved), "--out", str(out), "--scratch", str(tmp_path / "scratch")]  # fmt: skip
    summary = tmp_path / "summary.json"
    argv += ["--summary", str(summary)]
    assert tool.main(argv) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    got = sorted((r["hand_index"], r["variant"], r["status"]) for r in records)
    assert got == [(0, "fire", "skipped"), (0, "plain", "unsolved"), (1, "fire", "skipped"), (1, "plain", "unsolved")]
    assert summarize(records)["snake_eye plain"]["unsolved"] == 2
    with open(out, "a") as f:
        f.write('{"format": "ygorl-demo", "trunc')  # an interrupted write
    assert tool.main(argv + ["--hands", "3"]) == 0  # resumes: only hand 2 runs
    assert repair_jsonl(out) == 0
    keys = done_keys(out, None)
    assert {k[1] for k in keys} == {0, 1, 2} and len(out.read_text().splitlines()) == 6
    with pytest.raises(ValueError, match="environment"):
        done_keys(out, {"version": "md-2026-10", "fingerprint": "x"})
    assert json.loads(summary.read_text())["table"]["snake_eye plain"]["hands"] == 3
    frozen = out.read_bytes()
    # Validate identity before truncating even a partial record.
    out.write_bytes(frozen + b'{"partial"')
    for changed in (["--seed", "7"], ["--solve-ms", "200"], ["--solver-seed", "5"], ["--no-fire"]):
        assert tool.main(argv + changed) == 2
        assert out.read_bytes() == frozen + b'{"partial"'
    assert tool.main(argv + ["--hands", "3"]) == 0
    assert out.read_bytes() == frozen
    assert json.loads(summary.read_text())["new_records"] == 0
    assert json.loads(summary.read_text())["records"] == 6
    # An existing error remains a failed batch even with no new work.
    old = [json.loads(line) for line in frozen.splitlines()]
    old[0]["status"] = "error"
    out.write_text("".join(json.dumps(r) + "\n" for r in old))
    assert tool.main(argv + ["--hands", "3"]) == 1
    assert json.loads(summary.read_text())["table"]["snake_eye plain"]["error"] == 1
    # Version equality alone is insufficient.
    for r in old:
        r["environment"] = {"version": "md-2026-10", "fingerprint": "old"}
    out.write_text("".join(json.dumps(r) + "\n" for r in old))
    with pytest.raises(ValueError, match="environment"):
        done_keys(out, {"version": "md-2026-10", "fingerprint": "new"})
    with out.open("a") as stream:
        stream.write(json.dumps(old[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        done_keys(out, {"version": "md-2026-10", "fingerprint": "old"})


def test_batch_manifest_identity_and_exclusive_writer(tmp_path):
    from ygorl.solver.resume import bind_output, output_lock

    out = tmp_path / "batch.jsonl"
    identity = {"environment": {"version": "md", "fingerprint": "a"}, "decks": {"a": [1, 2]},
                "implementation": {"solver": "binary1"}, "search": {"seed": 2}}  # fmt: skip
    with output_lock(out):
        bind_output(out, identity)
        with pytest.raises(ValueError, match="active writer"), output_lock(out):
            pass
    with output_lock(out):
        bind_output(out, identity)
        for field in identity:
            with pytest.raises(ValueError, match="identity mismatch"):
                bind_output(out, {**identity, field: "changed"})
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text('{"partial":')
    with output_lock(legacy), pytest.raises(ValueError, match="no batch manifest"):
        bind_output(legacy, identity)
    assert legacy.read_text() == '{"partial":'


def test_solver_content_identity_tracks_edits(tmp_path):
    from ygorl.solver.resume import tree_digest

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.py").write_text("x = 1")
    (b / "one.py").write_text("x = 1")
    assert tree_digest(a, "*.py") == tree_digest(b, "*.py")
    (b / "one.py").write_text("x = 2")
    assert tree_digest(a, "*.py") != tree_digest(b, "*.py")


def test_resume_fire_uses_persisted_plain_line(tmp_path, monkeypatch):
    from ygorl.solver import HandJob
    from ygorl.solver import batch

    base = Demonstration.new(SNAKE, [], hand_index=0, hand_seed=7, variant="plain", targets=["48452496"])
    base.status = "unsolved"
    job = HandJob(deck_path=tmp_path / "deck.ydk", hand_index=0, hand_seed=7, targets=("48452496",),
                  workdir=tmp_path, scratch=tmp_path / "scratch", fire=(ASH_BLOSSOM,))  # fmt: skip

    def unexpected(*args, **kwargs):
        pytest.fail("resuming a missing fire record must not replace the plain search")

    monkeypatch.setattr(batch, "solve_hand", unexpected)
    result = batch.resume_job((job, base))
    assert result[0] == base.to_json()
    assert result[1]["variant"] == "fire" and result[1]["status"] == "skipped"
