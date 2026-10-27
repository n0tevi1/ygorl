"""Smoke tests of the phase-0 offline tools (docs/spikes/first-seat-interruption-plan-2026-10-24.md §8 T3): each tool
runs end to end on a two-hand sample with a tiny untrained checkpoint, and the same inputs and seed give exactly the
same report twice (timings aside).

``tests/data/line_demos_ash.jsonl`` holds two solved plain records with Ash windows (branded hand 41, sky-striker hand
41), ``line_labels_graded.jsonl`` their graded labels, ``line_demos_ash_fire.jsonl`` their ``fire`` records from
``out/claude-scale/handtrap/full_fire8.jsonl`` (verified recovery lines; the environment binding removed)."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import line_break_census  # noqa: E402
import turn1_board_modes  # noqa: E402
import turn1_exit_pilot  # noqa: E402
import window_counterfactual_probe  # noqa: E402
from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.nets import NetConfig  # noqa: E402
from ygorl.train.checkpoint import CriticConfig, build_actor_critic, save_checkpoint, vocab_to_text  # noqa: E402
from ygorl.train.trainer import TrainConfig  # noqa: E402

DATA = Path(__file__).parent / "data"
DEMOS = str(DATA / "line_demos_ash.jsonl")
FIRE = str(DATA / "line_demos_ash_fire.jsonl")
GRADED = str(DATA / "line_labels_graded.jsonl")
LINES = str(DATA / "line_demos.jsonl")
DECKS = [str(Path(__file__).parent / "decks" / "kashtira.ydk")]
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A PPO checkpoint (privileged critic, separate critic trunk as the runs use) around a fresh tiny actor-critic."""
    tmp = tmp_path_factory.mktemp("ckpt")
    vocab = CardVocab.from_db(CardDB.load())
    cfg = NetConfig(vocab_size=len(vocab), **TINY)
    critic = CriticConfig(shared_backbone=False)
    torch.manual_seed(3)
    model = build_actor_critic(cfg, None, critic)
    train = TrainConfig(decks=("x.ydk",), event_length=16, privileged_critic=True, privileged_dim=critic.privileged_dim,
                        critic_hidden=critic.hidden, shared_backbone=False)  # fmt: skip
    state = {"config": train.to_dict(), "net_config": cfg.to_dict(), "vocab": vocab_to_text(vocab, tmp / "v.json"),
             "environment": None, "learner": {"model": model.state_dict(), "updates": 1}}  # fmt: skip
    return str(save_checkpoint(state, tmp / "tiny.pt"))


def _twice(main, args, tmp_path):
    reports = []
    for i in range(2):
        out = tmp_path / f"r{i}.json"
        main([*args, "--out", str(out)])
        report = json.loads(out.read_text())
        report.pop("seconds")
        reports.append(report)
    assert reports[0] == reports[1]
    return reports[0]


def test_line_break_census(ckpt, tmp_path):
    r = _twice(line_break_census.main, [ckpt, "--lines", DEMOS, "--decks", *DECKS, "--games", "2", "--max-turns", "4", "--envs", "2",
                                        "--threads", "1", "--torch-threads", "1", "--rows"], tmp_path)  # fmt: skip
    assert r["lines"] == 2 and set(r["seats"]) == {"first", "second"}
    for seat in r["seats"].values():
        assert seat["games"] == 2
        assert sum(seat["status_counts"].values()) == 2
        assert sum(seat["cause_counts"].values()) == seat["status_counts"].get("broken", 0)
    rows = r["rows"]
    assert len(rows) == 4 and all(x["status"] in ("completed", "broken", "playing") for x in rows)
    assert all((x["cause"] is None) == (x["status"] != "broken") for x in rows)


def test_window_counterfactual_probe(ckpt, tmp_path):
    r = _twice(window_counterfactual_probe.main, [GRADED, ckpt, "--demos", DEMOS, FIRE, "--min-hand", "0",
                                                  "--rollouts", "2", "--workers", "1"], tmp_path)  # fmt: skip
    assert r["hands"] == 2 and r["hands_with_recovery_lines"] == 2
    assert r["windows"] == 5 and r["windows_reached"] == 5  # branded 3 windows, sky-striker 2
    assert r["windows_with_recovery"] >= 1
    labels = {(x["deck"], x["window"]): x["label"] for x in r["rows"]}
    assert labels[("sky-striker", 1)] == "hold" and labels[("sky-striker", 2)] == "fire"  # bait, then the starter
    assert labels[("branded", 1)] == "hold" and labels[("branded", 2)] == "fire"
    assert set(r["summary"]) == {f"{m}/{t}" for m in window_counterfactual_probe.MODES for t in ("critic", "board")}
    for s in r["summary"].values():
        assert s["windows"] == 5
        assert sum(sum(c.values()) for c in s["confusion"].values()) == 5
    # the recovery line is taken where Ash was fired at a recoverable window
    assert sum(x["recovery_followed"] for x in r["rows"]) >= 1
    assert all(x["recovery_mismatch"] == 0 for x in r["rows"])


def test_window_counterfactual_probe_joins_the_fire_records_of_a_hand(ckpt, tmp_path):
    """Several fire records of one hand (a graded run's levels): every record's windows are used, the first record
    recovering a window gives its line; one record per line, the last first, changes nothing here."""
    split = tmp_path / "split_fire.jsonl"
    with open(FIRE) as src, open(split, "w") as dst:
        for line in src:
            rec = json.loads(line)
            for ln in reversed(rec["lines"]):
                dst.write(json.dumps({**rec, "lines": [ln]}) + "\n")
    args = [GRADED, ckpt, "--min-hand", "0", "--rollouts", "1", "--workers", "1"]
    whole = window_counterfactual_probe.main([*args, "--demos", DEMOS, FIRE, "--out", str(tmp_path / "a.json")])
    parts = window_counterfactual_probe.main([*args, "--demos", DEMOS, str(split), "--out", str(tmp_path / "b.json")])
    assert parts["windows_with_recovery"] == whole["windows_with_recovery"] >= 1
    assert [r["has_recovery"] for r in parts["rows"]] == [r["has_recovery"] for r in whole["rows"]]
    assert sum(r["recovery_followed"] for r in parts["rows"]) >= 1
    assert all(r["recovery_mismatch"] == 0 for r in parts["rows"])


def test_turn1_board_modes(ckpt, tmp_path):
    r = _twice(turn1_board_modes.main, [DEMOS, "--ckpt", ckpt, "--hands", "2", "--samples", "3", "--min-hand", "0",
                                        "--max-hand", "100", "--workers", "1"], tmp_path)  # fmt: skip
    assert r["summary"]["hands"] == 2
    for row in r["rows"]:
        assert row["samples"] + row["failures"] == 3
        assert 0 < row["field_grave"]["top3"] <= 1 and row["field_grave"]["distinct"] >= 1


def test_turn1_exit_pilot(ckpt, tmp_path):
    r = _twice(turn1_exit_pilot.main, [DEMOS, "--ckpt", ckpt, "--lines", DEMOS, LINES, "--real-opponent", ckpt,
                                       "--decks", *DECKS, "--hands", "2", "--min-hand", "0", "--max-hand", "100",
                                       "--candidates", "3", "--opponents", "2", "--real-games", "1", "--max-turns", "4", "--envs", "4",
                                       "--threads", "1", "--torch-threads", "1"], tmp_path)  # fmt: skip
    s = r["summary"]
    assert s["hands"] == 2
    for row in r["rows"]:
        assert row["candidates"] == 3 and len(row["J"]) == 3 and all(len(j) == 2 for j in row["J"])
        assert row["pairs"] == 6
    assert set(s["realised"]) == {"line", "free"}
    assert all(v["games_per_arm"] == 2 for v in s["realised"].values())


def test_line_break_census_counts_interruption_cards_and_recovery_windows(ckpt, tmp_path):
    """First-seat plan §5.1b / §10.2c: activations on the line's turn by card, the recovery-line coverage of the
    windows of a playing line, and the switches (the collector's interrupt)."""
    r = _twice(line_break_census.main, [ckpt, "--lines", DEMOS, "--line-recovery", FIRE, "--decks", *DECKS, "--games",
                                        "4", "--max-turns", "4", "--envs", "2", "--threads", "1", "--torch-threads",
                                        "1", "--rows"], tmp_path)  # fmt: skip
    assert r["lines_with_recovery"] >= 1 and set(r["recovery_lines_by_card"]) == {"14558127"}
    for name, seat in r["seats"].items():
        rows = [x for x in r["rows"] if ("first" if x["seat"] == 0 else "second") == name]
        assert (
            sum(seat["fire_cards"].values()) == sum(x["fires"] for x in rows) == sum(len(x["fire_log"]) for x in rows)
        )
        assert sum(seat["fire_cards_playing"].values()) == sum(x["fires_before_break"] for x in rows)
        rec = seat["recovery"]
        assert rec["windows_with_recovery"] <= rec["windows_playing"]
        assert rec["switches"] == sum(f["switched"] for x in rows for f in x["fire_log"]) <= rec["fires_with_recovery"]


def test_window_counterfactual_probe_reads_one_card_of_a_multi_card_graded_file(ckpt, tmp_path):
    """``--fire`` picks the card's records of a graded file with several hand traps (default Ash): the Ash rows are
    those of the Ash-only file."""
    multi = tmp_path / "graded_multi.jsonl"
    with open(GRADED) as f, open(multi, "w") as out:
        for line in f:
            rec = json.loads(line)
            out.write(json.dumps({**rec, "fire": 23434538, "values": {k: 0 for k in rec["values"]}}) + "\n")
            out.write(line)
    args = [ckpt, "--demos", DEMOS, FIRE, "--min-hand", "0", "--rollouts", "1", "--workers", "1"]
    alone = window_counterfactual_probe.main([GRADED, *args, "--out", str(tmp_path / "a.json")])
    mixed = window_counterfactual_probe.main([str(multi), *args, "--out", str(tmp_path / "b.json")])
    assert mixed["fire"] == 14558127 and mixed["rows"] == alone["rows"]
