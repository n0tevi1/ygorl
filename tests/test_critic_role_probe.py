"""Game identity, weighting and role stratification for cached critic experiments."""

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_role_probe import combine


def corpus(role, lengths):
    ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    x = torch.arange(len(ids)).float()[:, None]
    return dict(
        ids=ids,
        context=x,
        candidate=x + 10,
        privileged={"x": x + 20},
        resources=x + 30,
        z=torch.ones(len(ids)),
        games=[dict(id=i, target="runick", role=role, draw=i) for i in range(len(lengths))],
        zero_games=[],
    )


def test_subsetting_and_equal_game_weights_with_colliding_source_ids():
    face, own = corpus("face-target", [1, 7]), corpus("own-target", [3, 2])
    d = combine([("face", face, lambda g: g["draw"] == 1), ("own", own, lambda g: True)])
    assert [g["id"] for g in d["games"]] == ["face:1", "own:0", "own:1"]
    assert d["ids"].tolist() == [0] * 7 + [1] * 3 + [2] * 2
    assert d["context"][:, 0].tolist() == list(range(1, 8)) + list(range(5))
    torch.testing.assert_close(d["privileged"]["x"], d["context"] + 20)
    mass = torch.zeros(3).index_add_(0, d["ids"], d["weight"])
    torch.testing.assert_close(mass, torch.full((3,), 4.0))
    assert len(d["group_labels"]) == 2
    assert d["groups"].tolist() == [0] * 7 + [1] * 5


def test_zero_idle_games_remain_in_totals_and_filters():
    d = corpus("own-target", [2])
    d.update(zero_games=[3], zero_records=[dict(id=3, target="runick", role="own-target", draw=3)])
    all_games = combine([("own", d, lambda g: True)])
    assert all_games["total_games"] == 2 and all_games["zero_games"] == ["own:3"]
    selected = combine([("own", d, lambda g: g["draw"] == 0)])
    assert selected["total_games"] == 1 and selected["zero_games"] == []


def test_duplicate_source_is_rejected():
    d = corpus("face-target", [2])
    with pytest.raises(AssertionError, match="duplicate source game"):
        combine([("face", d, lambda g: True), ("face", d, lambda g: True)])
