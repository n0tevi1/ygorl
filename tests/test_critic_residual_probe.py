# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_residual_probe import fit_residual, predict_residual


def test_intercept_is_weighted_residual_and_predictions_are_bounded():
    x = torch.empty(3, 0)
    p = torch.tensor([[0.8, 0.7], [-0.2, -0.1], [0.2, 0.1]])
    y = torch.tensor([1.0, -1.0, 1.0])
    w = torch.tensor([0.5, 0.5, 1.0])
    s = fit_residual(x, p, y, w, 0.01)
    expected = ((y.double()[:, None] - p.double()) * w.double()[:, None]).sum(0) / w.sum()
    torch.testing.assert_close(s["beta"][0], expected)
    torch.testing.assert_close(predict_residual(s, x, p), (p.double() + expected).clamp(-1, 1))


def test_duplicate_states_preserve_fit_when_game_weight_is_split():
    x = torch.tensor([[0.0, 2.0], [3.0, 2.0], [1.0, 2.0]])
    p = torch.zeros(3, 2)
    y = torch.tensor([-1.0, 1.0, 1.0])
    s = fit_residual(x, p, y, torch.ones(3), 0.1)
    ix = torch.tensor([0, 1, 1, 2])
    repeated = fit_residual(x[ix], p[ix], y[ix], torch.tensor([1.0, 0.5, 0.5, 1.0]), 0.1)
    for key in ["mean", "scale", "beta"]:
        torch.testing.assert_close(s[key], repeated[key], rtol=1e-10, atol=1e-10)
    assert s["scale"][1] == 1
    before = {k: s[k].clone() for k in ["mean", "scale", "beta"]}
    predict_residual(s, x * 100, p)
    for k in before:
        torch.testing.assert_close(s[k], before[k])


@pytest.mark.parametrize("ridge", [0, -1, float("nan"), float("inf")])
def test_invalid_regularization_rejected(ridge):
    with pytest.raises(ValueError, match="ridge"):
        fit_residual(torch.zeros(1, 1), torch.zeros(1, 2), torch.zeros(1), torch.ones(1), ridge)
