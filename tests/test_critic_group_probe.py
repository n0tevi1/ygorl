# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_group_probe import coefficients, objective


@pytest.mark.parametrize("temperature", [None, 0.25, 0.5])
def test_two_pass_gradient_matches_full_objective(temperature):
    pred = torch.tensor([[0.2, 0.3], [-0.4, 0.6], [0.9, 0.8], [-0.3, -0.5]], dtype=torch.double, requires_grad=True)
    target = torch.tensor([1, 1, -1, -1], dtype=torch.double)
    weight = torch.tensor([0.5, 0.5, 1.0, 2.0], dtype=torch.double)
    groups = torch.tensor([0, 0, 1, 2])
    loss, losses, shares, _ = objective(pred, target, weight, groups, temperature)
    (expected,) = torch.autograd.grad(loss, pred)
    coeff = coefficients(pred.detach(), target, weight, groups, temperature)
    for start, end in [(0, 1), (1, 4)]:
        (coeff[start:end] * pred[start:end]).sum().backward()
    torch.testing.assert_close(pred.grad, expected, rtol=1e-12, atol=1e-12)
    if temperature is None:
        direct = ((pred - target[:, None]).square().mean(-1) * weight).sum() / weight.sum()
        torch.testing.assert_close(loss, direct)
    else:
        assert shares.argmax() == losses.argmax()


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        objective(torch.zeros(1, 2), torch.zeros(1), torch.ones(1), torch.zeros(1, dtype=torch.long), temperature)
