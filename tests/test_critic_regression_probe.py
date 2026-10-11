# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
from critic_regression_probe import calibrated_squared_error


def test_wrong_saturation_has_a_recovery_gradient_only_without_clamping():
    raw = torch.tensor([[2.0, -2.0]], requires_grad=True)
    offset = torch.tensor([[0.5, -0.5]], requires_grad=True)
    target = torch.tensor([-1.0])
    clipped = calibrated_squared_error(raw, target, offset, clip_prediction=True)
    clipped.sum().backward()
    assert bool((raw.grad == 0).all())
    assert offset.grad is None
    raw.grad = None
    loss = calibrated_squared_error(raw, target, offset, clip_prediction=False)
    loss.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[3.5, -1.5]]))
    assert offset.grad is None
    before = raw.detach().clone()
    after = before - 0.1 * raw.grad
    assert calibrated_squared_error(after, target, offset, clip_prediction=False) < loss


def test_in_range_modes_match_values_and_gradients_and_reject_shape_broadcasting():
    raw = torch.tensor([[0.1, -0.2], [0.3, 0.4]], requires_grad=True)
    offset = torch.full_like(raw, 0.1)
    target = torch.tensor([1.0, -1.0])
    a = calibrated_squared_error(raw, target, offset, clip_prediction=True)
    b = calibrated_squared_error(raw, target, offset, clip_prediction=False)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a.sum(), raw)[0], torch.autograd.grad(b.sum(), raw)[0])
    with pytest.raises(ValueError):
        calibrated_squared_error(raw, target[:, None], offset, clip_prediction=False)
