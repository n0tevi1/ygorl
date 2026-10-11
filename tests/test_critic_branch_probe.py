# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
from critic_branch_probe import branch_squared_error, relative_branch_squared_error


def test_root_weight_does_not_grow_with_action_count_and_padding_is_ignored():
    pred = torch.tensor([[2.0, float("nan"), 9.0], [2.0, 2.0, 2.0]], requires_grad=True)
    target = torch.zeros_like(pred, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss = branch_squared_error(pred, target, mask)
    torch.testing.assert_close(loss, torch.tensor([4.0, 4.0], dtype=torch.double))
    loss.mean().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[2.0, 0.0, 0.0], [2 / 3, 2 / 3, 2 / 3]]))
    assert target.grad is None


def test_selected_control_updates_only_original_action_and_full_arm_recovers_other_action():
    pred = torch.tensor([[-2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    selected = torch.tensor([1])
    loss = branch_squared_error(pred, target, mask, selected=selected)
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), pred)[0], torch.tensor([[0.0, 6.0]]))
    all_loss = branch_squared_error(pred, target, mask)
    torch.testing.assert_close(torch.autograd.grad(all_loss.sum(), pred)[0], torch.tensor([[-3.0, 3.0]]))
    with pytest.raises(ValueError, match="supported"):
        branch_squared_error(pred, target, torch.tensor([[True, False]]), selected=selected)
    with pytest.raises(ValueError, match="shape"):
        branch_squared_error(pred, target[:, :1], mask)


def test_relative_objective_matches_pair_differences_and_ignores_state_offset():
    pred = torch.tensor([[3.0, 1.0, float("nan")]], requires_grad=True)
    target = torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = relative_branch_squared_error(pred, target, mask)
    # Pair errors are [0, 4, -4, 0]: half their mean square is 4.
    torch.testing.assert_close(loss, torch.tensor([4.0], dtype=torch.double))
    torch.testing.assert_close(relative_branch_squared_error(pred + 10, target - 3, mask), loss)
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[2.0, -2.0, 0.0]]))
    assert target.grad is None
