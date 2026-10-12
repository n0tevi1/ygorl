# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
from critic_gradient_projection import project_root_gradients


def sequential_reference(gradients, orders):
    original = gradients.double()
    output = original.clone()
    for i, order in enumerate(orders.tolist()):
        for j in order:
            norm = original[j].square().sum()
            if i != j and norm > 0:
                dot = output[i] @ original[j]
                if dot < 0:
                    output[i] -= dot / norm * original[j]
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_span_projection_matches_sequential_reference_without_mutation(dtype):
    rng = torch.Generator().manual_seed(731)
    gradients = torch.randn(12, 31, generator=rng, dtype=dtype)
    gradients[0].zero_()
    gradients.requires_grad_()
    before = gradients.detach().clone()
    orders = torch.stack([torch.randperm(12, generator=rng) for _ in range(12)])
    expected = sequential_reference(gradients.detach(), orders)
    output, diagnostic = project_root_gradients(gradients, orders)
    torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(gradients, before, rtol=0, atol=0)
    assert not output.requires_grad
    assert diagnostic["projections"] > 0


def test_known_conflict_and_zero_roots():
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, 0.0]])
    output, _ = project_root_gradients(gradients, torch.arange(3).expand(3, 3))
    torch.testing.assert_close(output, torch.tensor([[0.5, 0.5], [0.0, 1.0], [0.0, 0.0]], dtype=torch.double))


@pytest.mark.parametrize("gradients", [torch.zeros(3, 4), torch.ones(4, 3), torch.eye(4)])
def test_nonconflicting_roots_are_unchanged(gradients):
    n = len(gradients)
    output, diagnostic = project_root_gradients(gradients, torch.arange(n).expand(n, n))
    torch.testing.assert_close(output, gradients.double(), atol=0, rtol=0)
    assert diagnostic["projections"] == 0


@pytest.mark.parametrize(
    "gradients",
    [torch.empty(0, 3), torch.empty(3, 0), torch.tensor([[float("nan")]]), torch.ones(2, 3, dtype=torch.long)],
)
def test_invalid_gradient_inputs_are_rejected(gradients):
    with pytest.raises(ValueError):
        project_root_gradients(gradients, torch.zeros(2, 2, dtype=torch.long))


@pytest.mark.parametrize(
    "orders",
    [
        torch.zeros(2, 2, dtype=torch.long),
        torch.tensor([[0, 2], [0, 1]]),
        torch.eye(2),
        torch.zeros(2, 3, dtype=torch.long),
    ],
)
def test_invalid_permutations_are_rejected(orders):
    with pytest.raises(ValueError):
        project_root_gradients(torch.ones(2, 3), orders)
