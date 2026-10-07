"""Distribution, fallback, and frozen-feature contracts for experimental filtering."""

import pytest

torch = pytest.importorskip("torch")

from ygorl.nets.action_filter import ActionRiskHead, soft_filter  # noqa: E402


def test_filter_preserves_legality_support_and_actual_log_probs():
    logits = torch.tensor([[1.0, 0.0, 100.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    out = soft_filter(logits, torch.tensor([[1.0, 0.0, float("nan")]]), legal, legal, max_penalty=4)
    p = out.logits.softmax(-1)
    assert p[0, 2] == 0 and (p[legal] > 0).all()
    assert torch.allclose(p[0, :2], torch.tensor([-3.0, 0.0]).softmax(-1))
    logp = out.logits.log_softmax(-1)[0, 0]
    logp.backward()
    assert torch.allclose(logits.grad[0, :2], torch.tensor([1.0, 0.0]) - p.detach()[0, :2])
    assert not out.failed_rows.any()


def test_fallback_is_per_row_and_unsupported_scores_are_ignored():
    logits = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    legal = torch.ones_like(logits, dtype=torch.bool)
    scope = torch.tensor([[True, True, False], [True, True, False]])
    scores = torch.tensor([[1.0, float("nan"), 0.0], [1.0, 0.0, float("inf")]])
    out = soft_filter(logits, scores, legal, scope)
    assert torch.equal(out.logits[0].softmax(-1), logits[0].softmax(-1))
    assert out.failed_rows.tolist() == [True, False]
    assert out.penalty[1].tolist() == [4.0, 0.0, 0.0]
    assert torch.equal(
        soft_filter(logits, scores, legal, torch.zeros_like(scope)).logits.softmax(-1), logits.softmax(-1)
    )


def test_filter_is_equivariant_to_candidate_order():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    risk = torch.tensor([[0.95, 0.2, 0.99]])
    legal = torch.ones_like(logits, dtype=torch.bool)
    perm = torch.tensor([2, 0, 1])
    a = soft_filter(logits, risk, legal, legal).logits
    b = soft_filter(logits[:, perm], risk[:, perm], legal[:, perm], legal[:, perm]).logits
    assert torch.equal(a[:, perm], b)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_supported_score_reverts_to_base(bad):
    base = torch.tensor([[1.0, 2.0]])
    mask = torch.ones_like(base, dtype=torch.bool)
    out = soft_filter(base, torch.tensor([[bad, 0.99]]), mask, mask)
    assert torch.equal(out.logits.softmax(-1), base.softmax(-1)) and out.failed_rows.item()


def test_invalid_actor_or_no_legal_action_raises():
    x = torch.zeros(1, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError):
        soft_filter(x, x, ~mask, mask)
    with pytest.raises(ValueError):
        soft_filter(x + float("nan"), x, mask, mask)
    with pytest.raises(ValueError):
        soft_filter(x, x, mask, mask, max_penalty=float("inf"))


def test_risk_head_trains_without_mutating_frozen_features_and_roundtrips():
    torch.manual_seed(7)
    context, actions = torch.randn(4, 3), torch.randn(4, 2, 3)
    original = (context.clone(), actions.clone())
    head = ActionRiskHead(3)
    head.fit_normalization(head.inputs(context, actions).flatten(0, 1))
    target = torch.tensor([[0.0, 1.0]]).expand(4, 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    before = torch.nn.functional.binary_cross_entropy_with_logits(head(context, actions), target)
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(context, actions), target)
        loss.backward()
        optimizer.step()
    assert loss < before / 3
    assert torch.equal(context, original[0]) and torch.equal(actions, original[1])
    restored = ActionRiskHead(3)
    restored.load_state_dict(head.state_dict())
    assert torch.equal(head(context, actions), restored(context, actions))
    perm = torch.tensor([1, 0])
    assert torch.allclose(head(context, actions[:, perm]), head(context, actions)[:, perm], atol=1e-6)


def test_extreme_legal_logits_cannot_make_padding_selectable():
    x = torch.tensor([[-1e12, -1e12, 0.0]])
    legal = torch.tensor([[True, True, False]])
    out = soft_filter(x, torch.zeros_like(x), legal, legal)
    assert out.logits.softmax(-1).tolist() == [[0.5, 0.5, 0.0]]
    assert torch.isfinite(out.logits).all()


def test_filter_pilot_gate_uses_exact_counts_at_five_percent_boundary():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools/action_filter_pilot.py"
    spec = importlib.util.spec_from_file_location("filter_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {"n": 20, "good_flag_count": 1, "bad_flag_count": 10}
    assert module.gate({"self": row, "opponent": row})
    assert not module.gate({"self": row, "opponent": row}, 30)
    assert not module.gate({"n": 0})
    assert not module.gate({"self": row | {"good_flag_count": 2}, "opponent": row})


def test_extreme_spread_keeps_entropy_finite():
    x = torch.tensor([[3e38, -3e38, 0.0]])
    mask = torch.tensor([[True, True, False]])
    out = soft_filter(x, torch.zeros_like(x), mask, mask)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(-(out.logits.softmax(-1) * out.logits.log_softmax(-1)).sum())
    assert out.logits.softmax(-1).tolist() == [[1.0, 0.0, 0.0]]
