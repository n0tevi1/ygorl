import torch
from torch import nn

from ygorl.nets.battle_residual import BattleResidualHead


def test_correction_starts_at_exact_baseline_and_keeps_numeric_model_frozen():
    torch.manual_seed(1)
    base = nn.Sequential(nn.Linear(12, 8), nn.Dropout(0.5)).eval()
    model = BattleResidualHead(base, width=15).train()
    x, context = torch.randn(7, 12), torch.randn(7, 15)
    before = {k: v.clone() for k, v in model.numeric.state_dict().items()}
    out, gate = model(x, context)
    torch.testing.assert_close(out, base(x), rtol=0, atol=0)
    assert not model.numeric.training
    assert all(p.requires_grad for p in base.parameters())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    (out.square().mean() + gate.sigmoid().mean()).backward()
    assert all(p.grad is None for p in model.numeric.parameters())
    optimizer.step()
    for k, v in model.numeric.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert not torch.equal(model(x, context)[0], out)


def test_gate_scales_correction_and_checkpoint_roundtrip():
    base = nn.Linear(12, 8)
    model = BattleResidualHead(base, width=15)
    with torch.no_grad():
        model.correction[-1].bias[:8].fill_(2)
        model.correction[-1].bias[8] = 0
    x, context = torch.randn(4, 12), torch.randn(4, 15)
    out, _ = model(x, context)
    torch.testing.assert_close(out, base(x) + 1)
    restored = BattleResidualHead(base, width=15)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(out, restored(x, context)[0], rtol=0, atol=0)
