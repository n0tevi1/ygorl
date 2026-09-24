"""Split-K weight gradients (ygorl.nets.gemm) and training on a GPU device."""

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from ygorl.nets.gemm import SplitKLinear, tn_mm, use_split_k  # noqa: E402

rocm = pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip), reason="needs a ROCm GPU")


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 48, bias=False))


def _grads(model, x):
    x = x.detach().clone().requires_grad_()
    model(x).square().sum().backward()
    return [p.grad for p in model.parameters()] + [x.grad]


def test_use_split_k_keeps_parameters_and_cpu_results():
    plain = _mlp()
    split = use_split_k(copy.deepcopy(plain))
    assert [type(m) for m in split] == [SplitKLinear, nn.ReLU, SplitKLinear]
    assert split.state_dict().keys() == plain.state_dict().keys()
    x = torch.randn(5000, 64)
    for a, b in zip(_grads(plain, x), _grads(split, x)):
        assert torch.equal(a, b)  # off ROCm the plain path runs
    assert torch.equal(tn_mm(x, x), x.t() @ x)


@rocm
@pytest.mark.parametrize("rows", [4096, 51_213])
def test_tn_mm_matches_on_rocm(rows):
    a = torch.randn(rows, 64, device="cuda", dtype=torch.float64)
    b = torch.randn(rows, 48, device="cuda", dtype=torch.float64)
    torch.testing.assert_close(tn_mm(a, b), a.t() @ b)


@rocm
def test_split_k_linear_gradients_match_on_rocm():
    plain = _mlp().cuda()
    split = use_split_k(copy.deepcopy(plain))
    x = torch.randn(3, 20_000, 64, device="cuda")
    for a, b in zip(_grads(plain, x), _grads(split, x)):  # float32 sums over 60k rows in another order
        assert float((a - b).norm() / a.norm()) < 1e-5


@rocm
def test_trainer_step_on_gpu(tmp_path):
    from pathlib import Path

    from ygorl.train.checkpoint import load_checkpoint
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig, Trainer

    decks = tuple(str(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))[:2])
    cfg = TrainConfig(decks=decks, num_envs=4, steps=8, device="cuda", snapshot_every=1, checkpoint_every=0,
                      eval_every=0, ppo=PPOConfig(epochs=1, minibatch_size=16))  # fmt: skip
    trainer = Trainer(cfg, tmp_path, log=None)
    rec = trainer.step()
    assert rec["rows"] == 32 and rec["update"] == 1
    trainer.pool.add(trainer.model, update=1)
    path = trainer.save()
    state = load_checkpoint(path)  # loads on the CPU
    assert all(t.device.type == "cpu" for t in state["learner"]["model"].values())
    resumed = Trainer.resume(path, tmp_path / "resumed", log=None)
    assert next(resumed.model.parameters()).is_cuda
    assert resumed.step()["update"] == 2


@rocm
def test_overlapped_bf16_training_on_gpu(tmp_path):
    from pathlib import Path

    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig, Trainer

    decks = tuple(str(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))[:2])
    cfg = TrainConfig(decks=decks, num_envs=4, steps=8, device="cuda", overlap_collect=True, bf16=True,
                      snapshot_every=1, checkpoint_every=0, eval_every=0, ppo=PPOConfig(epochs=1, minibatch_size=16))  # fmt: skip
    trainer = Trainer(cfg, tmp_path, log=None)
    assert trainer._stream is not None  # noqa: SLF001
    records = [trainer.step() for _ in range(3)]
    assert [r["update"] for r in records] == [1, 2, 3] and all(r["rows"] == 32 for r in records)
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in records)
