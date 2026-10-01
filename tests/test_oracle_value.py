"""The offline oracle value network (docs/oracle-value.md): inputs, output range, save / load."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardVocab  # noqa: E402
from ygorl.env.privileged import P_NEXT, P_WIDTHS, PRIVILEGED_KEYS  # noqa: E402
from ygorl.nets.oracle_value import OracleValueNet, load_value_net, save_value_net  # noqa: E402

from tests.test_nets import V, random_batch, small  # noqa: E402


def random_privileged(b: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    widths = {**P_WIDTHS, "my_next": P_NEXT, "op_next": P_NEXT}
    out = {k: torch.as_tensor(np.stack([rng.integers(0, 2, (n, 3)) * [rng.integers(0, V), 1, 1]
                                        for _ in range(b)])).long() for k, n in widths.items()}  # fmt: skip
    out["counts"] = torch.as_tensor(rng.integers(0, 40, (b, 5))).long()
    assert set(out) == set(PRIVILEGED_KEYS)
    return out


@pytest.mark.parametrize("inputs", ["public", "privileged", "oracle"])
def test_value_in_range_and_trains(inputs):
    torch.manual_seed(0)
    net = OracleValueNet(small(), inputs=inputs, privileged_dim=16, hidden=32)
    obs, priv = random_batch(4), random_privileged(4)
    v = net(obs, priv)
    assert v.shape == (4,) and bool((v.abs() <= 1).all())
    (v - torch.tensor([1.0, -1.0, 0.0, 1.0])).pow(2).mean().backward()
    assert net.head[0].weight.grad is not None


def test_deck_order_reaches_only_the_oracle_net():
    torch.manual_seed(0)
    obs, priv = random_batch(2), random_privileged(2)
    changed = dict(priv, my_next=priv["my_next"].roll(1, dims=1))
    for draws in ("order", "identity"):
        oracle = OracleValueNet(small(), inputs="oracle", privileged_dim=16, hidden=32, draws=draws).eval()
        assert not torch.allclose(oracle(obs, priv), oracle(obs, changed))
    plain = OracleValueNet(small(), inputs="privileged", privileged_dim=16, hidden=32).eval()
    assert torch.allclose(plain(obs, priv), plain(obs, changed))
    with pytest.raises(ValueError):
        plain(obs, None)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = OracleValueNet(small(), inputs="oracle", privileged_dim=16, hidden=32).eval()
    vocab = CardVocab(range(1000, 1000 + V - CardVocab.FIRST_INDEX))
    save_value_net(net, vocab, 16, tmp_path / "vn.pt", epoch=3)
    loaded, vocab2, length = load_value_net(tmp_path / "vn.pt")
    obs, priv = random_batch(3), random_privileged(3)
    assert length == 16 and len(vocab2) == len(vocab)
    assert torch.allclose(net(obs, priv), loaded(obs, priv))


@pytest.mark.parametrize("inputs,draws", [("privileged", "order"), ("oracle", "order"), ("oracle", "identity")])
def test_from_actor_critic_computes_the_critics_v(inputs, draws):
    from ygorl.train.checkpoint import CriticConfig, build_actor_critic

    torch.manual_seed(0)
    ac = build_actor_critic(small(), None, CriticConfig(privileged_dim=16, hidden=32)).eval()
    net = OracleValueNet.from_actor_critic(ac, inputs, draws).eval()
    obs, priv = random_batch(4), random_privileged(4)
    assert torch.allclose(net(obs, priv), ac(obs, priv).v, atol=1e-6)
