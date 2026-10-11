# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from declaration_probe import chunked_logits, declaration_rows
from tests.test_nets import random_obs, small
from ygorl.cards.cdb import CardVocab
from ygorl.engine.actions import Action
from ygorl.nets import PolicyNet, collate


def test_declaration_keeps_engine_order_unknowns_and_tail():
    vocab = CardVocab([7, 9])
    actions = [Action("declare", value=x) for x in [9, 100, 7] * 100]
    rows = declaration_rows(actions, vocab)
    assert rows.shape == (300, 10)
    assert rows[:, 2].tolist() == [vocab.index(a.value) for a in actions]
    assert not rows[:, [1, 3, 4, 5, 6, 7, 8, 9]].any()
    with pytest.raises(ValueError):
        declaration_rows([Action("pass")], vocab)


@pytest.mark.parametrize("chunk", [1, 127, 128, 256, 1000])
def test_full_dense_and_chunked_logits_and_global_distribution_match(chunk):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    net = PolicyNet(small(history="none")).eval()
    obs = random_obs(np.random.default_rng(19))
    rows = np.zeros((301, 10), dtype=np.int32)
    from ygorl.env.encoding import ACTION_KINDS

    rows[:, 0] = ACTION_KINDS.index("declare") + 1
    rows[:, 2] = np.arange(301) % 299 + 1
    obs["globals"][20] = len(rows)
    batch = collate([obs])
    with torch.no_grad():
        f = net.features(batch)
        full = {
            **batch,
            "actions": torch.as_tensor(rows).unsqueeze(0),
            "action_mask": torch.ones(1, len(rows), dtype=torch.bool),
        }
        expected = net(full).logits
        actual = chunked_logits(net, f, full["actions"], chunk_size=chunk)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1))
    assert actual.softmax(-1)[:, 128:].sum() > 0.5
