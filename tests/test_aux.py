"""Auxiliary critic targets and heads (#440)."""

import pytest

torch = pytest.importorskip("torch")

from ygorl.train.aux import AUX_TARGETS, aux_targets  # noqa: E402


def _obs(rows):
    """rows: list of (turn, my_lp, op_lp, own_monsters, opp_monsters) in rollout order t * B + b."""
    n = len(rows)
    g = torch.zeros(n, 22)
    cards = torch.zeros(n, 160, 23)
    for i, (turn, my, op, own, opp) in enumerate(rows):
        g[i, 3], g[i, 5], g[i, 6] = turn, my, op
        k = 0
        for _ in range(own):
            cards[i, k, 1], cards[i, k, 4] = 3, 0
            k += 1
        for _ in range(opp):
            cards[i, k, 1], cards[i, k, 4] = 3, 1
            k += 1
    return {"globals": g, "cards": cards}


def test_targets_look_at_the_first_row_after_the_turn_ends_from_the_deciders_view():
    # one column, one game: seat 0 acts on turn 1 (two rows), seat 1 acts on turn 2, seat 0 on turn 3
    rows = [(1, 8000, 8000, 1, 0), (1, 8000, 8000, 2, 0), (2, 8000, 7000, 1, 2), (3, 6000, 7000, 2, 1)]
    players = torch.tensor([[0], [0], [1], [0]])
    dones = torch.zeros(4, 1, dtype=torch.bool)
    t, v = aux_targets(_obs(rows), players, dones)
    assert t.shape == (4, 1, len(AUX_TARGETS)) and v.tolist() == [[True], [True], [True], [False]]
    # turn-1 rows target the turn-2 row, seen by seat 1: seat 0's opponent monsters = seat 1's own (1), own = 2
    assert t[0, 0].tolist() == pytest.approx([1 / 5, 2 / 5, ((7000 - 8000) - 0) / 8000])
    assert t[1, 0].tolist() == pytest.approx([1 / 5, 2 / 5, ((7000 - 8000) - 0) / 8000])
    # the turn-2 row (seat 1, lp 8000/7000 from its view) targets the turn-3 row seen by seat 0 (6000/7000)
    assert t[2, 0].tolist() == pytest.approx([2 / 5, 1 / 5, ((7000 - 6000) - 1000) / 8000])


def test_targets_do_not_cross_an_episode_end():
    rows = [(1, 8000, 8000, 1, 0), (1, 8000, 8000, 1, 0), (1, 8000, 8000, 0, 3)]
    players = torch.tensor([[0], [0], [0]])
    dones = torch.tensor([[False], [True], [False]])  # game ends after row 1; row 2 starts a new game
    t, v = aux_targets(_obs(rows), players, dones)
    assert v.tolist() == [[False], [False], [False]]


def test_critic_with_aux_heads_outputs_one_vector_per_candidate():
    from ygorl.train.critic import Critic, aux_loss

    c = Critic(history_dim=8, action_dim=8, privileged_dim=0, hidden=16, aux_dim=3)
    out = c(torch.randn(5, 8), torch.randn(5, 4, 8), torch.ones(5, 4, dtype=torch.bool))
    assert out.aux.shape == (5, 4, 3) and out.q.shape == (5, 4)
    loss = aux_loss(out.aux, torch.zeros(5, dtype=torch.long), torch.zeros(5, 3), torch.ones(5, dtype=torch.bool))
    assert torch.isfinite(loss)
    assert Critic(8, 8, 0, 16)(torch.randn(2, 8), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool)).aux is None
