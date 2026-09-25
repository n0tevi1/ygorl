"""Policy-vs-policy cells of the agent matrix on the batched C++ path (#87)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.agents import AgentSpec  # noqa: E402
from ygorl.cards.cdb import CardVocab  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.engine.duel import DuelConfig, default_cards  # noqa: E402
from ygorl.eval import agent_matrix as am  # noqa: E402
from ygorl.nets import NetConfig, PolicyNet  # noqa: E402
from ygorl.nets.agent import save_checkpoint  # noqa: E402

DECKS = Path(__file__).parent / "decks"
THREE = [load_ydk(DECKS / f"{n}.ydk") for n in ("snake_eye", "kashtira", "yubel")]
SHORT = DuelConfig(max_turns=4)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    torch.set_num_threads(1)
    vocab = CardVocab.from_db(default_cards())
    out = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1))
        out.append(save_checkpoint(tmp_path_factory.mktemp("p") / "p.pt", net, vocab, event_length=16))
    return out


def _count_paths(monkeypatch):
    calls = {"batched": 0, "arena": 0}
    real_batched, real_arena = am._play_batched, am.Arena.play

    def batched(cells, *a, **k):
        out = real_batched(cells, *a, **k)
        calls["batched"] += len(out)
        return out

    def arena(self, specs):
        calls["arena"] += len(specs)
        return real_arena(self, specs)

    monkeypatch.setattr(am, "_play_batched", batched)
    monkeypatch.setattr(am.Arena, "play", arena)
    return calls


def test_policy_cells_take_the_batched_path_and_rule_agents_the_arena(ckpts, monkeypatch):
    calls = _count_paths(monkeypatch)
    agents = {"p": AgentSpec(f"policy:{ckpts[0]}"), "q": AgentSpec(f"policy:{ckpts[1]}"), "greedy": AgentSpec("greedy")}
    m = am.build_agent_matrix(agents, THREE, pairings=2, seed=1, config=SHORT, device="cpu")
    assert m.batched and calls["batched"] == 1  # the one policy-vs-policy cell
    assert calls["arena"] == 2 * 4 * 2  # greedy's two cells, 4 games per pairing
    i, j = m.index("p"), m.index("q")
    assert m.games[i][j] + m.errors[i][j] == 8 and m.win_rate[i][j] + m.win_rate[j][i] == 1.0


def test_greedy_policies_play_the_same_games_on_both_paths(ckpts):
    """argmax policies are deterministic, so the batched cell must reproduce the arena's cell game for game."""
    agents = {"p": AgentSpec(f"policy:{ckpts[0]}@greedy"), "q": AgentSpec(f"policy:{ckpts[1]}@greedy")}
    fast = am.build_agent_matrix(agents, THREE, pairings=3, seed=2, config=SHORT, device="cpu")
    slow = am.build_agent_matrix(agents, THREE, pairings=3, seed=2, config=SHORT)
    assert fast.batched and not slow.batched
    assert (fast.win_rate, fast.games, fast.errors) == (slow.win_rate, slow.games, slow.errors)


def test_common_random_numbers_hold_on_the_batched_path(ckpts):
    """Sampled decisions are seeded by deck slot: copies named before and after their opponent play the same games."""
    agents = {"a": AgentSpec(f"policy:{ckpts[0]}"), "m": AgentSpec(f"policy:{ckpts[1]}"),
              "z": AgentSpec(f"policy:{ckpts[0]}")}  # fmt: skip
    m = am.build_agent_matrix(agents, THREE, pairings=3, seed=3, config=SHORT, device="cpu")
    a, mid, z = m.index("a"), m.index("m"), m.index("z")
    assert m.win_rate[a][mid] == m.win_rate[z][mid] and m.win_rate[a][z] == 0.5


def test_a_batched_matrix_is_extended_on_the_batched_path(ckpts, monkeypatch):
    agents = {"p": AgentSpec(f"policy:{ckpts[0]}"), "greedy": AgentSpec("greedy")}
    m = am.build_agent_matrix(agents, THREE, pairings=1, seed=4, config=SHORT, device="cpu")
    calls = _count_paths(monkeypatch)
    grown = am.extend_agent_matrix(m, {"q": AgentSpec(f"policy:{ckpts[1]}")}, THREE, config=SHORT)
    assert grown.batched and calls["batched"] == 1 and calls["arena"] == 4  # q-p batched, q-greedy in the arena
    plain = am.build_agent_matrix(agents, THREE, pairings=1, seed=4, config=SHORT)
    with pytest.raises(ValueError, match="without the batched path"):
        am.extend_agent_matrix(plain, {"q": AgentSpec(f"policy:{ckpts[1]}")}, THREE, config=SHORT, device="cpu")
