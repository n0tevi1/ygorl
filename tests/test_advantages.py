"""Advantage estimators and the privileged critic (T4b.3) on toy games with analytic values.

Every toy game is finite and acyclic, so the exact Q^π / V^π come from the Bellman linear system and
every episode can be enumerated with its probability: expectations below are exact weighted sums, not
Monte-Carlo estimates. Values are always from the perspective of the seat to act (docs/training.md).
"""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.train import advantages as adv  # noqa: E402
from ygorl.train.critic import Critic, q_loss, v_loss  # noqa: E402

DT = torch.float64


@dataclass
class Game:
    """A finite acyclic game. ``outcomes[s][a]`` lists ``(prob, next_state | None, reward)``.

    The reward goes to ``players[s]`` (the seat that acted) and ``None`` ends the episode.
    """

    players: list[int]
    outcomes: list[list[list[tuple[float, int | None, float]]]]
    policy: list[list[float]]
    gamma: float = 1.0

    @property
    def n_actions(self) -> int:
        return max(len(o) for o in self.outcomes)

    def sign(self, s: int, s2: int) -> float:
        return 1.0 if self.players[s] == self.players[s2] else -1.0

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact (Q^π [S, A], V^π [S]) from (I - γ M) V = b; zero-sum: V_other = -V."""
        n = len(self.players)
        m, b = np.zeros((n, n)), np.zeros(n)
        for s in range(n):
            for a, outs in enumerate(self.outcomes[s]):
                for p, s2, r in outs:
                    w = self.policy[s][a] * p
                    b[s] += w * r
                    if s2 is not None:
                        m[s, s2] += w * self.sign(s, s2)
        v = np.linalg.solve(np.eye(n) - self.gamma * m, b)
        q = np.zeros((n, self.n_actions))
        for s in range(n):
            for a, outs in enumerate(self.outcomes[s]):
                q[s, a] = sum(p * (r + (0.0 if s2 is None else self.gamma * self.sign(s, s2) * v[s2]))
                              for p, s2, r in outs)  # fmt: skip
        return q, v

    def trajectories(self, start: int = 0) -> list[tuple[float, list[tuple[int, int, float]]]]:
        """Every episode from ``start`` with its probability, as ``(prob, [(state, action, reward)])``."""

        def rec(s):
            for a, outs in enumerate(self.outcomes[s]):
                if self.policy[s][a] == 0:
                    continue
                for p, s2, r in outs:
                    w = self.policy[s][a] * p
                    if s2 is None:
                        yield w, [(s, a, r)]
                    else:
                        for w2, tail in rec(s2):
                            yield w * w2, [(s, a, r), *tail]

        return list(rec(start))

    def batch(self, q: np.ndarray | None = None, v: np.ndarray | None = None, start: int = 0) -> dict:
        """Every episode as one column of a [T, B] layout, padded (valid = False) after its terminal step."""
        trajs = self.trajectories(start)
        T, B, A = max(len(t) for _, t in trajs), len(trajs), self.n_actions
        states = torch.zeros(T, B, dtype=torch.long)
        out = {
            "players": torch.zeros(T, B, dtype=torch.long),
            "actions": torch.zeros(T, B, dtype=torch.long),
            "rewards": torch.zeros(T, B, dtype=DT),
            "dones": torch.zeros(T, B, dtype=torch.bool),
            "valid": torch.zeros(T, B, dtype=torch.bool),
        }
        for b, (_, steps) in enumerate(trajs):
            for t, (s, a, r) in enumerate(steps):
                states[t, b], out["actions"][t, b], out["rewards"][t, b], out["valid"][t, b] = s, a, r, True
                out["players"][t, b] = self.players[s]
            out["dones"][len(steps) - 1, b] = True
        mask = torch.zeros(len(self.players), A, dtype=torch.bool)
        probs = torch.zeros(len(self.players), A, dtype=DT)
        for s, outs in enumerate(self.outcomes):
            mask[s, : len(outs)] = True
            probs[s, : len(outs)] = torch.tensor(self.policy[s], dtype=DT)
        out["action_mask"], out["probs"] = mask[states], probs[states]
        if q is not None:
            out["q"] = torch.as_tensor(q, dtype=DT)[states]
        if v is not None:
            out["values"] = torch.as_tensor(v, dtype=DT)[states]
        out["states"] = states
        out["weights"] = torch.tensor([w for w, _ in trajs], dtype=DT)
        return out


def chain(stochastic: bool = True) -> Game:
    """Single-player MDP: intermediate rewards, γ = 0.9, chance transitions, 2-4 legal actions per state."""
    outcomes = [
        [[(0.7, 1, 0.0), (0.3, 2, 0.5)], [(1.0, 2, 0.0)], [(1.0, None, 0.2)]],
        [[(0.5, 3, 0.0), (0.5, None, 1.0)], [(1.0, 3, -0.2)]],
        [[(1.0, 3, 0.1)], [(0.4, None, -1.0), (0.6, None, 1.0)]],
        [[(1.0, None, 1.0)], [(1.0, None, -1.0)], [(1.0, None, 0.0)], [(0.5, None, 0.3), (0.5, None, -0.3)]],
    ]
    if not stochastic:
        outcomes = [[[(1.0, *outs[0][1:])] for outs in state] for state in outcomes]
    policy = [[0.5, 0.3, 0.2], [0.6, 0.4], [0.25, 0.75], [0.1, 0.2, 0.3, 0.4]]
    return Game([0, 0, 0, 0], outcomes, policy, gamma=0.9)


def duel(policy: list[list[float]] | None = None) -> Game:
    """Two-player zero-sum game, terminal reward only (win +1 / loss -1 / draw 0 for the seat that acted).

    Seat 0 acts twice in a row at s0 -> s1 (a multi-step selection), seat 1 at s2 -> s3; coins at s1, s2, s3, s4.
    """
    outcomes = [
        [[(1.0, 1, 0.0)], [(1.0, 2, 0.0)]],
        [[(1.0, 2, 0.0)], [(0.5, 3, 0.0), (0.5, None, 1.0)]],
        [[(0.3, None, 1.0), (0.7, 4, 0.0)], [(1.0, 3, 0.0)], [(1.0, None, 0.0)]],
        [[(1.0, 4, 0.0)], [(0.5, None, -1.0), (0.5, None, 0.0)]],
        [[(1.0, None, 1.0)], [(1.0, None, -1.0)], [(0.5, None, 1.0), (0.5, None, -1.0)]],
    ]
    policy = policy or [[0.6, 0.4], [0.5, 0.5], [0.2, 0.5, 0.3], [0.7, 0.3], [0.3, 0.3, 0.4]]
    return Game([0, 0, 1, 1, 0], outcomes, policy, gamma=1.0)


def conditional(x: torch.Tensor, batch: dict) -> dict[tuple[int, int], tuple[float, float]]:
    """(state, action) -> (E[x_t | s_t, a_t], Var[x_t | s_t, a_t]) over all enumerated episodes."""
    w = batch["weights"].expand_as(x)
    acc: dict[tuple[int, int], list] = {}
    for t, b in zip(*torch.nonzero(batch["valid"], as_tuple=True), strict=True):
        key = (int(batch["states"][t, b]), int(batch["actions"][t, b]))
        acc.setdefault(key, []).append((float(w[t, b]), float(x[t, b])))
    out = {}
    for key, items in acc.items():
        ws, xs = np.array(items).T
        mean = float((ws * xs).sum() / ws.sum())
        out[key] = (mean, float((ws * (xs - mean) ** 2).sum() / ws.sum()))
    return out


def random_layout(seed: int = 0, T: int = 9, B: int = 6, A: int = 5) -> dict:
    """Random rollout segments: alternating / repeated players, episodes ending mid-column, bootstrap at T."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(T, B, A, generator=g) < 0.7
    mask[..., 0] = True
    logits = torch.randn(T, B, A, generator=g, dtype=DT).masked_fill(~mask, -torch.inf)
    probs = torch.softmax(logits, -1)
    return {
        "players": torch.randint(0, 2, (T, B), generator=g),
        "actions": torch.multinomial(probs.reshape(-1, A), 1, generator=g).reshape(T, B),
        "rewards": torch.randn(T, B, generator=g, dtype=DT),
        "dones": torch.rand(T, B, generator=g) < 0.25,
        "action_mask": mask,
        "probs": probs,
        "q": torch.randn(T, B, A, generator=g, dtype=DT),
        "values": torch.randn(T, B, generator=g, dtype=DT),
        "bootstrap_value": torch.randn(B, generator=g, dtype=DT),
        "bootstrap_player": torch.randint(0, 2, (B,), generator=g),
    }


def reference_returns(rewards, dones, players, gamma, boot_v, boot_p, lookahead=None):
    """Naive discounted return (n-step when ``lookahead`` values are given): every reward is converted to
    the perspective of the seat acting at t by comparing seats directly, not by chaining sign flips."""
    T, B = rewards.shape
    out = torch.zeros(T, B, dtype=DT)
    for b in range(B):
        for t in range(T):
            me = players[t, b]
            g, k = float(rewards[t, b]), t
            if not dones[t, b]:
                if k + 1 == T:
                    g += gamma * float(boot_v[b]) * (1 if boot_p[b] == me else -1)
                elif lookahead is not None:
                    g += gamma * float(lookahead[t + 1, b]) * (1 if players[t + 1, b] == me else -1)
                else:
                    disc = gamma
                    while True:
                        k += 1
                        if k == T:
                            g += disc * float(boot_v[b]) * (1 if boot_p[b] == me else -1)
                            break
                        g += disc * float(rewards[k, b]) * (1 if players[k, b] == me else -1)
                        if dones[k, b]:
                            break
                        disc *= gamma
            out[t, b] = g
    return out


def ev(q, probs, mask):
    return (torch.where(mask, probs * q, 0)).sum(-1) / torch.where(mask, probs, 0).sum(-1)


# ---------------------------------------------------------------------------------------------- GAE


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_gae_lambda_one_is_monte_carlo_return_minus_value(gamma):
    d = random_layout(1)
    a, ret = adv.gae(d["rewards"], d["values"], d["dones"], d["players"], gamma=gamma, lam=1.0,
                     bootstrap_value=d["bootstrap_value"], bootstrap_player=d["bootstrap_player"])  # fmt: skip
    mc = reference_returns(d["rewards"], d["dones"], d["players"], gamma, d["bootstrap_value"], d["bootstrap_player"])
    torch.testing.assert_close(ret, mc)
    torch.testing.assert_close(a, mc - d["values"])


def test_gae_lambda_zero_is_the_td_error():
    d = random_layout(2)
    a, ret = adv.gae(d["rewards"], d["values"], d["dones"], d["players"], gamma=0.95, lam=0.0,
                     bootstrap_value=d["bootstrap_value"], bootstrap_player=d["bootstrap_player"])  # fmt: skip
    td = reference_returns(d["rewards"], d["dones"], d["players"], 0.95, d["bootstrap_value"],
                           d["bootstrap_player"], lookahead=d["values"])  # fmt: skip
    torch.testing.assert_close(a, td - d["values"])
    torch.testing.assert_close(ret, td)


@pytest.mark.parametrize("game", [chain(), duel()], ids=["chain", "duel"])
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95])
def test_gae_with_the_true_value_is_unbiased_for_q_minus_v(game, lam):
    q, v = game.solve()
    d = game.batch(v=v)
    a, _ = adv.gae(d["rewards"], d["values"], d["dones"], d["players"], gamma=game.gamma, lam=lam, valid=d["valid"])
    for (s, act), (mean, _) in conditional(a, d).items():
        assert mean == pytest.approx(q[s, act] - v[s], abs=1e-12)
    assert torch.all(a[~d["valid"]] == 0)


# ------------------------------------------------------------------------------------ Expected-SARSA


def es_returns(d, game=None, lam=0.8, **kw):
    kw.setdefault("gamma", game.gamma if game else 1.0)
    return adv.expected_sarsa_returns(d["rewards"], d["q"], d["probs"], d["action_mask"], d["actions"], d["dones"],
                                      d["players"], lam=lam, valid=d.get("valid"), **kw)  # fmt: skip


def test_expected_values_are_the_policy_mean_over_legal_candidates():
    q, v = chain().solve()
    d = chain().batch(q=q, v=v)
    torch.testing.assert_close(adv.expected_values(d["q"], d["probs"], d["action_mask"]), d["values"])


@pytest.mark.parametrize("lam", [0.0, 0.6, 1.0])
def test_expected_sarsa_with_the_true_q_is_exact_under_deterministic_dynamics(lam):
    game = chain(stochastic=False)
    q, v = game.solve()
    d = game.batch(q=q)
    g = es_returns(d, game, lam)
    taken = d["q"].gather(-1, d["actions"].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(g[d["valid"]], taken[d["valid"]])  # every sample, not just on average


@pytest.mark.parametrize("game", [chain(), duel()], ids=["chain", "duel"])
@pytest.mark.parametrize("lam", [0.0, 0.7, 1.0])
def test_expected_sarsa_with_the_true_q_is_unbiased(game, lam):
    q, _ = game.solve()
    d = game.batch(q=q)
    for (s, a), (mean, _) in conditional(es_returns(d, game, lam), d).items():
        assert mean == pytest.approx(q[s, a], abs=1e-12)


def test_expected_sarsa_limits():
    d = random_layout(3)
    boot = {"bootstrap_value": d["bootstrap_value"], "bootstrap_player": d["bootstrap_player"]}
    # λ = 1 with Q = 0: plain discounted Monte-Carlo return (the control variates vanish)
    zero = dict(d, q=torch.zeros_like(d["q"]))
    mc = reference_returns(d["rewards"], d["dones"], d["players"], 0.9, d["bootstrap_value"], d["bootstrap_player"])
    torch.testing.assert_close(es_returns(zero, lam=1.0, gamma=0.9, **boot), mc)
    # λ = 0: one-step Expected-SARSA target r + γ σ Σ_a π(a|s') Q(s', a)
    vbar = ev(d["q"], d["probs"], d["action_mask"])
    one = reference_returns(d["rewards"], d["dones"], d["players"], 0.9, d["bootstrap_value"], d["bootstrap_player"],
                            lookahead=vbar)  # fmt: skip
    torch.testing.assert_close(es_returns(d, lam=0.0, gamma=0.9, **boot), one)
    # general λ: G_t - Q(s_t, a_t) = δ^Q_t + γ λ σ_t (G_{t+1} - Q(s_{t+1}, a_{t+1})), cut at terminals
    lam, g = 0.7, es_returns(d, lam=0.7, gamma=0.9, **boot)
    taken = d["q"].gather(-1, d["actions"].unsqueeze(-1)).squeeze(-1)
    sign = torch.where(d["players"][1:] == d["players"][:-1], 1.0, -1.0).to(DT)
    cont = 0.9 * sign * (~d["dones"][:-1])
    torch.testing.assert_close((g - taken)[:-1], (one - taken)[:-1] + lam * cont * (g - taken)[1:])


# ------------------------------------------------------------------------------------------ VRPO


def vrpo(d, game, lam=0.8, mode="return"):
    return adv.vrpo_advantages(d["rewards"], d["q"], d["probs"], d["action_mask"], d["actions"], d["dones"],
                               d["players"], gamma=game.gamma, lam=lam, valid=d["valid"], mode=mode)  # fmt: skip


@pytest.mark.parametrize("game", [chain(), chain(stochastic=False), duel()], ids=["chain", "deterministic", "duel"])
def test_vrpo_advantage_matches_the_analytic_q_minus_v(game):
    q, v = game.solve()
    d = game.batch(q=q)
    exact = torch.as_tensor(q - v[:, None], dtype=DT)[d["states"], d["actions"]]
    a_critic, targets = vrpo(d, game, mode="critic")
    torch.testing.assert_close(a_critic[d["valid"]], exact[d["valid"]])  # every sample
    torch.testing.assert_close(targets, es_returns(d, game, 0.8))
    for lam in (0.0, 0.8, 1.0):
        a_ret, g = vrpo(d, game, lam=lam)
        torch.testing.assert_close(a_ret, torch.where(d["valid"], g - ev(d["q"], d["probs"], d["action_mask"]), 0))
        for (s, act), (mean, _) in conditional(a_ret, d).items():
            assert mean == pytest.approx(q[s, act] - v[s], abs=1e-12)
    # all-action form: A(s, a) = Q(s, a) - Σ π Q for every legal candidate, π-weighted mean 0
    per_action = adv.q_advantages(d["q"], d["probs"], d["action_mask"])
    exact_all = torch.as_tensor(q - v[:, None], dtype=DT)[d["states"]]
    torch.testing.assert_close(per_action[d["action_mask"]], exact_all[d["action_mask"]])
    torch.testing.assert_close((per_action * d["probs"]).sum(-1), torch.zeros_like(d["rewards"]))


@pytest.mark.parametrize("game", [chain(stochastic=False), duel()], ids=["deterministic", "duel"])
def test_vrpo_removes_the_variance_of_policy_sampling(game):
    """With an exact critic, GAE's advantage still carries the noise of every later sampled action
    (Fan & Farina's point for stochastic self-play policies); the Q-boosted one does not."""
    q, v = game.solve()
    d = game.batch(q=q, v=v)
    a_gae, _ = adv.gae(d["rewards"], d["values"], d["dones"], d["players"], gamma=game.gamma, lam=0.95,
                       valid=d["valid"])  # fmt: skip
    a_vrpo, _ = vrpo(d, game, lam=0.95)
    var_gae = sum(var for _, var in conditional(a_gae, d).values())
    var_vrpo = sum(var for _, var in conditional(a_vrpo, d).values())
    assert var_gae > 0.1
    if game.players == [0, 0, 0, 0]:  # deterministic dynamics: no variance left at all
        assert var_vrpo == pytest.approx(0.0, abs=1e-20)
    assert var_vrpo < var_gae  # with chance nodes only the coin-flip noise remains (duel: 3.4 vs 5.7)


# --------------------------------------------------------------------------------- two-player signs


def negamax(game: Game) -> np.ndarray:
    """Expectiminimax values from the perspective of the seat to act, by direct recursion."""
    memo: dict[int, float] = {}

    def val(s: int) -> float:
        if s not in memo:
            memo[s] = max(sum(p * (r + (0.0 if s2 is None else game.sign(s, s2) * val(s2))) for p, s2, r in outs)
                          for outs in game.outcomes[s])  # fmt: skip
        return memo[s]

    return np.array([val(s) for s in range(len(game.players))])


def test_alternating_players_match_the_minimax_and_policy_values():
    base = duel()
    star = negamax(base)
    np.testing.assert_allclose(star, [0.75, 0.75, 0.0, -0.5, 1.0])
    greedy = []
    for s, outs in enumerate(base.outcomes):
        vals = [sum(p * (r + (0.0 if s2 is None else base.sign(s, s2) * star[s2])) for p, s2, r in o) for o in outs]
        greedy.append([1.0 if a == int(np.argmax(vals)) else 0.0 for a in range(len(outs))])
    game = duel(greedy)
    q, v = game.solve()
    np.testing.assert_allclose(v, star)  # the linear solve and the recursion agree on the sign convention

    # Monte-Carlo returns: the terminal reward, negated for the other seat's steps
    d = base.batch()
    _, mc = adv.gae(d["rewards"], torch.zeros_like(d["rewards"]), d["dones"], d["players"], lam=1.0, valid=d["valid"])
    last = d["valid"].sum(0) - 1
    cols = torch.arange(d["rewards"].shape[1])
    winner_view = d["rewards"][last, cols] * torch.where(d["players"][last, cols] == 0, 1.0, -1.0).to(DT)  # seat 0
    expect = winner_view * torch.where(d["players"] == 0, 1.0, -1.0).to(DT)
    torch.testing.assert_close(mc, torch.where(d["valid"], expect, 0))

    # Expected-SARSA with the minimax Q under the minimax policies reproduces Q* in expectation
    d = game.batch(q=q)
    for (s, a), (mean, _) in conditional(es_returns(d, game, 0.9), d).items():
        assert mean == pytest.approx(q[s, a], abs=1e-12)
    assert set(conditional(es_returns(d, game, 0.9), d)) == {(0, 0), (1, 1), (3, 1)}


def test_terminal_rewards_are_from_the_acting_seat():
    players = torch.tensor([[0, 1, 1], [1, 0, 0]])
    dones = torch.tensor([[False, True, True], [True, True, False]])
    winner = torch.tensor([[1, 1, -1], [1, 1, 0]])  # -1 = draw; only read where done
    r = adv.terminal_rewards(players, dones, winner)
    torch.testing.assert_close(r, torch.tensor([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0]]))


# ------------------------------------------------------------------------------- layout and masking


def test_illegal_candidates_are_ignored():
    d = random_layout(4)
    boot = {"bootstrap_value": d["bootstrap_value"], "bootstrap_player": d["bootstrap_player"]}
    noisy = dict(d)
    noisy["q"] = torch.where(d["action_mask"], d["q"], 1e6)
    noisy["probs"] = torch.where(d["action_mask"], d["probs"], 0.3)  # stray mass on illegal slots
    torch.testing.assert_close(adv.expected_values(noisy["q"], noisy["probs"], d["action_mask"]),
                               ev(d["q"], d["probs"], d["action_mask"]))  # fmt: skip
    torch.testing.assert_close(es_returns(noisy, lam=0.7, **boot), es_returns(d, lam=0.7, **boot))
    per_action = adv.q_advantages(noisy["q"], noisy["probs"], d["action_mask"])
    assert torch.all(per_action[~d["action_mask"]] == 0)
    assert torch.isfinite(per_action).all()


def test_episodes_packed_in_columns_match_separate_computation():
    """Fixed-T segments with auto-reset (several episodes per column), a bootstrapped tail and padding."""
    game = duel()
    q, v = game.solve()
    full = game.batch(q=q, v=v)
    lengths = full["valid"].sum(0).tolist()
    # column 0: episodes 0, 1 and the first 2 steps of 2 (cut by the segment end); column 1: episode 3 + padding
    T = lengths[0] + lengths[1] + 2
    assert lengths[3] < T
    keys = ("players", "actions", "rewards", "dones", "action_mask", "probs", "q", "values")
    seg = {k: torch.zeros((T, 2, *full[k].shape[2:]), dtype=full[k].dtype) for k in keys}
    valid = torch.zeros(T, 2, dtype=torch.bool)
    t0 = 0
    for ep, n in ((0, lengths[0]), (1, lengths[1]), (2, 2)):
        for k in keys:
            seg[k][t0 : t0 + n, 0] = full[k][:n, ep]
        t0 += n
    for k in keys:
        seg[k][: lengths[3], 1] = full[k][: lengths[3], 3]
    valid[:, 0], valid[: lengths[3], 1] = True, True
    boot_state = int(full["states"][2, 2])
    boot_player = torch.tensor([game.players[boot_state], 0])
    for lam in (0.0, 0.9):
        # GAE: the tail bootstraps with V(s_T); the episode was cut, so compare with the truncated λ-return
        a_seg, _ = adv.gae(seg["rewards"], seg["values"], seg["dones"], seg["players"], gamma=1.0, lam=lam,
                           valid=valid, bootstrap_value=torch.tensor([v[boot_state], 0.0], dtype=DT),
                           bootstrap_player=boot_player)  # fmt: skip
        a_full, _ = adv.gae(full["rewards"], full["values"], full["dones"], full["players"], gamma=1.0, lam=lam,
                            valid=full["valid"])  # fmt: skip
        t0 = 0
        for ep in (0, 1):
            torch.testing.assert_close(a_seg[t0 : t0 + lengths[ep], 0], a_full[: lengths[ep], ep])
            t0 += lengths[ep]
        torch.testing.assert_close(a_seg[: lengths[3], 1], a_full[: lengths[3], 3])
        assert torch.all(a_seg[lengths[3] :, 1] == 0)
        # the cut tail at λ = 0 is the TD error against the bootstrap value
        sign = 1.0 if game.players[boot_state] == int(seg["players"][T - 1, 0]) else -1.0
        tail_td = seg["rewards"][T - 1, 0] + sign * v[boot_state] - seg["values"][T - 1, 0]
        if lam == 0.0:
            assert float(a_seg[T - 1, 0]) == pytest.approx(float(tail_td))
        # Expected-SARSA: identical per-episode for the complete episodes
        q_boot = torch.as_tensor(q[boot_state], dtype=DT)
        vbar_boot = torch.tensor([float(ev(q_boot, full["probs"][2, 2], full["action_mask"][2, 2])), 0.0], dtype=DT)
        g_seg = es_returns(dict(seg, valid=valid), lam=lam, bootstrap_value=vbar_boot, bootstrap_player=boot_player)
        g_full = es_returns(full, lam=lam)
        torch.testing.assert_close(g_seg[: lengths[0], 0], g_full[: lengths[0], 0])
        torch.testing.assert_close(g_seg[: lengths[3], 1], g_full[: lengths[3], 3])


def test_layout_errors_are_reported():
    d = duel().batch(v=duel().solve()[1])
    r, v, dn, p, valid = d["rewards"], d["values"], d["dones"], d["players"], d["valid"]
    cut = dn.clone()
    cut[:, 0] = False  # episode 0 no longer ends, but is followed by padding
    with pytest.raises(ValueError, match="padding"):
        adv.gae(r, v, cut, p, valid=valid)
    hole = valid.clone()
    hole[0, 0] = False  # padding before real steps
    with pytest.raises(ValueError, match="padding"):
        adv.gae(r, v, dn, p, valid=hole)
    open_end = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="bootstrap"):
        adv.gae(torch.zeros(3, 2), torch.zeros(3, 2), open_end, torch.zeros(3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="bootstrap_player"):
        adv.gae(torch.zeros(3, 2), torch.zeros(3, 2), open_end, torch.zeros(3, 2, dtype=torch.long),
                bootstrap_value=torch.zeros(2))  # fmt: skip


def test_normalize_advantages():
    a = torch.tensor([[1.0, -2.0], [3.0, 100.0]], dtype=DT)
    valid = torch.tensor([[True, True], [True, False]])
    n = adv.normalize_advantages(a, valid, mode="standard")
    assert n[1, 1] == 0
    assert float(n[valid].mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(n[valid].std(correction=0)) == pytest.approx(1.0, abs=1e-6)
    s = adv.normalize_advantages(a, valid, mode="scale")
    torch.testing.assert_close(s[valid], a[valid] / a[valid].std(correction=0))
    assert torch.equal(adv.normalize_advantages(a, valid, mode="none"), torch.where(valid, a, 0))
    with pytest.raises(ValueError):
        adv.normalize_advantages(a, valid, mode="bogus")


def test_estimate_switches_between_gae_and_vrpo():
    game = duel()
    q, v = game.solve()
    d = game.batch(q=q, v=v)
    kw = {k: d[k] for k in ("rewards", "dones", "players", "actions", "action_mask", "probs", "q", "values", "valid")}
    g = adv.estimate("gae", gamma=1.0, lam=0.9, **kw)
    a_gae, ret = adv.gae(d["rewards"], d["values"], d["dones"], d["players"], gamma=1.0, lam=0.9, valid=d["valid"])
    torch.testing.assert_close(g.advantages, a_gae)
    torch.testing.assert_close(g.v_targets, ret)
    torch.testing.assert_close(g.q_targets, es_returns(d, game, 0.9))
    r = adv.estimate("vrpo", gamma=1.0, lam=0.9, **kw)
    a_vrpo, targets = vrpo(d, game, lam=0.9)
    torch.testing.assert_close(r.advantages, a_vrpo)
    torch.testing.assert_close(r.q_targets, targets)
    torch.testing.assert_close(r.v_targets, targets)
    n = adv.estimate("vrpo", gamma=1.0, lam=0.9, normalize="standard", **kw)
    torch.testing.assert_close(n.advantages, adv.normalize_advantages(a_vrpo, d["valid"], mode="standard"))
    gae_only = {k: kw[k] for k in ("rewards", "dones", "players", "values", "valid")}
    assert adv.estimate("gae", **gae_only).q_targets is None  # no Q head needed for the GAE control
    with pytest.raises(ValueError, match="estimator"):
        adv.estimate("vtrace", **kw)


# -------------------------------------------------------------------------------------------- critic


def test_critic_shapes_masks_and_privileged_input():
    torch.manual_seed(0)
    critic = Critic(history_dim=8, action_dim=6, privileged_dim=5, hidden=16)
    h, cand, priv = torch.randn(3, 4, 8), torch.randn(3, 4, 7, 6), torch.randn(3, 4, 5)
    mask = torch.rand(3, 4, 7) < 0.6
    mask[..., 0] = True
    out = critic(h, cand, mask, priv)
    assert out.q.shape == (3, 4, 7) and out.v.shape == (3, 4)
    assert torch.all(out.q[~mask] == 0)
    assert out.q.abs().max() < 1 and out.v.abs().max() < 1  # tanh: terminal rewards are in [-1, 1]
    other = critic(h, torch.where(mask.unsqueeze(-1), cand, 9.0), mask, priv)
    torch.testing.assert_close(other.q, out.q)  # illegal candidates do not leak into legal scores
    moved = critic(h, cand, mask, priv + 1)
    assert not torch.allclose(moved.q, out.q) and not torch.allclose(moved.v, out.v)
    with pytest.raises(ValueError, match="privileged"):
        critic(h, cand, mask)
    plain = Critic(history_dim=8, action_dim=6, hidden=16)
    assert plain(h, cand, mask).q.shape == (3, 4, 7)
    with pytest.raises(ValueError, match="privileged"):
        plain(h, cand, mask, priv)


def test_critic_losses_use_only_taken_actions_and_valid_steps():
    q = torch.randn(4, 3, 5, requires_grad=True)
    v = torch.randn(4, 3, requires_grad=True)
    actions = torch.randint(0, 5, (4, 3))
    targets = torch.randn(4, 3)
    valid = torch.rand(4, 3) < 0.7
    valid[0, 0] = True
    lq = q_loss(q, actions, targets, valid)
    taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lq, 0.5 * ((taken - targets) ** 2)[valid].mean())
    lq.backward()
    hit = torch.zeros(4, 3, 5, dtype=torch.bool).scatter_(-1, actions.unsqueeze(-1), True) & valid.unsqueeze(-1)
    assert torch.all(q.grad[~hit] == 0) and torch.all(q.grad[hit] != 0)
    lv = v_loss(v, targets, valid)
    torch.testing.assert_close(lv, 0.5 * ((v - targets) ** 2)[valid].mean())
    w = torch.rand(4, 3)
    torch.testing.assert_close(q_loss(q, actions, targets, w), 0.5 * (w * (taken - targets) ** 2).sum() / w.sum())


@pytest.fixture
def one_thread():
    """Tiny tensors: intra-op threads only add contention (seconds instead of ~1 s on a busy machine)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_critic_learns_q_pi_from_expected_sarsa_targets(one_thread):
    """Fitted policy evaluation on the stochastic chain: Q head -> Q^π, V head -> V^π."""
    torch.manual_seed(0)
    game = chain()
    q_true, v_true = game.solve()
    d = game.batch()
    S, A = len(game.players), game.n_actions
    history = torch.eye(S)[d["states"]]
    cand = torch.eye(S * A).reshape(S, A, S * A)[d["states"]]
    critic = Critic(history_dim=S, action_dim=S * A, hidden=32, squash=False)
    opt = torch.optim.Adam(critic.parameters(), lr=0.02)
    w = d["weights"].float().expand(d["valid"].shape) * d["valid"]
    for _ in range(400):
        out = critic(history, cand, d["action_mask"])
        with torch.no_grad():
            g = adv.expected_sarsa_returns(d["rewards"].float(), out.q, d["probs"].float(), d["action_mask"],
                                           d["actions"], d["dones"], d["players"], gamma=game.gamma, lam=0.8,
                                           valid=d["valid"])  # fmt: skip
        loss = q_loss(out.q, d["actions"], g, w) + v_loss(out.v, g, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = critic(torch.eye(S), torch.eye(S * A).reshape(S, A, S * A), torch.ones(S, A, dtype=torch.bool))
    legal = [(s, a) for s in range(S) for a in range(len(game.outcomes[s]))]
    for s, a in legal:
        assert float(out.q[s, a]) == pytest.approx(q_true[s, a], abs=2e-2)
    np.testing.assert_allclose(out.v.numpy(), v_true, atol=2e-2)
