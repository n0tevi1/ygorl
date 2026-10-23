"""Survival potential Φ and potential-based shaping at turn boundaries (docs/training.md §8.2, #442)."""

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.train.aux import phi_shaping, turn_boundaries  # noqa: E402
from ygorl.train.critic import phi_loss  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner  # noqa: E402

# ------------------------------------------------------------------------------------------ targets (hand-built)

# (player, turn, is_my_turn, my_lp, own_monsters, done, truncated, reward) per row; columns side by side
COLS = [
    [  # 0: a complete game; seat 0 keeps its board through turn 2, then seat 1 wins on turn 4
        (0, 1, 1, 8000, 2, 0, 0, 0),
        (0, 1, 1, 8000, 3, 0, 0, 0),  # row 1: seat 0's turn-1 boundary
        (1, 2, 1, 8000, 1, 0, 0, 0),
        (1, 2, 1, 8000, 1, 0, 0, 0),  # row 3: seat 1's turn-2 boundary
        (0, 2, 0, 8000, 3, 0, 0, 0),  # seat 0 answers in seat 1's turn: not a boundary
        (0, 3, 1, 7000, 2, 0, 0, 0),
        (0, 3, 1, 7000, 2, 0, 0, 0),  # row 6: seat 0's turn-3 boundary
        (1, 4, 1, 5000, 0, 1, 0, 1),  # seat 1 wins on its turn 4 (reward from its view)
    ],
    [  # 1: a game cut by a limit, then a game still running at the segment end
        (0, 1, 1, 8000, 1, 0, 0, 0),  # row 0: boundary (turn 2 follows); its next own turn is row 3
        (1, 2, 1, 8000, 1, 0, 0, 0),  # row 1: boundary, but the game is truncated before seat 1's turn 4
        (0, 2, 0, 8000, 1, 0, 0, 0),
        (0, 3, 1, 8000, 1, 1, 1, 0),  # truncated in turn 3: the turn did not end, no boundary
        (0, 1, 1, 8000, 1, 0, 0, 0),
        (0, 1, 1, 8000, 1, 0, 0, 0),  # row 5: boundary, the segment ends before turn 3
        (1, 2, 1, 8000, 0, 0, 0, 0),
        (1, 2, 1, 8000, 0, 0, 0, 0),  # turn 2 still running: no boundary
    ],
    [  # 2: seat 0 is broken on turn 2 but wins on turn 3; then a pool game with only seat 0's rows
        (0, 1, 1, 8000, 3, 0, 0, 0),  # row 0: boundary; at row 2 it has 1 monster (< 3 - 1) and 5000 LP (< 6000)
        (1, 2, 1, 8000, 2, 0, 0, 0),  # row 1: boundary; seat 1's next own turn is row 4
        (0, 3, 1, 5000, 1, 0, 0, 0),
        (0, 3, 1, 5000, 1, 0, 0, 0),  # row 3: boundary, then seat 0 wins
        (1, 4, 1, 7000, 2, 1, 0, -1),  # row 4: seat 1 loses on its turn 4 (its boundary: the game ended)
        (0, 1, 1, 8000, 2, 0, 0, 0),  # row 5: boundary; its next own turn is row 6 (2 monsters, same LP)
        (0, 3, 1, 8000, 2, 0, 0, 0),
        (0, 3, 1, 8000, 2, 0, 0, 0),  # turn 3 still running at the segment end
    ],
]
T, B = len(COLS[0]), len(COLS)


def rollout_tensors():
    g = torch.zeros(T * B, 22)
    cards = torch.zeros(T * B, 160, 23)
    players = torch.zeros(T, B, dtype=torch.long)
    dones = torch.zeros(T, B, dtype=torch.bool)
    truncated = torch.zeros(T, B, dtype=torch.bool)
    rewards = torch.zeros(T, B)
    for b, col in enumerate(COLS):
        for t, (p, turn, mine, lp, own, done, trunc, r) in enumerate(col):
            i = t * B + b
            g[i, 2], g[i, 3], g[i, 5], g[i, 6] = mine, turn, lp, 8000
            for k in range(own):
                cards[i, k, 1], cards[i, k, 4] = 3, 0  # monster zone, controlled by the viewer
            cards[i, 5, 1], cards[i, 5, 4] = 3, 1  # an opponent monster: never counted
            players[t, b], dones[t, b], truncated[t, b], rewards[t, b] = p, bool(done), bool(trunc), r
    return {"globals": g, "cards": cards}, players, dones, rewards, truncated


def test_boundaries_and_survival_targets_on_a_hand_built_rollout():
    tb = turn_boundaries(*rollout_tensors())
    expect = {  # (t, b): (survive, valid)
        (1, 0): (1, True),  # 2 monsters at turn 3 >= 3 - 1
        (3, 0): (1, True),  # seat 1 at turn 4: 0 monsters >= 1 - 1
        (6, 0): (0, True),  # seat 0 lost before its turn 5
        (7, 0): (1, True),  # seat 1 won
        (0, 1): (1, True),
        (1, 1): (0, False),  # truncated before seat 1's next turn: masked
        (5, 1): (0, False),  # the segment ends first: masked
        (0, 2): (0, True),  # broken: lost 2 monsters and 3000 LP
        (1, 2): (1, True),  # 7000 >= 8000 - 2000
        (3, 2): (1, True),  # seat 0 won before its next turn
        (4, 2): (0, True),  # seat 1 lost
        (5, 2): (1, True),
    }
    got = {(t, b) for t, b in tb.boundary.nonzero().tolist()}
    assert got == set(expect)
    for (t, b), (s, v) in expect.items():
        assert (float(tb.survive[t, b]), bool(tb.valid[t, b])) == (s, v), (t, b)
    assert not (tb.valid & ~tb.boundary).any() and float(tb.survive[~tb.valid].abs().sum()) == 0
    # links: each seat's next boundary in the same game; terminal after a finished game's last boundary
    assert tb.next_row[1, 0] == 6 and tb.next_row[3, 0] == 7 and tb.next_row[0, 2] == 3 and tb.next_row[1, 2] == 4
    assert {(t, b) for t, b in tb.terminal.nonzero().tolist()} == {(6, 0), (7, 0), (3, 2), (4, 2)}
    for t, b in ((0, 1), (1, 1), (5, 1), (5, 2)):  # open: no next boundary in the rollout, not terminal
        assert tb.next_row[t, b] == -1 and not tb.terminal[t, b]


def test_shaping_telescopes_over_a_game_and_leaves_other_rows_alone():
    tb = turn_boundaries(*rollout_tensors())
    torch.manual_seed(0)
    phi = torch.rand(T, B)
    f = phi_shaping(phi, tb, gamma=1.0)
    assert (f[~tb.boundary] == 0).all()
    players = rollout_tensors()[1]
    for seat, first in ((0, 1), (1, 3)):  # column 0 is one complete game
        rows = players[:, 0] == seat
        assert float(f[rows, 0].sum()) == pytest.approx(-float(phi[first, 0]))
    assert float(f[0:5, 2][players[0:5, 2] == 0].sum()) == pytest.approx(-float(phi[0, 2]))
    # open boundaries get nothing; γ < 1 discounts the next potential by the rows in between
    assert f[0, 1] == 0 and f[5, 2] == 0
    g = phi_shaping(phi, tb, gamma=0.9)
    assert float(g[1, 0]) == pytest.approx(0.9**5 * float(phi[6, 0]) - float(phi[1, 0]))
    assert float(g[6, 0]) == pytest.approx(-float(phi[6, 0]))
    # first seat only
    h = phi_shaping(phi, tb, gamma=1.0, rows=players == 0)
    assert (h[players == 1] == 0).all() and torch.equal(h[players == 0], f[players == 0])


def test_phi_loss_is_masked_bce():
    logit = torch.tensor([0.0, 2.0, -1.0])
    target = torch.tensor([1.0, 0.0, 1.0])
    valid = torch.tensor([True, False, True])
    expect = (torch.log1p(torch.exp(torch.tensor(-0.0))) + torch.log1p(torch.exp(torch.tensor(1.0)))) / 2
    assert float(phi_loss(logit, target, valid)) == pytest.approx(float(expect))
    assert float(phi_loss(logit, target, torch.zeros(3, dtype=torch.bool))) == 0.0
    with pytest.raises(ValueError):
        PPOConfig(phi_coef=-1.0)


# ------------------------------------------------------------------------------------------ the learner on real duels

DECK_DIR = Path(__file__).parent / "decks"
PAIR = (str(DECK_DIR / "snake_eye.ydk"), str(DECK_DIR / "kashtira.ydk"))
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model(vocab_size, phi):
    from ygorl.nets import NetConfig
    from ygorl.nets.actor_critic import ActorCritic

    torch.manual_seed(0)
    return ActorCritic(NetConfig(vocab_size=vocab_size, **TINY), privileged=True, privileged_dim=8, critic_hidden=16,
                       phi=phi)  # fmt: skip


@pytest.fixture(scope="module")
def duels():
    """A self-play rollout long enough to hold turn boundaries, collected by a model without the phi head."""
    from ygorl.cards.cdb import CardDB, CardVocab
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import DuelConfig
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.train.rollout import RolloutCollector
    from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool

    torch.set_num_threads(1)
    db = CardDB.load()
    vocab = CardVocab.from_db(db)
    m = model(len(vocab), phi=False)
    schedule = SelfPlaySchedule(DeckPool([load_ydk(p) for p in PAIR], "cross"), SnapshotPool(2),
                                DuelConfig(max_decisions=150), selfplay_fraction=1.0, seed=5)  # fmt: skip
    env = EncodedVecEnv(2, 1, cards=db, vocab=vocab, privileged=True, event_length=16)
    ro = RolloutCollector(env, m, schedule, 120, opponents=schedule.opponent, seed=5).collect()
    return len(vocab), ro


def learner(m, **kw):
    base = {"epochs": 1, "minibatch_size": 4096, "target_kl": None}
    return PPOLearner(m, PPOConfig(**{**base, **kw}))


def run(m, ro, **kw):
    lr = learner(m, **kw)
    torch.manual_seed(3)
    return lr.update(ro)


def test_the_rollout_has_boundaries_with_targets(duels):
    _, ro = duels
    tb = turn_boundaries(ro.obs, ro.players, ro.dones, ro.rewards, ro.truncated)
    assert int(tb.boundary.sum()) > 0 and int(tb.valid.sum()) > 0
    # boundaries are rows of the turn player
    assert (ro.obs["globals"][:, 2].view(ro.shape)[tb.boundary] == 1).all()


def test_zero_coefficients_leave_the_update_unchanged(duels):
    n, ro = duels
    base, with_head = model(n, phi=False), model(n, phi=True)
    names = [k for k, _ in base.named_parameters()]
    assert [k for k, _ in with_head.named_parameters()][: len(names)] == names  # the head is appended last
    assert all(torch.equal(a, with_head.state_dict()[k]) for k, a in base.state_dict().items())
    s0, s1 = run(base, ro), run(with_head, ro, phi_coef=0.0, phi_shape_coef=0.0, phi_shape_first_seat_only=True)
    for key in ("loss", "policy_loss", "entropy", "kl_ref", "q_loss", "v_loss", "grad_norm", "adv_mean", "adv_std"):
        assert s0[key] == s1[key], key
    assert not any(k.startswith("phi") for k in s0) and not any(k.startswith("phi") for k in s1)
    assert all(torch.equal(p, with_head.state_dict()[k]) for k, p in base.state_dict().items())


def test_shaping_changes_only_the_reward_seen_by_the_estimator(duels):
    n, ro = duels
    m = model(n, phi=True)
    plain = learner(m).targets(ro)
    lr = learner(m, phi_shape_coef=0.5)
    shaped = lr.targets(ro)
    assert lr._phi_stats["phi_shape_rows"] > 0 and not torch.equal(plain.advantages, shaped.advantages)
    f = lr.phi_shaping(ro)
    tb = lr._boundaries
    assert (f[~tb.boundary] == 0).all() and (f[tb.boundary] != 0).any()
    first = learner(m, phi_shape_coef=0.5, phi_shape_first_seat_only=True)
    first.targets(ro)
    assert (first.phi_shaping(ro)[ro.players == 1] == 0).all()
    with pytest.raises(ValueError, match="critic_phi"):
        learner(model(n, phi=False), phi_shape_coef=0.5).targets(ro)


def test_one_step_with_phi_coef_lowers_the_phi_loss(duels):
    from ygorl.nets.policy import trim_padding

    n, ro = duels
    tb = turn_boundaries(ro.obs, ro.players, ro.dones, ro.rewards, ro.truncated)
    rows = tb.valid.reshape(-1).nonzero().squeeze(1)
    obs = trim_padding(ro.obs)
    sub = {k: v[rows] for k, v in obs.items()}
    priv = {k: v[rows] for k, v in ro.privileged.items()}

    def loss(m):
        with torch.no_grad():
            return float(
                phi_loss(m(sub, priv).phi, tb.survive.reshape(-1)[rows], torch.ones_like(rows, dtype=torch.bool))
            )

    m = model(n, phi=True)
    before = loss(m)
    stats = run(m, ro, phi_coef=1.0, lr=1e-2, entropy_coef=0.0, kl_ref_coef=0.0, q_coef=0.0, v_coef=0.0)
    assert stats["phi_rows"] == len(rows) and 0 <= stats["phi_rate"] <= 1
    assert stats["phi_loss"] == pytest.approx(before, rel=1e-4)
    assert loss(m) < before
    with pytest.raises(ValueError, match="critic_phi"):
        run(model(n, phi=False), ro, phi_coef=1.0)


def test_a_learner_without_the_head_resumes_into_one_with_it(duels):
    n, ro = duels
    old = learner(model(n, phi=False))
    torch.manual_seed(3)
    old.update(ro)
    new = learner(model(n, phi=True))
    fresh_head = copy.deepcopy(new.model.critic.phi_head.state_dict())
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert all(torch.equal(v, new.model.state_dict()[k]) for k, v in old.model.state_dict().items())
    assert all(torch.equal(v, new.model.critic.phi_head.state_dict()[k]) for k, v in fresh_head.items())
    assert new.updates == old.updates and new.new_heads and all(k.startswith("critic.phi_head.") for k in new.new_heads)
    new.cfg = PPOConfig(epochs=1, minibatch_size=4096, target_kl=None, phi_coef=0.5)
    assert new.update(ro)["phi_rows"] > 0  # Adam steps the new head along with the restored parameters
