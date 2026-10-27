"""Look-ahead play (ygorl.train.lookahead, first-seat plan §13): best-of-K at a seat's decisions by snapshot branching.

- K = 1 is the plain policy bit for bit (same actions, same outcomes);
- the same games with the same keys give the same games (determinism), and branching leaves no snapshot behind;
- on a constructed turn 1 (a deck of level-4 normal monsters, a policy that mostly ends its turn) the look-ahead picks
  the candidate whose rollout ends the turn with the better board (a monster on the field);
- the rollout collector plays a look-ahead opponent (``Assignment.lookahead``) without touching the learner's rows.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from ygorl.cards.ydk import Deck, load_ydk  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.encoding import ACTION_KINDS  # noqa: E402
from ygorl.env.pool import GameSpec  # noqa: E402
from ygorl.nets.heads import MASKED_LOGIT  # noqa: E402
from ygorl.train.lookahead import (  # noqa: E402
    GameHooks,
    GameSetup,
    LookaheadConfig,
    SeatPlayer,
    Uniforms,
    look_ahead,
    lookahead_value,
    play,
)

DECKS = Path(__file__).parent / "decks"
ALEXANDRITE = 43096270  # Alexandrite Dragon: level 4 normal monster, 2000 ATK
END = ACTION_KINDS.index("end_phase") + 1


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def db():
    from ygorl.cards.cdb import CardDB

    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    from ygorl.cards.cdb import CardVocab

    return CardVocab.from_db(db)


class KindPolicy(nn.Module):
    """A fixed policy over action kinds: logit ``bias[kind]`` (0 for others) plus a small per-card term, so the
    rollouts have something to vary on."""

    def __init__(self, bias: dict[int, float]) -> None:
        super().__init__()
        self.bias = bias

    def policy_logits(self, obs):
        kinds = obs["actions"][..., 0]
        logits = torch.zeros(kinds.shape, dtype=torch.float32)
        for kind, b in self.bias.items():
            logits = torch.where(kinds == kind, torch.full_like(logits, b), logits)
        logits = logits + 0.01 * (obs["actions"][..., 2] % 7).float()
        return logits.masked_fill(~obs["action_mask"].bool(), MASKED_LOGIT)


def make_env(db, vocab, n: int = 4, snapshots: bool = True) -> EncodedVecEnv:
    return EncodedVecEnv(n, 2, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=True,
                         snapshots=snapshots)  # fmt: skip


def deck_games(policies, lookahead=(None, None), n: int = 4, max_decisions: int = 300):
    decks = [load_ydk(DECKS / f) for f in ("snake_eye.ydk", "tenpai.ydk", "kashtira.ydk", "purrely.ydk")]
    out = []
    for i in range(n):
        spec = GameSpec(seed=500 + i, deck_a=decks[i % 4], deck_b=decks[(i + 1) % 4], first=i % 2,
                        config=DuelConfig(max_decisions=max_decisions))  # fmt: skip
        seats = tuple(SeatPlayer(policies[s], lookahead[s]) for s in (0, 1))
        out.append(GameSetup(spec, seats, key=77 + i))
    return out


def trace(outcomes):
    return [(o.winner, o.reason, o.turns, o.actions) for o in outcomes]


def test_k1_is_the_plain_policy_bit_for_bit(db, vocab):
    pols = (KindPolicy({END: -1.0}), KindPolicy({END: 0.5}))
    env = make_env(db, vocab)
    plain, _ = play(env, deck_games(pols))
    k1 = LookaheadConfig(k=1)
    same, _ = play(env, deck_games(pols, (k1, k1)))
    assert trace(plain) == trace(same)
    assert all(o.stats.qualifying > 0 and o.stats.branched == 0 for o in same)


def test_lookahead_is_deterministic_and_releases_its_snapshots(db, vocab):
    pols = (KindPolicy({END: -1.0}), KindPolicy({END: 0.5}))
    cfg = LookaheadConfig(k=3, first_turn=1, last_turn=2)
    env = make_env(db, vocab)
    a, _ = play(env, deck_games(pols, (cfg, None)))
    b, _ = play(make_env(db, vocab, n=2), deck_games(pols, (cfg, None)))  # other slots, other batching
    assert trace(a) == trace(b)
    assert sum(o.stats.branched for o in a) > 0
    assert env.snapshot_count() == 0 and env.arena_escapes() == 0


def run_sync(env, env_id, gen, module):
    """Drive one look-ahead generator on one env, one request at a time."""
    from ygorl.nets.batch import collate, policy_logits

    value = None
    while True:
        try:
            req = gen.send(value)
        except StopIteration as stop:
            return stop.value
        kind = req[0]
        if kind == "forward":
            logits = policy_logits(module, collate([req[2].obs])).float()
            value = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
            continue
        if kind == "release":
            env.release(env_id, req[1])
            value = None
            continue
        if kind == "snapshot":
            sid = env.snapshot(env_id)
            (ev,) = env.recv(1)
            value = sid
            continue
        if kind == "step":
            env.step(env_id, int(req[1]))
        else:
            env.restore(env_id, req[1])
        (value,) = env.recv(1)


def test_lookahead_picks_the_better_board(db, vocab):
    """Turn 1, main phase, a hand of level-4 normal monsters: the policy ends the turn with probability ~0.9; ending
    the turn leaves an empty field (score 0), a summon or a set one monster (score 1). With K = 8 the candidates
    include a summon, the look-ahead plays it, and the environment is back at the decision."""
    deck = Deck((ALEXANDRITE,) * 40, (), (), "alexandrite")
    policy = KindPolicy({END: 3.0})
    env = make_env(db, vocab, n=1)
    env.reset(0, GameSpec(seed=11, deck_a=deck, deck_b=deck, first=0, config=DuelConfig(max_decisions=200)))
    (ev,) = env.recv(1)
    while ev.obs["globals"][3] == 1 and not (ev.obs["actions"][:, 0] == END).any():  # to the idle command
        env.step(0, int(np.flatnonzero(ev.obs["action_mask"])[0]))
        (ev,) = env.recv(1)
    assert int(ev.obs["globals"][3]) == 1 and int(ev.player) == 0
    from ygorl.nets.batch import collate, policy_logits

    probs = torch.softmax(policy_logits(policy, collate([ev.obs])).float(), -1)[0].numpy().astype(np.float64)
    legal = np.flatnonzero(ev.obs["action_mask"])
    end_row = next(int(i) for i in legal if int(ev.obs["actions"][i][0]) == END)
    assert probs[end_row] > 0.8
    cfg = LookaheadConfig(k=8, own_turn=True)

    def score(cur, seat, w):
        return lookahead_value(cur, seat, w, cfg, db, vocab)

    def draw(u):
        return int(np.searchsorted(np.cumsum(probs), u * probs.sum(), side="right"))

    for key in range(200):  # a game key whose own draw ends the turn (the policy's usual choice), not all K of them
        draws = Uniforms(key)
        picks = [draw(draws.candidate(0, 0, j)) for j in range(cfg.k)]
        if picks[0] == end_row and any(a != end_row for a in picks):
            break
    gen = look_ahead(cfg, 0, 0, ev, probs, draws, GameHooks(), score, record=True)
    action, info = run_sync(env, 0, gen, policy)
    kinds = [int(ev.obs["actions"][a][0]) for a in info["actions"]]
    assert info["branched"] and kinds[0] == END and info["actions"][0] == end_row
    values = dict(zip(kinds, info["values"]))
    assert values[END] == 0.0
    best = max(info["values"])
    assert best >= 1.0 and info["changed"]
    chosen = int(ev.obs["actions"][action][0])
    assert chosen != END and info["values"][info["actions"].index(action)] == best
    assert env.snapshot_count() == 0
    env.step(0, action)  # the main game goes on from the restored decision
    (nxt,) = env.recv(1)
    assert nxt.result is None


def test_k1_lookahead_draw_matches_the_main_draw():
    probs = np.array([0.2, 0.0, 0.5, 0.3])
    d = Uniforms(5)
    for n in range(20):
        assert d.candidate(1, n, 0) == d.main(1, n)
    assert len({d.candidate(0, 3, j) for j in range(8)}) == 8
    assert LookaheadConfig.segment("t3").applies is not None
    with pytest.raises(ValueError):
        LookaheadConfig(k=0)
    assert probs.sum() == 1.0


def test_collector_plays_a_lookahead_opponent(db, vocab):
    """Self-play dealing (``SelfPlaySchedule(lookahead_share=1)``): every pool game carries the look-ahead (turn 2 when
    the learner moves first; ``lookahead_second`` adds turn 1), the collector rolls its decisions out without adding
    learner rows, and every snapshot is released."""
    from ygorl.nets import NetConfig
    from ygorl.nets.actor_critic import ActorCritic
    from ygorl.train.rollout import RolloutCollector
    from ygorl.train.selfplay import DeckPool, SelfPlaySchedule, SnapshotPool

    torch.manual_seed(0)
    model = ActorCritic(NetConfig(vocab_size=len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1),
                        privileged=True, privileged_dim=8, critic_hidden=16)  # fmt: skip
    pool = SnapshotPool(2)
    pool.add(model, update=0)
    decks = [load_ydk(DECKS / f) for f in ("snake_eye.ydk", "tenpai.ydk")]
    cfg = LookaheadConfig(k=4)
    schedule = SelfPlaySchedule(DeckPool(decks), pool, DuelConfig(max_decisions=300), selfplay_fraction=0.0, seed=1,
                                lookahead=cfg, lookahead_share=1.0, lookahead_second=True)  # fmt: skip
    env = make_env(db, vocab, n=2)
    col = RolloutCollector(env, model, schedule, 64, opponents=schedule.opponent, seed=2, min_batch=2)
    ro = col.collect()
    assert ro.players.shape == (64, 2)
    assert ro.lookahead_decisions > 0 and ro.lookahead_branched > 0 and ro.lookahead_steps > 0
    assert env.snapshot_count() == 0 and env.arena_escapes() == 0
    first = [g for g in ro.games if g.assignment.learner_seat == 0]
    assert all(g.assignment.lookahead.first_turn == 2 for g in first)
    with pytest.raises(ValueError):
        SelfPlaySchedule(DeckPool(decks), pool, lookahead_share=0.5)


def test_candidates_start_with_the_policy_draw():
    from ygorl.train.lookahead import candidates

    probs = np.array([0.05, 0.0, 0.6, 0.25, 0.1])
    mask = np.array([1, 0, 1, 1, 1])
    draws = Uniforms(9)

    def draw(j):
        return draws.candidate(0, 4, j)

    own = candidates(LookaheadConfig(k=1), probs, mask, draw)
    for mode in ("sample", "top"):
        for k in (1, 2, 8):
            got = candidates(LookaheadConfig(k=k, candidates=mode), probs, mask, draw)
            assert got[0] == own[0] and len(set(got)) == len(got) <= k and 1 not in got
    top = candidates(LookaheadConfig(k=3, candidates="top"), probs, mask, lambda j: 0.99)  # own draw: the last row
    assert top == [4, 2, 3]
    assert sorted(candidates(LookaheadConfig(k=8, candidates="top"), probs, mask, draw)) == [0, 2, 3, 4]
