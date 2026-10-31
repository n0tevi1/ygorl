"""Window branching in the rollout collector (T4; ygorl.train.rollout.BranchConfig, docs/training.md §8.1 「窗口分支」).

On the toy game (Nim, ``ygorl.train.toy``: the best take from ``n`` is ``n % 4``) the branch values must rank the best
take first; on the engine (the recovery-line hands of ``tests/data/line_recovery.jsonl``, a learner holding 40 Ash
Blossom) branching must leave the main game bit for bit as it was, give every branched window a value for the pass and
the Ash, put the line player's state back, and release every snapshot.
"""

import copy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from ygorl.engine.constants import MSG_SELECT_CHAIN  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.nets import NetConfig  # noqa: E402
from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.nets.heads import MASKED_LOGIT  # noqa: E402
from ygorl.train.line_labels import FIRE  # noqa: E402
from ygorl.train.lines import CHAIN_KIND, LineLibrary, LinePlayer  # noqa: E402
from ygorl.train.rollout import PASS_KIND, Assignment, BranchConfig, Rollout, RolloutCollector  # noqa: E402
from ygorl.train.selfplay import SnapshotPool  # noqa: E402
from ygorl.train.toy import N_ACTIONS, NimBranchHooks, NimEnv, NimModel, optimal_action  # noqa: E402

from ygorl.cards.ydk import Deck  # noqa: E402

DATA = Path(__file__).parent / "data" / "line_recovery.jsonl"  # as tests/test_line_recovery.py
TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}
ASH_DECK = Deck((FIRE,) * 40, (), (), "ash")  # the learner holds Ash at every window

# the Rollout fields collection had before branching
OLD_FIELDS = [f.name for f in fields(Rollout) if not f.name.startswith("branch_") and f.name != "seconds"]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def assert_same_rollout(a: Rollout, b: Rollout) -> None:
    for name in OLD_FIELDS:
        if name == "games":
            assert [(g.winner, g.reason, g.learner_rows, g.line, g.assignment.spec) for g in a.games] == [
                (g.winner, g.reason, g.learner_rows, g.line, g.assignment.spec) for g in b.games
            ]
        else:
            assert same(getattr(a, name), getattr(b, name)), name


# ------------------------------------------------------------------------------------------ toy


class FixedNim(nn.Module):
    """A Nim opponent: the best take where there is one, else ``fallback`` stones (deterministic)."""

    def __init__(self, fallback: int = 1) -> None:
        super().__init__()
        self.fallback = fallback

    def policy_logits(self, obs):
        pile = obs["pile"].long()
        take = torch.where(pile % 4 != 0, pile % 4, torch.full_like(pile, self.fallback)).clamp(max=pile)
        logits = torch.full(obs["action_mask"].shape, -30.0)
        logits[torch.arange(len(pile)), take - 1] = 30.0
        return logits.masked_fill(~obs["action_mask"], MASKED_LOGIT)


def nim_collector(branch, *, opponent=None, num_envs=4, steps=12, seed=3, min_batch=None):
    torch.manual_seed(0)
    model = NimModel(15)
    rng = np.random.default_rng(seed)
    opp = opponent if opponent is not None else model

    def next_game():
        return Assignment(int(rng.integers(1, 16)), 0, int(rng.integers(0, 2)))

    env = NimEnv(num_envs, snapshots=True)
    col = RolloutCollector(env, model, next_game, steps, opponents=lambda sid: opp, seed=seed,
                           min_batch=num_envs if min_batch is None else min_batch, branch=branch,
                           branch_hooks=NimBranchHooks())  # fmt: skip
    return col, env, model


def test_branch_values_rank_the_best_nim_take_first():
    """Against a perfect opponent the best take leaves the learner a winning pile at its next decision (board 0),
    every other take a lost one (board 1) or the game, and taking the last stones wins outright (+1)."""
    cfg = BranchConfig(k=N_ACTIONS, rollouts=2, require_recovery=False, board_scale=0.5)
    col, env, _ = nim_collector(cfg, opponent=FixedNim())
    ro = col.collect()
    piles = ro.obs["pile"].reshape(ro.shape)
    valid = ro.branch_valid
    assert ro.branch_windows == int(valid.sum()) > 10 and env.snapshot_count() == 0
    checked = 0
    for t, b in valid.nonzero().tolist():
        pile = int(piles[t, b])
        value = ro.branch_value[t, b]
        legal = ro.action_mask[t, b]
        assert torch.isfinite(value[legal]).all() and torch.isnan(value[~legal]).all()
        if pile % 4:
            best = optimal_action(pile) - 1
            assert int(value[legal].argmax()) == best and value[best] > value[legal].sort().values[-2]
            assert value[best] == (1.0 if pile <= N_ACTIONS else 0.0)
            # any other take: the opponent wins at once (-1) or leaves a multiple of 4 (board 1)
            assert all(float(value[a]) in (-0.5, -1.0) for a in range(N_ACTIONS) if legal[a] and a != best)
            checked += 1
    assert checked > 5
    assert (ro.branch_pass == -1).all() and ro.branch_decisions > ro.branch_windows


def test_critic_horizon_values_are_the_learners_value_there():
    """With a deterministic opponent every rollout ends on a known pile: the value is the learner's V there."""
    cfg = BranchConfig(k=N_ACTIONS, rollouts=1, require_recovery=False, horizon="critic")
    col, _, model = nim_collector(cfg, opponent=FixedNim())
    ro = col.collect()
    piles, seats = ro.obs["pile"].reshape(ro.shape), ro.players
    checked = 0
    for t, b in ro.branch_valid.nonzero().tolist():
        pile = int(piles[t, b])
        for a in ro.action_mask[t, b].nonzero().flatten().tolist():
            left = pile - (a + 1)
            if left == 0:
                assert ro.branch_value[t, b, a] == 1.0
                continue
            opp_take = left % 4 if left % 4 else 1
            after = left - opp_take
            if after == 0:
                assert ro.branch_value[t, b, a] == -1.0
                continue
            obs = NimEnv.observe(after, int(seats[t, b]), 15)
            with torch.no_grad():
                v = model(model.collate([obs]))[2]
            torch.testing.assert_close(ro.branch_value[t, b, a], v[0].float(), rtol=0, atol=1e-6)
            checked += 1
    assert checked > 5


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_branching_leaves_the_main_toy_game_unchanged(fraction):
    plain, _, _ = nim_collector(None)
    branched, env, _ = nim_collector(BranchConfig(fraction=fraction, k=2, rollouts=3, require_recovery=False))
    for _ in range(2):  # across segments too (the bootstrap state, the held decisions)
        a, b = plain.collect(), branched.collect()
        assert_same_rollout(a, b)
        assert b.branch_windows > 0 and env.snapshot_count() == 0
        assert a.branch_value is None and b.branch_value.shape == b.q.shape


def test_zero_fraction_collects_the_rollout_without_branching():
    plain, _, _ = nim_collector(None, min_batch=2)
    zero, env, _ = nim_collector(BranchConfig(fraction=0.0, require_recovery=False), min_batch=2)
    a, b = plain.collect(), zero.collect()
    assert_same_rollout(a, b)
    assert b.branch_windows == b.branch_decisions == 0 and not b.branch_valid.any()
    assert torch.isnan(b.branch_value).all() and (b.branch_pass == -1).all()


def test_asynchronous_batches_branch_too():
    """With min_batch below num_envs the other slots keep playing while one branches (no lockstep)."""
    col, env, _ = nim_collector(BranchConfig(require_recovery=False), min_batch=1)
    ro = col.collect()
    assert ro.branch_windows > 0 and env.snapshot_count() == 0 and ro.players.shape == (12, 4)


def test_branching_needs_snapshots_and_segments():
    with pytest.raises(ValueError, match="snapshots"):
        RolloutCollector(NimEnv(2), NimModel(15), lambda: Assignment(5), 4, branch=BranchConfig())
    with pytest.raises(ValueError, match="complete_games"):
        RolloutCollector(NimEnv(2, snapshots=True), NimModel(15), lambda: Assignment(5), 4, complete_games=True,
                         branch=BranchConfig())  # fmt: skip
    for bad in ({"fraction": 1.5}, {"k": 1}, {"rollouts": 0}, {"horizon": "far"}, {"board_scale": 0.0}):
        with pytest.raises(ValueError):
            BranchConfig(**bad)


# ------------------------------------------------------------------------------------------ engine


@pytest.fixture(scope="module")
def db():
    from ygorl.cards.cdb import CardDB

    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    from ygorl.cards.cdb import CardVocab

    return CardVocab.from_db(db)


@pytest.fixture(scope="module")
def library(db, vocab):
    return LineLibrary.load([DATA], vocab, cards=db, recovery=[DATA])


@pytest.fixture(scope="module")
def plain(db, vocab):
    return LineLibrary.load([DATA], vocab, cards=db)


class AshOrPass(ActorCritic):
    """The tiny actor-critic, nudged to fire Ash at chain windows about half the time (so both kinds are sampled)."""

    ash = 0

    def forward(self, obs, privileged=None):
        out = super().forward(obs, privileged)
        actions = obs["actions"][:, : out.logits.shape[1]]
        hit = (actions[..., 0] == CHAIN_KIND) & (actions[..., 2] == self.ash) & obs["action_mask"].bool()
        passes = (actions[..., 0] == PASS_KIND) & obs["action_mask"].bool()
        logits = torch.where(hit | passes, out.logits + 20.0, out.logits)
        return out._replace(logits=logits)


def engine_collector(
    db, vocab, scripts, branch, *, first_seat=False, steps=48, num_envs=2, skip_forced=True, opp_cpu_threads=0
):
    torch.manual_seed(0)
    model = AshOrPass(NetConfig(vocab_size=len(vocab), **TINY), privileged=True, privileged_dim=8, critic_hidden=16)
    model.ash = vocab.index(FIRE)
    pool = SnapshotPool(2)
    sid = pool.add(model, update=0)
    seat = 0 if first_seat else 1
    it = iter(range(10**6))

    def next_game():
        k = next(it)
        script = scripts[k % len(scripts)]
        spec = script.spec(ASH_DECK, 100 + k, DuelConfig(max_decisions=80), learner_seat=seat)
        return Assignment(spec, sid, seat, {"line": script.id}, script)

    env = EncodedVecEnv(num_envs, 1, cards=db, vocab=vocab, privileged=True, event_length=16, skip_forced=skip_forced,
                        snapshots=branch is not None)  # fmt: skip
    col = RolloutCollector(env, model, next_game, steps, opponents=pool.get, seed=5, min_batch=num_envs,
                           branch=branch, opp_cpu_threads=opp_cpu_threads)  # fmt: skip
    return col, env


@pytest.mark.parametrize("first_seat", [False, True])
def test_engine_branching_leaves_the_main_game_unchanged_and_values_fire_and_hold(db, vocab, library, first_seat):
    plain_col, _ = engine_collector(db, vocab, library.scripts, None, first_seat=first_seat)
    col, env = engine_collector(db, vocab, library.scripts, BranchConfig(rollouts=2), first_seat=first_seat)
    a, b = plain_col.collect(), col.collect()
    assert_same_rollout(a, b)
    assert env.snapshot_count() == 0 and env.arena_escapes() == 0
    valid = b.branch_valid
    assert b.branch_windows >= int(valid.sum()) > 0
    ash = vocab.index(FIRE)
    for t, c in valid.nonzero().tolist():
        value, i = b.branch_value[t, c], t * b.shape[1] + c
        rows = torch.isfinite(value).nonzero().flatten().tolist()
        assert len(rows) == 2 and int(b.branch_pass[t, c]) in rows
        kinds = {int(b.obs["actions"][i, r, 0]) for r in rows}
        assert kinds == {PASS_KIND, CHAIN_KIND}
        fire = next(r for r in rows if int(b.obs["actions"][i, r, 0]) == CHAIN_KIND)
        assert int(b.obs["actions"][i, fire, 2]) == ash
        g = b.obs["globals"][i]
        assert int(g[3]) == (2 if first_seat else 1) and int(g[2]) == 0  # the line's turn
        assert b.line_effect[t, c] >= 0  # a counted window of the line (its recovery key)
        assert (value[rows].abs() <= 1.0 + 0.1 * 40).all()
    assert not valid[b.line_effect < 0].any()


def test_threaded_opponents_keep_branching_rollouts_identical(db, vocab, library):
    """--opp-cpu-threads also serves the snapshot's decisions inside branches (one job per module, split after):
    on a CPU learner the threaded rollout, branch values included, is the synchronous one bit for bit."""
    sync, _ = engine_collector(db, vocab, library.scripts, BranchConfig(rollouts=2))
    threaded, env = engine_collector(db, vocab, library.scripts, BranchConfig(rollouts=2), opp_cpu_threads=1)
    a, b = sync.collect(), threaded.collect()
    assert_same_rollout(a, b)
    assert b.branch_windows > 0 and env.snapshot_count() == 0
    torch.testing.assert_close(a.branch_value, b.branch_value, rtol=0, atol=0, equal_nan=True)
    assert same(a.branch_valid, b.branch_valid) and int(b.branch_valid.sum()) > 0


def test_engine_branching_needs_a_recovery_line_unless_told_otherwise(db, vocab, plain):
    col, env = engine_collector(db, vocab, plain.scripts, BranchConfig())
    ro = col.collect()
    assert ro.branch_windows == 0 and not ro.branch_valid.any()
    col, env = engine_collector(db, vocab, plain.scripts, BranchConfig(require_recovery=False, horizon="both"))
    ro = col.collect()
    assert ro.branch_windows > 0 and ro.branch_valid.any() and env.snapshot_count() == 0


def test_line_player_restores_a_copy_of_its_state(library):
    script = library[0]
    (rec,) = script.recovery
    p = LinePlayer(script, True)
    p.pos, p.played = 3, 3
    before = copy.copy(p.__dict__)
    state = p.copy()
    assert p.interrupt(*rec.key) and p.status == "recovering" and p.steps is rec.steps
    p.pos, p.played, p.extra = 1, 4, 2
    p.restore(state)
    assert p.__dict__ == before and p.steps is script.steps and p.status == "playing"
    assert p.interrupt(*rec.key) and p.recoveries == 1  # the copy is still the state before the switch
    p.restore(state)
    assert p.__dict__ == before
    with pytest.raises(ValueError):
        p.restore(LinePlayer(library[1], True))


def test_board_score_reads_the_line_seats_board_from_the_card_table_and_the_ground_truth(db, vocab):
    """The phase-0 probe's board units: 3 per interruption, 1 per monster, 0.5 per spell / trap zone card; the hand
    and the set cards come from the privileged tensors."""
    from ygorl.env.encoding import CARD_INDEX, CONTROLLER, F_CARD, LOCATION, N_CARDS, POSITION, SEQUENCE
    from ygorl.env.privileged import P_COLS, P_HAND, P_SET
    from ygorl.solver.blocking import board_interruptions
    from ygorl.train.rollout import board_score

    imperm = 10045474  # Infinite Impermanence: a trap that interrupts when set
    cards = np.zeros((N_CARDS, F_CARD), dtype=np.int32)

    def put(i, index, loc, pos, controller=1, seq=0):
        cards[i, [CARD_INDEX, LOCATION, POSITION, CONTROLLER, SEQUENCE]] = index, loc, pos, controller, seq

    put(0, vocab.index(FIRE), 3, 1)  # a face-up monster
    put(1, 1, 3, 4)  # a face-down monster (unknown)
    put(2, 1, 4, 0, seq=2)  # a set spell / trap (unknown to the learner)
    put(3, vocab.index(FIRE), 5, 0)  # graveyard
    put(4, vocab.index(imperm), 4, 0, controller=0)  # the learner's own card: not counted
    priv = {"op_hand": np.zeros((P_HAND, P_COLS), dtype=np.int32), "op_set": np.zeros((P_SET, P_COLS), dtype=np.int32)}
    priv["op_hand"][0, 0] = vocab.index(FIRE)
    priv["op_set"][7 + 2, 0] = vocab.index(imperm)
    side = {"mzone": [{"code": FIRE, "position": 0x1}], "szone": [{"code": imperm, "position": 0xA}],
            "hand": [FIRE], "grave": [FIRE]}  # fmt: skip
    bi = board_interruptions({"players": [side]}, db, 0)
    assert bi.interruptions >= 2  # Ash in hand, Impermanence set
    assert board_score({"cards": cards}, priv, db, vocab) == 3 * bi.interruptions + 2 + 0.5
    # without the ground truth the hand and the set card are unknown
    blind = board_interruptions({"players": [{**side, "szone": [], "hand": []}]}, db, 0)
    assert board_score({"cards": cards}, None, db, vocab) == 3 * blind.interruptions + 2 + 0.5


# ------------------------------------------------------------------------------------------ branch cards (T13)

VEILER = 97268402  # Effect Veiler: a covered card the learner's deck does not hold


class _Line:
    first_seat, status = False, "playing"

    def __init__(self, recovery=()):
        self.recovery = set(recovery)


class _Slot:
    def __init__(self, line):
        self.line = line


class _Ev:
    def __init__(self, rows):
        """``rows``: (kind, card vocab index) per action row, all legal; the line's turn 1, its chain window."""
        from ygorl.env.encoding import A_ACTION

        actions = np.zeros((len(rows), A_ACTION), dtype=np.int64)
        for i, (kind, index) in enumerate(rows):
            actions[i, 0], actions[i, 2] = kind, index
        g = np.zeros(32, dtype=np.int64)
        g[18], g[3], g[2] = MSG_SELECT_CHAIN, 1, 0
        self.obs = {"globals": g, "actions": actions, "action_mask": np.ones(len(rows), dtype=bool)}


def test_branch_cards_need_every_activation_of_the_set_to_be_a_listed_card(db, vocab):
    """The pass row is not counted; only the activations that made the branch set (the k - 1 likeliest) are."""
    from ygorl.train.rollout import EncodedBranchHooks

    hooks = EncodedBranchHooks(vocab, db)
    ash, veiler = vocab.index(FIRE), vocab.index(VEILER)
    ev = _Ev([(PASS_KIND, 0), (CHAIN_KIND, ash), (CHAIN_KIND, veiler)])
    probs = torch.tensor([0.2, 0.5, 0.3])
    slot = _Slot(_Line())

    def found(k, cards, require=False, at=None):
        return hooks.candidates(at or slot, ev, probs, k, require, 4, 1, frozenset(cards))

    assert found(2, ()) == (0, [0, 1])
    assert found(2, (FIRE,)) == (0, [0, 1])  # Veiler is not in the branch set
    assert found(3, (FIRE,)) is None
    assert found(3, (FIRE, VEILER)) == (0, [0, 1, 2])
    assert found(2, (VEILER,)) is None
    # with require_recovery both conditions: a recovery line for the Ash and the Ash listed
    covered = _Slot(_Line([(4, 1, ash)]))
    assert found(2, (FIRE,), True, covered) == (0, [0, 1]) and found(2, (VEILER,), True, covered) is None
    assert found(2, (FIRE,), True) is None
    assert hooks.recovered(covered, ev, [1], 4, 1) and not hooks.recovered(slot, ev, [1], 4, 1)
    assert not hooks.recovered(covered, ev, [1, 2], 4, 1)


def test_engine_branch_cards_branch_the_listed_cards_without_a_recovery_line(db, vocab, plain):
    """Without recovery lines the listed card's windows are branched (all of them without one); an unlisted card's
    are not."""
    col, env = engine_collector(db, vocab, plain.scripts, BranchConfig(require_recovery=False, cards=(FIRE,)))
    ro = col.collect()
    assert ro.branch_windows > 0 and ro.branch_valid.any() and env.snapshot_count() == 0
    assert ro.branch_no_recovery == ro.branch_windows
    col, env = engine_collector(db, vocab, plain.scripts, BranchConfig(require_recovery=False, cards=(VEILER,)))
    ro = col.collect()
    assert ro.branch_windows == 0 and ro.branch_no_recovery == 0 and not ro.branch_valid.any()


def test_engine_branch_cards_with_require_recovery_take_both_conditions(db, vocab, library):
    """With recovery lines for the Ash and require_recovery, listing the Ash changes nothing (bit for bit, branch
    fields included); listing only Veiler leaves no window."""
    a, _ = engine_collector(db, vocab, library.scripts, BranchConfig())
    b, _ = engine_collector(db, vocab, library.scripts, BranchConfig(cards=(FIRE, VEILER)))
    ra, rb = a.collect(), b.collect()
    assert_same_rollout(ra, rb)
    for name in ("branch_valid", "branch_pass", "branch_windows", "branch_decisions", "branch_no_recovery"):
        assert same(getattr(ra, name), getattr(rb, name)), name
    assert torch.equal(ra.branch_value.nan_to_num(9.0), rb.branch_value.nan_to_num(9.0))
    assert ra.branch_windows > 0 and ra.branch_no_recovery == 0
    c, _ = engine_collector(db, vocab, library.scripts, BranchConfig(cards=(VEILER,)))
    assert c.collect().branch_windows == 0


def test_branch_no_recovery_share_in_the_rollout_stats():
    from ygorl.train.trainer import Trainer

    col, _, _ = nim_collector(BranchConfig(require_recovery=False))
    ro = col.collect()
    assert ro.branch_windows > 0 and ro.branch_no_recovery == ro.branch_windows  # Nim has no recovery lines
    assert Trainer._rollout_stats(ro, 0.0)["branch_no_recovery_share"] == 1.0
    ro.branch_windows, ro.branch_no_recovery = 4, 1
    assert Trainer._rollout_stats(ro, 0.0)["branch_no_recovery_share"] == 0.25
    ro.branch_windows = ro.branch_no_recovery = 0
    assert Trainer._rollout_stats(ro, 0.0)["branch_no_recovery_share"] is None


def test_branch_cards_config_and_flag():
    import argparse

    from ygorl.train import cli
    from ygorl.train.trainer import TrainConfig

    cfg = TrainConfig(line_demos=("l.jsonl",), line_share=1.0, line_recovery=("r.jsonl",), branch_windows=True,
                      branch_require_recovery=False, branch_cards=(14558127, VEILER))  # fmt: skip
    assert cfg.branch_config().cards == (14558127, VEILER)
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    argv = cli.to_argv(cfg)
    i = argv.index("--branch-cards")
    assert argv[i + 1 : i + 3] == ["14558127", str(VEILER)]
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    assert cli.from_args(parser.parse_args(argv), cfg.decks).branch_cards == (14558127, VEILER)
    assert cli.from_args(parser.parse_args([]), ()).branch_cards == ()
    with pytest.raises(ValueError, match="branch_cards"):
        TrainConfig(branch_cards=(14558127,))
    with pytest.raises(ValueError, match="branch_cards"):
        TrainConfig(line_demos=("l.jsonl",), line_share=1.0, branch_windows=True, branch_require_recovery=False,
                    branch_cards=(14558127,))  # fmt: skip
