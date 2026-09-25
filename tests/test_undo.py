"""No-op undos are masked (docs/encoding.md 「撤销类空操作」): backing out of a command just started from the
main / battle menu, and undoing the previous select / unselect of a SELECT_UNSELECT_CARD. Taking one returns the
game to exactly the decision it came from, so a deterministic policy could otherwise loop forever. Also masked
from the menus: shuffling the hand, and an effect already activated MAX_MENU_ACTIVATIONS times this turn."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import MAX_MENU_ACTIVATIONS, MAX_SELECTION_STEPS, Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.encoding import ObservationEncoder

CELTIC = 91152256  # Celtic Guardian: a plain Level 4 monster
DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
MENUS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def lockstep(db, vocab, duel, cfg):
    host = _core.HostDuel(db.to_core(), default_scripts(), [vocab.password(i) for i in range(2, len(vocab))])
    decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
    host.start(expand_seed(duel.seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1), decks, cfg.max_turns,
               cfg.max_decisions)  # fmt: skip
    return host


class Attacker:
    """Summons, attacks, then tries to back out of the attack; records (point, Python obs, C++ obs)."""

    def __init__(self, encoder, duel, host):
        self.encoder, self.duel, self.host, self.seen = encoder, duel, host, []

    def act(self, point):
        obs = self.encoder.encode(point, self.duel._core)
        cpp = self.host.observe()
        for k in ("cards", "globals", "actions", "action_mask"):
            np.testing.assert_array_equal(np.asarray(cpp[k]).reshape(obs[k].shape), obs[k], err_msg=k)
        self.seen.append((point, obs))
        kinds = [a.kind for a in point.actions]
        legal = set(np.flatnonzero(obs["action_mask"]).tolist())
        pick = None
        for kind in ("summon", "attack", "battle_phase", "cancel", "pass", "no", "end_phase"):
            if kind in kinds and kinds.index(kind) in legal:
                pick = kinds.index(kind)
                break
        pick = pick if pick is not None else min(legal)
        self.host.act(pick)
        return pick


def test_backing_out_of_an_attack_just_declared_is_masked(db, vocab):
    cfg = DuelConfig(max_turns=3, max_decisions=400, shuffle_decks=False)
    duel = Duel(1, None, Deck(main=(CELTIC,) * 40), Deck(main=(CELTIC,) * 40), cards=db, config=cfg)
    agent = Attacker(ObservationEncoder(db, vocab), duel, lockstep(db, vocab, duel, cfg))
    result = duel.run(agent, agent)
    assert result.reason in ("win", "turn_limit")  # no attack / cancel loop up to the decision limit
    targets = [(p, o) for p, o in agent.seen if p.decision.TYPE == C.MSG_SELECT_CARD
               and "cancel" in [a.kind for a in p.actions]]  # fmt: skip
    assert targets, "the scenario should reach a cancelable attack-target selection"
    for point, obs in targets:
        cancel = [a.kind for a in point.actions].index("cancel")
        assert point.undo == (cancel,) and obs["action_mask"][cancel] == 0
        assert obs["action_mask"].sum() >= 1  # the targets stay choosable


@pytest.mark.parametrize("seed, a, b", [(3, "snake_eye", "kashtira"), (5, "labrynth", "tenpai"),
                                        (7, "branded_despia", "purrely"), (9, "tearlaments", "yubel")])  # fmt: skip
def test_a_masked_undo_really_returns_to_the_same_decision(db, seed, a, b):
    """Random play that ignores the mask: every time it takes a row the mask hides, the player's next decision
    is the one it came from (the menu after a cancel, the selection before the toggled step)."""
    history = []  # (player, decision, chosen kind, was undo)
    rng = RandomAgent(seed)

    class Probe:
        def act(self, point):
            i = rng.act(point)
            history.append((point, i))
            return i

    Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=4000)).run(Probe(), Probe())
    taken = 0
    for k, (point, i) in enumerate(history):
        if i not in point.undo or point.actions[i].kind in ("shuffle", "activate"):
            continue  # rules 3-4 (shuffle, repeated activation) are no-ops, not undos: no round trip to check
        if point.actions[i].kind == "unselect":
            prev_point, prev_i = next(((p, j) for p, j in reversed(history[:k]) if p.player == point.player), (None, None))
            prev = prev_point.actions[prev_i] if prev_point is not None else None
            if prev is None or prev.kind != "select" or prev.card != point.actions[i].card:
                continue  # rule 5 (a long selection moves forward only), not the reversal of the previous step
        taken += 1
        nxt = next((p for p, _ in history[k + 1:] if p.player == point.player), None)
        if nxt is None:
            continue
        kind = point.actions[i].kind
        if kind == "cancel":  # back at the menu the command was started from, unchanged
            menu = next(p for p, _ in reversed(history[:k]) if p.player == point.player and p.decision.TYPE in MENUS)
            assert nxt.decision == menu.decision, (point.decision.name, nxt.decision.name)
        else:  # the selection before the undone step, unchanged
            before = next(p for p, _ in reversed(history[:k]) if p.player == point.player)
            assert nxt.decision == before.decision
        assert len(point.undo) < len(point.actions)  # never every row
    assert sum(bool(p.undo) for p, _ in history) > 0



def test_shuffle_and_the_repeated_activation_limit_are_masked_in_the_menus(db):
    """Random play records the menus; a tracker fed the same effect's activation MAX_MENU_ACTIVATIONS times in one
    turn masks it (and only it), a new turn lifts the limit; shuffle is always masked in a menu."""
    seen = []
    rng = RandomAgent(3)

    class Probe:
        def act(self, point):
            if point.decision.TYPE in MENUS:
                seen.append(point)
            return rng.act(point)

    duel = Duel(3, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db, config=DuelConfig(max_decisions=3000))
    duel.run(Probe(), Probe())
    shuffles = [p for p in seen if "shuffle" in [a.kind for a in p.actions]]
    assert shuffles and all([a.kind for a in p.actions].index("shuffle") in p.undo for p in shuffles)
    point = next(p for p in seen if sum(a.kind == "activate" for a in p.actions) >= 2)
    first, other = [i for i, a in enumerate(point.actions) if a.kind == "activate"][:2]
    tracker = duel.tracker()
    tracker.turn = 5
    for _ in range(MAX_MENU_ACTIVATIONS - 1):
        tracker._note_undo(point.decision, point.actions[first])
    assert first not in tracker._undo(point.decision, point.actions)
    tracker._note_undo(point.decision, point.actions[first])
    masked = tracker._undo(point.decision, point.actions)
    assert first in masked and other not in masked
    tracker.turn = 6
    assert first not in tracker._undo(point.decision, point.actions)


def test_a_long_selection_can_only_move_forward(db):
    """Rule 5: after MAX_SELECTION_STEPS steps in one SELECT_UNSELECT_CARD selection, unselecting is masked (a policy
    that never picks the card a finish needs would otherwise toggle the rest forever)."""
    seen = []
    rng = RandomAgent(5)

    class Probe:
        def act(self, point):
            if point.decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and any(a.kind == "unselect" for a in point.actions):
                seen.append(point)
            return rng.act(point)

    for seed, a, b in ((3, "snake_eye", "kashtira"), (5, "labrynth", "tenpai"), (7, "branded_despia", "purrely")):
        duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=3000))
        duel.run(Probe(), Probe())
        if seen:
            break
    assert seen, "random play should reach a selection with an unselect row"
    point = seen[0]
    unselect = [i for i, a in enumerate(point.actions) if a.kind == "unselect"]
    tracker = duel.tracker()
    tracker._selection_player = point.decision.player
    tracker._selection_steps = MAX_SELECTION_STEPS - 1
    assert not set(unselect) & set(tracker._undo(point.decision, point.actions))
    tracker._note_undo(point.decision, point.actions[0])
    masked = set(tracker._undo(point.decision, point.actions))
    if len(unselect) < len(point.actions):
        assert set(unselect) <= masked  # every unselect row, nothing else from rule 5
