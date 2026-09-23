"""No-op undos are masked (docs/encoding.md 「撤销类空操作」): backing out of a command just started from the
main / battle menu, and undoing the previous select / unselect of a SELECT_UNSELECT_CARD. Taking one returns the
game to exactly the decision it came from, so a deterministic policy could otherwise loop forever."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
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
        if i not in point.undo:
            continue
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

