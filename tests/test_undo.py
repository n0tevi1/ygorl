"""No-op undos are masked (docs/encoding.md 「撤销类空操作」): backing out of a command just started from the
main / battle menu, and undoing the previous select / unselect of a SELECT_UNSELECT_CARD. Taking one returns the
game to exactly the decision it came from, so a deterministic policy could otherwise loop forever. Also masked
from the menus: shuffling the hand, and an effect already activated MAX_MENU_ACTIVATIONS times this turn. The rules
live in engine.undo.UndoRules (C++ UndoRules); a lockstep test with lowered limits keeps the two hosts equal."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed, seat_of_deck
from ygorl.engine.undo import DEFAULT_LIMITS, UndoLimits, UndoRules, override_limits
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
            prev_point, prev_i = next(
                ((p, j) for p, j in reversed(history[:k]) if p.player == point.player), (None, None)
            )
            prev = prev_point.actions[prev_i] if prev_point is not None else None
            if prev is None or prev.kind != "select" or prev.card != point.actions[i].card:
                continue  # rule 5 (a long selection moves forward only), not the reversal of the previous step
        taken += 1
        nxt = next((p for p, _ in history[k + 1 :] if p.player == point.player), None)
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
    """Random play records the menus; rules fed the same effect's activation menu_activations times in one turn mask
    it (and only it), a new turn lifts the limit; shuffle is always masked in a menu."""
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
    rules = UndoRules(DEFAULT_LIMITS)
    rules.on_decision(point.decision)
    for _ in range(DEFAULT_LIMITS.menu_activations - 1):
        rules.on_action(point.decision, point.actions[first], 5)
    assert first not in rules.undo(point.decision, point.actions, 5)
    rules.on_action(point.decision, point.actions[first], 5)
    masked = rules.undo(point.decision, point.actions, 5)
    assert first in masked and other not in masked
    assert first not in rules.undo(point.decision, point.actions, 6)


def test_a_long_selection_can_only_move_forward(db):
    """Rule 5: after selection_steps steps in one SELECT_UNSELECT_CARD selection, unselecting is masked (a policy
    that never picks the card a finish needs would otherwise toggle the rest forever)."""
    seen = []
    rng = RandomAgent(5)

    class Probe:
        def act(self, point):
            kinds = {a.kind for a in point.actions}
            if point.decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and {"unselect", "select"} <= kinds:
                seen.append(point)
            return rng.act(point)

    for seed, a, b in ((3, "snake_eye", "kashtira"), (5, "labrynth", "tenpai"), (7, "branded_despia", "purrely")):
        duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=3000))
        duel.run(Probe(), Probe())
        if seen:
            break
    assert seen, "random play should reach a selection with select and unselect rows"
    point = seen[0]
    unselect = [i for i, a in enumerate(point.actions) if a.kind == "unselect"]
    limits = UndoLimits(DEFAULT_LIMITS.menu_activations, 3, DEFAULT_LIMITS.selection_cancels)
    rules = UndoRules(limits)
    rules.on_decision(point.decision)
    for _ in range(limits.selection_steps - 1):
        rules.on_action(point.decision, point.actions[0], 1)
        rules.on_decision(point.decision)
    assert not set(unselect) & set(rules.undo(point.decision, point.actions, 1)) - {
        i
        for i in unselect
        if point.actions[i].card == point.actions[0].card  # rule 2: the reversal of the last step
    }
    rules.on_action(point.decision, point.actions[0], 1)
    masked = set(rules.undo(point.decision, point.actions, 1))
    assert set(unselect) <= masked  # every unselect row is masked; the select rows keep the selection moving
    assert any(point.actions[i].kind == "select" and i not in masked for i in range(len(point.actions)))


class UndoLockstep:
    """Random play (masked rows included) on the Python host, mirrored on a C++ HostDuel: both hosts must mask the
    same rows at every decision. Records which rules fired."""

    def __init__(self, seed, host, fired):
        self.rng, self.host, self.fired = RandomAgent(seed), host, fired

    def act(self, point):
        assert self.host.actions() is not None
        assert list(self.host.undo()) == list(point.undo), (point.index, point.decision.name)
        for i in point.undo:
            self.fired.add((point.decision.TYPE, point.actions[i].kind))
        if sum(point.actions[i].kind == "unselect" for i in point.undo) >= 2:
            self.fired.add("rule 5")  # rule 2 masks one unselect at most
        idx = self.rng.act(point)
        self.host.act(idx)
        return idx


def test_both_hosts_mask_the_same_rows_with_every_rule_firing(db, vocab):
    """With the limits of rules 3, 5 and 6 lowered (for both hosts at once), random play reaches rules 1–5; the
    Python DecisionPoint.undo and the C++ tracker's undo() agree at every decision, and so do the results. (Rule 6
    needs repeated cancels of one prompt: tests/test_target_cancel_guard.py and test_material_cancel_guard.py.)"""
    fired: set = set()
    games = [(11, "snake_eye", "kashtira", 0), (12, "labrynth", "tenpai", 1), (13, "branded_despia", "purrely", 0),
             (14, "tearlaments", "yubel", 0), (15, "fiendsmith_ryzeal", "voiceless_voice", 1)]  # fmt: skip
    with override_limits(menu_activations=1, selection_steps=2, selection_cancels=1):
        for seed, a, b, first in games:
            cfg = DuelConfig(max_decisions=3000)
            duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=cfg)
            host = lockstep(db, vocab, duel, cfg)
            seats = [UndoLockstep(seed, host, fired), UndoLockstep(seed + 1, host, fired)]
            result = duel.run(seats[seat_of_deck(first, 0)], seats[seat_of_deck(first, 1)])
            assert host.done() and host.result()["decisions"] == result.decisions
            assert host.result()["responses"] == result.responses
    # rule 1 (cancel in a selection), 2 (unselect), 3 (activate in a menu), 4 (shuffle) and 5
    assert {(C.MSG_SELECT_CARD, "cancel"), (C.MSG_SELECT_UNSELECT_CARD, "unselect"), (C.MSG_SELECT_IDLECMD, "activate"),
            (C.MSG_SELECT_IDLECMD, "shuffle"), "rule 5"} <= fired  # fmt: skip


def test_undo_limits_are_shared_by_both_hosts():
    """The limits are written once (C++): the Python rules read them when a duel starts, and override_limits
    restores them."""
    assert UndoLimits.current() == DEFAULT_LIMITS
    with override_limits(selection_steps=5) as limits:
        assert limits == UndoLimits(DEFAULT_LIMITS.menu_activations, 5, DEFAULT_LIMITS.selection_cancels)
        assert UndoRules().limits == limits and _core.undo_limits()["selection_steps"] == 5
    assert UndoLimits.current() == DEFAULT_LIMITS
    with pytest.raises(ValueError, match="at least 1"):
        _core.set_undo_limits(menu_activations=0)
    assert UndoLimits.current() == DEFAULT_LIMITS


@pytest.mark.parametrize("spec", ["random:choosable", "greedy:choosable"])
def test_baselines_can_skip_the_masked_rows(db, spec):
    """``random:choosable`` / ``greedy:choosable`` pick only DecisionPoint.choosable rows (what a network policy may
    pick); plain ``random`` keeps taking masked rows, so its recorded numbers stay comparable."""
    from ygorl.agents import make_agent

    def play(spec):
        taken, offered = [], 0
        agents = [make_agent(spec, 3), make_agent(spec, 4)]

        class Probe:
            def __init__(self, agent):
                self.agent = agent

            def act(self, point):
                nonlocal offered
                i = self.agent.act(point)
                offered += bool(point.undo)
                assert point.choosable and set(point.choosable).isdisjoint(point.undo)
                taken.append(i in point.undo)
                return i

        Duel(3, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db,
             config=DuelConfig(max_decisions=3000)).run(Probe(agents[0]), Probe(agents[1]))  # fmt: skip
        return sum(taken), offered

    taken, offered = play(spec)
    assert offered > 0 and taken == 0
    assert play("random")[0] > 0


def test_only_choosable_maps_indices_back(db):
    point = next(p for p in _menu_points(db) if p.undo)
    restricted, keep = point.only_choosable()
    assert keep == point.choosable and not restricted.undo
    assert [restricted.actions[i] for i in range(len(keep))] == [point.actions[i] for i in keep]
    unmasked = next(p for p in _menu_points(db) if not p.undo)
    assert unmasked.only_choosable() == (unmasked, tuple(range(len(unmasked.actions))))


def _menu_points(db):
    seen, rng = [], RandomAgent(3)

    class Probe:
        def act(self, point):
            seen.append(point)
            return rng.act(point)

    Duel(3, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db, config=DuelConfig(max_decisions=600)).run(
        Probe(), Probe()
    )
    return seen
