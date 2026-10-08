"""Adversarial responses, incomplete search and independent native proof replay."""

import copy
from types import SimpleNamespace

import pytest

from ygorl.agents.lethal import passive_action
from ygorl.cards.ydk import Deck
from ygorl.data.environment import PlayerRules
from ygorl.engine.actions import Action
from ygorl.engine.duel import Duel, DuelConfig, DuelResult, DuelSession
from ygorl.eval.win_search import SearchBudget, position, search_action, verify_certificate


class TreeSession:
    def __init__(self, graph):
        self.graph, self.state = graph, "root"

    @property
    def done(self):
        return self.state in ("win", "loss", "error")

    @property
    def tracker(self):
        return SimpleNamespace(result=self.result(), _engine_winner=0 if self.state == "win" else 1, turn=2)

    def result(self):
        return DuelResult(
            winner=0 if self.state == "win" else 1, reason="error" if self.state == "error" else "win", turns=2
        )

    @property
    def point(self):
        if self.done:
            return None
        player, children = self.graph[self.state]
        return SimpleNamespace(
            index=0,
            player=player,
            turn_player=0,
            turn=2,
            phase=1,
            lp=(1000, 1000),
            decision=Action(self.state),
            actions=[Action("select", i) for i in range(len(children))],
        )

    def snapshot(self):
        return self.state

    def restore(self, state):
        self.state = state

    def act(self, i):
        self.state = self.graph[self.state][1][i]

    def close(self):
        pass


def test_passive_win_is_not_a_win_against_all_responses():
    graph = {"root": (0, ["opponent"]), "opponent": (1, ["win", "loss"])}
    session = TreeSession(graph)
    cert = search_action(session, 0, 0)
    assert cert["status"] == "no_forced_win" and session.state == "root"
    assert verify_certificate(cert, lambda: TreeSession(graph))["cold_verified"]


def test_universal_proof_requires_every_response_and_cold_checks_endpoints():
    graph = {"root": (0, ["opponent"]), "opponent": (1, ["win", "win"])}
    cert = search_action(TreeSession(graph), 0, 0)
    assert cert["status"] == "win"
    assert verify_certificate(cert, lambda: TreeSession(graph))["leaves"] == 2
    incomplete = copy.deepcopy(cert)
    incomplete["tree"]["children"].pop()
    with pytest.raises(ValueError, match="incomplete universal"):
        verify_certificate(incomplete, lambda: TreeSession(graph))
    changed = {**graph, "opponent": (1, ["win", "loss"])}
    with pytest.raises(ValueError, match="endpoint mismatch"):
        verify_certificate(cert, lambda: TreeSession(changed))


def test_budget_or_health_failure_is_unknown_not_a_negative_label():
    graph = {"root": (0, ["loop"]), "loop": (0, ["loop", "loss"])}
    cert = search_action(TreeSession(graph), 0, 0, SearchBudget(nodes=2))
    assert cert["status"] == "unknown"
    with pytest.raises(ValueError, match="unknown"):
        verify_certificate(cert, lambda: TreeSession(graph))
    graph = {"root": (0, ["error"])}
    assert search_action(TreeSession(graph), 0, 0)["status"] == "unknown"


def native_root(snapshots=False):
    deck = Deck(main=(91152256,) * 40)  # Celtic Guardian, 1400 ATK
    cfg = DuelConfig(max_turns=4, shuffle_decks=False, player=PlayerRules(starting_lp=1000))
    session = DuelSession(Duel(3, None, deck, deck, config=cfg, snapshots=snapshots))
    while (p := session.point) is not None:
        kinds = [a.kind for a in p.actions]
        if p.turn == 2 and p.player == 1:
            if "attack" in kinds:
                return session
            if "summon" in kinds:
                session.act(kinds.index("summon"))
                continue
            if "battle_phase" in kinds:
                session.act(kinds.index("battle_phase"))
                continue
        session.act(passive_action(p))
    session.close()
    raise AssertionError("native fixture never reached attack")


def test_native_search_restores_root_and_certificate_cold_replays_without_snapshots():
    session = native_root(snapshots=True)
    try:
        before = position(session)
        attack = next(i for i, a in enumerate(session.point.actions) if a.kind == "attack")
        cert = search_action(session, attack, 1)
        assert cert["status"] == "win" and position(session) == before
        assert verify_certificate(cert, native_root)["cold_verified"]
        main2 = next(i for i, a in enumerate(session.point.actions) if a.kind == "main2")
        assert search_action(session, main2, 1, SearchBudget(depth=1))["status"] == "unknown"
        assert position(session) == before
    finally:
        session.close()
