"""Bounded full-state, fixed-RNG AND/OR audit. Never a public-information teacher.

An existential player node and universal opponent node certify a same-turn win
only in the actual engine state. Exhausted budgets remain unknown. Certificates
can be checked using fresh sessions and action prefixes, without snapshot reuse.
"""

import json
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchBudget:
    nodes: int = 256
    depth: int = 48
    seconds: float = 20.0

    def __post_init__(self):
        if self.nodes < 1 or not 1 <= self.depth <= 128 or self.seconds <= 0:
            raise ValueError("positive budgets and depth <= 128 required")


def position(session):
    p = session.point
    if p is None:
        r = session.result()
        return json.loads(
            json.dumps(
                {
                    "result": {
                        k: getattr(r, k)
                        for k in (
                            "winner",
                            "reason",
                            "turns",
                            "decisions",
                            "lp",
                            "win_reason",
                            "retries",
                            "unknown_messages",
                            "undecodable_messages",
                            "script_errors",
                            "error",
                        )
                    }
                }
            )
        )
    return json.loads(
        json.dumps(
            {
                "index": p.index,
                "player": p.player,
                "turn": p.turn,
                "phase": p.phase,
                "lp": p.lp,
                "decision": asdict(p.decision),
                "options": [asdict(a) for a in p.actions],
            }
        )
    )


def _leaf(session, player, turn):
    # Health flags may already be present before a nominal terminal outcome.
    r = session.tracker.result
    if r.error or r.retries or r.unknown_messages or r.undecodable_messages or r.script_errors:
        return "unknown", "health"
    if session.done:
        if r.reason != "win":
            return "unknown", r.reason
        won = session.tracker._engine_winner == player and session.tracker.turn == turn
        return ("win" if won else "no_forced_win"), "terminal"
    if session.point.turn != turn:
        return "no_forced_win", "next_turn"
    return None


def _combine(player_node, statuses, complete):
    decisive = "win" if player_node else "no_forced_win"
    if decisive in statuses:
        return decisive
    other = "no_forced_win" if player_node else "win"
    return other if complete and statuses and all(s == other for s in statuses) else "unknown"


def search_action(session, action, player, budget=SearchBudget()):
    """Force one root action, solve its continuation, and always restore the root."""
    p = session.point
    if p is None or p.player != player or p.turn_player != player or not 0 <= action < len(p.actions):
        raise ValueError("expected a legal action on the searching player's turn")
    start, turn, used = time.monotonic(), p.turn, 0
    root = position(session)
    snap = session.snapshot()

    def visit(depth):
        nonlocal used
        here = position(session)
        leaf = _leaf(session, player, turn)
        if leaf:
            return {"position": here, "status": leaf[0], "reason": leaf[1]}
        if used >= budget.nodes or time.monotonic() - start >= budget.seconds or depth >= budget.depth:
            return {"position": here, "status": "unknown", "reason": "budget"}
        used += 1
        p = session.point
        own = p.player == player
        # Order affects coverage, never completeness: low-policy choices remain included.
        priority = ("attack", "battle_phase", "yes", "finish", "pass", "no", "main2", "end_phase")
        order = sorted(
            range(len(p.actions)),
            key=lambda i: (priority.index(p.actions[i].kind) if p.actions[i].kind in priority else len(priority), i),
        )
        saved, children = session.snapshot(), []
        try:
            for i in order:
                session.restore(saved)
                session.act(i)
                child = visit(depth + 1)
                children.append({"action": i, "child": child})
                if child["status"] == ("win" if own else "no_forced_win"):
                    break
                if used >= budget.nodes or time.monotonic() - start >= budget.seconds:
                    break
        finally:
            session.restore(saved)
        status = _combine(own, [c["child"]["status"] for c in children], len(children) == len(order))
        return {"position": here, "status": status, "children": children}

    try:
        session.act(action)
        tree = visit(1)
    finally:
        session.restore(snap)
    return {
        "scope": "actual-hidden-state-fixed-rng",
        "root": root,
        "action": action,
        "player": player,
        "turn": turn,
        "status": tree["status"],
        "tree": tree,
        "nodes": used,
        "seconds": time.monotonic() - start,
        "budget": asdict(budget),
    }


def verify_certificate(certificate, make_root):
    """Cold replay every necessary proof leaf. make_root must create a fresh root.

    Only decisive branches of existential claims are required; universal claims
    require every legal action. Unknown trees are not certified or treated as negatives.
    """
    player, turn = certificate["player"], certificate["turn"]
    if certificate["status"] == "unknown":
        raise ValueError("an unknown search is not a certificate")
    paths = []

    def walk(node, path):
        if "children" not in node:
            if node["status"] == "unknown":
                raise ValueError("unknown leaf in proof")
            paths.append((path, node))
            return node["status"]
        own = node["position"]["player"] == player
        decisive = "win" if own else "no_forced_win"
        children = node["children"]
        indices = [c["action"] for c in children]
        n = len(node["position"]["options"])
        if len(set(indices)) != len(indices) or any(not 0 <= i < n for i in indices):
            raise ValueError("invalid proof actions")
        if node["status"] == decisive:
            children = [c for c in children if c["child"]["status"] == decisive][:1]
            if not children:
                raise ValueError("missing decisive branch")
        elif set(indices) != set(range(n)):
            raise ValueError("incomplete universal proof")
        statuses = [walk(c["child"], path + [(node["position"], c["action"])]) for c in children]
        actual = _combine(own, statuses, len(children) == n)
        if actual != node["status"]:
            raise ValueError("invalid proof reduction")
        return actual

    if walk(certificate["tree"], [(certificate["root"], certificate["action"])]) != certificate["status"]:
        raise ValueError("root status mismatch")
    for path, leaf in paths:
        session = make_root()
        try:
            for expected, action in path:
                if position(session) != expected:
                    raise ValueError("cold replay position mismatch")
                session.act(action)
            if position(session) != leaf["position"] or _leaf(session, player, turn) != (
                leaf["status"],
                leaf["reason"],
            ):
                raise ValueError("cold replay endpoint mismatch")
        finally:
            session.close()
    return {"cold_verified": True, "leaves": len(paths)}
