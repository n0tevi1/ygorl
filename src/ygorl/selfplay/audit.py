"""Offline audit of the Solo player's in-action decisions on recorded traces (design 10 「识别与执行的系统化」).

For every question MD asked inside an action or in the opponent's turn (the attack target, a Summon's position, the
effect choice, activate-or-pass prompts, card lists), the trace says who decided it then (the policy through the
shadow, or a rule); the audit replays what :mod:`ygorl.selfplay.decide` and the tolerant lockstep of
:mod:`ygorl.selfplay.observe` would do on the board recorded in the step (the memory digest, rebuilt into a
:class:`ygorl.md.memory.RawDuel`), and counts who decides now. No game client is involved.

What the digest does not hold is approximated: the advisor's Deck is the advisor's cards seen in the trace (its
order and the exact list are not recorded), banished cards and Xyz materials are missing, flags (Set this turn,
position locked) are clear. The shadow itself is not replayed: a lockstep answer recorded then still counts, and a
lockstep that failed only because of a difference the observer now tolerates counts as the shadow's.

    python -m ygorl.selfplay.audit [--policy CHECKPOINT] [--samples 2] out/mdplay/2026102*.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ygorl.md.memory import PHASES, RawCard, RawDuel, RawPlayer
from ygorl.md.offsets import FIELD_SLOTS, PILES, Offsets
from ygorl.selfplay.trace import read

OFFSETS = Path(__file__).resolve().parents[1] / "md" / "offsets_data"
QUESTIONS = ("target", "position", "option", "prompt", "list")


def raw_from_digest(rec: dict, offsets: Offsets, deck: list[int]) -> RawDuel:
    """The recorded step as a :class:`RawDuel` (the client is player 0); ``deck``: MD card ids for the advisor's
    Deck pile (the digest records its count only)."""
    sem = offsets.semantics

    def player(side: dict, mine: bool) -> RawPlayer:
        slots = []
        for s in side["field"]:
            if s is None:
                slots.append(None)
                continue
            cid, face, sideways = s
            slots.append(RawCard(cid, 0, 0, face=face, sideways=sideways, slot=bytes(sem["locked_byte"] + 1)))
        slots += [None] * (FIELD_SLOTS - len(slots))
        counts = {z: int(side["counts"].get(z, 0)) for z in PILES}
        piles: dict[str, tuple[RawCard, ...]] = {z: () for z in PILES}
        piles["grave"] = tuple(RawCard(c, 0, 0) for c in side.get("grave", ()))
        if mine:
            piles["hand"] = tuple(RawCard(c, 0, 0) for c in side.get("hand", ()))
            n = counts.get("deck", 0)
            piles["deck"] = tuple(RawCard(deck[k % len(deck)], 0, 0) for k in range(n)) if deck else ()
        return RawPlayer(lp=max(1, int(side["lp"])), slots=tuple(slots[:FIELD_SLOTS]), piles=piles, counts=counts,
                         header=bytes(sem["normal_summons"] + 4))  # fmt: skip

    mem = rec["memory"]
    turn = rec["turn"] - sem["turn_base"]
    phase = PHASES.index(rec["phase"]) if rec["phase"] in PHASES else 2
    return RawDuel(turn=turn, phase=phase, turn_player=0 if rec["turn_player"] == "me" else 1, me=0,
                   players=(player(mem["me"], True), player(mem["opp"], False)), offsets=offsets)  # fmt: skip


def load_offsets() -> Offsets:
    path = next(iter(sorted(OFFSETS.glob("*.json"))))
    return Offsets.from_json(json.loads(path.read_text(encoding="utf-8")))


def _mix(rec: dict, root: Path, offsets: Offsets) -> tuple:
    """The dialog's mix entries and choices from the step's saved memory block (MD's own dialogs: not in md_request)."""
    import struct

    blob = rec.get("memory_blob")
    if not blob or not (root / blob).is_file():
        return (), ()
    s = np.load(root / blob)["S"].tobytes()

    def u32(o: int) -> int:
        return struct.unpack_from("<I", s, o)[0]

    mix = tuple((v >> 28, v & 0xFFFFFFF) for v in (u32(offsets.dialog_mix + 4 * k)
                                                    for k in range(min(u32(offsets.dialog_mix_count), 8))))  # fmt: skip
    choices = tuple(u32(offsets.dialog_choices + 4 * k) for k in range(min(u32(offsets.dialog_choice_count), 16)))
    return mix, choices


@dataclass
class Tally:
    before: Counter
    after: Counter
    agree: Counter  # rebuilt decisions: same answer as the rule's then / different
    picks: Counter  # rebuilt decisions by situation and answer

    @classmethod
    def new(cls) -> Tally:
        return cls(Counter(), Counter(), Counter(), Counter())


def audit(paths: list[Path], decider, passwords: dict[int, int], cards, *, limit: int = 0) -> dict[str, Tally]:
    """Who decided each in-action question then (``before``) and who decides now (``after``)."""
    from ygorl.selfplay.observe import md_option_desc

    offsets = load_offsets()
    out: dict[str, Tally] = defaultdict(Tally.new)
    for path in paths:
        recs = [r for r in read(path) if r.get("kind") == "step" and "memory" in r]
        pool = [c for r in recs for c in r["memory"]["me"].get("hand", []) + r["memory"]["me"]["grave"]
                + [s[0] for s in r["memory"]["me"]["field"] if s]]  # fmt: skip
        deck = [c for c in pool if passwords.get(c) in cards and not cards[passwords[c]].is_extra_deck]
        last = None  # the last policy decision (the action the questions belong to)
        last_rec = None  # and its step
        for rec in recs:
            d, o = rec.get("decision") or {}, rec.get("observed") or {}
            ls = o.get("lockstep") or {}
            if d.get("by") == "policy":
                last, last_rec = d, rec
            question = _question(rec, d, o)
            if question is None:
                continue
            tally = out[question]
            if limit and sum(tally.before.values()) >= limit:
                continue
            was = "shadow" if d.get("by") == "shadow" or ls.get("pick") is not None else "rule"
            if question == "target":
                was = "rule"
            tally.before[was] += 1
            if was == "shadow":  # a lockstep answer then is one now
                tally.after["shadow"] += 1
                continue
            try:
                raw = raw_from_digest(rec, offsets, deck)
            except Exception as exc:  # noqa: BLE001
                tally.after[f"rule (digest: {type(exc).__name__})"] += 1
                continue
            mix, choices = _mix(rec, path.parent, offsets)
            if question == "position" and not any(k == 5 for k, _ in mix) and last_rec is not None:
                # the monster in limbo since the policy's action (the Deck pile is not recorded: hand, GY only)
                from ygorl.selfplay.decide import in_limbo

                limbo = in_limbo(raw_from_digest(last_rec, offsets, []), raw_from_digest(rec, offsets, []), passwords,
                                 cards)  # fmt: skip
                listed = {passwords.get(c[0]) for r in recs[recs.index(last_rec) : recs.index(rec)]
                          for c in (r.get("observed") or {}).get("list_candidates") or ()}  # fmt: skip
                if len(limbo) != 1 and len(listed) == 1 and cards.get(next(iter(listed))) is not None:
                    limbo = listed if cards[next(iter(listed))].is_monster else limbo
                if len(limbo) == 1:
                    cid = next(c for c, p in passwords.items() if p == next(iter(limbo)))
                    mix = (*mix, (5, cid))
            now, same, label = _now(question, rec, d, o, ls, raw, last, decider, passwords, cards, mix, choices,
                                    md_option_desc)  # fmt: skip
            tally.after[now] += 1
            if same is not None:
                tally.agree["same as the rule" if same else "differs from the rule"] += 1
            if label:
                tally.picks[label] += 1
    return out


def _question(rec: dict, d: dict, o: dict) -> str | None:
    ls = o.get("lockstep") or {}
    if d.get("by") == "policy" and d.get("action") == "attack" and "target_slot" in o:
        return "target"
    if rec.get("dialog") == "position" or o.get("kind") == "position" or ls.get("lockstep") == "position":
        return "position"
    if (d.get("rule") or "").startswith("option") or ls.get("lockstep") == "option":
        return "option"
    if (d.get("rule") or "").startswith("prompt") or d.get("dialog") == "prompt" or ls.get("lockstep") == "prompt":
        return "prompt"
    if ls.get("lockstep") == "list" or (d.get("rule") or "").startswith("list"):
        return "list"
    return None


def _now(question, rec, d, o, ls, raw, last, decider, passwords, cards, mix, choices, md_option_desc):
    """(who decides now, whether a rebuilt answer agrees with the rule's then or None, the rebuilt answer)."""
    if question == "target":
        attacker = next((s for s, c in enumerate(rec["memory"]["me"]["field"][:7])
                         if c and passwords.get(c[0]) == d.get("card")), None)  # fmt: skip
        if attacker is None:
            return "rule (attacker not on the field)", None, ""
        choice = decider.attack_target(raw, attacker)
        occupied = {s for s, c in enumerate(rec["memory"]["opp"]["field"][:7]) if c}
        if choice.pick in occupied:
            p = choice.probs[choice.pick]
            return "rebuild", choice.pick == o.get("target_slot"), f"p(target) {'>= 0.9' if p >= 0.9 else '< 0.9'}"
        return f"rule ({(choice.notes or ['no target'])[0][:60]})", None, ""
    if question == "position":
        if ls.get("ygo") == "SelectPlace":
            return "shadow (zone first)", None, ""
        action = None
        if last is not None and last.get("action") == "spsummon":
            action = SimpleNamespace(kind="spsummon", card=SimpleNamespace(code=last.get("card")))
        choice = decider.position(raw, SimpleNamespace(params=mix), action, None)
        if choice.pick is None:
            return f"rule ({(choice.notes or ['?'])[0][:60]})", None, ""
        name = cards[choice.detail["card"]].name if choice.detail.get("card") in cards else "?"
        p = choice.probs.get(1, 0.0)
        label = f"{name}: {'defense' if choice.pick == 1 else 'attack'} (p(defense) {'< 0.1' if p < 0.1 else '>= 0.1'})"
        return "rebuild", choice.pick == 0, label
    if question == "option":
        tabs = [md_option_desc(c, passwords) for c in choices or [int(x, 16) for x in d.get("choices", [])]]
        card = last.get("card") if last is not None and last.get("action") == "activate" else None
        if card and any(t is not None and t >> 20 == card for t in tabs):
            return "policy (activate description)", None, ""
        return "rule", None, ""
    if question == "prompt":
        req = o.get("md_request") or {}
        cands = [SimpleNamespace(cid=c[0]) for c in req.get("candidates", [])]
        choice = decider.respond(raw, SimpleNamespace(candidates=cands))
        if choice.pick is None:
            return f"rule ({(choice.notes or ['?'])[0][:70]})", None, ""
        rule = d.get("rule")
        who = "opponent" if choice.detail.get("theirs") else "own"
        label = f"{who} {choice.detail.get('phase')} {req.get('kind')}: {'activate' if choice.pick == 0 else 'pass'}"
        label += f" (rule then: {'activate' if rule == 'prompt-respond' else 'pass'})" if rule else ""
        return "rebuild", (choice.pick == 0) == (rule == "prompt-respond") if rule else None, label
    if question == "list":
        cands = o.get("list_candidates") or []
        if ls.get("desync") and len({passwords.get(c[0]) for c in cands}) == 1 and cands:
            return "forced (one card)", None, ""
        return "rule", None, ""
    return "rule", None, ""


def format_audit(tallies: dict[str, Tally]) -> str:
    lines = []
    for q in QUESTIONS:
        t = tallies.get(q)
        if t is None:
            continue
        n = sum(t.before.values())
        lines.append(f"{q}: {n} asked; then " + ", ".join(f"{k} {v}" for k, v in t.before.most_common()))
        lines += [f"  now {k:58} {v}" for k, v in t.after.most_common()]
        if t.agree:
            lines.append("  rebuilt answers: " + ", ".join(f"{k} {v}" for k, v in t.agree.most_common()))
        lines += [f"    {v:4}  {k}" for k, v in sorted(t.picks.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="who decides the Solo player's in-action questions (design 10)")
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--policy", type=Path, help="score with this checkpoint (default: uniform, reachability only)")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="most questions audited per kind (0: all)")
    args = ap.parse_args(argv)
    from ygorl.advisor.reconstruct import OpponentPrior
    from ygorl.data.environment import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.md.memory import load_md_cid
    from ygorl.selfplay.decide import Decider, policy_scorer

    random.seed(0)
    env = load_environment(args.env)
    cards = default_cards()
    passwords = load_md_cid(env.artifact_path("md_cid.json"))
    scorer = None
    if args.policy is not None:
        from ygorl.train.checkpoint import load_actor

        scorer = policy_scorer(load_actor(args.policy), cards, env.player.starting_lp)
    decider = Decider(env.rule_flags, cards, passwords, OpponentPrior.from_environment(env), scorer=scorer,
                      samples=args.samples)  # fmt: skip
    print(format_audit(audit(args.trace, decider, passwords, cards, limit=args.limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
