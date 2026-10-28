"""Read-only self-audit of a Solo player run (design 10 「自审」): the checks a human auditor made on each trace.

Each check reads only the trace (:mod:`ygorl.selfplay.trace`): a step's ``observed`` notes and the memory digest
before and after it (the next step's ``memory``). The card database and the MD card id -> password map name the cards
and give the ATK / Field Spell type some checks need; without them those checks are skipped. Nothing touches the
game or the running player.

Findings carry a severity: ``major`` (the game did something other than what the player decided, or the player gave
up), ``minor`` (a recovery, an unverified pick, a repeated refusal), ``info`` (known and accepted: Ancient Chant's
Graveyard refusals, the rule picks listed in :data:`ACCEPTED`, policy picks after masks). A game's verdict is its
worst finding: ``correct`` (none above info), ``minor`` or ``major``.

    uv run --no-sync python -m ygorl.selfplay.selfaudit out/mdplay/<run>.jsonl [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ygorl.selfplay.trace import read

SEVERITIES = ("major", "minor", "info")
TYPE_FIELD = 0x80000  # = ygorl.engine.constants.TYPE_FIELD
FIELD_ZONE = 12  # the memory digest's Field Zone slot (5 Monster, 2 Extra Monster, 5 Spell & Trap zones before it)

# passwords of the cards the accepted items name
ANCIENT_CHANT = 78665705
CARD_ADVANCE = 52112003
RAS_DISCIPLE = 74875003
THE_TRUE_NAME = 39913299
DARK_SPELL_REGENERATION = 12930501
POT_OF_AVARICE = 67169062
MOUND_OF_THE_BOUND_CREATOR = 269012
MILLENNIUM_REVELATION = 41044418

# rules that answer a screen with one sensible click (no choice between cards or effects): counted, never findings
MECHANICAL_RULES = frozenset({
    "draw-continue", "notice-confirm", "zone-first-free", "zone-again", "banner-wait", "restore-minimised",
    "phase-menu-cancel", "yesno-yes-tribute", "yesno-yes-own-effect", "position-faceup-attack",
})  # fmt: skip
# rules accepted for the card the step belongs to (the user's list of known rule picks): info
ACCEPTED: dict[str, frozenset] = {
    "list-set-first-accepted": frozenset({CARD_ADVANCE, RAS_DISCIPLE, POT_OF_AVARICE}),
    "declare-most-common-in-deck": frozenset({THE_TRUE_NAME}),
    "list-first-on-screen": frozenset({DARK_SPELL_REGENERATION, MOUND_OF_THE_BOUND_CREATOR, THE_TRUE_NAME}),
    "field-pick-opponent-first": frozenset({DARK_SPELL_REGENERATION}),
    "prompt-respond": frozenset({MILLENNIUM_REVELATION}),
}
ACCEPTED_ANY = frozenset({"coin-call-first"})  # whatever the card


@dataclass
class Finding:
    severity: str  # major / minor / info
    game: int
    step: int | None
    seq: int | None
    kind: str
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class GameReport:
    game: int
    exit: int | None = None  # the game record's exit code (None: the trace ends inside this game)
    result: int | None = None
    turns: int | None = None
    steps: int = 0
    policy: int = 0  # decisions by the policy (rebuild and shadow answers included)
    rules: Counter = field(default_factory=Counter)  # rule decisions by rule name
    recoveries: Counter = field(default_factory=Counter)  # recoveries by cause
    findings: list[Finding] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        sev = {f.severity for f in self.findings}
        return "major" if "major" in sev else "minor" if "minor" in sev else "correct"

    def to_json(self) -> dict:
        return {"game": self.game, "verdict": self.verdict, "exit": self.exit, "result": self.result,
                "turns": self.turns, "steps": self.steps, "policy": self.policy, "rules": dict(self.rules),
                "recoveries": dict(self.recoveries), "findings": [asdict(f) for f in self.findings]}  # fmt: skip


@dataclass
class Report:
    path: str
    games: list[GameReport]

    @property
    def findings(self) -> list[Finding]:
        return [f for g in self.games for f in g.findings]

    def to_json(self) -> dict:
        return {"path": self.path, "verdicts": dict(Counter(g.verdict for g in self.games)),
                "games": [g.to_json() for g in self.games]}  # fmt: skip


# --- trace helpers ---------------------------------------------------------------------------------------------


def _split_games(records: Iterable[dict]) -> list[tuple[list[dict], dict | None]]:
    """(the game's step records, its game record or None) in order; a game ends at its ``game`` record."""
    out: list[tuple[list[dict], dict | None]] = []
    steps: list[dict] = []
    for r in records:
        if r.get("kind") == "step":
            steps.append(r)
        elif r.get("kind") == "game":
            out.append((steps, r))
            steps = []
    if steps:
        out.append((steps, None))
    return out


def _decisions(d: dict | None) -> list[dict]:
    """The step's decision and the ones made inside it (``then``)."""
    if not d:
        return []
    return [d, *(t for t in d.get("then") or () if isinstance(t, dict))]


def _piles(mem: dict) -> dict[str, Counter]:
    me, op = mem.get("me") or {}, mem.get("opp") or {}
    return {
        "hand": Counter(me.get("hand") or ()),
        "grave": Counter(me.get("grave") or ()),
        "field": Counter(s[0] for s in me.get("field") or () if s),
        "opp_grave": Counter(op.get("grave") or ()),
        "opp_field": Counter(s[0] for s in op.get("field") or () if s and s[0]),
    }


def _near(a, b, tol: int = 80) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) < tol


def _own_monster_slot(xy) -> int | None:
    from ygorl.md.ui import my_monster_xy

    return next((s for s in range(7) if _near(xy, my_monster_xy(s))), None)


def _own_zone(xy) -> tuple[str, int] | None:
    """The own Main Monster / Spell & Trap zone a "zone for" click hit: ("mzone" | "szone", 0-4)."""
    from ygorl.md import layout as L

    for s, x in enumerate(L.ZONE_X):
        if abs(xy[0] - x) < 40 and abs(xy[1] - L.MY_MZONE_Y) < 40:
            return "mzone", s
        if abs(xy[0] - x) < 40 and abs(xy[1] - L.MY_SZONE_Y) < 40:
            return "szone", s
    return None


class _Ctx:
    """Card names, ATK and types from the optional card database and cid map."""

    def __init__(self, passwords: Mapping[int, int], cards):
        self.passwords = passwords
        self.cards = cards
        self.cid_of: dict[int, set] = {}
        for cid, pw in passwords.items():
            self.cid_of.setdefault(pw, set()).add(cid)

    def pw(self, cid: int | None) -> int | None:
        return self.passwords.get(cid) if cid is not None else None

    def card(self, pw: int | None):
        return self.cards.get(pw) if self.cards is not None and pw is not None else None

    def name(self, pw: int | None) -> str:
        k = self.card(pw)
        return k.name if k is not None else str(pw)

    def cname(self, cid: int | None) -> str:
        pw = self.pw(cid)
        return self.name(pw) if pw is not None else f"cid {cid}"

    def same(self, cid_a: int | None, cid_b: int | None) -> bool:
        """Two MD card ids of one card (copies share the password)."""
        if cid_a == cid_b:
            return True
        pa, pb = self.pw(cid_a), self.pw(cid_b)
        return pa is not None and pa == pb

    def attack(self, cid: int | None) -> int | None:
        k = self.card(self.pw(cid))
        if k is None or getattr(k, "attack", None) is None or k.attack < 0:
            return None  # unknown or "?" (the Egyptian Gods): not compared
        return int(k.attack)


# --- the checks --------------------------------------------------------------------------------------------------


class _Game:
    def __init__(self, n: int, steps: list[dict], game: dict | None, ctx: _Ctx):
        self.ctx = ctx
        self.steps = steps
        self.rep = GameReport(game=n, steps=sum(1 for s in steps if "seq" in s))
        if game is not None:
            self.rep.exit, self.rep.result, self.rep.turns = game.get("exit"), game.get("result"), game.get("turn")
        self.game = game

    def add(self, severity: str, rec: dict | None, kind: str, message: str, **evidence) -> None:
        self.rep.findings.append(Finding(severity, self.rep.game, rec.get("step") if rec else None,
                                         rec.get("seq") if rec else None, kind, message, evidence))  # fmt: skip

    def later_memories(self, i: int, n: int) -> list[dict]:
        """The memory digests of the next ``n`` steps after step ``i`` (this game) that have one, up to the next
        policy decision (a later action's moves are not this pick's: run s85, a revealed card stays in the hand and
        the next action's summon of another hand card read as the pick's)."""
        out = []
        for r in self.steps[i + 1 :]:
            if "memory" in r:
                out.append(r["memory"])
            if (r.get("decision") or {}).get("by") == "policy" or len(out) >= n:
                break
        return out

    def next_memory(self, i: int) -> dict | None:
        """The memory digest of the first step after step ``i`` (this game) that has one."""
        for r in self.steps[i + 1 :]:
            if "memory" in r:
                return r["memory"]
        return None

    def run(self) -> GameReport:
        last_card: int | None = None  # the card the latest decision acted with (rule picks belong to it)
        refusals: dict[tuple, list] = {}
        failed: dict[tuple, list] = {}
        recoveries: dict[str, list] = {}
        low_p: list[int] = []
        for i, rec in enumerate(self.steps):
            d, o = rec.get("decision") or {}, rec.get("observed") or {}
            pc = o.get("prompt_card") or {}
            self.prev_lockstep = getattr(self, "this_lockstep", None)  # (the shadow dropped a step before the pick)
            self.this_lockstep = o.get("lockstep")
            if d.get("by") == "policy" and d.get("card"):
                last_card = d["card"]
            elif d.get("by") == "shadow" and isinstance(d.get("action"), list) and len(d["action"]) > 1:
                last_card = d["action"][1] or last_card
            if pc.get("want"):
                last_card = pc["want"]
            for dd in _decisions(d):
                if dd.get("by") in ("policy", "shadow", "rebuild"):
                    self.rep.policy += 1
                elif dd.get("by") == "rule" or dd.get("rule"):  # (a rule once recorded with by= overwritten)
                    self.rule(rec, dd.get("rule") or "?", dd.get("card") or last_card)
            if d.get("by") == "policy" and float(d.get("p", 1.0)) < 0.01:
                low_p.append(rec.get("seq"))
                masked = o.get("md_masked") or []
                if (
                    d.get("action") in ("battle_phase", "main2", "end_phase")
                    and masked
                    and o.get("md_nothing_lit", True)
                    and not o.get("md_lit_this_phase")  # (trusted: the client had lit cards in this phase)
                ):
                    # the phase was left because MD's highlight masked the cards' actions (run s83: an empty
                    # highlight during the Draw Phase's animation skipped whole turns): the policy wanted otherwise
                    self.add("major", rec, "masked-phase", f"{d['action']} at p={float(d.get('p', 0)):.3f} with "
                             f"{len(masked)} card actions masked by MD's highlight", masked=masked)  # fmt: skip
            if (gu := o.get("graveyard_unverified")) is not None:
                self.add("minor", rec, "graveyard-unverified", f"{self.ctx.name(gu.get('want'))} in the Graveyard was "
                         "taken as refused without reading which card was open (masked for the phase)")  # fmt: skip
            if (r := rec.get("refusal")) is not None:
                refusals.setdefault((rec.get("turn"), r.get("action"), r.get("card"), r.get("reason")), []).append(rec)
            if (c := rec.get("check")) is not None and not c.get("ok", True):
                failed.setdefault((rec.get("turn"), c.get("action"), c.get("card")), []).append(rec)
            if o.get("menu") == "turn_order":
                self.add("info", rec, "rule-accepted", "turn-order menu answered by the script (known, accepted)")
            if isinstance(fl := o.get("failure"), dict):  # a failure by known cause, recovered by masking the action
                recoveries.setdefault(f"failure: {fl.get('cause')}", []).append(rec)
            if (rv := o.get("recovery")) is not None:
                recoveries.setdefault(str(rv.get("why")), []).append(rec)
            self.check_selected(rec, o)
            self.check_prompt_card(rec, d, o)
            self.check_list(i, rec, d, o)
            self.check_attack(rec, d, o)
            self.check_tributes(rec, d, o)
            self.check_zones(rec, d, self.next_memory(i))
        self.report_grouped(refusals, failed, recoveries, low_p)
        if self.game is not None and self.game.get("exit") not in (0, None):
            self.add("major", self.steps[-1] if self.steps else None, "duel-given-up",
                     f"the duel was given up (exit {self.game.get('exit')}, turn {self.game.get('turn')})",
                     exit=self.game.get("exit"), finished=self.game.get("finished"), lp=self.game.get("lp"))  # fmt: skip
        return self.rep

    # rules ------------------------------------------------------------------------------------------------------
    def rule(self, rec: dict, rule: str, card: int | None) -> None:
        self.rep.rules[rule] += 1
        if rule == "position-faceup-attack":
            notes = ((rec.get("observed") or {}).get("rebuild") or {}).get("notes") or []
            if any("not known" in str(n) for n in notes):
                # the position fell to a rule because the summoned monster was not named (run s57: Mahad drawn
                # and summoned by its own prompt); the policy could have chosen it
                self.add("minor", rec, "position-by-rule", "a rule chose the position: the summoned monster was "
                         "not known to the rebuild", rule=rule)  # fmt: skip
                return
        if rule in MECHANICAL_RULES or rule.startswith("recover-"):
            return
        if rule in ACCEPTED_ANY or card in ACCEPTED.get(rule, ()):
            self.add("info", rec, "rule-accepted", f"{rule} for {self.ctx.name(card)} (known, accepted)",
                     rule=rule, card=card)  # fmt: skip
            return
        if rule == "prompt-respond":
            pc = (rec.get("observed") or {}).get("prompt_card") or {}
            if pc.get("want") and pc.get("got") == pc.get("want"):
                self.add("info", rec, "rule-accepted", f"{rule}: {self.ctx.name(card)} (the offered card, confirmed)",
                         rule=rule, card=card)  # fmt: skip
                return
        if rule in ("tribute-lowest-atk", "field-pick-own-lowest-atk"):
            o = rec.get("observed") or {}
            lock = o.get("lockstep") or getattr(self, "prev_lockstep", None) or {}
            if lock.get("ygo") == "SelectTribute":
                # the shadow had reached the Tributes, so the policy could have picked them: a rule did (run s53:
                # Dark Magician tributed while Soul Crossing allowed the opponent's monster)
                # (an opponent monster MD's request offers, memory; a frame seen on their side can be a glow:
                # Ra's Tributes, which cannot include theirs, were flagged on one)
                req = o.get("md_request") or {}
                theirs = [c for c in req.get("candidates") or [] if len(c) > 3 and c[3] not in (0, None)]
                self.add("major" if theirs else "minor", rec, "tribute-by-rule",
                         f"a rule picked the Tributes though the shadow asked them ({lock.get('desync') or 'dropped'})"
                         + (": the opponent's monster offered was not considered" if theirs else ""),
                         rule=rule, theirs=theirs)  # fmt: skip
            return  # otherwise checked by check_tributes against the board
        if rule == "summon-pick-first-offered":
            o = rec.get("observed") or {}
            ygo = ((o.get("shadow_vs_md") or {}).get("ygo") or {}).get("decision")
            if ygo == "SelectCard":
                # the shadow asked which monster to summon: the policy could have picked it (run s54: a guess
                # between the two Ra cards summoned Sphere Mode)
                self.add("minor", rec, "summon-pick-by-rule", "a rule picked the monster to summon though the "
                         "shadow asked it", rule=rule)  # fmt: skip
                return
        if rule == "prompt-respond-hand-trigger":
            k = self.ctx.card(self.ctx.pw(card)) or self.ctx.card(card)  # the rule records an MD card id
            if k is not None and not k.is_monster:
                # a hand trigger is a monster's: a Spell taken by this rule is a play no policy chose (Soul
                # Crossing at an open timing, run s51)
                self.add("major", rec, "hand-trigger-not-monster",
                         f"the hand-trigger rule activated {k.name}, which is no monster", rule=rule, card=card)  # fmt: skip
                return
        self.add("info", rec, "rule-decision", f"a rule decided: {rule} (in the action of {self.ctx.name(card)})",
                 rule=rule, card=card)  # fmt: skip

    # per-step checks -----------------------------------------------------------------------------------------------
    def check_selected(self, rec: dict, o: dict) -> None:
        s = o.get("selected")
        if isinstance(s, dict) and "got" in s and not self.ctx.same(s.get("want"), s.get("got")):
            self.add("major", rec, "selected-mismatch",
                     f"opened {self.ctx.cname(s.get('got'))} for an action of {self.ctx.cname(s.get('want'))}",
                     want=s.get("want"), got=s.get("got"))  # fmt: skip

    def check_prompt_card(self, rec: dict, d: dict, o: dict) -> None:
        pc = o.get("prompt_card")
        if not isinstance(pc, dict) or not pc.get("want"):
            return
        want, got = pc["want"], pc.get("got")
        if got == want:
            return
        whys = [str(c.get("why", "")) for c in rec.get("clicks") or ()]
        passed = d.get("rule") == "prompt-pass" or any(w == "pass" or w.startswith("pass") for w in whys)
        listed = o.get("prompt_list") or []
        only_want = bool(listed) and all(self.ctx.pw(c) == want for c in listed)
        ev = {"want": want, "got": got, "prompt_list": listed, "passed": passed}
        if got is not None and only_want and passed:
            # the client's list offered exactly the meant card, yet a disagreeing read made the player pass (run
            # s57: a stale open-card read named a Mirror Force from an earlier duel)
            self.add("major", rec, "prompt-card", f"{self.ctx.name(want)} was offered (client list) and meant, but "
                     f"the prompt was passed on a read of {self.ctx.name(got)}", **ev)  # fmt: skip
        elif got is not None:
            self.add("minor", rec, "prompt-card", f"MD offered {self.ctx.name(got)}, the player meant "
                     f"{self.ctx.name(want)}" + (" (passed)" if passed else ""), **ev)  # fmt: skip
        elif passed:
            self.add("major", rec, "prompt-card", f"{self.ctx.name(want)} was meant but the prompt was passed "
                     "(the offered card was not confirmed)", **ev)  # fmt: skip
        elif only_want or want == MILLENNIUM_REVELATION:
            self.add("info", rec, "prompt-card", f"{self.ctx.name(want)} activated without a confirming read "
                     "(the only card offered)", **ev)  # fmt: skip
        else:
            self.add("minor", rec, "prompt-card", f"{self.ctx.name(want)} activated without a confirming read among "
                     f"{len(listed)} offered cards", **ev)  # fmt: skip

    def check_list(self, i: int, rec: dict, d: dict, o: dict) -> None:
        if any(dd.get("rule") == "list-wanted-card-not-displayed" for dd in _decisions(d)):
            self.add("minor", rec, "list-pick", "the wanted card is not displayed in MD's list (a rule picked)")
        lc = o.get("list_check")
        if not isinstance(lc, dict) or not lc.get("want"):
            return
        want = lc["want"]
        if lc.get("unverified"):
            self.add("minor", rec, "list-pick", f"list pick of {self.ctx.cname(want)} unverified (position guessed)",
                     want=want)  # fmt: skip
        elif lc.get("ok") is False:
            self.add("major", rec, "list-pick", f"list pick of {self.ctx.cname(want)} not confirmed by the client list",
                     want=want)  # fmt: skip
        if "memory" not in rec:
            return
        cands = o.get("list_candidates") or []
        # the picked card can move only once MD has asked the rest (a Fusion Monster's position and zone): the
        # first of the next few steps where a candidate moved (run s77's wrong Fusion Monster moved 4 steps later)
        moved = []
        for after in self.later_memories(i, 6):
            moved = self.moved(rec["memory"], after, cands)
            if moved:
                break
        if moved and not any(self.ctx.same(c, want) for c in moved):
            self.add("major", rec, "list-pick", f"the list pick wanted {self.ctx.cname(want)} but "
                     + ", ".join(self.ctx.cname(c) for c in moved) + " moved", want=want, moved=moved)  # fmt: skip

    @staticmethod
    def moved(before: dict, after: dict, cands: list) -> list[int]:
        """The candidates' card ids that left their pile (or, from the Deck, turned up in hand / GY / field)."""
        b, a = _piles(before), _piles(after)
        out: list[int] = []
        for c in cands:
            cid, loc = c[0], c[1] if len(c) > 1 else ""
            if not cid or cid in out:
                continue
            if loc in ("deck", "extra"):
                if any(a[k][cid] > b[k][cid] for k in ("hand", "grave", "field")):
                    out.append(cid)
                continue
            keys = {"hand": ("hand",), "grave": ("grave", "opp_grave"), "mzone": ("field", "opp_field"),
                    "szone": ("field", "opp_field")}.get(loc, ())  # fmt: skip
            if any(a[k][cid] < b[k][cid] for k in keys):
                out.append(cid)
        return out

    def check_attack(self, rec: dict, d: dict, o: dict) -> None:
        if d.get("by") != "policy" or d.get("action") != "attack" or "target_slot" not in o:
            return
        slot, target = o["target_slot"], d.get("target") if isinstance(d.get("target"), dict) else {}
        if target.get("by") == "rebuild" and target.get("pick") is not None and target["pick"] != slot:
            self.add("major", rec, "attack-target", f"the attack went to zone {slot}, the rebuild picked zone "
                     f"{target['pick']}", rebuild=target["pick"], target_slot=slot)  # fmt: skip
        field_ = ((rec.get("memory") or {}).get("opp") or {}).get("field") or []
        if isinstance(slot, int) and 0 <= slot < len(field_) and field_[slot] is None:
            self.add("major", rec, "attack-target", f"the attack target zone {slot} is empty", target_slot=slot)

    def check_tributes(self, rec: dict, d: dict, o: dict) -> None:
        """A rule's Tribute / own field pick: the monster taken has the lowest ATK among the candidates."""
        rules = {dd.get("rule") for dd in _decisions(d)}
        mem = rec.get("memory") or {}
        own = (mem.get("me") or {}).get("field") or []
        if "tribute-lowest-atk" in rules:
            picks = [(c["xy"], "(policy)" in str(c.get("why")))
                     for c in rec.get("clicks") or () if str(c.get("why", "")).startswith("tribute ")]  # fmt: skip
            cands = o.get("tribute_candidates")
        elif "field-pick-own-lowest-atk" in rules:
            picks = [(c["xy"], False) for c in rec.get("clicks") or () if c.get("why") == "highlighted card"][:1]
            before = o.get("picked_before") or []
            lit = [xy for xy in o.get("highlighted") or () if not any(_near(xy, p, 40) for p in before)]
            cands = sorted({s for xy in lit if (s := _own_monster_slot(xy)) is not None})
        else:
            return
        if cands is None:  # older traces: every monster of ours on the field
            cands = [s for s in range(min(7, len(own))) if own[s]]
        taken: set = set()
        for xy, by_policy in picks:
            s = _own_monster_slot(xy)
            if s is None or s >= len(own) or own[s] is None:
                continue
            if by_policy:  # the policy's own pick (through the shadow): not the rule's
                taken.add(s)
                continue
            pool = [c for c in cands if c not in taken and c < len(own) and own[c] is not None] + [s]
            atk = {c: self.ctx.attack(own[c][0]) for c in set(pool)}
            mine = atk.get(s)
            known = {c: a for c, a in atk.items() if a is not None}
            if mine is not None and known and mine > min(known.values()):
                low = min(known, key=known.get)
                self.add("major", rec, "tribute-not-lowest",
                         f"the rule took {self.ctx.cname(own[s][0])} ({mine}) while {self.ctx.cname(own[low][0])} "
                         f"({known[low]}) was a candidate", slot=s, atk=mine, candidates=sorted(known.items()))  # fmt: skip
                return  # one finding per step: the later picks follow from the same candidates
            taken.add(s)

    def check_zones(self, rec: dict, d: dict, after_mem: dict | None) -> None:
        own = ((rec.get("memory") or {}).get("me") or {}).get("field") or []
        field_spell = False
        if d.get("by") == "policy" and d.get("action") in ("activate", "sset"):
            k = self.ctx.card(d.get("card"))
            field_spell = k is not None and bool(getattr(k, "type", 0) & TYPE_FIELD)
        for c in rec.get("clicks") or ():
            if not str(c.get("why", "")).startswith("zone for") or not c.get("xy"):
                continue
            z = _own_zone(c["xy"])
            if z is None:
                continue
            loc, s = z
            idx = s if loc == "mzone" else 7 + s
            if field_spell and loc == "szone":
                # harmless when the zone was empty and the card reached the Field Zone (MD ignores the click)
                after = ((after_mem or {}).get("me") or {}).get("field") or []
                landed = len(after) > FIELD_ZONE and after[FIELD_ZONE] is not None
                empty = not (idx < len(own) and own[idx] is not None)
                self.add("minor" if empty and landed else "major", rec, "field-spell-zone",
                         f"a Spell & Trap zone was clicked for the Field Spell {self.ctx.name(d.get('card'))}"
                         + ("" if empty and landed else " (the zone was taken or the card did not reach the Field Zone)"),
                         zone=s, card=d.get("card"), landed=landed)  # fmt: skip
                continue
            if idx < len(own) and own[idx] is not None:
                self.add("minor", rec, "zone-occupied", f"clicked the occupied {loc} {s} "
                         f"({self.ctx.cname(own[idx][0])}) for {c.get('why')}", zone=[loc, s])  # fmt: skip

    # grouped ----------------------------------------------------------------------------------------------------
    def report_grouped(self, refusals: dict, failed: dict, recoveries: dict, low_p: list) -> None:
        for (turn, action, card, reason), recs in refusals.items():
            n, seqs = len(recs), [r.get("seq") for r in recs]
            if card == ANCIENT_CHANT and "Graveyard" in str(reason):
                self.add("info", recs[0], "refusal", f"{reason} x{n} in turn {turn} (known, accepted)", seqs=seqs)
            elif n > 1:
                self.add("minor", recs[0], "refusal-repeated", f"{action} {self.ctx.name(card)} refused {n} times in "
                         f"turn {turn}: {reason}", seqs=seqs)  # fmt: skip
            else:
                self.add("info", recs[0], "refusal", f"{action} {self.ctx.name(card)} refused in turn {turn}: {reason}",
                         seqs=seqs)  # fmt: skip
        for (turn, action, card), recs in failed.items():
            n, seqs = len(recs), [r.get("seq") for r in recs]
            problem = (recs[0].get("check") or {}).get("problem")
            self.add("major", recs[0], "check-failed", f"{action} {self.ctx.name(card)} did not take"
                     + (f" ({n} times)" if n > 1 else "") + f" in turn {turn}: {problem}", seqs=seqs)  # fmt: skip
        for why, recs in recoveries.items():
            self.rep.recoveries[why] += len(recs)
            self.add("minor", recs[0], "recovery", f"{len(recs)} recover{'ies' if len(recs) > 1 else 'y'}: {why}",
                     seqs=[r.get("seq") for r in recs])  # fmt: skip
        if low_p:
            self.add("info", None, "low-p", f"{len(low_p)} policy picks with p < 0.01 (the preferred actions were "
                     "masked or refused)", seqs=low_p)  # fmt: skip


def audit_records(records: Iterable[dict], passwords: Mapping[int, int] | None = None, cards=None,
                  path: str = "") -> Report:  # fmt: skip
    """:func:`audit_trace` on records already read."""
    ctx = _Ctx(passwords or {}, cards)
    games = [_Game(n, steps, game, ctx).run() for n, (steps, game) in enumerate(_split_games(records), 1)]
    return Report(path, games)


def audit_trace(path: str | Path, passwords: Mapping[int, int] | None = None, cards: Any = None) -> Report:
    """Audit one trace (JSONL, :mod:`ygorl.selfplay.trace`): per-game findings. ``passwords`` (MD card id ->
    password) and ``cards`` (the card database) name cards and enable the ATK / card type checks; None loads the
    run's environment map and the default database (when loadable)."""
    records = list(read(path))
    if passwords is None or cards is None:
        env_name = next((r.get("env") for r in records if r.get("kind") == "start" and r.get("env")), "md-2026-09")
        passwords = passwords if passwords is not None else _load_passwords(env_name)
        cards = cards if cards is not None else _load_cards()
    return audit_records(records, passwords, cards, str(path))


def _load_passwords(env_name: str) -> dict[int, int]:
    try:
        from ygorl.data.environment import load_environment
        from ygorl.md.memory import load_md_cid

        return load_md_cid(load_environment(env_name).artifact_path("md_cid.json"))
    except Exception:  # noqa: BLE001 - names and card checks are optional
        return {}


def _load_cards():
    try:
        from ygorl.engine.duel import default_cards

        return default_cards()
    except Exception:  # noqa: BLE001
        return None


def format_report(rep: Report, show_info: bool = False) -> str:
    lines = [f"== {rep.path}"]
    lines.append(f"{'game':>4}  {'verdict':8} {'exit':>4} {'result':>6} {'turns':>5} {'steps':>5} {'policy':>6} "
                 f"{'rules':>5} {'recov':>5} {'major':>5} {'minor':>5} {'info':>4}")  # fmt: skip
    for g in rep.games:
        n = Counter(f.severity for f in g.findings)
        lines.append(f"{g.game:>4}  {g.verdict:8} {g.exit if g.exit is not None else '-':>4} "
                     f"{g.result if g.result is not None else '-':>6} {g.turns if g.turns is not None else '-':>5} "
                     f"{g.steps:>5} {g.policy:>6} {sum(g.rules.values()):>5} {sum(g.recoveries.values()):>5} "
                     f"{n['major']:>5} {n['minor']:>5} {n['info']:>4}")  # fmt: skip
    verdicts = Counter(g.verdict for g in rep.games)
    lines.append("verdicts: " + ", ".join(f"{v} {verdicts[v]}" for v in ("correct", "minor", "major") if verdicts[v]))
    rules: Counter = sum((g.rules for g in rep.games), Counter())
    if rules:
        lines.append("rule decisions: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
    lines.append("findings:")
    for f in rep.findings:
        if f.severity == "info" and not show_info:
            continue
        where = f"seq {f.seq}" if f.seq is not None else "-"
        lines.append(f"  [{f.severity}] game {f.game} {where:>8} {f.kind}: {f.message}")
    if not show_info:
        info = Counter(f.kind for f in rep.findings if f.severity == "info")
        if info:
            lines.append("  info (--info to list): " + ", ".join(f"{k} {v}" for k, v in info.most_common()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="self-audit of Solo player traces (design 10 「自审」); read-only")
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true", help="one JSON report per trace")
    ap.add_argument("--info", action="store_true", help="list info findings too")
    args = ap.parse_args(argv)
    passwords, cards = None, None
    for path in args.trace:
        try:
            rep = audit_trace(path, passwords, cards)
        except Exception as exc:  # noqa: BLE001 - the audit never fails a run
            print(f"== {path}: cannot audit: {type(exc).__name__}: {exc}")
            continue
        print(json.dumps(rep.to_json(), ensure_ascii=False) if args.json else format_report(rep, args.info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
