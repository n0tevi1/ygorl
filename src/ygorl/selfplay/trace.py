"""Per-step trace of the Solo player (design 10, Solo self-play): one JSONL file per run, one record per loop step.

A run writes ``<out>/<run>.jsonl``: a ``start`` record (run metadata), one ``step`` record per loop step and an
``end`` record (exit code, LP). A step record::

    {"kind": "step", "step": 7, "t": 1761000000.12, "turn": 3, "phase": "main1", "turn_player": "me",
     "input": {"mode": 2, "accepting": true, "player": 0}, "dialog": null,
     "memory": {"me": {"lp": 8000, "field": [[cid, face, sideways] | null, ... 13], "hand": [cid, ...],
                       "grave": [cid, ...], "counts": {...}},
                "opp": {"lp": 6100, "field": [... face-down cards as [0, face, sideways]], "grave": [...],
                        "counts": {...}}},
     "decision": {"by": "policy", "action": "summon", "card": 89631139, "p": 0.61, "phase": "main1",
                  "top": [["summon", 89631139, 0.61], ...]}            # or {"by": "rule", "rule": "prompt-pass"}, {"by": "shadow", "dialog": "list", "pick": 1, ...},
                                                                        # {"by": "rebuild", "question": "respond", "pick": 0, "probs": {...}, ...};
                                                                        # an attack carries "target": {"by": "rebuild" | "rule", ...}
     "observed": {"icons": [[x, y], ...]},                              # what the UI layer reported
     "clicks": [{"t": ..., "xy": [x, y], "why": "summon"}, ...],
     "check": {"action": "summon", "card": 89631139, "ok": true, "problem": null},   # the previous action's check
     "refusal": {"action": "summon", "card": 89631139, "reason": "..."},            # MD showed no command
     "stuck": null,                                                     # or why the player stopped
     "screen": "run-screens/0007.npz"}                                  # when a capture was saved

The memory digest holds what the client may see: the opponent's hand, deck and face-down cards are never written
(only their counts). Captures are saved every ``screen_every`` steps (and on :meth:`Trace.save_screen`) as
compressed ``.npz`` at half resolution of the reference frame.

    python -m ygorl.selfplay.trace SUMMARY out/mdplay/<run>.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PHASE_NAMES = ("draw", "standby", "main1", "battle", "main2", "end")  # = ygorl.md.memory.PHASES


def phase_name(phase: int) -> str:
    return PHASE_NAMES[phase] if 0 <= phase < len(PHASE_NAMES) else str(phase)


def digest(raw) -> dict:
    """A compact, client-visible view of a :class:`ygorl.md.memory.RawDuel` (MD card ids)."""
    faceup = (getattr(raw.offsets, "semantics", None) or {}).get("faceup", 1)

    def side(p: int) -> dict:
        pl, mine = raw.players[p], p == raw.me
        slots = []
        for c in pl.slots:
            if c is None:
                slots.append(None)
            else:
                known = mine or bool(c.face & faceup)
                slots.append([c.cid if known else 0, c.face, c.sideways])
        out: dict[str, Any] = {"lp": pl.lp, "field": slots}
        if mine:
            out["hand"] = [c.cid for c in pl.piles["hand"]]
        out["grave"] = [c.cid for c in pl.piles["grave"]]
        out["counts"] = dict(pl.counts)
        return out

    return {"me": side(raw.me), "opp": side(1 - raw.me)}


def _card(action) -> int:
    return action.card.code if getattr(action, "card", None) is not None else 0


@dataclass
class Trace:
    """Writes one JSONL file per run; :meth:`begin` opens a step record, :meth:`end_step` writes it."""

    out: Path
    run: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    seq: int = 0  # records written so far: names the screens and memory snapshots uniquely across duels
    screen_every: int = 0  # save the reference-frame capture every N steps (0: only on demand)
    rec: dict | None = None
    last: dict | None = None  # the last record written (for stuck records)
    rules: Counter = field(default_factory=Counter)  # rule-made decisions of the duel under way (a fallback count)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / f"{self.run}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.last = rec

    def start(self, **meta) -> None:
        self._write({"kind": "start", "t": round(time.time(), 2), "run": self.run, **meta})

    def begin(self, step: int, raw, inp, dialog: str | None) -> dict:
        """Open the record of loop step ``step`` from what was read before deciding."""
        self.flush()
        self.seq += 1  # run-wide: the step number restarts with every duel of a series
        self.rec = {"kind": "step", "seq": self.seq, "step": step, "t": round(time.time(), 2), "turn": raw.turn + 1,
                    "phase": phase_name(raw.phase), "turn_player": "me" if raw.turn_player == raw.me else "opp",
                    "input": {"mode": inp.mode, "accepting": bool(inp.accepting), "player": inp.player},
                    "dialog": dialog, "memory": digest(raw), "decision": None, "observed": {}, "clicks": [],
                    "check": None, "refusal": None, "stuck": None, "screen": None}  # fmt: skip
        return self.rec

    def _current(self) -> dict:
        if self.rec is None:  # notes outside a step (should not happen): keep them in an orphan record
            self.rec = {"kind": "step", "step": -1, "t": round(time.time(), 2), "decision": None, "observed": {},
                        "clicks": [], "check": None, "refusal": None, "stuck": None, "screen": None}  # fmt: skip
        return self.rec

    # --- notes on the open step -------------------------------------------------------------------------------
    def policy(self, action, prob: float, phase: str, top: Iterable = ()) -> None:
        """The policy's choice; ``top``: the allowed suggestions (``.action``, ``.prob``), best first."""
        self._current()["decision"] = {
            "by": "policy", "action": action.kind, "card": _card(action), "p": round(float(prob), 4), "phase": phase,
            "top": [[s.action.kind, _card(s.action), round(float(s.prob), 4)] for s in list(top)[:5]],
            # which effect (an activation's card string): an option dialog's tab is checked against it
            "description": int(getattr(action, "description", 0) or 0),
        }  # fmt: skip

    def rule(self, rule: str, **detail) -> None:
        """A hard-coded rule decided this step (prompt, dialog, field pick, ...)."""
        self.rules[rule] += 1
        self._decide({"by": "rule", "rule": rule, **detail})

    def shadow(self, dialog: str, pick: int, **detail) -> None:
        """The policy decided this dialog through the shadow duel (design 10 「边角情况由 player 自己学会」)."""
        self._decide({"by": "shadow", "dialog": dialog, "pick": pick, **detail})

    def rebuilt(self, **record) -> None:
        """The policy decided this step on a rebuild of the current board (:class:`ygorl.selfplay.decide.Choice`
        record: question, pick, averaged probabilities)."""
        self._decide(_jsonable({"by": "rebuild", **record}))

    def _decide(self, decision: dict) -> None:
        """Record this step's decision; one made inside the policy's action this step (its Tributes, a list it
        asks) goes under the policy decision's ``then`` instead of replacing it (an audit read the Tribute rule as
        the step's decision and the summon was lost)."""
        rec = self._current()
        if rec["decision"] is not None and rec["decision"].get("by") == "policy":
            rec["decision"].setdefault("then", []).append(decision)
        else:
            rec["decision"] = decision

    def sub(self, name: str, **detail) -> None:
        """A part of this step's decision decided separately (the attack target of a policy attack): ``detail``
        holds its own ``by``."""
        rec = self._current()
        if rec["decision"] is None:
            rec["decision"] = {"by": detail.get("by"), "question": name}
        rec["decision"][name] = _jsonable(detail)

    def observe(self, **seen) -> None:
        """What the UI layer reported (icons found, list cards, highlighted cards, ...)."""
        self._current()["observed"].update(_jsonable(seen))

    def check(self, action, problem: str | None) -> None:
        self._current()["check"] = {"action": action.kind, "card": _card(action), "ok": problem is None,
                                    "problem": problem}  # fmt: skip

    def refusal(self, action, reason: str) -> None:
        self._current()["refusal"] = {"action": action.kind, "card": _card(action), "reason": reason}

    def game(self, **summary) -> None:
        """One duel's outcome (several per trace with --games), with the rules that decided in it (each a fallback
        where no policy decision was available: the count to drive down)."""
        self._write({"kind": "game", "t": round(time.time(), 2), **summary, "rules": dict(self.rules)})
        self.rules = Counter()

    def stuck(self, why: str, record: str | None = None) -> None:
        rec = self._current()
        rec["stuck"] = why
        if record is not None:
            rec["stuck_record"] = record

    def save_screen(self, img: np.ndarray) -> str:
        """Save a reference-frame capture (half resolution, compressed) next to the trace; returns its path."""
        rec = self._current()
        d = self.out / f"{self.run}-screens"
        d.mkdir(exist_ok=True)
        name = d / f"{rec.get('seq', 0):05d}.npz"
        np.savez_compressed(name, screen=np.ascontiguousarray(img[::2, ::2]))
        rec["screen"] = str(name.relative_to(self.out))
        return rec["screen"]

    def save_memory(self, blocks: dict[str, bytes]) -> str:
        """Raw engine memory of this step (the duel state block and the like), for decoding what the snapshot does
        not read yet: once-per-turn usage, turn-long effects, activity counters (design 10 「识别与执行的系统化」)."""
        rec = self._current()
        d = self.out / f"{self.run}-memory"
        d.mkdir(exist_ok=True)
        name = d / f"{rec.get('seq', 0):05d}.npz"
        np.savez_compressed(name, **{k: np.frombuffer(v, np.uint8) for k, v in blocks.items()})
        rec["memory_blob"] = str(name.relative_to(self.out))
        return rec["memory_blob"]

    def wants_screen(self, step: int) -> bool:
        return self.screen_every > 0 and step % self.screen_every == 0

    def end_step(self, clicks: Iterable[dict] = ()) -> None:
        """Write the open step with the clicks issued during it."""
        if self.rec is None:
            return
        self.rec["clicks"].extend(clicks)
        rec, self.rec = self.rec, None
        self._write(rec)

    def flush(self) -> None:
        if self.rec is not None:
            self.end_step()

    def recent(self, n: int = 30) -> list[dict]:
        """The last ``n`` records of this run (for stuck records)."""
        self._fh.flush()
        return list(read(self.path))[-n:]

    def close(self, **result) -> None:
        self.flush()
        if not self._fh.closed:
            self._write({"kind": "end", "t": round(time.time(), 2), **result})
            self._fh.close()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def read(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def summarize(records: Iterable[dict]) -> dict:
    """Counts over a run: decisions by kind, checks, refusals by reason, stuck, steps per turn."""
    decisions: Counter = Counter()
    checks: Counter = Counter()
    failed: Counter = Counter()
    refusals: Counter = Counter()
    per_turn: Counter = Counter()
    stuck: list[str] = []
    steps = clicks = 0
    end = None
    for r in records:
        if r.get("kind") == "end":
            end = r
            continue
        if r.get("kind") != "step":
            continue
        steps += 1
        clicks += len(r.get("clicks") or ())
        per_turn[r.get("turn", 0)] += 1
        d = r.get("decision")
        if d:
            by = d.get("by")
            label = {"policy": f"policy:{d.get('action')}", "shadow": f"shadow:{d.get('dialog')}",
                     "rebuild": f"rebuild:{d.get('question')}"}  # fmt: skip
            decisions[label.get(by, f"rule:{d.get('rule')}")] += 1
            target = d.get("target")
            if isinstance(target, dict):  # an attack's target, decided on its own
                decisions[f"target:{target.get('by')}"] += 1
        c = r.get("check")
        if c:
            checks["ok" if c.get("ok") else "failed"] += 1
            if not c.get("ok"):
                failed[c.get("action", "?")] += 1
        f = r.get("refusal")
        if f:
            refusals[f.get("reason", "?")] += 1
        if r.get("stuck"):
            stuck.append(r["stuck"])
    return {"steps": steps, "clicks": clicks, "decisions": dict(decisions), "checks": dict(checks),
            "failed_checks": dict(failed), "refusals": dict(refusals), "stuck": len(stuck), "stuck_reasons": stuck,
            "steps_per_turn": dict(sorted(per_turn.items())), "end": end}  # fmt: skip


def format_summary(s: dict) -> str:
    lines = [f"steps {s['steps']}, clicks {s['clicks']}, stuck {s['stuck']}"]
    lines.append("decisions:")
    lines += [f"  {k:32} {v}" for k, v in sorted(s["decisions"].items(), key=lambda kv: (-kv[1], kv[0]))]
    ok, bad = s["checks"].get("ok", 0), s["checks"].get("failed", 0)
    lines.append(f"checks: ok {ok}, failed {bad}")
    lines += [f"  failed {k:25} {v}" for k, v in sorted(s["failed_checks"].items())]
    lines.append(f"refusals: {sum(s['refusals'].values())}")
    lines += [f"  {v:3}  {k}" for k, v in sorted(s["refusals"].items(), key=lambda kv: (-kv[1], kv[0]))]
    lines += [f"stuck: {why}" for why in s["stuck_reasons"]]
    lines.append("steps per turn: " + ", ".join(f"{t}:{n}" for t, n in s["steps_per_turn"].items()))
    if s["end"] is not None:
        lines.append("end: " + ", ".join(f"{k} {v}" for k, v in s["end"].items() if k not in ("kind", "t")))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solo player traces (design 10)")
    ap.add_argument("command", choices=("SUMMARY",))
    ap.add_argument("trace", type=Path, nargs="+")
    args = ap.parse_args(argv)
    for path in args.trace:
        if len(args.trace) > 1:
            print(f"== {path}")
        print(format_summary(summarize(read(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
