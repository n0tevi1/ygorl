"""Freeze and audit selected natural stop decisions; full-state diagnostic only."""

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path

from behavior_replay import sha
from ygorl import _core
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession
from ygorl.engine.replay import Replay
from ygorl.eval.win_search import SearchBudget, position, search_action, verify_certificate
from ygorl.train.registration import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def select(source, count):
    candidates = []
    for case in sorted((source / "replays").glob("seed-0-cold-u128-*/*")):
        if not (case / "report.json").exists():
            continue
        report = read(case / "report.json")
        if not report["original_result_equal"] or not report["cold_replay_equal"]:
            raise ValueError("source has not passed replay checks")
        for name, digest in report["files"].items():
            if sha(case / name) != digest:
                raise ValueError("source artifact changed")
        raw = read(case / "source.json")["row"]
        player = raw["result"]["first"]
        with gzip.open(case / "trace.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                if (
                    row["chosen"]
                    and row["player"] == player
                    and row["chosen"]["kind"] in ("main2", "end_phase")
                    and any(a["kind"] == "attack" and row["mask"][i] for i, a in enumerate(row["options"]))
                ):
                    candidates.append(
                        {
                            "source": str(case),
                            "decision": row["index"],
                            "turn": row["turn"],
                            "player": player,
                            "opponent_lp": row["lp"][1 - player],
                            "selection": report["selection"],
                            "original_action": row["choice"],
                            "source_sha256": {
                                n: sha(case / n) for n in ("source.json", "trace.jsonl.gz", "replay.json.gz")
                            },
                        }
                    )
    candidates.sort(key=lambda c: (c["opponent_lp"], c["source"], c["decision"]))
    selected, used = [], set()
    for c in candidates:
        if c["source"] not in used:
            selected.append(c)
            used.add(c["source"])
        if len(selected) == count:
            break
    return selected


def audit_case(case, out, budget, env):
    source = Path(case["source"])
    for name, digest in case["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("frozen source changed")
    replay = Replay.load(source / "replay.json.gz")
    with gzip.open(source / "trace.jsonl.gz", "rt") as f:
        trace = [json.loads(line) for line in f]

    def make_root(snapshots=False):
        session = DuelSession(replay.duel(env, snapshots=snapshots))
        try:
            for row in trace[: case["decision"] + 1]:
                p = session.point
                if p is None or (p.index, p.player, p.turn, p.phase, list(p.lp), [asdict(a) for a in p.actions]) != (
                    row["index"],
                    row["player"],
                    row["turn"],
                    row["phase"],
                    row["lp"],
                    row["options"],
                ):
                    raise ValueError("original action prefix does not match")
                events = [{"type": type(e).__name__, **asdict(e)} for e in p.events]
                if json.loads(json.dumps(events)) != row["events"]:
                    raise ValueError("original event prefix does not match")
                if p.index == case["decision"]:
                    return session
                session.act(row["choice"])
        except BaseException:
            session.close()
            raise
        session.close()
        raise ValueError("prefix did not reach target")

    out.mkdir()
    session = make_root(snapshots=True)
    reports = []
    try:
        original = position(session)
        atomic_json(out / "root.json", {"case": case, "position": original, "policy_row": trace[case["decision"]]})
        for action in range(len(session.point.actions)):
            certificate = search_action(session, action, case["player"], budget)
            if position(session) != original:
                raise ValueError("search failed to restore original root")
            atomic_json(out / f"action-{action}.json", certificate)
            verification = {"cold_verified": False, "reason": "unknown search"}
            if certificate["status"] != "unknown":
                # Reload the serialized certificate; verify with fresh no-snapshot duels.
                verification = verify_certificate(read(out / f"action-{action}.json"), make_root)
            item = {k: certificate[k] for k in ("action", "status", "nodes", "seconds")}
            item.update(verification=verification, certificate_sha256=sha(out / f"action-{action}.json"))
            reports.append(item)
            atomic_json(out / "progress.json", reports)
            print(json.dumps({"source": case["source"], "decision": case["decision"], **item}), flush=True)
    finally:
        session.close()
    original_status = next(r["status"] for r in reports if r["action"] == case["original_action"])
    alternatives = [r["action"] for r in reports if r["action"] != case["original_action"] and r["status"] == "win"]
    summary = {
        "case": case,
        "actions": reports,
        "winning_alternatives": alternatives,
        "original_status": original_status,
        "missed_actual_state_forced_win": bool(alternatives) and original_status == "no_forced_win",
    }
    atomic_json(out / "report.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    if args.cases < 1:
        ap.error("cases must be positive")
    budget = SearchBudget(args.nodes, args.depth, args.seconds)
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        env = load_environment("md-2026-09")
        identity = read(source / "identity.json")
        if identity["native_sha256"] != sha(_core.__file__) or identity["environment"] != env.stamp():
            raise ValueError("native/environment differs from replay source")
        selected = select(source, args.cases)
        from ygorl.eval import win_search

        bindings = {str(p.resolve()): sha(p) for p in (Path(__file__), Path(win_search.__file__))}
        atomic_json(
            out / "protocol.json",
            {
                "scope": "actual-hidden-state-fixed-rng; development selected examples; no teacher labels",
                "selection": "cold128 four opponents; lowest opponent LP first; one stop decision per game; ties source/index",
                "requested_cases": args.cases,
                "budget_per_root_action": asdict(budget),
                "cases": selected,
                "environment": env.stamp(),
                "native_sha256": sha(_core.__file__),
                "bindings": bindings,
                "source_identity_sha256": sha(source / "identity.json"),
            },
        )
        reports = []
        for n, case in enumerate(selected):
            if any(sha(p) != h for p, h in bindings.items()):
                raise ValueError("registered audit source changed")
            reports.append(audit_case(case, out / str(n), budget, env))
            atomic_json(out / "report.json", reports)
        atomic_json(out / "complete.json", {"cases": len(reports), "protocol_sha256": sha(out / "protocol.json")})
    except BaseException as e:
        atomic_json(out / "STOP.json", {"error": repr(e)})
        raise


if __name__ == "__main__":
    main()
