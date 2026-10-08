"""Fixed opened-state 2x2 search coverage comparison, never actor training."""

import argparse
import gzip
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from audit_win_opportunities import read, sha
from ygorl import _core
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession
from ygorl.engine.replay import Replay
from ygorl.eval import win_search
from ygorl.eval.win_search import SearchBudget, position, search_action, verify_certificate
from ygorl.train.registration import atomic_json

ARMS = [(depth, allocation) for depth in ("host", "branch") for allocation in ("dfs", "fair")]


def jobs_from_parent(parent):
    jobs = []
    for folder, stratum in ((parent, "stop"), (parent / "entry-controls", "entry")):
        for case_dir in sorted(folder.glob("[0-9]*")):
            case = read(case_dir / "root.json")["case"]
            for path in sorted(case_dir.glob("action-*.json")):
                cert = read(path)
                jobs.append(
                    {
                        "case": case,
                        "action": cert["action"],
                        "stratum": stratum,
                        "previous_status": cert["status"],
                        "previous_certificate": str(path),
                        "previous_sha256": sha(path),
                    }
                )
    folder = parent / "terminal-controls"
    for n, case in enumerate(read(folder / "protocol.json")["cases"]):
        path = folder / f"control-{n}.json"
        cert = read(path)
        jobs.append(
            {
                "case": case,
                "action": cert["action"],
                "stratum": "terminal",
                "previous_status": cert["status"],
                "previous_certificate": str(path),
                "previous_sha256": sha(path),
            }
        )
    if Counter(j["stratum"] for j in jobs) != {"stop": 27, "entry": 23, "terminal": 8}:
        raise ValueError("expected frozen 58-root development panel")
    return jobs


def root_factory(case, env):
    source = Path(case["source"])
    if any(sha(source / name) != digest for name, digest in case["source_sha256"].items()):
        raise ValueError("registered source changed")
    replay = Replay.load(source / "replay.json.gz")
    with gzip.open(source / "trace.jsonl.gz", "rt") as f:
        prefix = [json.loads(line) for line in f][: case["decision"] + 1]

    def make_root(snapshots=False):
        session = DuelSession(replay.duel(env, snapshots=snapshots))
        try:
            for row in prefix:
                p = session.point
                if p is None or (p.index, p.player, p.turn, p.phase, list(p.lp), [asdict(a) for a in p.actions]) != (
                    row["index"],
                    row["player"],
                    row["turn"],
                    row["phase"],
                    row["lp"],
                    row["options"],
                ):
                    raise ValueError("original prefix position mismatch")
                events = [{"type": type(e).__name__, **asdict(e)} for e in p.events]
                if json.loads(json.dumps(events)) != row["events"]:
                    raise ValueError("original prefix events mismatch")
                if p.index == case["decision"]:
                    return session
                session.act(row["choice"])
            raise ValueError("original prefix did not reach root")
        except BaseException:
            session.close()
            raise

    return make_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        env = load_environment("md-2026-09")
        old = read(parent / "protocol.json")
        if old["native_sha256"] != sha(_core.__file__) or old["environment"] != env.stamp():
            raise ValueError("parent native/environment mismatch")
        jobs = jobs_from_parent(parent)
        budget = SearchBudget(nodes=256, depth=48, seconds=20, transitions=1024, path_steps=128)
        bindings = {
            str(p.resolve()): sha(p)
            for p in (
                Path(__file__),
                Path(win_search.__file__),
                Path("tools/audit_win_opportunities.py"),
                Path("tools/behavior_replay.py"),
            )
        }
        atomic_json(
            out / "protocol.json",
            {
                "scope": "opened actual-hidden-state/fixed-RNG development comparison; no training labels",
                "parent_protocol_sha256": sha(parent / "protocol.json"),
                "budget": asdict(budget),
                "arms": ARMS,
                "jobs": jobs,
                "bindings": bindings,
                "environment": env.stamp(),
                "native_sha256": sha(_core.__file__),
                "trials": 232,
                "order": "rotate the four arms by root index, no outcome-dependent expansion",
            },
        )
        reports = []
        for n, job in enumerate(jobs):
            if (
                any(sha(p) != h for p, h in bindings.items())
                or sha(job["previous_certificate"]) != job["previous_sha256"]
            ):
                raise ValueError("registered code/parent changed")
            make_root = root_factory(job["case"], env)
            outcomes = set()
            offset = n % len(ARMS)
            for depth_mode, allocation in ARMS[offset:] + ARMS[:offset]:
                start = time.monotonic()
                session = make_root(snapshots=True)
                try:
                    before = position(session)
                    cert = search_action(
                        session,
                        job["action"],
                        job["case"]["player"],
                        budget,
                        depth_mode=depth_mode,
                        allocation=allocation,
                    )
                    if position(session) != before:
                        raise ValueError("root not restored")
                finally:
                    session.close()
                arm = f"{depth_mode}-{allocation}"
                path = out / f"{n:03d}-{arm}.json.gz"
                with gzip.open(path, "wt") as f:
                    json.dump(cert, f)
                proof = {"cold_verified": False, "reason": "unknown"}
                if cert["status"] != "unknown":
                    with gzip.open(path, "rt") as f:
                        proof = verify_certificate(json.load(f), make_root)
                    outcomes.add(cert["status"])
                if len(outcomes) > 1:
                    raise ValueError("contradictory decisive statuses between arms")
                if cert["status"] != "unknown" and job["previous_status"] not in (cert["status"], "unknown"):
                    raise ValueError("decisive status contradicts previous audit")
                report = {
                    "root": n,
                    "arm": arm,
                    "stratum": job["stratum"],
                    **{k: cert[k] for k in ("status", "nodes", "transitions", "stats", "leaf_reasons", "seconds")},
                    "verification": proof,
                    "certificate_sha256": sha(path),
                    "total_seconds": time.monotonic() - start,
                }
                reports.append(report)
                atomic_json(out / "report.json", reports)
                print(json.dumps(report), flush=True)
        atomic_json(out / "complete.json", {"trials": len(reports), "protocol_sha256": sha(out / "protocol.json")})
    except BaseException as e:
        atomic_json(out / "STOP.json", {"error": repr(e)})
        raise


if __name__ == "__main__":
    main()
