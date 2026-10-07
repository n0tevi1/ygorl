"""Hash-checked outcome screening of reused natural-policy traces; not labels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from ygorl.eval.action_outcomes import action_intervals
from ygorl.train.registration import atomic_json


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    root = a.output.resolve()
    protocol = read(root / "protocol.json")
    panel = Path(protocol["panel"]).resolve()
    collection = read(panel / "collection.json")
    assert len(collection["games"]) == protocol["games"]
    assert not (root / "identity.json").exists(), "preserve prior run"
    atomic_json(
        root / "identity.json",
        {
            "driver_sha256": sha(__file__),
            "extractor_sha256": sha("src/ygorl/eval/action_outcomes.py"),
            "protocol_sha256": sha(root / "protocol.json"),
            "collection_sha256": sha(panel / "collection.json"),
            "panel_identity_sha256": sha(panel / "panel-identity.json"),
            "panel": str(panel),
        },
    )
    all_rows, selected, seen = [], [], set()
    counts, stops, codes = Counter(), Counter(), Counter()
    damage_outside = [0, 0]
    try:
        for game in collection["games"]:
            base = panel / "games" / f"{game['game']:04d}"
            for name in ("trace.jsonl.gz", "spec.json", "result.json", "replay.json.gz"):
                assert sha(base / name) == game["files"][name]
            spec, result = read(base / "spec.json"), read(base / "result.json")
            assert result["reason"] == "win" and not any(
                result[k] for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
            )
            learner = int(spec["first"])
            with gzip.open(base / "trace.jsonl.gz", "rt") as f:
                intervals, coverage = action_intervals(json.loads(line) for line in f)
            counts["games"] += 1
            counts["decisions"] += coverage["trace_rows"]
            for player in range(2):
                damage_outside[int(player != learner)] += coverage["damage_outside_tracked_intervals"][player]
            for interval in intervals:
                row = {"game": game["game"], "pair": game["pair"], "learner": learner, **interval}
                all_rows.append(row)
                if row["initiator"] != learner:
                    continue
                kind = row["kind"]
                counts[kind + "_initiated"] += 1
                stops[kind + ":" + row["stop"]] += 1
                if not row["complete"]:
                    continue
                counts[kind + "_complete"] += 1
                if row["lp_cost"][learner] > 0:
                    counts[kind + "_with_lp_cost"] += 1
                if row["damage"][learner] > 0:
                    counts[kind + "_with_self_damage"] += 1
                    code = row["action"]["card"]["code"] if row["action"]["card"] else 0
                    codes[f"{kind}:{code}"] += 1
                    key = (kind, code)
                    if key not in seen and len(selected) < 12:
                        selected.append(row)
                    seen.add(key)
        atomic_json(root / "intervals.json", all_rows)
        atomic_json(root / "selected.json", selected)
        report = {
            "identity_sha256": sha(root / "identity.json"),
            "counts": dict(counts),
            "stops": dict(stops),
            "self_damage_by_initiator_card": dict(codes),
            "damage_outside_tracked_intervals_learner_opponent": damage_outside,
            "selected": len(selected),
            "intervals_sha256": sha(root / "intervals.json"),
            "selected_sha256": sha(root / "selected.json"),
            "interpretation": protocol["interpretation"],
        }
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
