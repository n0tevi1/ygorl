"""Independent behavior observer; never mutates or controls the registered training study."""

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from behavior_replay import replay_game, sha
from colab_checkpoint import run_lock
from ygorl import _core
from ygorl.data import load_environment
from ygorl.eval.behavior import length_stats
from ygorl.train.registration import atomic_json

HEALTH = ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")


def read_rows(path):
    rows = []
    incomplete = False
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:
            for line in f:
                if line.endswith("\n"):
                    rows.append(json.loads(line))
                else:
                    incomplete = True
    except EOFError:
        incomplete = True  # a still-being-written gzip member is not a corrupt completed record
    return rows, incomplete


def snapshot(study):
    training = {}
    cells = {}
    alerts = []
    jobs = []
    for p in sorted(study.glob("seed-*/*/run/games.jsonl.gz")):
        rows, partial = read_rows(p)
        key = str(p.parent.parent.relative_to(study))
        metrics, _ = read_rows(p.with_name("metrics.jsonl"))
        training[key] = {
            "all": length_stats([r["turns"] for r in rows]),
            "selfplay": length_stats([r["turns"] for r in rows if r["opponent"] is None]),
            "pool": length_stats([r["turns"] for r in rows if r["opponent"] is not None]),
            "last_1000": length_stats([r["turns"] for r in rows[-1000:]]),
            "partial_gzip_tail": partial,
            "health_flags": sum(any(r.get(k) for k in HEALTH) or r["truncated"] for r in rows),
            "last_metrics": metrics[-1] if metrics else None,
        }
        if training[key]["health_flags"]:
            alerts.append(key + ": unhealthy or truncated training games")
        if any(isinstance(v, float) and not math.isfinite(v) for r in metrics for v in r.values()):
            alerts.append(key + ": nonfinite training metric")
    for p in sorted(study.glob("evaluation/*/*.json")):
        cell = json.loads(p.read_text())
        raw = p.with_suffix(".jsonl")
        assert sha(raw) == cell["raw_sha256"], "completed evaluation changed"
        rows, partial = read_rows(raw)
        assert not partial and len(rows) == 256
        key = p.parent.name + "-" + p.stem
        cells[key] = {
            "turns": length_stats([r["result"]["turns"] for r in rows]),
            "decisions": length_stats([r["result"]["decisions"] for r in rows]),
            "deck_out": sum(r["result"]["win_reason"] == 2 for r in rows),
            "candidate_counts": dict(sum((Counter(r["candidate_counts"]) for r in rows), Counter())),
        }
        if any(any(r["result"].get(k) for k in HEALTH) or r["result"]["reason"] != "win" for r in rows):
            alerts.append(key + ": unhealthy evaluation")
        chosen = {r["game_id"]: "fixed-index" for r in rows if r["game_id"] < 16}
        for field in ("turns", "decisions"):
            worst = max(rows, key=lambda r: (r["result"][field], -r["game_id"]))
            chosen.setdefault(worst["game_id"], "selected-" + field)
        # Investigate tails first, while retaining fixed-index denominators separately.
        for r in sorted(rows, key=lambda r: (chosen.get(r["game_id"]) == "fixed-index", r["game_id"])):
            if r["game_id"] in chosen:
                jobs.append((key, r, chosen[r["game_id"]]))
    retention = study / "retention/progress.json"
    return {
        "checked_unix": time.time(),
        "training": training,
        "evaluation": cells,
        "alerts": alerts,
        "retention": json.loads(retention.read_text()) if retention.exists() else None,
    }, jobs


def publish(root, status):
    text = [
        "持续行为监测（独立于正式棋力主终点）",
        "",
        f"核验时间 Unix {status['checked_unix']:.0f}；完整评估cell {len(status['evaluation'])}；"
        f"轨迹冷回放通过 {status['replays']['completed']}，运行中 {status['replays']['running']}。",
    ]
    for name, r in status["training"].items():
        s = r["all"]
        if s["n"]:
            text.append(
                f"{name}: {s['n']}个已结束训练对局，平均{s['mean']:.2f}回合，P50={s['median']}，P90={s['p90']}，P99={s['p99']}，最长{s['max']}；≤4回合{s['le4']}，>20回合{s['gt20']}。"
            )
    text += [
        "",
        "回合为引擎单方turn；未结束对局右删失。循环/自灰/自伤/提前停手均为调查信号，不自动标错。",
        "固定索引样本与最长/最多决策选例分开保存；原结果匹配及冷回放核验后才纳入轨迹报告。",
        f"行为信号（含选例，不能估算总体频率）：{status['replays']['signals']}。",
        f"异常：{status['alerts']}。",
        "协议：docs/spikes/behavior-monitor-2026-10-08.md；证据：out/research/behavior-monitor-2026-10-08/。",
    ]
    start = "<!-- ygorl-behavior-monitor-20261008:start -->"
    end = "<!-- ygorl-behavior-monitor-20261008:end -->"
    body = json.loads(
        subprocess.check_output(
            ["gh", "issue", "view", "218", "--repo", "n0tevi1/ygorl", "--json", "body"], text=True, timeout=30
        )
    )["body"]
    block = start + "\n" + "\n".join(text) + "\n" + end
    if start in body:
        before, rest = body.split(start, 1)
        _, after = rest.split(end, 1)
        body = before + block + after
    else:
        body += "\n\n" + block
    path = root / "issue-body.md"
    path.write_text(body)
    subprocess.run(
        ["gh", "issue", "edit", "218", "--repo", "n0tevi1/ygorl", "--body-file", str(path)],
        check=True,
        timeout=30,
        capture_output=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    study = a.study.resolve()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    bindings = {
        str(repo / x): sha(repo / x)
        for x in (
            "tools/behavior_watch.py",
            "tools/behavior_replay.py",
            "src/ygorl/eval/behavior.py",
            "src/ygorl/eval/action_outcomes.py",
        )
    }
    protocol = root / "protocol.md"
    if not protocol.exists():
        shutil.copyfile(repo / "docs/spikes/behavior-monitor-2026-10-08.md", protocol)
    identity = {
        "study": str(study),
        "study_sha256": sha(study / "identity.json"),
        "bindings": bindings,
        "environment": load_environment("md-2026-09").stamp(),
        "native_sha256": sha(_core.__file__),
        "protocol_sha256": sha(protocol),
    }
    with run_lock(root):
        ip = root / "identity.json"
        if ip.exists():
            assert json.loads(ip.read_text()) == identity, "observer identity changed"
        else:
            atomic_json(ip, identity)
        pending = {}
        reports = {}
        last_publish = 0
        deadline = time.monotonic() + 47 * 3600
        with cf.ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
            while time.monotonic() < deadline:
                assert sha(study / "identity.json") == identity["study_sha256"]
                assert all(sha(path) == digest for path, digest in bindings.items())
                status, jobs = snapshot(study)
                for f, key in list(pending.items()):
                    if f.done():
                        reports[key] = f.result()
                        del pending[f]
                keys = set(pending.values())
                for cell, row, selection in jobs:
                    key = (cell, row["game_id"])
                    if key in reports or key in keys:
                        continue
                    path = root / "replays" / cell / str(row["game_id"])
                    report = path / "report.json"
                    if report.exists():
                        rec = json.loads(report.read_text())
                        assert all(sha(path / k) == v for k, v in rec["files"].items())
                        reports[key] = rec
                        continue
                    if (path / "STOP.json").exists():
                        raise RuntimeError("prior replay failure: " + str(path))
                    if len(pending) < 2:
                        pending[pool.submit(replay_game, (root, cell, row, selection))] = key
                        keys.add(key)
                signals = Counter(f["kind"] for r in reports.values() for f in r["findings"])
                status["replays"] = {
                    "completed": len(reports),
                    "running": len(pending),
                    "queued": len(jobs) - len(reports) - len(pending),
                    "signals": dict(signals),
                    "fixed_index_games": sum(r["selection"] == "fixed-index" for r in reports.values()),
                }
                atomic_json(root / "latest.json", status)
                # All records remain separate; selected tails never enter fixed-index frequency denominators.
                atomic_json(
                    root / "replay-index.json",
                    [
                        {
                            "cell": k[0],
                            "game_id": k[1],
                            "selection": r["selection"],
                            "counts": r["counts"],
                            "findings": r["findings"],
                        }
                        for k, r in sorted(reports.items())
                    ],
                )
                print(
                    json.dumps({"time": time.time(), "replays": status["replays"], "alerts": status["alerts"]}),
                    flush=True,
                )
                if a.publish and time.monotonic() - last_publish > 600:
                    try:
                        publish(root, status)
                        last_publish = time.monotonic()
                    except subprocess.SubprocessError as exc:
                        atomic_json(root / "publication-error.json", {"error": repr(exc)})
                state = json.loads((study / "pipeline-status.json").read_text())["stage"]
                if a.once or (state in ("complete", "stopped") and not pending and len(reports) == len(jobs)):
                    break
                time.sleep(5 if pending else 60)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        if "--root" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--root") + 1])
            atomic_json(target / "STOP.json", {"error": traceback.format_exc(), "time": time.time()})
        traceback.print_exc()
        raise
