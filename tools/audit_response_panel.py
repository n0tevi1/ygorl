"""Audit matched first-response windows and unchanged full-game outcomes in a four-arm panel."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    arms = manifest["arms"]
    assert set(arms) == {"control", "reranked", "pass_bias", "always_activate"}
    count = 4 * len(manifest["pairings"])
    report = {
        "base_games": 0,
        "eligible_base_games": 0,
        "matched_windows": 0,
        "same_action_outcome_comparisons": 0,
        "turns": Counter(),
        "chain_lengths": Counter(),
        "changed": Counter(),
        "errors": Counter(),
        "error_pairings": set(),
        "choice_disagreements": Counter(),
        "all_game_difference_bounds": {arm: [0.0, 0.0] for arm in arms if arm != "control"},
        "cell_sha256": {},
    }
    for opponent in manifest["opponents"]:
        cells = {}
        for arm in arms:
            path = directory / f"{arm}-{opponent}.json"
            report["cell_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            cells[arm] = json.loads(path.read_text())["records"]
            assert len(cells[arm]) == count
        for i in range(count):
            report["base_games"] += 1
            control = cells["control"][i]

            def score_bounds(record):
                if record["reason"] == "error":
                    return 0.0, 1.0
                value = 0.5 if record["winner"] is None else float(record["winner"] == 0)
                return value, value

            control_low, control_high = score_bounds(control)
            for arm, bounds in report["all_game_difference_bounds"].items():
                low, high = score_bounds(cells[arm][i])
                bounds[0] += low - control_high
                bounds[1] += high - control_low
            window = control["intervention"]
            if window is not None:
                report["eligible_base_games"] += 1
                report["turns"][window["turn"]] += 1
                report["chain_lengths"][window["chain_length"]] += 1
                assert not window["applied"]
                for other in ("pass_bias", "always_activate"):
                    report["choice_disagreements"][f"reranked-{other}"] += (
                        window["choices"]["reranked"] != window["choices"][other]
                    )
            by_action = {}
            for arm in arms:
                record = cells[arm][i]
                assert record["pair"] == i // 4
                assert (record["seed"], record["first"]) == (control["seed"], control["first"])
                actual = record["intervention"]
                assert (actual is None) == (window is None), (opponent, i, arm, "window presence")
                if actual is not None:
                    for key in ("signature", "turn", "chain_length", "sampled", "choices"):
                        assert actual[key] == window[key], (opponent, i, arm, key)
                    assert actual["applied"] == (arm != "control")
                    report["matched_windows"] += 1
                    chosen = actual["sampled"] if arm == "control" else actual["choices"][arm]
                    if arm != "control":
                        assert actual["changed"] == (chosen != actual["sampled"])
                        report["changed"][arm] += actual["changed"]
                else:
                    chosen = None
                # Same intervention (or none) must preserve every recorded full-game outcome field.
                # Raw trajectories are not stored by this evaluator, so this is not byte-level replay parity.
                outcome = {k: record[k] for k in ("winner", "reason", "error", "turns", "decisions", "lp")}
                if chosen in by_action:
                    assert outcome == by_action[chosen], (opponent, i, arm, "same-action outcome")
                    report["same_action_outcome_comparisons"] += 1
                by_action[chosen] = outcome
                if record["reason"] == "error":
                    report["errors"][arm] += 1
                    report["error_pairings"].add(record["pair"])
    report["error_pairings"] = sorted(report["error_pairings"])
    report["all_game_difference_bounds"] = {
        arm: [v / report["base_games"] for v in bounds] for arm, bounds in report["all_game_difference_bounds"].items()
    }
    report["bounds_scope"] = "finite-panel missing-outcome bounds, not sampling confidence intervals"
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.panel)
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
