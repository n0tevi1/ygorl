"""Training-only nested group validation and continuation-noise diagnostics for response teachers.

No final-game panel is read and no model is exported. Optional repaired continuations must preserve
all original successful records and carry the original collection's provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ygorl.eval.response_probe import ResponseRidge, common_valid

MODELS = ("full", "interaction", "action")
REGULARIZATION = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training(directory, retries=None):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path): digest(manifest_path)}
    provenance = None
    retry_sources = {}
    used_retries = set()
    if retries is not None:
        path = retries / "provenance.json"
        provenance = json.loads(path.read_text())
        hashes[str(path)] = digest(path)
        if (
            provenance["parent_manifest_sha256"] != digest(manifest_path)
            or provenance["environment"] != manifest["environment"]
        ):
            raise ValueError("retry collection provenance mismatch")
        retry_sources = {(Path(p).parent.name, Path(p).name): sha for p, sha in provenance["sources"].items()}
        if len(retry_sources) != len(provenance["sources"]):
            raise ValueError("ambiguous retry sources")
    excluded, rows = set(), []
    total = errors = replaced = 0
    for opponent in manifest["opponents"]:
        path = directory / opponent / "states.json"
        states = json.loads(path.read_text())
        hashes[str(path)] = digest(path)
        excluded.update(int(i) // 4 for i, s in states["outcomes"].items() if s["status"] == "error")
        for key, root in states["roots"].items():
            path = directory / opponent / f"rollouts-{key}.json"
            hashes[str(path)] = digest(path)
            records = json.loads(path.read_text())["records"]
            if retries is not None and (retry := retries / opponent / path.name).exists():
                expected_hash = retry_sources.get((opponent, path.name))
                if expected_hash != digest(path):
                    raise ValueError(f"retry source mismatch: {path}")
                retry_states_path = retries / opponent / "states.json"
                hashes[str(retry_states_path)] = digest(retry_states_path)
                retry_states = json.loads(retry_states_path.read_text())
                if retry_states["roots"][key] != root:
                    raise ValueError("retry root changed")
                after = json.loads(retry.read_text())["records"]
                if len(after) != len(records):
                    raise ValueError("retry must include every action and seed of the root")
                for a, b in zip(records, after, strict=True):
                    if a["reason"] != "error" and any(
                        a[k] != b[k] for k in ("reason", "winner", "turns", "decisions", "lp")
                    ):
                        raise ValueError("retry changed an originally successful outcome")
                records = after
                hashes[str(retry)] = digest(retry)
                replaced += len(after)
                used_retries.add((opponent, path.name))
            total += len(records)
            errors += sum(r["reason"] == "error" for r in records)
            q = np.array(
                [
                    np.nan
                    if r["reason"] == "error"
                    else 0.5
                    if r["winner"] is None
                    else float(r["winner"] == root["player"])
                    for r in records
                ]
            ).reshape(len(root["actions"]), manifest["continuations"])
            valid = common_valid(q)
            if valid.sum() * 2 < manifest["continuations"]:
                excluded.add(root["pairing"])
                continue
            q = q[:, valid]
            x = np.asarray(root["features"], dtype=float)
            x = x - x[root["pass"]]
            p = np.asarray(root["probs"], dtype=float)
            if (
                x.ndim != 2
                or x.shape[1] % 2
                or not np.isfinite(x).all()
                or (p < 0).any()
                or not np.isfinite(p).all()
                or p.sum() <= 0
            ):
                raise ValueError("invalid public features or policy probabilities")
            rows.append(dict(pair=root["pairing"], x=x, passed=root["pass"], p=p / p.sum(), q=q, values=q.mean(axis=1)))
    if used_retries != set(retry_sources):
        raise ValueError("retry provenance lists missing or unused replacement files")
    pairs = np.array([i for i in range(len(manifest["pairings"])) if i not in excluded])
    if len(pairs) < 10:
        raise ValueError("need at least ten complete pairing clusters for nested 5x4 validation")
    return (
        [r for r in rows if r["pair"] not in excluded],
        pairs,
        manifest,
        dict(
            continuations=total,
            error_continuations=errors,
            replaced_continuations=replaced,
            excluded_pairings=sorted(excluded),
            source_hashes=hashes,
        ),
    )


def columns(x, model):
    half = x.shape[1] // 2
    return x if model == "full" else x[:, :half] if model == "interaction" else x[:, half:]


def xy(rows, model):
    x, y = [], []
    for row in rows:
        for j in range(len(row["x"])):
            if j != row["passed"]:
                x.append(columns(row["x"], model)[j])
                y.append(row["values"][j] - row["values"][row["passed"]])
    return np.asarray(x), np.asarray(y)


def select(training, pairs):
    scores = []
    for model in MODELS:
        for regularization in REGULARIZATION:
            error, count = 0.0, 0
            for held in np.array_split(pairs, 4):
                train = [r for r in training if r["pair"] not in held]
                test = [r for r in training if r["pair"] in held]
                x, y = xy(train, model)
                xt, yt = xy(test, model)
                teacher = ResponseRidge.fit(x, y, regularization=regularization)
                error += np.square(teacher.scores(xt) - yt).sum()
                count += len(yt)
            scores.append(dict(model=model, regularization=regularization, mse=float(error / count)))
    return min(scores, key=lambda s: s["mse"]), scores


def gain(rows, teacher, model):
    return float(
        sum(r["values"][np.argmax(teacher.scores(columns(r["x"], model)))] - r["p"] @ r["values"] for r in rows)
    )


def analyze(rows, pairs, opponent_count, *, seed):
    pairs = pairs.copy()
    np.random.default_rng(seed).shuffle(pairs)
    folds = []
    for k, held in enumerate(np.array_split(pairs, 5)):
        train = [r for r in rows if r["pair"] not in held]
        test = [r for r in rows if r["pair"] in held]
        selected, scores = select(train, np.array([p for p in pairs if p not in held]))
        model = selected["model"]
        teacher = ResponseRidge.fit(*xy(train, model), regularization=selected["regularization"])
        fixed = ResponseRidge.fit(*xy(train, "full"), regularization=10)
        denominator = len(held) * 4 * opponent_count
        folds.append(
            dict(
                fold=k,
                pairings=held.tolist(),
                selected=selected,
                inner_scores=scores,
                selected_gain=gain(test, teacher, model) / denominator,
                fixed_gain=gain(test, fixed, "full") / denominator,
                base_games=denominator,
            )
        )
    halves, noise, deltas = [], [], []
    oracle = optimistic = 0.0
    for r in rows:
        q = r["q"]
        middle = q.shape[1] // 2
        a, b = q[:, :middle].mean(axis=1), q[:, middle:].mean(axis=1)
        passed = r["passed"]
        for j in range(len(q)):
            if j == passed:
                continue
            delta = q[j] - q[passed]
            deltas.append(delta.mean())
            noise.append(delta.var(ddof=1) / len(delta))
            halves.append([a[j] - a[passed], b[j] - b[passed]])
        oracle += (b[np.argmax(a)] - r["p"] @ b + a[np.argmax(b)] - r["p"] @ a) / 2
        optimistic += max(r["values"]) - r["p"] @ r["values"]
    x, y = xy(rows, "full")
    denominator = len(pairs) * 4 * opponent_count
    final, all_scores = select(rows, pairs)
    return dict(
        roots=len(rows),
        pairings=len(pairs),
        labels=len(y),
        features=x.shape[1],
        feature_rank=int(np.linalg.matrix_rank(x)),
        label_variance=float(np.var(deltas, ddof=1)),
        mean_label_noise_variance=float(np.mean(noise)),
        independent_half_delta_correlation=float(np.corrcoef(np.asarray(halves).T)[0, 1]),
        cross_half_oracle_gain=float(oracle / denominator),
        same_sample_oracle_gain=float(optimistic / denominator),
        nested_selected_gain=sum(f["selected_gain"] * f["base_games"] for f in folds) / denominator,
        fixed_full_lam10_gain=sum(f["fixed_gain"] * f["base_games"] for f in folds) / denominator,
        full_training_selection=final,
        full_training_inner_scores=all_scores,
        folds=folds,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--retries", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026100406)
    a = p.parse_args()
    rows, pairs, manifest, provenance = load_training(a.training, a.retries)
    report = analyze(rows, pairs, len(manifest["opponents"]), seed=a.seed)
    report.update(
        environment=manifest["environment"],
        provenance=provenance,
        tool_sha256=digest(Path(__file__)),
        protocol=dict(
            seed=a.seed,
            models=MODELS,
            regularization=REGULARIZATION,
            outer_folds=5,
            inner_folds=4,
            selection="inner held-out paired action-minus-pass MSE",
            denominator="all base games, including no-window zeros",
        ),
        scope="Exploratory training-only diagnostic; overlapping fits, no nominal CIs or full-game strength claims. No model exported.",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("provenance", "folds", "full_training_inner_scores")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
