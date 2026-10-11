"""Compose cached critic corpora without mixing game weights or local IDs."""

from __future__ import annotations

import torch


def combine(parts):
    """Combine (corpus name, dataset, game predicate) parts with equal game weight.

    Dataset zero_records contains metadata for every zero-idle game. They remain
    in planned totals, but cannot contribute a prediction or training gradient.
    Source IDs are namespaced; target/role groups are loss labels, never inputs.
    """
    games, zeros, tensors, privates, ids = [], [], [], [], []
    seen = set()
    for name, data, keep in parts:
        zero_records = data.get("zero_records", [])
        assert {g["id"] for g in zero_records} == set(data["zero_games"])
        selected = [i for i, g in enumerate(data["games"]) if keep(g)]
        mapping = torch.full((len(data["games"]),), -1, dtype=torch.long)
        mapping[selected] = torch.arange(len(games), len(games) + len(selected))
        mask = mapping[data["ids"]] >= 0
        for g in [data["games"][i] for i in selected] + [g for g in zero_records if keep(g)]:
            key = f"{name}:{g['id']}"
            assert key not in seen, f"duplicate source game: {key}"
            seen.add(key)
        games.extend({**data["games"][i], "id": f"{name}:{data['games'][i]['id']}"} for i in selected)
        zeros.extend({**g, "id": f"{name}:{g['id']}"} for g in zero_records if keep(g))
        tensors.append({k: data[k][mask] for k in ["context", "candidate", "z", "resources"]})
        privates.append({k: v[mask] for k, v in data["privileged"].items()})
        ids.append(mapping[data["ids"][mask]])
    assert games, "no eligible games in selected corpora"
    out = {k: torch.cat([t[k] for t in tensors]) for k in tensors[0]}
    out["privileged"] = {k: torch.cat([p[k] for p in privates]) for k in privates[0]}
    out["ids"] = torch.cat(ids)
    counts = torch.bincount(out["ids"], minlength=len(games))
    assert bool((counts > 0).all())
    out["weight"] = (len(out["ids"]) / len(games)) / counts[out["ids"]].float()
    labels = sorted({(g["target"], g["role"]) for g in games})
    groups = torch.tensor([labels.index((g["target"], g["role"])) for g in games])
    out.update(
        games=games,
        zero_records=zeros,
        zero_games=[g["id"] for g in zeros],
        total_games=len(games) + len(zeros),
        groups=groups[out["ids"]],
        group_labels=labels,
    )
    return out
