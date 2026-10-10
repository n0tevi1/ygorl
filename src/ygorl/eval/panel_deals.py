"""Explicit, paired opening deals for serialized full-game evaluation jobs.

Changing a core RNG seed does not reshuffle an already prepared deck. Generate
deck orders here, by deck role rather than engine seat, and verify actual loaded
orders before treating draw labels as independent opening samples.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping

from ygorl.eval.arena import derive_seed
from ygorl.engine.duel import shuffle_deck


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deck_orders(job: Mapping) -> list[list[int]]:
    return [list(job[key]["main"]) for key in ("deck_a", "deck_b")]


def deal_key(job: Mapping) -> list:
    return [job[key] for key in ("target", "anchor", "role", "draw")]


def redeal_job(job: Mapping, panel_seed: int) -> dict:
    """Return a fresh deal without mutating a template or changing its deck lists.

    Checkpoints, opponents and seats share a deal for a target/anchor/role/draw.
    Sorting the multiset before shuffling makes previous template order irrelevant.
    Extra-deck order is preserved; the engine never draws an opening hand from it.
    """
    out = copy.deepcopy(dict(job))
    key = deal_key(out)
    seed = int(fingerprint(["panel-deal-v1", panel_seed, key])[:15], 16)
    for slot, name in enumerate(("deck_a", "deck_b")):
        out[name]["main"] = shuffle_deck(sorted(out[name]["main"]), seed, slot)
    out["config"]["shuffle_decks"] = False
    out["seed"] = derive_seed(seed, 200)
    out["agent_seeds"] = [derive_seed(seed, 100), derive_seed(seed, 101)]
    out["opening_deal"] = dict(version=1, panel_seed=panel_seed, key=key, sha256=fingerprint(deck_orders(out)))
    return out


def validate_panel(jobs: Iterable[Mapping], *, prior_jobs: Iterable[Mapping] = ()) -> dict:
    """Fail on stale metadata, reused prior orders, or mislabeled independent deals.

    Reject duplicate orders across distinct registered draws rather than silently
    counting them twice. Actual opening hands can legitimately coincide even when
    full orders differ; report their number, without rejecting or resampling them.
    """
    prior = list(prior_jobs)
    if any(j["config"].get("shuffle_decks", True) for j in prior):
        raise ValueError("prior jobs must record explicit loaded deck orders")
    excluded = {fingerprint(deck_orders(j)) for j in prior}
    by_key, by_order, hands = {}, {}, set()
    count = 0
    for job in jobs:
        count += 1
        meta = job.get("opening_deal")
        if not meta or meta.get("version") != 1:
            raise ValueError("missing explicit opening deal; new RNG seeds alone do not imply new deals")
        if job["config"].get("shuffle_decks", True):
            raise ValueError("prepared deals must not be shuffled again by engine seat")
        expected = redeal_job(job, meta["panel_seed"])
        if any(job[k] != expected[k] for k in ("opening_deal", "seed", "agent_seeds", "deck_a", "deck_b")):
            raise ValueError("opening deal or RNG binding mismatch")
        key = fingerprint([meta["panel_seed"], deal_key(job)])
        order = fingerprint(deck_orders(job))
        if order in excluded:
            raise ValueError("opening deal overlaps prior panel")
        if key in by_key and by_key[key] != order:
            raise ValueError("paired candidates/opponents/seats have different opening deals")
        if order in by_order and by_order[order] != key:
            raise ValueError("different draw labels reuse one opening deal")
        by_key[key], by_order[order] = order, key
        n = job["config"]["player"]["starting_hand"]
        hands.add(fingerprint([sorted(d[-n:]) if n else [] for d in deck_orders(job)]))
    if not count:
        raise ValueError("empty opening panel")
    return dict(
        games=count,
        unique_deals=len(by_order),
        unique_opening_hand_pairs=len(hands),
        prior_unique_deals=len(excluded),
        prior_overlap=0,
    )


def validate_loaded_deal(job: Mapping, duel) -> None:
    """Compare the actual engine input, converting engine seats back to deck roles."""
    loaded = duel.loaded_decks()
    for seat in (0, 1):
        deck = job[("deck_a", "deck_b")[duel.deck_of(seat)]]
        if list(loaded[seat][0]) != list(deck["main"]) or list(loaded[seat][1]) != list(deck["extra"]):
            raise ValueError("actual engine-loaded deck differs from registered opening deal")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-seed", type=int, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    args = parser.parse_args()
    jobs = [redeal_job(j, args.panel_seed) for j in json.loads(args.input.read_text())]
    # A new panel must at least be disjoint from the source it was copied from.
    priors = list(dict.fromkeys([args.input, *args.prior]))
    audit = validate_panel(jobs, prior_jobs=[j for p in priors for j in json.loads(p.read_text())])
    with args.output.open("x") as f:
        f.write(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(audit, sort_keys=True))
