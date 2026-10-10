"""Regression: changing engine RNG cannot create fresh pre-shuffled openings."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession
from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location
from ygorl.eval.panel_deals import redeal_job, validate_loaded_deal, validate_panel


def template():
    decks = Path(__file__).parent / "decks"
    cfg = DuelConfig(shuffle_decks=False)
    cfg = replace(cfg, player=replace(cfg.player, draw_per_turn=0))
    return dict(
        target="kashtira",
        anchor="snake_eye",
        role="face-target",
        draw=0,
        candidate="start",
        opponent="greedy",
        first=0,
        seed=42,
        agent_seeds=[43, 44],
        config=asdict(cfg),
        deck_a=asdict(load_ydk(decks / "snake_eye.ydk")),
        deck_b=asdict(load_ydk(decks / "kashtira.ydk")),
    )


def actual_hands(job):
    from ygorl.data.environment import PlayerRules

    cfg = DuelConfig(**{**job["config"], "player": PlayerRules(**job["config"]["player"])})
    duel = Duel(job["seed"], None, Deck(**job["deck_a"]), Deck(**job["deck_b"]), first=job["first"], config=cfg)
    validate_loaded_deal(job, duel)
    session = DuelSession(duel)
    try:
        hands = [None, None]
        for seat in (0, 1):
            cards = parse_query_location(session.core.query_location(CARD_QUERY_FLAGS, seat, C.LOCATION_HAND))
            hands[duel.deck_of(seat)] = sorted(c["code"] for c in cards if c)
        return hands
    finally:
        session.close()


def test_legacy_seed_only_change_reuses_actual_engine_hands():
    old = template()
    changed = {**old, "seed": 999, "agent_seeds": [1000, 1001], "draw": 1}
    assert actual_hands(old) == actual_hands(changed)
    with pytest.raises(ValueError, match="new RNG seeds alone"):
        validate_panel([old, changed])


def test_fresh_draws_change_hands_but_seats_and_candidates_remain_paired():
    base = template()
    panel, hands = [], []
    for draw in range(3):
        original = redeal_job({**base, "draw": draw}, 700)
        expected = [sorted(original[name]["main"][-5:]) for name in ("deck_a", "deck_b")]
        hands.append(expected)
        for first in (0, 1):
            for candidate in ("start", "trained"):
                for opponent in ("greedy", "historical"):
                    j = redeal_job(
                        {**base, "draw": draw, "first": first, "candidate": candidate, "opponent": opponent}, 700
                    )
                    assert actual_hands(j) == expected
                    panel.append(j)
    assert len({str(h) for h in hands}) == 3
    assert validate_panel(panel, prior_jobs=[base])["unique_deals"] == 3
    assert base == template()  # no template mutation


def test_redeal_preserves_multisets_and_ignores_old_order():
    base = template()
    reversed_decks = copy.deepcopy(base)
    for name in ("deck_a", "deck_b"):
        reversed_decks[name]["main"] = tuple(reversed(reversed_decks[name]["main"]))
    a, b = redeal_job(base, 701), redeal_job(reversed_decks, 701)
    assert a == b
    for name in ("deck_a", "deck_b"):
        assert sorted(a[name]["main"]) == sorted(base[name]["main"])
        assert a[name]["extra"] == base[name]["extra"]


def test_panel_rejects_prior_overlap_and_stale_bindings():
    j = redeal_job(template(), 702)
    with pytest.raises(ValueError, match="overlaps prior"):
        validate_panel([j], prior_jobs=[j])
    for change in ({"draw": 1}, {"seed": 5}, {"agent_seeds": [0, 1]}):
        with pytest.raises(ValueError, match="binding mismatch"):
            validate_panel([{**j, **change}])
    bad = copy.deepcopy(j)
    bad["config"]["shuffle_decks"] = True
    with pytest.raises(ValueError, match="shuffled again"):
        validate_panel([bad])


def test_panel_rejects_identical_deals_disguised_as_different_draws():
    base = template()
    for name in ("deck_a", "deck_b"):
        base[name]["main"] = [base[name]["main"][0]] * 40
    # Distinct seed/metadata, but the loaded orders really are identical.
    with pytest.raises(ValueError, match="reuse one opening"):
        validate_panel([redeal_job({**base, "draw": d}, 703) for d in (0, 1)])


def test_actual_loaded_order_validation_rejects_accidental_reshuffle():
    j = redeal_job(template(), 704)
    duel = Duel(j["seed"], None, Deck(**j["deck_a"]), Deck(**j["deck_b"]), config=DuelConfig())
    with pytest.raises(ValueError, match="actual engine-loaded"):
        validate_loaded_deal(j, duel)


def test_cli_serializes_fresh_jobs_and_refuses_to_overwrite(tmp_path):
    source, output = tmp_path / "old.json", tmp_path / "new.json"
    source.write_text(json.dumps([template()]))
    args = [
        sys.executable,
        "-m",
        "ygorl.eval.panel_deals",
        "--input",
        str(source),
        "--output",
        str(output),
        "--panel-seed",
        "705",
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["prior_overlap"] == 0
    jobs = json.loads(output.read_text())
    assert validate_panel(jobs, prior_jobs=json.loads(source.read_text()))["unique_deals"] == 1
    prior_bytes = output.read_bytes()
    assert subprocess.run(args, capture_output=True).returncode != 0
    assert output.read_bytes() == prior_bytes


def test_runner_fails_closed_on_legacy_panel_claiming_fresh_deals(tmp_path, monkeypatch):
    import card_generalization_games as games

    monkeypatch.setattr(games, "verify", lambda root: None)
    (tmp_path / "panel.json").write_text(json.dumps([{**template(), "id": 0}]))
    (tmp_path / "registration.json").write_text(json.dumps(dict(games=1, fresh_opening_deals=True, prior_panels=[])))
    with pytest.raises(ValueError, match="new RNG seeds alone"):
        games.run(tmp_path)
    assert "new RNG seeds alone" in json.loads((tmp_path / "STOP.json").read_text())["reason"]
    assert json.loads((tmp_path / "pipeline-status.json").read_text())["stage"] == "stopped"
    assert not (tmp_path / "games").exists()
