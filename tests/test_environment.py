import json
from pathlib import Path

import pytest

from ygorl.data import EnvironmentConfigError, EnvironmentFileMissing, load_environment
from ygorl.engine import constants as C

MAIN = [1001] * 3 + [1002] * 3 + list(range(2000, 2034))  # 40 cards
EXTRA = [3001, 3002]


def make_env(root: Path, version: str = "test-2026-09", **overrides) -> Path:
    d = root / version
    (d / "meta").mkdir(parents=True)
    manifest = {
        "version": version,
        "format": "md",
        "rules": {"mode": "MR5"},
        "player": {"starting_lp": 8000, "starting_hand": 5, "draw_per_turn": 1},
        "deck": {"main_min": 40, "main_max": 60, "extra_max": 15, "side_max": 15, "max_copies": 3},
    }
    manifest.update(overrides.pop("manifest", {}))
    (d / "environment.json").write_text(json.dumps(manifest))
    pool = sorted(set(MAIN + EXTRA))
    (d / "pool.json").write_text(json.dumps({"cards": pool[:-1] + [{"password": pool[-1], "name": "X"}]}))
    (d / "banlist.lflist.conf").write_text("!test\n1001 1\n1002 0\n")
    (d / "meta" / "alpha.ydk").write_text("#main\n" + "\n".join(map(str, MAIN)) + "\n#extra\n3001\n3002\n!side\n")
    meta = {"decks": [{"name": "Alpha", "file": "meta/alpha.ydk", "share": 0.4}]}
    meta.update(overrides.pop("meta", {}))
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_load_ok(tmp_path):
    d = make_env(tmp_path)
    env = load_environment(d)
    assert env.version == "test-2026-09"
    assert env.format == "md"
    assert env.rule_flags == C.DUEL_MODE_MR5
    assert 1001 in env.card_pool and env.in_pool(3002)
    assert env.banlist.limit(1001) == 1 and env.banlist.limit(1002) == 0
    assert env.player.starting_lp == 8000
    assert env.deck_rules.main_min == 40
    (alpha,) = env.meta_decks
    assert alpha.name == "Alpha" and alpha.share == 0.4 and len(alpha.deck.main) == 40
    assert env.meta_deck("Alpha") is alpha
    assert len(env.fingerprint) == 64


def test_load_by_version_name(tmp_path, monkeypatch):
    make_env(tmp_path)
    assert load_environment("test-2026-09", root=tmp_path).version == "test-2026-09"
    monkeypatch.setenv("YGORL_ENVIRONMENTS", str(tmp_path))
    assert load_environment("test-2026-09").version == "test-2026-09"


def test_artifacts_and_stamp(tmp_path):
    env = load_environment(make_env(tmp_path))
    p = env.artifact_path("matrix", "result.json")
    assert p.parent.is_dir() and p.parent.parent == env.root / "artifacts"
    env.check_stamp(env.stamp())
    with pytest.raises(EnvironmentConfigError, match="belongs to environment"):
        env.check_stamp({"environment": "other", "fingerprint": env.fingerprint})
    with pytest.raises(EnvironmentConfigError, match="different revision"):
        env.check_stamp({"environment": env.version, "fingerprint": "0" * 64})


def test_fingerprint_changes_with_content(tmp_path):
    d = make_env(tmp_path)
    before = load_environment(d).fingerprint
    (d / "banlist.lflist.conf").write_text("!test\n1001 2\n")
    assert load_environment(d).fingerprint != before


@pytest.mark.parametrize("name", ["environment.json", "pool.json", "banlist.lflist.conf", "meta.json"])
def test_missing_required_file(tmp_path, name):
    d = make_env(tmp_path)
    (d / name).unlink()
    with pytest.raises(EnvironmentFileMissing, match=name):
        load_environment(d)


def test_missing_directory(tmp_path):
    with pytest.raises(EnvironmentFileMissing, match="not found"):
        load_environment(tmp_path / "nope")


def test_missing_meta_deck_file(tmp_path):
    d = make_env(tmp_path)
    (d / "meta" / "alpha.ydk").unlink()
    with pytest.raises(EnvironmentFileMissing, match="alpha.ydk"):
        load_environment(d)


@pytest.mark.parametrize(
    "overrides, msg",
    [
        ({"manifest": {"version": "other-1"}}, "does not match directory"),
        ({"manifest": {"rules": {"mode": "MR9"}}}, "unknown rules.mode"),
        ({"manifest": {"rules": {"mode": "MR5", "extra_flags": ["NOPE"]}}}, "unknown rules.extra_flags"),
        ({"manifest": {"player": {"starting_lp": -1}}}, "non-negative"),
        ({"manifest": {"deck": {"bogus": 1}}}, "unknown keys"),
        ({"meta": {"decks": [{"name": "A", "file": "meta/alpha.ydk", "share": 1.5}]}}, r"outside \[0, 1\]"),
        (
            {"meta": {"decks": [{"name": "A", "file": "meta/alpha.ydk", "share": 0.6}, {"name": "B", "file": "meta/alpha.ydk", "share": 0.6}]}},
            "sum to",
        ),
        (
            {"meta": {"decks": [{"name": "A", "file": "meta/alpha.ydk", "share": 0.1}, {"name": "A", "file": "meta/alpha.ydk", "share": 0.1}]}},
            "duplicate deck name",
        ),
        ({"meta": {"decks": [{"name": "A", "file": "../x.ydk", "share": 0.1}]}}, "escapes"),
        ({"meta": {"decks": [{"name": "A", "share": 0.1}]}}, "missing key 'file'"),
    ],
)
def test_invalid_config(tmp_path, overrides, msg):
    d = make_env(tmp_path, **overrides)
    with pytest.raises(EnvironmentConfigError, match=msg):
        load_environment(d)


def test_extra_flags(tmp_path):
    d = make_env(tmp_path, manifest={"rules": {"mode": "MR5", "extra_flags": ["TCG_SEGOC_NONPUBLIC"]}})
    assert load_environment(d).rule_flags == C.DUEL_MODE_MR5 | C.DUEL_TCG_SEGOC_NONPUBLIC


def test_invalid_pool_and_version(tmp_path):
    d = make_env(tmp_path)
    (d / "pool.json").write_text(json.dumps({"cards": [1, 1]}))
    with pytest.raises(EnvironmentConfigError, match="duplicate"):
        load_environment(d)
    (d / "pool.json").write_text("{not json")
    with pytest.raises(EnvironmentConfigError, match="invalid JSON"):
        load_environment(d)
    bad = make_env(tmp_path, version="Bad_Version")
    with pytest.raises(EnvironmentConfigError, match="must match"):
        load_environment(bad)


@pytest.mark.parametrize(
    "manifest, msg",
    [
        ({"player": [1, 2]}, "player must be an object"),
        ({"deck": "40"}, "deck must be an object"),
        ({"rules": {"mode": "MR5", "extra_flags": "TCG_SEGOC_NONPUBLIC"}}, "rules.extra_flags must be a list"),
        ({"rules": {"mode": "MR5", "extra_flags": [7]}}, "unknown rules.extra_flags entry 7"),
        ({"rules": {"mode": ["MR5"]}}, "unknown rules.mode"),
        ({"player": {"starting_lp": 0}}, "starting_lp must be at least 1"),
        ({"player": {"starting_lp": 2**31}}, "starting_lp must be at most"),
        ({"player": {"starting_hand": 70}}, "starting_hand 70 is larger than deck.main_min 40"),
        ({"player": {"starting_hand": 21}, "deck": {"main_min": 20}}, "starting_hand 21 is larger than deck.main_min 20"),
        ({"player": {"draw_per_turn": 41}}, "draw_per_turn 41 is larger than deck.main_min 40"),
        ({"player": {"starting_hand": 1.5}}, "non-negative integer"),
        ({"deck": {"main_min": 61}}, "main_min 61 is larger than deck.main_max 60"),
        ({"deck": {"max_copies": 0}}, "max_copies must be at least 1"),
        ({"description": 5}, "key 'description' has wrong type int"),
        ({"banlist_name": 5}, "key 'banlist_name' has wrong type int"),
    ],
)
def test_degenerate_manifest(tmp_path, manifest, msg):
    with pytest.raises(EnvironmentConfigError, match=msg):
        load_environment(make_env(tmp_path, manifest=manifest))


def test_sane_bounds_are_accepted(tmp_path):
    manifest = {"player": {"starting_lp": 1, "starting_hand": 0, "draw_per_turn": 0}, "deck": {"main_min": 40}}
    env = load_environment(make_env(tmp_path, manifest=manifest))
    assert (env.player.starting_lp, env.player.starting_hand, env.player.draw_per_turn) == (1, 0, 0)


@pytest.mark.parametrize(
    "name, content, msg",
    [
        ("banlist.lflist.conf", b"!ban\n1001 1\n# caf\xe9\n", "banlist.lflist.conf: not valid UTF-8"),
        ("environment.json", b'{"version": "caf\xe9"}', "environment.json: not valid UTF-8"),
        ("pool.json", b"[1001, 1002]", "pool.json: expected a JSON object"),
        ("pool.json", b'"cards"', "pool.json: expected a JSON object"),
        ("pool.json", b'{"cards": [1001, 100000000]}', r"cards\[1\] is not a valid card password"),
        ("meta.json", b'[{"name": "A"}]', "meta.json: expected a JSON object"),
        ("meta/alpha.ydk", b"#main\n1001\n# caf\xe9\n", "alpha.ydk: not valid UTF-8"),
        ("meta/alpha.ydk", b"#main\n99999999999\n", "invalid card password 99999999999"),
    ],
)
def test_malformed_files_are_reported(tmp_path, name, content, msg):
    d = make_env(tmp_path)
    (d / name).write_bytes(content)
    with pytest.raises(EnvironmentConfigError, match=msg):
        load_environment(d)


@pytest.mark.parametrize("parts", [("..", "x.json"), ("matrix", "../../escape.json"), ("/tmp/abs.json",), ()])
def test_artifact_path_refuses_to_escape(tmp_path, parts):
    env = load_environment(make_env(tmp_path))
    with pytest.raises(EnvironmentConfigError, match="escapes"):
        env.artifact_path(*parts)
    assert not (env.root / "x.json").exists() and not (tmp_path / "escape.json").exists()


def test_artifact_path_refuses_a_symlink_out(tmp_path):
    env = load_environment(make_env(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    (env.artifacts_dir / "matrix").symlink_to(outside)
    with pytest.raises(EnvironmentConfigError, match="escapes"):
        env.artifact_path("matrix", "x.json")


def test_matrix_save_refuses_a_name_with_a_path(tmp_path):
    from ygorl.eval.matchup import MatchupMatrix, MetaGame

    env = load_environment(make_env(tmp_path))
    half = ((0.5, 0.5), (0.5, 0.5))
    matrix = MatchupMatrix(decks=("a", "b"), win_rate=half, games=((0, 2), (2, 0)), ci_low=half, ci_high=half,
                           agent="random", seed=0, pairs=1, environment=env.stamp())  # fmt: skip
    meta = MetaGame(matrix=matrix, nash=(0.5, 0.5), alpha_rank=(0.5, 0.5))
    with pytest.raises(EnvironmentConfigError, match="escapes"):
        meta.save(env=env, name="../../evil")
    assert not (env.root / "evil.json").exists()
    assert meta.save(env=env, name="ok") == env.root / "artifacts" / "matrix" / "ok.json"


def fake_cards(env):
    """Card table in which every pool password is a main-deck monster (>= 3000: an Extra Deck Link monster)."""
    from types import SimpleNamespace

    def kind(p):
        return C.TYPE_MONSTER | (C.TYPE_LINK if p >= 3000 else 0)

    return {p: SimpleNamespace(name=f"C{p}", alias=0, type=kind(p)) for p in env.card_pool}


def test_meta_decks_are_validated_with_a_card_table(tmp_path):
    d = make_env(tmp_path)
    cards = fake_cards(load_environment(d))  # the fixture meta deck breaks the banlist (1001 limited, 1002 forbidden)
    with pytest.raises(EnvironmentConfigError) as exc:
        load_environment(d, cards=cards)
    text = str(exc.value)
    assert str(d / "meta" / "alpha.ydk") in text and "'Alpha'" in text
    assert "forbidden" in text and "banlist allows 1" in text
    (d / "banlist.lflist.conf").write_text("!test\n")
    assert load_environment(d, cards=cards).meta_decks[0].name == "Alpha"
