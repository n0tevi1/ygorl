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
