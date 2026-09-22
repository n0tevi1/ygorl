"""Replay format, environment binding and .yrpX export (T1.8)."""

import gzip
import json
import struct
from pathlib import Path

import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, default_scripts
from ygorl.engine.replay import Replay, ReplayEnvironmentMismatch

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
A, B = DECKS["snake_eye"], DECKS["labrynth"]


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def make_env(root: Path, version: str) -> Path:
    d = root / version
    (d / "meta").mkdir(parents=True)
    (d / "environment.json").write_text(json.dumps({"version": version, "format": "md", "rules": {"mode": "MR5"}}))
    pool = sorted(set(A.main + A.extra + B.main + B.extra))
    (d / "pool.json").write_text(json.dumps({"cards": pool}))
    (d / "banlist.lflist.conf").write_text("!none\n")
    (d / "meta.json").write_text(json.dumps({"decks": []}))
    return d


@pytest.fixture
def env(tmp_path):
    return load_environment(make_env(tmp_path, "test-2026-09"))


def record(db, env, seed=17, **kwargs):
    duel = Duel(seed, env, A, B, cards=db, first=1, record_messages=True, **kwargs)
    result = duel.run(RandomAgent(seed), RandomAgent(seed + 1))
    return duel, result


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_roundtrip_and_replay_to_same_end(db, env, tmp_path, suffix):
    duel, result = record(db, env)
    rep = Replay.from_duel(duel, result)
    path = tmp_path / f"game{suffix}"
    rep.save(path)
    loaded = Replay.load(path)
    assert loaded == rep
    assert loaded.environment == {"version": env.version, "fingerprint": env.fingerprint}
    again = loaded.play(env=env, cards=db, record_messages=True)
    assert again.message_log == result.message_log
    assert (again.winner, again.reason, again.turns, again.lp) == (result.winner, result.reason, result.turns, result.lp)
    assert loaded.result["winner"] == result.winner


def test_replay_without_environment(db):
    duel = Duel(5, None, A, B, cards=db)
    result = duel.run(RandomAgent(1), RandomAgent(2))
    rep = Replay.from_duel(duel, result)
    assert rep.environment is None
    assert rep.play(cards=db).responses == result.responses


def test_cross_environment_load_is_a_clear_error(db, env, tmp_path):
    duel, result = record(db, env)
    rep = Replay.from_duel(duel, result)
    other = load_environment(make_env(tmp_path, "other-2026-10"))
    with pytest.raises(ReplayEnvironmentMismatch, match="test-2026-09.*other-2026-10"):
        rep.play(env=other, cards=db)
    with pytest.raises(ReplayEnvironmentMismatch, match="requires environment test-2026-09"):
        rep.play(cards=db)
    (env.root / "banlist.lflist.conf").write_text("!changed\n")
    changed = load_environment(env.root)
    with pytest.raises(ReplayEnvironmentMismatch, match="fingerprint"):
        rep.play(env=changed, cards=db)


def test_unknown_format_version(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"format": "ygorl-replay", "format_version": 999}))
    with pytest.raises(ValueError, match="format_version 999"):
        Replay.load(tmp_path / "r.json")


def minimal_replay_json():
    decks = {s: {"name": s, "main": list(d.main), "extra": list(d.extra), "side": []} for s, d in (("a", A), ("b", B))}
    return {"format": "ygorl-replay", "format_version": 1, "seed": 3, "first": 1, "rule_flags": C.DUEL_MODE_MR5,
            "player": {"starting_lp": 8000, "starting_hand": 5, "draw_per_turn": 1}, "shuffle_decks": True,
            "decks": decks, "responses": ["00000000", "0100"], "environment": None,
            "result": {"winner": None, "lp": [8000, 8000]}}  # fmt: skip


def test_minimal_replay_json_loads():
    rep = Replay.from_json(minimal_replay_json())
    assert rep.responses == [b"\0\0\0\0", b"\x01\x00"] and rep.decks["a"]["main"] == list(A.main)


@pytest.mark.parametrize(
    "change, msg",
    [
        ({"seed": "abc"}, "'seed' must be an integer, not str"),
        ({"seed": True}, "'seed' must be an integer, not bool"),
        ({"first": 2}, "'first' must be 0 or 1"),
        ({"rule_flags": -1}, "'rule_flags' must be"),
        ({"player": [8000, 5, 1]}, "'player' must be an object"),
        ({"player": {"starting_lp": "8000", "starting_hand": 5, "draw_per_turn": 1}}, r"'player.starting_lp' must be"),
        ({"player": {"starting_lp": 8000}}, "'player' needs keys"),
        ({"shuffle_decks": "yes"}, "'shuffle_decks' must be true or false"),
        ({"decks": [1, 2]}, "'decks' must be an object with decks 'a' and 'b'"),
        ({"decks": {"a": {"main": [], "extra": []}}}, "'decks' must be an object with decks 'a' and 'b'"),
        ({"decks": {"a": {"main": [1], "extra": []}, "b": {"main": "1", "extra": []}}},
         r"'decks.b.main' must be a list"),
        ({"decks": {"a": {"main": [2**32], "extra": []}, "b": {"main": [], "extra": []}}}, r"'decks.a.main\[0\]'"),
        ({"responses": "00"}, "'responses' must be a list"),
        ({"responses": ["zz"]}, r"'responses\[0\]' is not a hex string"),
        ({"responses": [5]}, r"'responses\[0\]' is not a hex string"),
        ({"environment": "v1"}, "'environment' must be null or an object"),
        ({"environment": {"fingerprint": "x"}}, "'environment.version' must be a string"),
        ({"seed_words": [1, 2, 3]}, "'seed_words' must be null or four"),
        ({"max_turns": "200"}, "'max_turns' must be"),
        ({"result": []}, "'result' must be an object"),
        ({"result": {"lp": 8000}}, "'result.lp' must be"),
        ({"steps": {}}, "'steps' must be a list"),
        ({"engine": {"ocgcore": "11.0"}}, "'engine.ocgcore' must be"),
        ({"bogus": 1}, "unknown replay field"),
    ],
)
def test_malformed_replay_fields_are_value_errors(change, msg):
    data = minimal_replay_json()
    data.update(change)
    with pytest.raises(ValueError, match=msg):
        Replay.from_json(data)


def test_missing_replay_field():
    data = minimal_replay_json()
    del data["responses"]
    with pytest.raises(ValueError, match="missing field 'responses'"):
        Replay.from_json(data)


@pytest.mark.parametrize(
    "name, content, msg",
    [
        ("r.json.gz", gzip.compress(b'{"format": "ygorl-replay"}')[:12], "truncated or corrupt gzip"),
        ("r.json.gz", b"not gzip at all", "truncated or corrupt gzip"),
        ("r.json", b"[1, 2]", "expected a JSON object, not list"),
        ("r.json", b"\xff\xfe", "not valid JSON"),
        ("r.json", b"{", "not valid JSON"),
    ],
)
def test_unreadable_replay_files_are_value_errors(tmp_path, name, content, msg):
    (tmp_path / name).write_bytes(content)
    with pytest.raises(ValueError, match=msg):
        Replay.load(tmp_path / name)


class ProbAgent(RandomAgent):
    """Exposes the distribution it sampled from, as a policy agent would."""

    def act(self, point):
        n = len(point.actions)
        self.last_probs = [1.0 / n] * n
        return super().act(point)


def test_step_records_with_candidates_and_probabilities(db):
    duel = Duel(8, None, A, B, cards=db, record_steps=True, config=None)
    result = duel.run(ProbAgent(1), RandomAgent(2))
    assert len(result.steps) == result.decisions
    step = result.steps[0]
    assert set(step) >= {"index", "player", "decision", "actions", "chosen"}
    assert 0 <= step["chosen"] < len(step["actions"])
    assert {"kind", "code", "description", "value"} <= set(step["actions"][0])
    by_agent = {s["player"]: s for s in result.steps}
    first_player_step = next(s for s in result.steps if s["player"] == 0)
    assert first_player_step.get("probs") and abs(sum(first_player_step["probs"]) - 1) < 1e-9
    assert all("probs" not in s for s in result.steps if s["player"] == 1)
    assert by_agent
    rep = Replay.from_duel(duel, result)
    assert rep.steps == result.steps


# ------------------------------------------------------------------------ .yrpX


def parse_header(buf):
    ident, version, flag, stamp, datasize, hsh = struct.unpack_from("<6I", buf, 0)
    props = buf[24:32]
    header_version = struct.unpack_from("<Q", buf, 32)[0]
    seed = list(struct.unpack_from("<4Q", buf, 40))
    return {"id": ident, "version": version, "flag": flag, "datasize": datasize, "props": props,
            "header_version": header_version, "seed": seed}, buf[72:]  # fmt: skip


def read_names(body, pos):
    names = []
    for _ in range(2):
        (count,) = struct.unpack_from("<I", body, pos)
        pos += 4
        for _ in range(count):
            names.append(body[pos : pos + 40].decode("utf-16-le").split("\0")[0])
            pos += 40
    return names, pos


def parse_yrp1(buf):
    header, body = parse_header(buf)
    names, pos = read_names(body, 0)
    lp, hand, draw, flags = struct.unpack_from("<IIIQ", body, pos)
    pos += 20
    decks = []
    for _ in range(2):
        (n,) = struct.unpack_from("<I", body, pos)
        main = list(struct.unpack_from(f"<{n}I", body, pos + 4))
        pos += 4 + 4 * n
        (n,) = struct.unpack_from("<I", body, pos)
        extra = list(struct.unpack_from(f"<{n}I", body, pos + 4))
        pos += 4 + 4 * n
        decks.append((main, extra))
    (rules,) = struct.unpack_from("<I", body, pos)
    pos += 4 + 4 * rules
    responses = []
    while pos < len(body):
        n = body[pos]
        if n == 0:
            break
        responses.append(body[pos + 1 : pos + 1 + n])
        pos += 1 + n
    return header, names, (lp, hand, draw, flags), decks, responses


def parse_yrpx(buf):
    header, body = parse_header(buf)
    names, pos = read_names(body, 0)
    (flags,) = struct.unpack_from("<Q", body, pos)
    pos += 8
    packets = []
    while pos < len(body):
        msg = body[pos]
        (n,) = struct.unpack_from("<I", body, pos + 1)
        packets.append((msg, body[pos + 5 : pos + 5 + n]))
        pos += 5 + n
    return header, names, flags, packets


YRPX, YRP1, OLD_REPLAY_MODE = 0x58707279, 0x31707279, 231
EXPECTED_FLAGS = 0x10 | 0x20 | 0x100 | 0x200  # LUA64 | NEWREPLAY | 64BIT_DUELFLAG | EXTENDED_HEADER


def test_yrpx_export_structure(db, env, tmp_path):
    duel, result = record(db, env)
    rep = Replay.from_duel(duel, result)
    path = tmp_path / "game.yrpX"
    rep.to_yrpx(path, names=("Alice", "Bob"), env=env)
    header, names, flags, packets = parse_yrpx(path.read_bytes())
    assert header["id"] == YRPX and header["flag"] == EXPECTED_FLAGS and header["header_version"] == 1
    assert header["seed"] == rep.core_seed and header["datasize"] == len(path.read_bytes()) - 72
    assert names == ["Bob", "Alice"] and flags == rep.rule_flags  # seat order: b moved first (first=1)
    assert packets[0][0] == C.MSG_START and len(packets[0][1]) == 17  # u8 type, 2x u32 LP, 4x u16 deck sizes
    assert packets[-1][0] == OLD_REPLAY_MODE
    kinds = [m for m, _ in packets[1:-1]]
    assert not set(kinds) & M.DECISION_TYPES and C.MSG_RETRY not in kinds
    assert kinds.count(C.MSG_NEW_TURN) == result.turns
    assert kinds[-1] == C.MSG_WIN

    y_header, y_names, params, decks, responses = parse_yrp1(packets[-1][1])
    assert y_header["id"] == YRP1 and y_header["seed"] == rep.core_seed and y_names == ["Bob", "Alice"]
    assert params == (8000, 5, 1, rep.rule_flags)
    # engine player 0 is deck b because first=1; decks are stored in load (post-shuffle) order
    assert sorted(decks[0][0]) == sorted(B.main) and decks[0][1] == list(B.extra)
    assert sorted(decks[1][0]) == sorted(A.main) and decks[1][1] == list(A.extra)
    assert responses == rep.responses


def test_embedded_yrp1_resimulates_like_edopro_old_mode(db, env, tmp_path):
    """Re-run the core exactly as EDOPro's old replay mode would, from the file alone."""
    duel, result = record(db, env, seed=23)
    path = tmp_path / "g.yrpX"
    Replay.from_duel(duel, result).to_yrpx(path, env=env)
    *_, packets = parse_yrpx(path.read_bytes())
    header, _, (lp, hand, draw, flags), decks, responses = parse_yrp1(packets[-1][1])
    core = _core.Duel(header["seed"], flags, (lp, hand, draw), (lp, hand, draw), db.to_core(), default_scripts())
    assert core.load_script("constant.lua") and core.load_script("utility.lua")
    for team, (main, extra) in enumerate(decks):
        for code in main:
            core.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
        for code in extra:
            core.new_card(team, 0, code, team, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
    core.start()
    stream, feed = [], iter(responses)
    while True:
        status = core.process()
        stream.append(core.get_message())
        if any(isinstance(m, M.Win) for m in M.decode_buffer(stream[-1])) or status == _core.DUEL_STATUS_END:
            break
        if status == _core.DUEL_STATUS_AWAITING:
            core.set_response(next(feed))
    assert stream == result.message_log
