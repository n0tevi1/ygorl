"""Replay format, environment binding and .yrpX export (T1.8)."""

import gzip
import json
import lzma
import struct
import time
from pathlib import Path

import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, default_scripts
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
    assert (again.winner, again.reason, again.turns, again.lp) == (
        result.winner,
        result.reason,
        result.turns,
        result.lp,
    )
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
        (
            {"decks": {"a": {"main": [1], "extra": []}, "b": {"main": "1", "extra": []}}},
            r"'decks.b.main' must be a list",
        ),
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
        ({"recorded_at": "2026-09-22"}, "'recorded_at' must be null or unix seconds"),
        ({"recorded_at": -1}, "'recorded_at' must be null or unix seconds"),
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


def lzma_uncompress(stream, props, size):
    """EDOPro's LzmaUncompress: 5 property bytes kept apart, a bare LZMA1 stream, decoding stops at ``size`` bytes."""
    d, dict_size = props[0], struct.unpack_from("<I", props, 1)[0]
    lzma1 = {"id": lzma.FILTER_LZMA1, "lc": d % 9, "lp": d // 9 % 5, "pb": d // 45, "dict_size": dict_size}
    return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[lzma1]).decompress(stream, max_length=size)


def parse_header(buf):
    ident, version, flag, stamp, datasize, hsh = struct.unpack_from("<6I", buf, 0)
    props = buf[24:32]
    header_version = struct.unpack_from("<Q", buf, 32)[0]
    seed = list(struct.unpack_from("<4Q", buf, 40))
    body = lzma_uncompress(buf[72:], props, datasize) if flag & COMPRESSED else buf[72:]
    assert len(body) == datasize
    return {"id": ident, "version": version, "flag": flag, "datasize": datasize, "props": props,
            "header_version": header_version, "seed": seed}, body  # fmt: skip


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
COMPRESSED = 0x1
EXPECTED_FLAGS = 0x10 | 0x20 | 0x100 | 0x200  # LUA64 | NEWREPLAY | 64BIT_DUELFLAG | EXTENDED_HEADER
# LzmaCompress(level 5, dictSize 1 << 24, lc 3, lp 0, pb 2) in EDOPro's Replay::EndRecord: (pb * 5 + lp) * 9 + lc, dictSize
EDOPRO_LZMA_PROPS = bytes([93]) + (1 << 24).to_bytes(4, "little")


@pytest.mark.parametrize("compress", [True, False])
def test_yrpx_export_structure(db, env, tmp_path, compress):
    duel, result = record(db, env)
    rep = Replay.from_duel(duel, result)
    path = tmp_path / "game.yrpX"
    rep.to_yrpx(path, names=("Alice", "Bob"), env=env, compress=compress)
    header, names, flags, packets = parse_yrpx(path.read_bytes())
    flag, props = (EXPECTED_FLAGS | COMPRESSED, EDOPRO_LZMA_PROPS) if compress else (EXPECTED_FLAGS, bytes(5))
    assert header["id"] == YRPX and header["flag"] == flag and header["header_version"] == 1
    assert header["props"] == props + bytes(3)
    assert header["seed"] == rep.core_seed
    if not compress:
        assert header["datasize"] == len(path.read_bytes()) - 72
    assert names == ["Bob", "Alice"] and flags == rep.rule_flags  # seat order: b moved first (first=1)
    assert packets[0][0] == C.MSG_START and len(packets[0][1]) == 17  # u8 type, 2x u32 LP, 4x u16 deck sizes
    assert packets[-1][0] == OLD_REPLAY_MODE
    kinds = [m for m, _ in packets[1:-1]]
    assert not set(kinds) & M.DECISION_TYPES and C.MSG_RETRY not in kinds
    assert kinds.count(C.MSG_NEW_TURN) == result.turns
    assert kinds[-1] == C.MSG_WIN

    y_header, y_names, params, decks, responses = parse_yrp1(packets[-1][1])
    assert y_header["id"] == YRP1 and y_header["seed"] == rep.core_seed and y_names == ["Bob", "Alice"]
    assert y_header["flag"] == flag and y_header["props"] == props + bytes(3)  # EDOPro compresses both replays
    assert params == (8000, 5, 1, rep.rule_flags)
    # engine player 0 is deck b because first=1; decks are stored in load (post-shuffle) order
    assert sorted(decks[0][0]) == sorted(B.main) and decks[0][1] == list(B.extra)
    assert sorted(decks[1][0]) == sorted(A.main) and decks[1][1] == list(A.extra)
    assert responses == rep.responses


def test_compression_changes_only_the_encoding_and_shrinks_the_file(db, env, tmp_path):
    """Same header fields and packets either way; the LZMA file of a full game is a small fraction of the raw one."""
    duel, result = record(db, env, seed=23)
    rep = Replay.from_duel(duel, result)
    packed, raw = tmp_path / "packed.yrpX", tmp_path / "raw.yrpX"
    rep.to_yrpx(packed, env=env)
    rep.to_yrpx(raw, env=env, compress=False)
    (hp, *rest_p, pp), (hr, *rest_r, pr) = parse_yrpx(packed.read_bytes()), parse_yrpx(raw.read_bytes())
    assert rest_p == rest_r and pp[:-1] == pr[:-1] and len(pp) > 1000  # names, rule flags, every packet but the last

    def fixed(header):  # everything but the encoding
        return {k: v for k, v in header.items() if k not in ("flag", "props", "datasize")}

    assert hp["flag"] == hr["flag"] | COMPRESSED and fixed(hp) == fixed(hr)
    # the embedded yrp1 differs only by its own compression: same header fields, same body
    (yp, body_p), (yr, body_r) = parse_header(pp[-1][1]), parse_header(pr[-1][1])
    assert yp["flag"] == yr["flag"] | COMPRESSED and fixed(yp) == fixed(yr) and body_p == body_r
    assert hp["datasize"] == hr["datasize"] - len(pr[-1][1]) + len(pp[-1][1])
    assert len(raw.read_bytes()) > 500_000 and len(packed.read_bytes()) * 20 < len(raw.read_bytes())


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


# ------------------------------------------------ .yrpX: header date, host refreshes, limit results

TIMESTAMP_OFFSET = 12  # u32 after id, version, flag


def header_timestamp(buf):
    return struct.unpack_from("<I", buf, TIMESTAMP_OFFSET)[0]


def test_recorded_at_is_the_header_date_of_both_replays(db, env, tmp_path):
    duel, result = record(db, env, seed=29, config=DuelConfig.from_environment(env, max_turns=2))
    rep = Replay.from_duel(duel, result)
    assert isinstance(rep.recorded_at, int) and abs(rep.recorded_at - time.time()) < 600
    rep.recorded_at = 1_790_000_000
    first, second = tmp_path / "a.yrpX", tmp_path / "b.yrpX"
    rep.to_yrpx(first, env=env)
    Replay.load(_saved(rep, tmp_path)).to_yrpx(second, env=env)
    assert first.read_bytes() == second.read_bytes()  # the date travels with the replay file: exports are reproducible
    *_, packets = parse_yrpx(first.read_bytes())
    assert header_timestamp(first.read_bytes()) == header_timestamp(packets[-1][1]) == 1_790_000_000
    # a round trip through the EDOPro file keeps the date; the seed words are explicit, not the timestamp
    back = Replay.from_yrp(first)
    assert back.recorded_at == 1_790_000_000 and back.core_seed == rep.core_seed and back.seed == 0
    # replays saved before recorded_at existed load unchanged and export with date 0 (unknown)
    data = rep.to_json()
    del data["recorded_at"]
    old = Replay.from_json(data)
    assert old.recorded_at is None and "recorded_at" not in old.to_json()
    old.to_yrpx(second, env=env)
    assert header_timestamp(second.read_bytes()) == 0


def _saved(rep, tmp_path):
    path = tmp_path / "r.json.gz"
    rep.save(path)
    return path


def parse_queries(blob):
    """Split a query blob (EDOPro QueryStream / Query layout) into per-card [(flag, data)] lists; [] = empty slot."""
    cards, pos = [], 0
    while pos < len(blob):
        (size,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        if size == 0:
            cards.append([])
            continue
        fields = []
        while True:
            (flag,) = struct.unpack_from("<I", blob, pos)
            fields.append((flag, bytes(blob[pos + 4 : pos + size])))
            pos += size
            if flag == C.QUERY_END:
                break
            (size,) = struct.unpack_from("<H", blob, pos)
            pos += 2
        cards.append(fields)
    return cards


def location_queries(payload):
    (size,) = struct.unpack_from("<I", payload, 2)
    assert size == len(payload) - 6
    return parse_queries(payload[6:])


def host_form(fields):
    """What EDOPro's CoreUtils::Query::GenerateBuffer(check_hidden=false) makes of one card's raw core query."""
    kept = [(f, d) for f, d in fields if not (f in (C.QUERY_REASON_CARD, C.QUERY_EQUIP_CARD) and d[1] == 0)]
    return sorted(kept)


def edopro_core(db, yrp1_packet):
    header, _, (lp, hand, draw, flags), decks, _ = parse_yrp1(yrp1_packet)
    core = _core.Duel(header["seed"], flags, (lp, hand, draw), (lp, hand, draw), db.to_core(), default_scripts())
    assert core.load_script("constant.lua") and core.load_script("utility.lua")
    for team, (main, extra) in enumerate(decks):
        for code in main:
            core.new_card(team, 0, code, team, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
        for code in extra:
            core.new_card(team, 0, code, team, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
    return core


Q_MZONE, Q_SZONE, Q_HAND, Q_EXTRA, Q_DECK = 0x3981FFF, 0x3F81FFF, 0x3781FFF, 0x381FFF, 0x1181FFF


def test_stream_carries_the_host_refresh_packets(db, env, tmp_path):
    """The packets EDOPro's host (GenericDuel) records around the engine messages, checkable without EDOPro."""
    duel, result = record(db, env, seed=41)
    path = tmp_path / "g.yrpX"
    Replay.from_duel(duel, result).to_yrpx(path, env=env)
    *_, packets = parse_yrpx(path.read_bytes())
    refresh = {C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD}

    # the engine messages themselves are unchanged: all but decisions, MSG_RETRY and private hints, in order
    engine = [(r[0], r[1:]) for buf in result.message_log for r in M.split_messages(buf)
              if r[0] not in M.DECISION_TYPES and r[0] != C.MSG_RETRY and not (r[0] == C.MSG_HINT and r[1] in (1, 2, 3, 5))]  # fmt: skip
    assert [p for p in packets[1:-1] if p[0] not in refresh] == engine

    # after MSG_START: deck of both players (raw core query), then both extra decks, as at the start of the duel
    core = edopro_core(db, packets[-1][1])
    core.start()
    assert [(m, p[:2]) for m, p in packets[1:5]] == [(C.MSG_UPDATE_DATA, bytes([0, C.LOCATION_DECK])),
                                                     (C.MSG_UPDATE_DATA, bytes([1, C.LOCATION_DECK])),
                                                     (C.MSG_UPDATE_DATA, bytes([0, C.LOCATION_EXTRA])),
                                                     (C.MSG_UPDATE_DATA, bytes([1, C.LOCATION_EXTRA]))]  # fmt: skip
    for p in (0, 1):
        assert packets[1 + p][1][2:] == core.query_location(Q_DECK, p, C.LOCATION_DECK)
        direct = parse_queries(core.query_location(Q_EXTRA, p, C.LOCATION_EXTRA)[4:])
        assert location_queries(packets[3 + p][1]) == [host_form(c) for c in direct]
    core.close()

    def data(i):
        m, p = packets[i]
        return (m, p[0], p[1]) if m == C.MSG_UPDATE_DATA else None

    field_before = [(C.MSG_UPDATE_DATA, 0, C.LOCATION_MZONE), (C.MSG_UPDATE_DATA, 1, C.LOCATION_MZONE),
                    (C.MSG_UPDATE_DATA, 0, C.LOCATION_SZONE), (C.MSG_UPDATE_DATA, 1, C.LOCATION_SZONE)]  # fmt: skip
    turns = draws = moves = 0
    for i, (m, p) in enumerate(packets):
        if m == C.MSG_NEW_TURN:  # recorded last, after the monster and spell/trap zone refreshes
            assert [data(j) for j in range(i - 4, i)] == field_before
            turns += 1
        elif m == C.MSG_DRAW:  # followed by the hand of the drawing player
            assert data(i + 1) == (C.MSG_UPDATE_DATA, p[0], C.LOCATION_HAND)
            draws += 1
        elif m == C.MSG_MOVE:
            prev_con, prev_loc = p[4], p[5]
            con, loc, seq = p[14], p[15], struct.unpack_from("<I", p, 16)[0]
            if loc and not loc & C.LOCATION_OVERLAY and (loc != prev_loc or con != prev_con):
                nm, np_ = packets[i + 1]
                assert nm == C.MSG_UPDATE_CARD and np_[:3] == bytes([con, loc, seq])
                assert len(parse_queries(np_[3:])) == 1
                moves += 1
        if m == C.MSG_UPDATE_DATA and p[1] != C.LOCATION_DECK:
            for card in location_queries(p):  # EDOPro's re-serialisation: ascending flags, ending with QUERY_END
                flags = [f for f, _ in card]
                assert flags == sorted(flags) and (not card or flags[-1] == C.QUERY_END)
    assert turns == result.turns and draws and moves
    # the refreshes EDOPro's host sends before every idle/battle command: both monster, spell/trap zones and hands
    idle = sum(
        r[0] in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
        for buf in result.message_log
        for r in M.split_messages(buf)
    )
    six = field_before + [(C.MSG_UPDATE_DATA, 0, C.LOCATION_HAND), (C.MSG_UPDATE_DATA, 1, C.LOCATION_HAND)]
    runs = sum([data(j) for j in range(i, i + 6)] == six for i in range(len(packets) - 6))
    assert runs >= idle > 0


@pytest.mark.parametrize(
    ("limit", "reason"), [({"max_turns": 2}, "turn_limit"), ({"max_decisions": 40}, "decision_limit")]
)
def test_limit_games_end_with_a_host_msg_win(db, env, tmp_path, limit, reason):
    duel, result = record(db, env, seed=37, config=DuelConfig.from_environment(env, **limit))
    assert result.reason == reason
    rep = Replay.from_duel(duel, result)
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path, env=env)
    *_, packets = parse_yrpx(path.read_bytes())
    wins = [p for m, p in packets if m == C.MSG_WIN]
    seat = 2 if result.winner is None else (result.winner - duel.first) % 2
    assert wins == [bytes([seat, 0x3])] and packets[-2][0] == C.MSG_WIN  # [player, reason] like the host's timeout
    # the winner comes from the recorded result, not from a rule baked into the exporter
    rep.result = {**rep.result, "winner": None}
    rep.to_yrpx(path, env=env)
    *_, packets = parse_yrpx(path.read_bytes())
    assert packets[-2] == (C.MSG_WIN, bytes([2, 0x3]))
    # a replay read back from another host's file records no end, so it gets none
    again = tmp_path / "again.yrpX"
    Replay.from_yrp(path).to_yrpx(again, cards=db)
    *_, packets = parse_yrpx(again.read_bytes())
    assert C.MSG_WIN not in [m for m, _ in packets]


def test_games_ended_by_the_engine_get_no_extra_msg_win(db, env, tmp_path):
    duel, result = record(db, env)
    assert result.reason == "win"
    path = tmp_path / "g.yrpX"
    Replay.from_duel(duel, result).to_yrpx(path, env=env)
    *_, packets = parse_yrpx(path.read_bytes())
    assert [m for m, _ in packets].count(C.MSG_WIN) == 1
    again = tmp_path / "again.yrpX"
    Replay.from_yrp(path).to_yrpx(again, cards=db)
    *_, packets = parse_yrpx(again.read_bytes())
    assert [m for m, _ in packets].count(C.MSG_WIN) == 1


def test_verify_accepts_an_old_decision_limit_win_by_lp(db, env, tmp_path, capsys):
    """Before 2026-09-24 a decision-limit end went to the higher LP; it is a draw now. ``replay --verify`` still
    checks the responses, turns and LP of such a replay, not its winner."""
    from ygorl.cli import main

    for seed in range(37, 60):
        duel, result = record(db, env, seed=seed, config=DuelConfig.from_environment(env, max_decisions=300))
        if result.reason == "decision_limit" and result.lp[0] != result.lp[1]:
            break
    else:
        pytest.skip("no decision-limit game with unequal LP among these seeds")
    assert result.winner is None
    rep = Replay.from_duel(duel, result)
    rep.result = {**rep.result, "winner": 0 if result.lp[0] > result.lp[1] else 1}  # as recorded by the old rule
    path = tmp_path / "old.json"
    rep.save(path)
    code = main(["replay", str(path), "--verify", "--env", str(env.root)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "verify     ok" in out
    rep.result = {**rep.result, "turns": rep.result["turns"] + 1}  # anything else still has to match
    rep.save(path)
    assert main(["replay", str(path), "--verify", "--env", str(env.root)]) == 1
