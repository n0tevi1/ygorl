"""Reading EDOPro replays (.yrp / .yrpX, LZMA) and replaying them from explicit core seed words (T4a.1)."""

import lzma
import struct
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, expand_seed
from ygorl.engine.replay import (
    REPLAY_YRP1,
    REPLAY_YRPX,
    Replay,
    YrpError,
    load_yrp,
    parse_yrp,
)

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
A, B = DECKS["snake_eye"], DECKS["kashtira"]
FLAG_COMPRESSED = 0x1


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def game(db):
    duel = Duel(31, None, A, B, cards=db, first=1, config=DuelConfig(max_turns=3))
    result = duel.run(RandomAgent(3), RandomAgent(4))
    return duel, result, Replay.from_duel(duel, result)


@pytest.mark.parametrize("compress", [True, False])
def test_parse_exported_yrpx_and_its_embedded_yrp1(game, tmp_path, compress):
    duel, result, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path, names=("Alice", "Bob"), compress=compress)
    yrpx = load_yrp(path)
    assert yrpx.id == REPLAY_YRPX and yrpx.kind == "yrpX"
    assert bool(yrpx.flag & FLAG_COMPRESSED) == bool(yrpx.embedded.flag & FLAG_COMPRESSED) == compress
    assert yrpx.names == ("Bob", "Alice")  # seat order: b moved first
    assert yrpx.seed == tuple(rep.core_seed) and yrpx.rule_flags == rep.rule_flags
    assert yrpx.packets[0][0] == 4  # MSG_START
    inner = yrpx.replayable()
    assert inner is yrpx.embedded and inner.id == REPLAY_YRP1
    assert inner.player == (8000, 5, 1) and inner.rule_flags == rep.rule_flags
    assert inner.decks == tuple((tuple(m), tuple(x)) for m, x in duel.loaded_decks())
    assert inner.responses == tuple(result.responses)


def test_lzma_compressed_body_is_read(game, tmp_path):
    """A body compressed with other LZMA settings (dictionary 1 MiB, liblzma defaults) than ours is read too."""
    _, result, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path, compress=False)
    raw = path.read_bytes()
    inner_raw = next(p for m, p in parse_yrp(raw).packets if m == 231)
    header, body = bytearray(inner_raw[:72]), inner_raw[72:]
    alone = lzma.compress(body, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20}])
    props, stream = alone[:5], alone[13:]  # .lzma header: 5 property bytes + u64 size, as EDOPro stores them apart
    struct.pack_into("<I", header, 8, struct.unpack_from("<I", header, 8)[0] | FLAG_COMPRESSED)
    header[24:29] = props
    compressed = parse_yrp(bytes(header) + stream)
    assert compressed.responses == tuple(result.responses)
    assert compressed.decks == parse_yrp(inner_raw).decks


def test_replay_from_yrp_resimulates_the_game(game, db, tmp_path):
    _, result, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path)
    again = Replay.from_yrp(path)
    assert again.core_seed == rep.core_seed and again.shuffle_decks is False and again.first == 0
    played = again.play(cards=db)
    assert played.responses == result.responses and played.retries == 0
    # the file knows nothing of the recorded turn limit, so the re-run stops when the responses run out
    assert played.turns == result.turns and played.reason == "log_exhausted"
    assert sorted(played.lp) == sorted(result.lp)  # sides are seats here: deck a of the yrp is engine player 0


def test_explicit_core_seed_words(db):
    words = [11, 22, 33, 44]
    duel = Duel(0, None, A, B, cards=db, core_seed=words, config=DuelConfig(max_turns=2))
    assert duel.core_seed == words
    result = duel.run(RandomAgent(1), RandomAgent(2))
    rep = Replay.from_duel(duel, result)
    assert rep.core_seed == words and rep.to_json()["seed_words"] == words
    assert rep.play(cards=db).responses == result.responses
    plain = Replay.from_duel(*_plain(db))
    assert "seed_words" not in plain.to_json() and plain.core_seed == expand_seed(plain.seed)
    with pytest.raises(ValueError, match="four"):
        Duel(0, None, A, B, cards=db, core_seed=[1, 2])
    with pytest.raises(ValueError, match="all-zero"):
        Duel(0, None, A, B, cards=db, core_seed=[0, 0, 0, 0])


def _plain(db):
    duel = Duel(5, None, A, B, cards=db, config=DuelConfig(max_turns=1))
    return duel, duel.run(RandomAgent(1), RandomAgent(2))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b[:20], "truncated header"),
        (lambda b: b"abcd" + b[4:], "not an EDOPro replay"),
        (lambda b: b[:-30], "truncated"),
    ],
)
def test_malformed_files_raise_clear_errors(game, tmp_path, mutate, message):
    _, _, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path, compress=False)
    inner = next(p for m, p in parse_yrp(path.read_bytes()).packets if m == 231)
    with pytest.raises(YrpError, match=message):
        parse_yrp(mutate(inner))


def test_truncated_lzma_body_raises(game, tmp_path):
    _, _, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path)
    raw = path.read_bytes()
    with pytest.raises(YrpError, match="LZMA decompression failed"):
        parse_yrp(raw[: len(raw) // 2])


def test_yrpx_without_embedded_yrp1_is_not_replayable(game, tmp_path):
    _, _, rep = game
    path = tmp_path / "g.yrpX"
    rep.to_yrpx(path, compress=False)
    raw = path.read_bytes()
    parsed = parse_yrp(raw)
    last = parsed.packets[-1]
    cut = raw[: len(raw) - (5 + len(last[1]))]
    stripped = bytearray(cut)
    struct.pack_into("<I", stripped, 16, len(cut) - 72)
    with pytest.raises(YrpError, match="no embedded yrp1"):
        parse_yrp(bytes(stripped)).replayable()
