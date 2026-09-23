"""T5.1 data acquisition: parsers on small offline fixtures, fetchers with a fake network, the offline build.

The network is only touched by the tests marked ``network``, which run when ``YGORL_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
from collections import Counter
from pathlib import Path

import pytest

from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.cli import main
from ygorl.data import load_environment
from ygorl.data import masterduelmeta as mdm
from ygorl.data import yugipedia, ygoprodeck
from ygorl.data.build import OVERRIDES, REVIEW_DIR, BuildError, BuildOptions, build
from ygorl.data.cardmap import CardMapper, normalize_name
from ygorl.data.fetch import FetchError, Http, provenance, read_raw, write_raw

ROOT = Path(__file__).resolve().parents[1]
DECKS = ROOT / "tests" / "decks"
SNAPSHOT = ROOT / "environments" / "md-2026-09"

FENRIR, UNICORN, BIRTH = 32909498, 68304193, 69540484  # Kashtira Fenrir / Unicorn / Birth
MAXX_C, VEILER, DROLL, DROLL_ALT = 23434538, 97268402, 94145021, 94145022
BRANDED_FUSION, ALUBER, ALBAZ, BRANDED_OPENING = 44362883, 62962630, 68468459, 36637374
BARREL_DRAGON_YGOPRODECK, BARREL_DRAGON = 81480461, 81480460  # YGOPRODECK's id differs from BabelCDB's
TOKEN = 44586427  # Batteryman Token
UNKNOWN = 99999998


@pytest.fixture(scope="module")
def db() -> CardDB:
    return CardDB.load()


@pytest.fixture(scope="module")
def mapper(db) -> CardMapper:
    return CardMapper(db)


# --------------------------------------------------------------------------- fixtures: raw source files


def _slots(passwords, db):
    return [{"card": {"_id": f"m{p}", "name": db[p].name}, "amount": n} for p, n in Counter(passwords).items()]


def _list(dtype, created, main, extra, db, weight=10, include=True):
    return {"deckType": {"name": dtype}, "created": created, "url": f"/decks/{dtype.lower()}/{created[:10]}/",
            "rankedType": {"name": "Master I", "statsWeight": weight, "includeInStats": include},
            "main": _slots(main, db), "extra": _slots(extra, db)}  # fmt: skip


def write_fixture_raw(raw: Path, db: CardDB) -> None:
    """A small but complete set of raw files, as the fetchers would write them (with sidecars)."""
    branded, kashtira, yubel = (load_ydk(DECKS / f"{n}.ydk") for n in ("branded_despia", "kashtira", "yubel"))
    pool = sorted(set(branded.main + branded.extra + kashtira.main + kashtira.extra + yubel.main + yubel.extra))
    cards = [{"id": p, "name": db[p].name, "frameType": "effect"} for p in pool]
    cards += [{"id": TOKEN, "name": db[TOKEN].name, "frameType": "token"},
              {"id": UNKNOWN, "name": "Card From The Future", "frameType": "effect"},
              {"id": BARREL_DRAGON_YGOPRODECK, "name": "Barrel Dragon", "frameType": "effect"}]  # fmt: skip
    write_raw(raw / ygoprodeck.RAW_FILE, {"data": cards}, ["https://example.invalid/cardinfo"])

    mdm_cards = [{"_id": f"m{p}", "konamiID": str(p), "name": db[p].name} for p in pool]
    write_raw(raw / mdm.CARDS_FILE, mdm_cards, ["https://example.invalid/cards"])
    banlist = [{"konamiID": str(FENRIR), "name": "Kashtira Fenrir", "banStatus": "Limited 1"},
               {"konamiID": str(DROLL), "name": "Droll & Lock Bird", "banStatus": "Limited 2"},
               {"konamiID": str(DROLL_ALT), "name": "Droll & Lock Bird", "banStatus": "Limited 2"},
               {"konamiID": "99999997", "name": "Made Up Card", "banStatus": "Forbidden"}]  # fmt: skip
    write_raw(raw / mdm.BANLIST_FILE, banlist, ["https://example.invalid/cards"])
    articles = [{"title": "Master Duel: Forbidden / Limited List Update", "date": "2026-09-01T09:00:00Z",
                 "url": "/news/md-banlist/"}, {"title": "TCG: Forbidden & Limited List", "date": "2026-09-10"}]  # fmt: skip
    write_raw(raw / mdm.ARTICLES_FILE, articles, ["https://example.invalid/articles"])

    variant = list(branded.main)
    variant.remove(MAXX_C)
    variant.append(VEILER)  # 1 Maxx "C", 3 Effect Veiler
    kashtira_legal = [p for p in kashtira.main if p != FENRIR] + [FENRIR, UNICORN, BIRTH]  # 1 Fenrir
    decks = [
        _list("Branded", "2026-09-20T10:00:00Z", branded.main, branded.extra, db),
        _list("Branded", "2026-09-18T10:00:00Z", variant, branded.extra, db),
        _list("Branded", "2026-09-05T10:00:00Z", branded.main, branded.extra, db),
        _list("Kashtira", "2026-09-12T10:00:00Z", kashtira.main, kashtira.extra, db),  # 3 Fenrir: illegal
        _list("Kashtira", "2026-09-11T10:00:00Z", kashtira_legal, kashtira.extra, db),
        _list("Tenpai", "2026-09-13T10:00:00Z", branded.main, branded.extra, db, weight=0, include=False),
        _list("Old", "2026-08-20T10:00:00Z", branded.main, branded.extra, db),  # before the window
    ]
    yubel_list = _list("Yubel", "2026-09-14T10:00:00Z", yubel.main, yubel.extra, db)
    yubel_list["main"].append({"card": {"_id": "nope", "name": "Mystery Card"}, "amount": 1})
    decks.append(yubel_list)
    write_raw(raw / mdm.TOP_DECKS_FILE, decks, ["https://example.invalid/top-decks"], since="2026-08-01")

    ask = {
        "archetype_support": {"Branded Fusion": {"printouts": {"Password": [str(BRANDED_FUSION)],
                                                               "Archetype support": [{"fulltext": "Fallen of Albaz"}]}},
                              "Anime Card": {"printouts": {"Password": ["123"], "Archetype support": ["Nobody"]}}},
        "anti_support": [],
        "archseries_related": {"Branded Opening": {"printouts": {"Password": [str(BRANDED_OPENING)],
                                                                 "Archseries related": ["Albaz Dragon"]}}},
        "archseries": {"Aluber the Jester of Despia": {"printouts": {"Password": [str(ALUBER)],
                                                                     "Archseries": [{"fulltext": "Despia"}]}},
                       "Fallen of Albaz": {"printouts": {"Password": [str(ALBAZ)],
                                                         "Archseries": [{"fulltext": "Albaz Dragon"}]}}},
    }  # fmt: skip
    for key, results in ask.items():
        write_raw(raw / yugipedia.raw_file(key), {"query": {"results": results}}, ["https://example.invalid/ask"])


@pytest.fixture
def raw(tmp_path, db) -> Path:
    path = tmp_path / "raw"
    write_fixture_raw(path, db)
    return path


# --------------------------------------------------------------------------- card mapping


def test_card_mapper(mapper, db):
    assert mapper.map(ALUBER).password == ALUBER and mapper.map(ALUBER).how == "password"
    alt = mapper.map(str(DROLL_ALT), "Droll & Lock Bird")
    assert (alt.password, alt.how, alt.source_password) == (DROLL, "alias", DROLL_ALT)
    assert mapper.map(BARREL_DRAGON_YGOPRODECK, "Barrel Dragon").password == BARREL_DRAGON
    assert mapper.map(None, "barrel  DRAGON!").how == "name"  # case / punctuation insensitive
    assert mapper.map(UNKNOWN, "Card From The Future") is None
    assert mapper.map(None, db[TOKEN].name) is None  # tokens are never matched by name
    assert {DROLL, DROLL_ALT} <= mapper.arts(DROLL_ALT)
    assert normalize_name("Cú Chulainn the Awakened") == normalize_name("Cu Chulainn the Awakened")


# --------------------------------------------------------------------------- parsers


def test_parse_md_pool(raw, mapper):
    res = ygoprodeck.parse_md_pool(read_raw(raw / ygoprodeck.RAW_FILE), mapper)
    assert res.tokens == 1 and TOKEN not in res.passwords
    assert [u["id"] for u in res.unmapped] == [UNKNOWN]
    assert res.by_name == [{"id": BARREL_DRAGON_YGOPRODECK, "name": "Barrel Dragon", "password": BARREL_DRAGON}]
    assert BARREL_DRAGON in res.passwords and ALUBER in res.passwords
    with pytest.raises(ValueError, match="'data' list"):
        ygoprodeck.parse_md_pool({"cards": []}, mapper)


def test_parse_banlist(raw, mapper):
    res = mdm.parse_banlist(read_raw(raw / mdm.BANLIST_FILE), mapper, "2026.09 MD")
    assert dict(res.banlist.limits) == {FENRIR: 1, DROLL: 2}  # the alternate artwork folds into the original
    assert res.merged == [{"konamiID": str(DROLL_ALT), "name": "Droll & Lock Bird", "password": DROLL}]
    assert [u["name"] for u in res.unmapped] == ["Made Up Card"]
    # hand-made files may use other status spellings or an integer limit; conflicts keep the stricter limit
    hand = [{"konamiID": ALUBER, "banStatus": "Semi-Limited"}, {"konamiID": ALUBER, "limit": 1},
            {"name": "Fallen of Albaz", "banStatus": "Forbidden"}, {"konamiID": MAXX_C, "banStatus": "Unlimited"}]  # fmt: skip
    res = mdm.parse_banlist(hand, mapper, "x")
    assert dict(res.banlist.limits) == {ALUBER: 1, ALBAZ: 0} and len(res.conflicts) == 1 and res.by_name
    with pytest.raises(ValueError, match="unknown banStatus"):
        mdm.parse_banlist([{"konamiID": ALUBER, "banStatus": "Sort of banned"}], mapper, "x")


def test_last_banlist_update(raw):
    update = mdm.last_banlist_update(read_raw(raw / mdm.ARTICLES_FILE))
    assert update == {"date": "2026-09-01", "title": "Master Duel: Forbidden / Limited List Update",
                      "url": "https://www.masterduelmeta.com/news/md-banlist/"}  # fmt: skip
    assert mdm.last_banlist_update([{"title": "OCG: Forbidden / Limited List", "date": "2026-09-01"}]) is None


def test_parse_top_decks_and_summaries(raw, mapper):
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    records = mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, mapper, since="2026-09-01")
    assert Counter(r.type for r in records) == {"Branded": 3, "Kashtira": 2, "Tenpai": 1, "Yubel": 1}
    yubel = next(r for r in records if r.type == "Yubel")
    assert yubel.unmapped == ("Mystery Card",)
    assert next(r for r in records if r.type == "Tenpai").weight == 0

    def problems(r):  # stand-in legality: at most one Kashtira Fenrir
        return ["over_limit"] if r.counts()[FENRIR] > 1 else []

    summaries = {s.name: s for s in mdm.summarize_types(records, problems)}
    assert [s.name for s in mdm.summarize_types(records, problems)][:2] == ["Branded", "Kashtira"]
    assert summaries["Branded"].share == pytest.approx(0.5) and summaries["Tenpai"].share == 0
    # the medoid of three Branded lists is the (newest copy of the) list that appears twice
    assert summaries["Branded"].chosen.created.startswith("2026-09-20")
    assert summaries["Kashtira"].legal == 1 and summaries["Kashtira"].chosen.counts()[FENRIR] == 1
    assert summaries["Kashtira"].problems == {"over_limit": 1}
    assert summaries["Yubel"].chosen is None and summaries["Yubel"].problems == {"unmapped_card": 1}


def test_yugipedia_relations(raw, mapper):
    parsed, unmapped = {}, {}
    for key in yugipedia.PROPERTIES:
        parsed[key], unmapped[key] = yugipedia.parse_property(read_raw(raw / yugipedia.raw_file(key)), mapper, key)
    assert parsed["archetype_support"] == {BRANDED_FUSION: ["Fallen of Albaz"]}
    assert unmapped["archetype_support"] == ["Anime Card"] and parsed["anti_support"] == {}
    rel = yugipedia.relations(parsed, pool={BRANDED_FUSION, ALUBER, ALBAZ})
    assert rel["archseries"] == {"Albaz Dragon": [ALBAZ], "Despia": [ALUBER]}
    assert rel["archseries_related"] == {}  # Branded Opening is outside this pool
    with pytest.raises(ValueError, match="results"):
        yugipedia.parse_property({"results": {}}, mapper, "archseries")


# --------------------------------------------------------------------------- fetchers (fake network)


class FakeHttp:
    def __init__(self, pages):
        self.pages, self.calls, self.urls, self.user_agent = list(pages), [], [], "test"

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        self.urls.append(url)
        return self.pages.pop(0)


def test_fetch_top_decks_stops_at_the_window(tmp_path, monkeypatch):
    monkeypatch.setattr(mdm, "DECKS_PAGE", 2)
    pages = [[{"created": "2026-09-20"}, {"created": "2026-09-10"}], [{"created": "2026-09-05"}, {"created": "2026-08-30"}]]
    http = FakeHttp(pages + [[{"created": "never fetched"}]])
    path = mdm.fetch_top_decks(tmp_path, "2026-09-01", http=http)
    assert [d["created"] for d in read_raw(path)] == ["2026-09-20", "2026-09-10", "2026-09-05"]
    assert [c[1]["page"] for c in http.calls] == [1, 2]
    assert json.loads(path.with_name(path.name + ".source.json").read_text())["since"] == "2026-09-01"


def test_fetch_cards_pages_and_splits_the_banlist(tmp_path, monkeypatch):
    monkeypatch.setattr(mdm, "CARDS_PAGE", 2)
    pages = [[{"_id": "a", "konamiID": "1", "name": "A", "banStatus": "Forbidden"}, {"_id": "b", "name": "B"}],
             [{"_id": "c", "konamiID": "3", "name": "C"}]]  # fmt: skip
    cards, banlist = mdm.fetch_cards(tmp_path, http=FakeHttp(pages))
    assert len(read_raw(cards)) == 3
    assert read_raw(banlist) == [{"konamiID": "1", "name": "A", "banStatus": "Forbidden"}]
    with pytest.raises(ValueError, match="duplicate"):
        mdm.fetch_cards(tmp_path, http=FakeHttp([[{"_id": "a"}, {"_id": "a"}], []]))


def test_yugipedia_fetch_detects_the_offset_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(yugipedia, "PARTITIONS", ("[[Card type::Spell Card]]",))
    ok = [{"query": {"results": {"A": {"printouts": {}}}, "meta": {"offset": 0}}, "query-continue-offset": 1000},
          {"query": {"results": {"B": {"printouts": {}}}, "meta": {"offset": 1000}}}]  # fmt: skip
    path = yugipedia.fetch_property(tmp_path, "archseries", http=FakeHttp(ok))
    assert set(read_raw(path)["query"]["results"]) == {"A", "B"}
    reset = [{"query": {"results": {}, "meta": {"offset": 0}}, "query-continue-offset": 1000},
             {"query": {"results": {}, "meta": {"offset": 0}}}]  # fmt: skip
    with pytest.raises(FetchError, match="ignored offset"):
        yugipedia.fetch_property(tmp_path, "archseries", http=FakeHttp(reset))


def test_http_retries_and_rate_limit(monkeypatch):
    import ygorl.data.fetch as fetch

    sleeps, calls = [], []
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)

    def urlopen(req, timeout):
        calls.append(req.full_url)
        assert "ygorl" in req.get_header("User-agent")
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "7"}, None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    http = Http(0.0)
    assert http.get_json("https://example.invalid/api", {"format": "master duel"}) == {"ok": True}
    assert calls[0].endswith("?format=master%20duel") and 7.0 in sleeps and http.urls == calls[1:]

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)))  # fmt: skip
    with pytest.raises(FetchError, match="HTTP 404"):
        http.get_json("https://example.invalid/missing")


def test_provenance_marks_hand_edited_files(tmp_path):
    path = write_raw(tmp_path / "src" / "x.json", {"a": 1}, ["https://example.invalid/x"])
    assert provenance(path).get("manual") is None and provenance(path)["urls"] == ["https://example.invalid/x"]
    path.write_text('{"a": 2}', encoding="utf-8")
    assert provenance(path)["manual"] is True
    hand = tmp_path / "src" / "hand.json"
    hand.write_text("[]", encoding="utf-8")
    assert provenance(hand)["manual"] is True and provenance(hand)["retrieved"] is None


# --------------------------------------------------------------------------- the offline build


def _build(tmp_path, raw, db, **kw):
    opts = BuildOptions(version="md-2026-09", root=tmp_path / "envs", raw_dir=raw, offline=True, **kw)
    return build(opts, cards=db, log=lambda _msg: None)


def test_offline_build(tmp_path, raw, db):
    result = _build(tmp_path, raw, db)
    env = load_environment(result.path, cards=db)  # T0.4 validation, meta decks legal
    assert env.version == "md-2026-09" and env.banlist.name == "2026.09 MD"
    assert {m.name: m.share for m in env.meta_decks} == {"Branded": 0.5, "Kashtira": 0.3333}
    assert env.meta_deck("Kashtira").deck.counts()[FENRIR] == 1
    assert env.meta_deck("Branded").deck.counts()[MAXX_C] == 2
    assert BARREL_DRAGON in env.card_pool and TOKEN not in env.card_pool and UNKNOWN not in env.card_pool
    assert dict(env.banlist.limits) == {FENRIR: 1, DROLL: 2}
    s = result.stats
    assert s["pool_unmapped"][0]["id"] == UNKNOWN and s["banlist_unmapped"][0]["name"] == "Made Up Card"
    assert s["meta_lists"] == 7 and [n for n, *_ in s["meta_skipped"]] == ["Yubel"]
    assert s["meta_unmapped"] == {"Mystery Card": 1} and s["review"] == "pending"
    assert any("not reviewed" in w for w in result.warnings)
    manifest = env.manifest
    assert manifest["sources"]["build"]["since"] == "2026-09-01"  # from the banlist article
    assert manifest["sources"]["raw"]["ygoprodeck/cardinfo.json"]["requests"] == 1
    rel = json.loads((result.path / "relations.json").read_text(encoding="utf-8"))
    assert rel["archetype_support"] == {"Fallen of Albaz": [BRANDED_FUSION]}
    report = (result.path / REVIEW_DIR / "report.md").read_text(encoding="utf-8")
    assert "待校对" in report and "Made Up Card" in report and f"| {FENRIR} | Kashtira Fenrir | 限制 |" in report
    assert "#source https://www.masterduelmeta.com/decks/branded/2026-09-20/" in (
        result.path / "meta" / "branded.ydk").read_text(encoding="utf-8")  # fmt: skip


def test_rebuild_applies_overrides_and_keeps_review_and_options(tmp_path, raw, db):
    first = _build(tmp_path, raw, db, since="2026-09-10", reviewed_by="alice")
    env = load_environment(first.path)
    assert env.manifest["review"]["banlist"]["by"] == "alice"
    assert env.manifest["sources"]["build"]["since"] == "2026-09-10"
    # same inputs: the review status and the window survive a rebuild without flags
    again = _build(tmp_path, raw, db)
    assert again.stats["review"] == "reviewed"
    assert load_environment(again.path).manifest["sources"]["build"]["since"] == "2026-09-10"
    # a correction lifts Kashtira Fenrir: the banlist changes, so it needs a new review
    (first.path / REVIEW_DIR / OVERRIDES).write_text(f"{FENRIR} 3 --not limited in game\n{ALUBER} 2\n", encoding="utf-8")
    third = _build(tmp_path, raw, db)
    env = load_environment(third.path, cards=db)
    assert dict(env.banlist.limits) == {DROLL: 2, ALUBER: 2}
    assert third.stats["review"] == "pending" and third.stats["banlist_overrides"] == 2
    assert "人工修正" in (third.path / REVIEW_DIR / "report.md").read_text(encoding="utf-8")


def test_offline_build_needs_raw_files(tmp_path, raw, db):
    (raw / mdm.BANLIST_FILE).unlink()
    with pytest.raises(BuildError, match="raw file missing"):
        _build(tmp_path, raw, db)
    with pytest.raises(BuildError, match="md-YYYY-MM"):
        build(BuildOptions(version="tcg-2026-09", root=tmp_path, offline=True), cards=db)


def test_hand_replaced_raw_file(tmp_path, raw, db):
    """A source that broke can be replaced by a hand-made file with the documented minimal keys."""
    (raw / mdm.BANLIST_FILE).write_text(json.dumps([{"name": "Kashtira Fenrir", "banStatus": "Forbidden"}]), encoding="utf-8")
    (raw / mdm.BANLIST_FILE).with_name("banlist.json.source.json").unlink()
    result = _build(tmp_path, raw, db, relations=False)
    env = load_environment(result.path, cards=db)
    assert dict(env.banlist.limits) == {FENRIR: 0} and "Kashtira" not in {m.name for m in env.meta_decks}
    assert env.manifest["sources"]["raw"]["masterduelmeta/banlist.json"]["manual"] is True
    assert not (result.path / "relations.json").exists()


# --------------------------------------------------------------------------- command line and snapshot


def test_cli_env_build_and_check(tmp_path, raw, capsys):
    code = main(["env", "build", "md-2026-09", "--root", str(tmp_path), "--raw", str(raw), "--offline"])
    out = capsys.readouterr().out
    assert code == 0 and "meta       2 decks" in out and "1 limited" in out
    assert main(["env", "check", str(tmp_path / "md-2026-09")]) == 0
    assert "Branded" in capsys.readouterr().out
    assert main(["env", "build", "tcg-2026-09", "--root", str(tmp_path), "--offline"]) == 2
    assert "md-YYYY-MM" in capsys.readouterr().err
    assert main(["env", "build", "md-2026-09", "--offline", "--refresh"]) == 2


def test_snapshot_is_a_valid_environment(db):
    """The committed md-2026-09 snapshot passes the T0.4 validation, meta decks included."""
    env = load_environment(SNAPSHOT, cards=db)
    assert env.format == "md" and len(env.card_pool) > 13_000
    assert 10 <= len(env.meta_decks) <= 20 and 0.5 < sum(m.share for m in env.meta_decks) <= 1
    assert len(env.banlist.limits) > 150
    assert env.manifest["sources"]["build"]["since"] == "2026-09-04"


# --------------------------------------------------------------------------- network (opt-in)

network = pytest.mark.skipif(os.environ.get("YGORL_NETWORK_TESTS") != "1", reason="set YGORL_NETWORK_TESTS=1")


@network
def test_network_masterduelmeta_articles(tmp_path):
    update = mdm.last_banlist_update(read_raw(mdm.fetch_articles(tmp_path)))
    assert update is not None and update["date"] >= "2026-01-01"


@network
def test_network_ygoprodeck_card(mapper):
    data = Http(ygoprodeck.REQUEST_INTERVAL).get_json(ygoprodeck.API, {"id": ALUBER})
    assert ygoprodeck.parse_md_pool(data, mapper).passwords == {ALUBER}
