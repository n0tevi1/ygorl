"""tools/crosscheck_banlist.py on a small offline fixture (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ygorl.data.cardmap import CardMapper
from ygorl.engine import constants as C

TOOL = Path(__file__).resolve().parents[1] / "tools" / "crosscheck_banlist.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("crosscheck_banlist", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _card(name: str, alias: int = 0, type_: int = C.TYPE_MONSTER) -> SimpleNamespace:
    return SimpleNamespace(name=name, alias=alias, type=type_)


CARDS = {
    1000: _card("Alpha"),
    1001: _card("Alpha", alias=1000),  # alternate artwork
    2000: _card("Maliss <Q> Red Ransom"),
    3000: _card("Beta"),
    4000: _card("Gamma"),
    5000: _card("Delta"),
}

WIKITEXT = """{{Infobox Limited list
| effective_date = September 3, 2026
| medium         = Yu-Gi-Oh! Master Duel
| prev           = August 2026 Lists (Master Duel)
| next           = %s
}}

{{Master Duel Limitation status list
| date      = September 3, 2026
| cards     =
Alpha; Forbidden
Maliss ＜Q＞ Red Ransom; Forbidden
Beta; Limited; Semi-Limited
Gamma; Semi-Limited
Delta; Unlimited; Limited
}}
"""


def _write(tmp_path: Path, banlist: str, ygoprodeck: list, next_list: str = "", newest: str = "2026-09-03") -> Path:
    env = tmp_path / "envs" / "md-2026-09"
    env.mkdir(parents=True)
    (env / "banlist.lflist.conf").write_text("!2026.09 MD\n" + banlist)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ygoprodeck-md.json").write_text(json.dumps(ygoprodeck))
    dates = [{"date": "2026-08-05", "type": "Master Duel"}, {"date": newest, "type": "Master Duel"},
             {"date": "2026-10-01", "type": "OCG"}]  # fmt: skip
    (raw / "ygoprodeck-md-dates.json").write_text(json.dumps(dates))
    (raw / "yugipedia-md.json").write_text(json.dumps({"parse": {"wikitext": WIKITEXT % next_list}}))
    return tmp_path


def _ypd(pw: int, name: str, status: str) -> dict:
    return {"name": name, "id": pw, "status_text": status}


AGREEING = [_ypd(1001, "Alpha", "Forbidden"), _ypd(2000, "Maliss <Q> Red Ransom", "Forbidden"),
            _ypd(3000, "Beta", "Limited"), _ypd(4000, "Gamma", "Semi-Limited")]  # fmt: skip


def _run(tool, root: Path, *extra: str) -> int:
    argv = ["md-2026-09", "--root", str(root / "envs"), "--from", str(root / "raw"), "--date", "2026-09-03", *extra]
    return tool.main(argv, cards=CARDS)


def test_parsers_and_mapping(tool):
    entries, info = tool.parse_yugipedia(WIKITEXT % "")
    assert [(e.name, e.limit) for e in entries] == [("Alpha", 0), ("Maliss ＜Q＞ Red Ransom", 0), ("Beta", 1),
                                                    ("Gamma", 2), ("Delta", 3)]  # fmt: skip
    assert info == {"effective_date": "September 3, 2026", "prev": "August 2026 Lists (Master Duel)", "next": ""}
    mapper = CardMapper(CARDS)
    got = tool.map_entries(entries, mapper)
    assert got.limits == {1000: 0, 2000: 0, 3000: 1, 4000: 2}  # full-width brackets map by loose name; Delta dropped
    ypd = tool.map_entries(tool.parse_ygoprodeck(AGREEING + [_ypd(9999, "Nobody", "Limited")]), mapper)
    assert ypd.limits == got.limits  # the alternate artwork 1001 folds to 1000
    assert [e.name for e in ypd.unmapped] == ["Nobody"]
    with pytest.raises(ValueError, match="unknown status"):
        tool.status_limit("Banned")
    assert tool.diff({1000: 0, 3000: 2}, {"a": {1000: 0, 3000: 2}, "b": {1000: 0}}) == [(3000, 2, {"a": 2, "b": 3})]


def test_agreement_exits_zero(tool, tmp_path, capsys):
    root = _write(tmp_path, "1000 0\n2000 0\n3000 1\n4000 2\n", AGREEING)
    assert _run(tool, root) == 0
    assert "0 disagreements" in capsys.readouterr().out


def test_disagreements_and_newer_lists_are_reported(tool, tmp_path, capsys):
    ypd = [_ypd(1001, "Alpha", "Forbidden"), _ypd(3000, "Beta", "Limited"), _ypd(4000, "Gamma", "Limited"),
           _ypd(9999, "Nobody", "Forbidden")]  # fmt: skip
    root = _write(tmp_path, "1000 0\n3000 1\n4000 2\n", ypd, next_list="October 2026 Lists (Master Duel)",
                  newest="2026-10-01")  # fmt: skip
    assert _run(tool, root) == 1
    out = capsys.readouterr().out
    assert "| 2000 | Maliss <Q> Red Ransom | Unlimited | Unlimited | Forbidden |" in out
    assert "| 4000 | Gamma | Semi-Limited | Limited | Semi-Limited |" in out
    assert "| 1000 |" not in out and "| 3000 |" not in out
    assert "UNMAPPED: Nobody (9999, Forbidden)" in out
    assert "NEWER LIST on Yugipedia: October 2026 Lists (Master Duel)" in out
    assert "NEWER LIST on YGOPRODECK: 2026-10-01 > 2026-09-03" in out
    assert "2 disagreements" in out
