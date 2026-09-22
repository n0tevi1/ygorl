"""Tests for deck legality checks (T1.3)."""

import pytest

from ygorl.cards.cdb import Card
from ygorl.cards.lflist import Banlist
from ygorl.cards.legality import DeckRules, IllegalDeck, check_deck, validate_deck
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

MONSTER = C.TYPE_MONSTER | C.TYPE_EFFECT


def mk(pw, name, type_=MONSTER, alias=0):
    return Card(pw, name, "", ("",) * 16, alias, 3, (), type_, 0, 0, 4, 0, 0, C.RACE_DRAGON, C.ATTRIBUTE_DARK, 0, 0)


CARDS = {c.password: c for c in [mk(i, f"Filler {i}") for i in range(100, 160)]}
CARDS.update(
    {
        1: mk(1, "Ash Blossom"),
        2: mk(2, "Forbidden One"),
        3: mk(3, "Semi Card"),
        4: mk(4, "Ash Blossom", alias=1),  # alternate artwork of 1
        10: mk(10, "Link Boss", C.TYPE_MONSTER | C.TYPE_LINK | C.TYPE_EFFECT),
        11: mk(11, "Xyz Boss", C.TYPE_MONSTER | C.TYPE_XYZ),
        12: mk(12, "Token", C.TYPE_MONSTER | C.TYPE_TOKEN),
        13: mk(13, "Spell", C.TYPE_SPELL),
    }
)
BANLIST = Banlist("test", {1: 1, 2: 0, 3: 2})
POOL = frozenset(CARDS) - {159}


def fillers(n, start=100):
    return tuple(start + i // 3 for i in range(n))  # 3 copies each


def deck(main=None, extra=(10, 11), side=()):
    return Deck(fillers(40) if main is None else tuple(main), tuple(extra), tuple(side))


def codes(violations):
    return [v.code for v in violations]


def test_legal_deck():
    assert validate_deck(deck(), cards=CARDS, banlist=BANLIST, pool=POOL) == []
    check_deck(deck(), cards=CARDS, banlist=BANLIST, pool=POOL)  # does not raise


@pytest.mark.parametrize("n, code", [(39, "main_too_small"), (61, "main_too_large")])
def test_main_deck_size(n, code):
    main = [100 + i // 3 for i in range(n)]
    vs = validate_deck(deck(main), cards=CARDS, banlist=BANLIST, pool=None)
    assert code in codes(vs)
    msg = next(v.message for v in vs if v.code == code)
    assert str(n) in msg and ("40" in msg or "60" in msg)


def test_extra_and_side_size():
    extra = [10] * 3 + [11] * 3 + list(range(100, 110))  # 16 cards (fillers are not extra-deck monsters)
    vs = validate_deck(deck(extra=extra, side=[13] * 3 + list(fillers(13, 140))), cards=CARDS, banlist=BANLIST)
    assert "extra_too_large" in codes(vs) and "side_too_large" in codes(vs)


def test_copy_limits_count_main_extra_side_and_alternate_arts():
    main = list(fillers(37)) + [1, 4, 3]
    vs = validate_deck(deck(main, side=[3, 3]), cards=CARDS, banlist=BANLIST)
    by_code = {v.password: v for v in vs}
    assert by_code[1].code == "over_limit" and "Ash Blossom (1)" in by_code[1].message
    assert "2 copies" in by_code[1].message and "limited" in by_code[1].message
    assert by_code[3].code == "over_limit" and "3 copies" in by_code[3].message and "semi-limited" in by_code[3].message


def test_forbidden_card():
    vs = validate_deck(deck(list(fillers(39)) + [2]), cards=CARDS, banlist=BANLIST)
    assert codes(vs) == ["forbidden"]
    assert "Forbidden One (2) is forbidden" in vs[0].message


def test_more_than_three_without_banlist():
    vs = validate_deck(deck([100] * 4 + list(fillers(36, 110))), cards=CARDS, banlist=None)
    assert codes(vs) == ["over_limit"] and "4 copies" in vs[0].message and "maximum 3" in vs[0].message


def test_whitelist_banlist():
    wl = Banlist("wl", {p: 3 for p in CARDS if p != 105}, whitelist=True)
    vs = validate_deck(deck(), cards=CARDS, banlist=wl)
    assert codes(vs) == ["forbidden"] and vs[0].password == 105


def test_unknown_card_and_pool():
    vs = validate_deck(deck(list(fillers(38)) + [159, 999]), cards=CARDS, banlist=BANLIST, pool=POOL)
    assert sorted(codes(vs)) == ["not_in_pool", "unknown_card"]
    assert "999" in next(v.message for v in vs if v.code == "unknown_card")


def test_alternate_art_counts_as_pool_member():
    pool = POOL - {4}
    vs = validate_deck(deck(list(fillers(39)) + [4]), cards=CARDS, banlist=BANLIST, pool=pool)
    assert vs == []


def test_wrong_deck_section():
    vs = validate_deck(deck(list(fillers(39)) + [10], extra=(11, 13)), cards=CARDS, banlist=BANLIST)
    assert sorted(codes(vs)) == ["extra_in_main", "main_in_extra"]


def test_tokens_not_allowed():
    vs = validate_deck(deck(list(fillers(39)) + [12]), cards=CARDS, banlist=BANLIST)
    assert "token" in codes(vs)


def test_custom_rules():
    rules = DeckRules(main_min=20, main_max=30, extra_max=5, side_max=0, max_copies=1)
    vs = validate_deck(deck([100 + i for i in range(20)], extra=(10,)), cards=CARDS, banlist=None, rules=rules)
    assert vs == []
    vs = validate_deck(deck([100 + i for i in range(19)] + [100], extra=(10,)), cards=CARDS, banlist=None, rules=rules)
    assert codes(vs) == ["over_limit"]


def test_check_deck_raises_with_all_messages():
    with pytest.raises(IllegalDeck) as exc:
        check_deck(deck(list(fillers(38)) + [2]), cards=CARDS, banlist=BANLIST)
    text = str(exc.value)
    assert "forbidden" in text and "39" in text
    assert len(exc.value.violations) == 2


def test_environment_validates_meta_decks(tmp_path):
    """Environment exposes a helper that checks a deck against its own pool/banlist/rules."""
    from tests.test_environment import make_env
    from ygorl.data import load_environment

    env = load_environment(make_env(tmp_path))
    main = env.meta_decks[0].deck
    fake_cards = {p: mk(p, f"C{p}", C.TYPE_MONSTER | (C.TYPE_LINK if p >= 3000 else 0)) for p in env.card_pool}
    vs = env.validate_deck(main, cards=fake_cards)
    assert sorted(codes(vs)) == ["forbidden", "over_limit"]  # fixture banlist: 1001 limited, 1002 forbidden
