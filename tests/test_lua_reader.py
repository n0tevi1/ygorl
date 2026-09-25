"""Tests for the lightweight Lua reader used by the script miner (T5.3)."""

import pytest

from ygorl.build import lua
from ygorl.build.lua import BinOp, Call, Const, FuncExpr, Index, Method, Name, Table, UnOp


def test_tokenize_strips_comments_and_long_strings():
    src = """--[[ block
    comment ]] local x = 0x1F --line comment
    --[==[ another ]==]
    y = [[long
    string]] .. "a\\"b" """
    toks = lua.tokenize(src)
    assert [t[1] for t in toks] == ["local", "x", "=", "0x1F", "y", "=", "long\n    string", "..", 'a\\"b']
    assert [t[0] for t in toks][:4] == ["name", "name", "op", "number"]


def test_parse_filter_expression():
    toks = lua.tokenize(
        "c:IsSetCard(SET_SNAKE_EYE) and not c:IsCode(id) and (c:IsLevel(1) or c:IsRace(RACE_FIEND|RACE_DRAGON))"
    )
    expr, end = lua.parse_expr(toks, 0)
    assert end == len(toks)
    assert isinstance(expr, BinOp) and expr.op == "and"
    left, right = expr.left, expr.right
    # `and` is left-associative
    assert isinstance(left, BinOp) and left.op == "and"
    assert left.left == Method(Name("c"), "IsSetCard", (Name("SET_SNAKE_EYE"),))
    assert left.right == UnOp("not", Method(Name("c"), "IsCode", (Name("id"),)))
    assert isinstance(right, BinOp) and right.op == "or"
    assert right.right == Method(Name("c"), "IsRace", (BinOp("|", Name("RACE_FIEND"), Name("RACE_DRAGON")),))


def test_operator_precedence():
    expr, _ = lua.parse_expr(lua.tokenize("a or b and c == 1 + 2 * 3"), 0)
    assert expr == BinOp(
        "or",
        Name("a"),
        BinOp("and", Name("b"), BinOp("==", Name("c"), BinOp("+", Const(1), BinOp("*", Const(2), Const(3))))),
    )
    expr, _ = lua.parse_expr(lua.tokenize("LOCATION_HAND|LOCATION_DECK&0xff"), 0)
    assert expr == BinOp("|", Name("LOCATION_HAND"), BinOp("&", Name("LOCATION_DECK"), Const(0xFF)))
    expr, _ = lua.parse_expr(lua.tokenize("-1 - -2"), 0)
    assert expr == BinOp("-", UnOp("-", Const(1)), UnOp("-", Const(2)))


def test_parse_stops_at_expression_end():
    toks = lua.tokenize("return c:IsAbleToHand() end function s.x() end")
    expr, end = lua.parse_expr(toks, 1)
    assert expr == Method(Name("c"), "IsAbleToHand", ())
    assert toks[end][1] == "end"


def test_calls_tables_and_anonymous_functions():
    src = "Fusion.CreateSummonEff({handler=c,fusfilter=aux.FilterBoolFunction(Card.IsSetCard,SET_X),[1]=2,3}, function(c) return c:IsFaceup() end)"
    toks = lua.tokenize(src)
    expr, end = lua.parse_expr(toks, 0)
    assert end == len(toks)
    assert isinstance(expr, Call) and expr.func == Index(Name("Fusion"), "CreateSummonEff")
    table, fn = expr.args
    assert isinstance(table, Table)
    assert table.get("handler") == Name("c")
    assert table.get("fusfilter") == Call(
        Index(Name("aux"), "FilterBoolFunction"), (Index(Name("Card"), "IsSetCard"), Name("SET_X"))
    )
    assert table.positional() == [Const(3)]
    assert isinstance(fn, FuncExpr) and fn.params == ("c",)
    assert [lua.unparse(r) for r in lua.returns(toks, fn.body)] == ["c:IsFaceup()"]
    # table-call sugar and string-call sugar
    expr, _ = lua.parse_expr(lua.tokenize('Ritual.AddProcGreater{handler=c,lv=8} f"x"'), 0)
    assert isinstance(expr, Call) and isinstance(expr.args[0], Table)


def test_dotted_name():
    assert lua.dotted(Index(Index(Name("a"), "b"), "c")) == "a.b.c"
    assert lua.dotted(Method(Name("a"), "b", ())) is None
    assert lua.dotted(Name("id")) == "id"


def test_top_level_functions_and_blocks():
    src = """
local s,id=GetID()
function s.initial_effect(c)
    local e1=Effect.CreateEffect(c)
    for i=1,2 do
        if i==1 then e1:SetRange(LOCATION_HAND) elseif i==2 then e1:SetRange(LOCATION_GRAVE) else end
    end
    while false do end
    repeat local x=1 until true
    e1:SetTarget(function(e,tp) return true end)
end
local function helper(c,tp)
    return c:IsFaceup()
end
s.listed_names={id}
function s.filter(c)
    if c:IsLocation(LOCATION_GRAVE) then return c:IsMonster() end
    return c:IsSetCard(0x10)
        and c:IsAbleToHand()
end
"""
    script = lua.Script.parse(src)
    assert list(script.functions) == ["s.initial_effect", "helper", "s.filter"]
    f = script.functions["s.filter"]
    assert f.params == ("c",)
    assert [lua.unparse(r) for r in script.returns(f)] == ["c:IsMonster()", "c:IsSetCard(0x10) and c:IsAbleToHand()"]
    assert script.function_at(script.functions["helper"].body[0]) == "helper"
    # the anonymous function inside initial_effect does not end the enclosing function
    init = script.functions["s.initial_effect"]
    assert script.tokens[init.body[1]][1] == "end"
    assert script.top_level_assignments() == {"s.listed_names": Table(((None, Name("id")),))}


def test_constant_evaluation():
    env = {"A": 1, "B": 4}

    def val(src):
        return lua.const_eval(lua.parse_expr(lua.tokenize(src), 0)[0], env)

    assert val("A|B") == 5
    assert val("A+B") == 5
    assert val("0x10|A<<3") == 0x18
    assert val("B&~A") == 4
    assert val("UNKNOWN|A") is None
    assert val('"x"') is None
    consts = lua.read_constants("X = 0x3\nY = X|0x10 -- c\nZ = Y + 1\nlocal w = 2\nQ = 'str'\n")
    assert consts == {"X": 3, "Y": 0x13, "Z": 0x14}


@pytest.mark.parametrize("bad", ["(a", "f(a,", "{a=", "a +"])
def test_parse_errors_raise(bad):
    with pytest.raises(lua.LuaSyntaxError):
        lua.parse_expr(lua.tokenize(bad), 0)
