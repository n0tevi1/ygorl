"""A lightweight Lua reader for mining card scripts (no Lua runtime).

Card scripts are regular enough that we only need: a tokenizer, an expression
parser (Lua 5.4 precedence), block matching to find where functions end, and a
constant folder. Statements are *not* parsed; callers locate interesting
constructs (``Duel.SelectMatchingCard(...)``, ``e1:SetCategory(...)``,
``return <expr>``) by token patterns and parse the expressions there.

AST nodes are small frozen dataclasses so they can be compared, hashed and
pickled (the graph builder parses scripts in worker processes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

# ---------------------------------------------------------------- tokenizer

KEYWORDS = frozenset(
    "and break do else elseif end false for function goto if in local nil not or repeat return then true until while".split()
)

_TOKEN_RE = re.compile(
    r"""(?P<ws>\s+)
    |(?P<comment>--\[(?P<ceq>=*)\[.*?\](?P=ceq)\]|--[^\n]*)
    |(?P<lstring>\[(?P<seq>=*)\[\n?(?P<lbody>.*?)\](?P=seq)\])
    |(?P<string>"(?P<dq>(?:\\.|[^"\\\n])*)"|'(?P<sq>(?:\\.|[^'\\\n])*)')
    |(?P<number>0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)
    |(?P<name>[A-Za-z_]\w*)
    |(?P<op>\.\.\.|\.\.|==|~=|<=|>=|<<|>>|//|::|[-+*/%^\#&~|<>=(){}\[\];:,.])
    |(?P<bad>.)""",
    re.S | re.X,
)

Token = tuple[str, str]  # (kind, value); kind in name/number/string/op


class LuaSyntaxError(ValueError):
    pass


def tokenize(src: str) -> list[Token]:
    """Split Lua source into ``(kind, value)`` tokens; comments and whitespace are dropped.

    String tokens carry their raw contents (escapes are not decoded).
    """
    out: list[Token] = []
    append = out.append
    for m in _TOKEN_RE.finditer(src):
        kind = m.lastgroup
        if kind in ("ws", "comment", "ceq"):
            continue
        if kind == "name":
            append(("name", m.group("name")))
        elif kind == "op":
            append(("op", m.group("op")))
        elif kind == "number":
            append(("number", m.group("number")))
        elif kind in ("string", "dq", "sq"):
            dq = m.group("dq")
            append(("string", dq if dq is not None else m.group("sq")))
        elif kind in ("lstring", "seq", "lbody"):
            append(("string", m.group("lbody")))
        elif kind == "bad":
            continue  # stray characters (e.g. a BOM) are ignored
        else:  # pragma: no cover - every alternative is handled above
            raise AssertionError(kind)
    return out


# ---------------------------------------------------------------- AST


@dataclass(frozen=True, slots=True)
class Name:
    id: str


@dataclass(frozen=True, slots=True)
class Const:
    value: object  # int, float, str, bool or None (nil)
    raw: str | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class Index:
    obj: Node
    key: str


@dataclass(frozen=True, slots=True)
class IndexExpr:
    obj: Node
    key: Node


@dataclass(frozen=True, slots=True)
class Call:
    func: Node
    args: tuple


@dataclass(frozen=True, slots=True)
class Method:
    obj: Node
    name: str
    args: tuple


@dataclass(frozen=True, slots=True)
class BinOp:
    op: str
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class UnOp:
    op: str
    operand: Node


@dataclass(frozen=True, slots=True)
class Table:
    items: tuple  # ((key, value), ...); key is a str (name key), a Node ([expr] key) or None

    def get(self, key: str, default=None):
        for k, v in self.items:
            if k == key:
                return v
        return default

    def positional(self) -> list:
        return [v for k, v in self.items if k is None]


@dataclass(frozen=True, slots=True)
class FuncExpr:
    params: tuple
    body: tuple  # (first body token index, index of the closing `end`)


Node = Union[Name, Const, Index, IndexExpr, Call, Method, BinOp, UnOp, Table, FuncExpr]

# (left, right) binding priorities, as in lparser.c
_BINARY = {
    "or": (1, 1), "and": (2, 2),
    "<": (3, 3), ">": (3, 3), "<=": (3, 3), ">=": (3, 3), "~=": (3, 3), "==": (3, 3),
    "|": (4, 4), "~": (5, 5), "&": (6, 6), "<<": (7, 7), ">>": (7, 7),
    "..": (9, 8), "+": (10, 10), "-": (10, 10),
    "*": (11, 11), "/": (11, 11), "//": (11, 11), "%": (11, 11),
    "^": (14, 13),
}  # fmt: skip
_UNARY_PRIORITY = 12
_UNARY = frozenset(("not", "-", "#", "~"))
_BLOCK_OPENERS = frozenset(("function", "if", "do", "repeat"))
_BLOCK_CLOSERS = frozenset(("end", "until"))


def _number(raw: str) -> int | float:
    low = raw.lower()
    if low.startswith("0x"):
        return int(low, 16)
    if any(ch in low for ch in ".e"):
        return float(low)
    return int(low)


class _Parser:
    __slots__ = ("toks", "pos", "n")

    def __init__(self, toks: list[Token], pos: int) -> None:
        self.toks = toks
        self.pos = pos
        self.n = len(toks)

    def peek(self) -> Token | None:
        return self.toks[self.pos] if self.pos < self.n else None

    def is_op(self, value: str) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] == value

    def expect_op(self, value: str) -> None:
        if not self.is_op(value):
            raise LuaSyntaxError(f"expected {value!r} at token {self.pos}, got {self.peek()!r}")
        self.pos += 1

    def name(self) -> str:
        tok = self.peek()
        if tok is None or tok[0] != "name" or tok[1] in KEYWORDS:
            raise LuaSyntaxError(f"expected a name at token {self.pos}, got {tok!r}")
        self.pos += 1
        return tok[1]

    # expr := subexpr(0)
    def expr(self, limit: int = 0) -> Node:
        tok = self.peek()
        if tok is None:
            raise LuaSyntaxError("unexpected end of input")
        if tok[1] in _UNARY and (tok[0] == "op" or tok[1] == "not"):
            self.pos += 1
            left: Node = UnOp(tok[1], self.expr(_UNARY_PRIORITY))
        else:
            left = self.simple()
        while True:
            tok = self.peek()
            if tok is None or tok[0] not in ("op", "name"):
                break
            prio = _BINARY.get(tok[1])
            if prio is None or (tok[0] == "name" and tok[1] not in ("and", "or")) or prio[0] <= limit:
                break
            self.pos += 1
            left = BinOp(tok[1], left, self.expr(prio[1]))
        return left

    def simple(self) -> Node:
        kind, value = self.peek()
        if kind == "number":
            self.pos += 1
            return Const(_number(value), value)
        if kind == "string":
            self.pos += 1
            return Const(value, None)
        if kind == "name":
            if value == "nil":
                self.pos += 1
                return Const(None)
            if value in ("true", "false"):
                self.pos += 1
                return Const(value == "true")
            if value == "function":
                self.pos += 1
                return self.funcbody()
        if kind == "op":
            if value == "...":
                self.pos += 1
                return Name("...")
            if value == "{":
                return self.table()
        return self.suffixed()

    def primary(self) -> Node:
        tok = self.peek()
        if tok is not None and tok[0] == "op" and tok[1] == "(":
            self.pos += 1
            e = self.expr()
            self.expect_op(")")
            return e
        return Name(self.name())

    def suffixed(self) -> Node:
        e = self.primary()
        while True:
            tok = self.peek()
            if tok is None:
                return e
            kind, value = tok
            if kind == "op" and value == ".":
                self.pos += 1
                e = Index(e, self.name())
            elif kind == "op" and value == "[":
                self.pos += 1
                key = self.expr()
                self.expect_op("]")
                e = IndexExpr(e, key)
            elif kind == "op" and value == ":":
                self.pos += 1
                meth = self.name()
                e = Method(e, meth, self.args())
            elif (kind == "op" and value in ("(", "{")) or kind == "string":
                e = Call(e, self.args())
            else:
                return e

    def args(self) -> tuple:
        kind, value = self.peek() or ("", "")
        if kind == "string":
            self.pos += 1
            return (Const(value),)
        if kind == "op" and value == "{":
            return (self.table(),)
        self.expect_op("(")
        out = []
        if not self.is_op(")"):
            out.append(self.expr())
            while self.is_op(","):
                self.pos += 1
                out.append(self.expr())
        self.expect_op(")")
        return tuple(out)

    def table(self) -> Table:
        self.expect_op("{")
        items = []
        while not self.is_op("}"):
            tok = self.peek()
            if tok is None:
                raise LuaSyntaxError("unterminated table")
            if tok[0] == "op" and tok[1] == "[":
                self.pos += 1
                key = self.expr()
                self.expect_op("]")
                self.expect_op("=")
                items.append((key, self.expr()))
            elif (
                tok[0] == "name" and tok[1] not in KEYWORDS and self.pos + 1 < self.n
                and self.toks[self.pos + 1] == ("op", "=")
            ):  # fmt: skip
                self.pos += 2
                items.append((tok[1], self.expr()))
            else:
                items.append((None, self.expr()))
            if self.is_op(",") or self.is_op(";"):
                self.pos += 1
            elif not self.is_op("}"):
                raise LuaSyntaxError(f"expected ',' or '}}' in table at token {self.pos}")
        self.pos += 1
        return Table(tuple(items))

    def funcbody(self) -> FuncExpr:
        self.expect_op("(")
        params = []
        while not self.is_op(")"):
            if self.is_op("..."):
                self.pos += 1
                params.append("...")
            else:
                params.append(self.name())
            if self.is_op(","):
                self.pos += 1
        self.pos += 1
        start = self.pos
        end = block_end(self.toks, start)
        self.pos = end + 1
        return FuncExpr(tuple(params), (start, end))


def parse_expr(toks: list[Token], pos: int) -> tuple[Node, int]:
    """Parse one expression starting at token ``pos``; returns ``(node, next_pos)``."""
    p = _Parser(toks, pos)
    try:
        node = p.expr()
    except (TypeError, IndexError) as exc:  # peek() returned None mid-construct
        raise LuaSyntaxError(f"unexpected end of input near token {p.pos}") from exc
    return node, p.pos


def parse_args(toks: list[Token], pos: int) -> tuple[tuple, int]:
    """Parse a call's argument list (``(...)``, ``{...}`` or a string) starting at ``pos``."""
    p = _Parser(toks, pos)
    try:
        args = p.args()
    except (TypeError, IndexError) as exc:
        raise LuaSyntaxError(f"unexpected end of input near token {p.pos}") from exc
    return args, p.pos


def block_end(toks: list[Token], pos: int, depth: int = 1) -> int:
    """Index of the token closing the block that is open at ``pos`` (``end`` / ``until``)."""
    for j in range(pos, len(toks)):
        kind, value = toks[j]
        if kind != "name":
            continue
        if value in _BLOCK_OPENERS:
            depth += 1
        elif value in _BLOCK_CLOSERS:
            depth -= 1
            if depth == 0:
                return j
    raise LuaSyntaxError("unterminated block")


def returns(toks: list[Token], body: tuple[int, int]) -> list[Node]:
    """First expression of every ``return`` directly in ``body`` (nested functions skipped)."""
    out = []
    j, end = body
    while j < end:
        kind, value = toks[j]
        if kind == "name":
            if value == "function":
                # skip nested function definitions: `function name(...)` or `function(...)`
                k = j + 1
                while k < end and toks[k] != ("op", "("):
                    k += 1
                j = block_end(toks, k + 1) + 1
                continue
            if value == "return":
                nxt = toks[j + 1] if j + 1 < len(toks) else None
                if nxt is None or nxt[1] in ("end", "else", "elseif", "until", ";"):
                    j += 1
                    continue
                try:
                    node, j = parse_expr(toks, j + 1)
                except LuaSyntaxError:
                    j += 1
                    continue
                out.append(node)
                continue
        j += 1
    return out


def dotted(node: Node) -> str | None:
    """``a.b.c`` for a chain of names/indexes, else ``None``."""
    parts = []
    while isinstance(node, Index):
        parts.append(node.key)
        node = node.obj
    if not isinstance(node, Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


# ---------------------------------------------------------------- unparse (diagnostics, tests)


def _prio(node: Node) -> int:
    if isinstance(node, BinOp):
        return _BINARY[node.op][0]
    if isinstance(node, UnOp):
        return _UNARY_PRIORITY
    return 100


def unparse(node: Node) -> str:
    if isinstance(node, Name):
        return node.id
    if isinstance(node, Const):
        if node.raw is not None:
            return node.raw
        if node.value is None:
            return "nil"
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, str):
            return '"' + node.value + '"'
        return repr(node.value)
    if isinstance(node, Index):
        return f"{unparse(node.obj)}.{node.key}"
    if isinstance(node, IndexExpr):
        return f"{unparse(node.obj)}[{unparse(node.key)}]"
    if isinstance(node, Call):
        return f"{unparse(node.func)}({','.join(unparse(a) for a in node.args)})"
    if isinstance(node, Method):
        return f"{unparse(node.obj)}:{node.name}({','.join(unparse(a) for a in node.args)})"
    if isinstance(node, UnOp):
        inner = unparse(node.operand)
        if _prio(node.operand) < _UNARY_PRIORITY:
            inner = f"({inner})"
        return f"not {inner}" if node.op == "not" else f"{node.op}{inner}"
    if isinstance(node, BinOp):
        lp, rp = _BINARY[node.op]
        left, right = unparse(node.left), unparse(node.right)
        if _prio(node.left) < lp:
            left = f"({left})"
        if _prio(node.right) <= rp and not (_prio(node.right) == rp and node.op in ("..", "^")):
            if _prio(node.right) < 100:
                right = f"({right})"
        sep = f" {node.op} " if node.op in ("and", "or") else node.op
        return f"{left}{sep}{right}"
    if isinstance(node, Table):
        parts = []
        for k, v in node.items:
            if k is None:
                parts.append(unparse(v))
            elif isinstance(k, str):
                parts.append(f"{k}={unparse(v)}")
            else:
                parts.append(f"[{unparse(k)}]={unparse(v)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(node, FuncExpr):
        return f"function({','.join(node.params)}) ... end"
    raise TypeError(node)  # pragma: no cover


# ---------------------------------------------------------------- constants


def const_eval(node: Node, env: dict[str, int]) -> int | None:
    """Fold an integer constant expression (names from ``env``); ``None`` if not constant."""
    if isinstance(node, Const):
        v = node.value
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    if isinstance(node, Name):
        return env.get(node.id)
    if isinstance(node, UnOp):
        v = const_eval(node.operand, env)
        if v is None:
            return None
        if node.op == "-":
            return -v
        if node.op == "~":
            return ~v
        return None
    if isinstance(node, BinOp):
        a = const_eval(node.left, env)
        if a is None:
            return None
        b = const_eval(node.right, env)
        if b is None:
            return None
        op = node.op
        if op == "|":
            return a | b
        if op == "&":
            return a & b
        if op == "~":
            return a ^ b
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "<<":
            return a << b
        if op == ">>":
            return a >> b
        if op == "//" and b:
            return a // b
        return None
    return None


def read_constants(src: str, env: dict[str, int] | None = None) -> dict[str, int]:
    """Top-level ``NAME = <integer constant expression>`` definitions, evaluated in order.

    ``env`` supplies names defined elsewhere (e.g. constant.lua for later files);
    only the names defined in ``src`` are returned.
    """
    toks = tokenize(src)
    scope = dict(env or {})
    out: dict[str, int] = {}
    j, n = 0, len(toks)
    while j < n:
        kind, value = toks[j]
        if kind == "name" and value == "function":
            k = j + 1
            while k < n and toks[k] != ("op", "("):
                k += 1
            try:
                j = block_end(toks, k + 1) + 1
            except LuaSyntaxError:
                break
            continue
        if kind == "name" and value == "local":
            j += 2
            continue
        if (
            kind == "name" and value not in KEYWORDS and j + 1 < n and toks[j + 1] == ("op", "=")
            and (j == 0 or toks[j - 1][1] not in (".", ":", ","))
        ):  # fmt: skip
            try:
                node, k = parse_expr(toks, j + 2)
            except LuaSyntaxError:
                j += 2
                continue
            val = const_eval(node, scope)
            if val is not None:
                scope[value] = val
                out[value] = val
            j = k
            continue
        j += 1
    return out


# ---------------------------------------------------------------- scripts


@dataclass(frozen=True, slots=True)
class FunctionDef:
    name: str
    params: tuple
    body: tuple  # (first body token, closing `end` token)


class Script:
    """A tokenized script with its top-level function definitions located."""

    def __init__(self, tokens: list[Token], functions: dict[str, FunctionDef], assignments: dict[str, tuple[int, int]]):
        self.tokens = tokens
        self.functions = functions
        self._assign_spans = assignments
        self._ranges = sorted((f.body[0], f.body[1], f.name) for f in functions.values())

    @classmethod
    def parse(cls, src: str) -> Script:
        toks = tokenize(src)
        functions: dict[str, FunctionDef] = {}
        assignments: dict[str, tuple[int, int]] = {}
        j, n = 0, len(toks)
        while j < n:
            kind, value = toks[j]
            if kind == "name" and value in ("function", "local"):
                k = j + 1
                if value == "local":
                    if k < n and toks[k] == ("name", "function"):
                        k += 1
                    else:
                        j += 1
                        continue
                # function <name>{.<name>}[:<name>] ( params ) body end
                parts = []
                while k < n and toks[k][0] == "name":
                    parts.append(toks[k][1])
                    k += 1
                    if k < n and toks[k][0] == "op" and toks[k][1] in (".", ":"):
                        k += 1
                    else:
                        break
                if k < n and toks[k] == ("op", "(") and parts:
                    fn = _Parser(toks, k).funcbody()
                    functions.setdefault(".".join(parts), FunctionDef(".".join(parts), fn.params, fn.body))
                    j = fn.body[1] + 1
                    continue
                j += 1
                continue
            if kind == "name" and value not in KEYWORDS and (j == 0 or toks[j - 1][1] not in (".", ":", ",", "local")):
                # <name>{.<name>} = <expr>  at top level
                k = j
                parts = [value]
                while k + 2 < n and toks[k + 1] == ("op", ".") and toks[k + 2][0] == "name":
                    parts.append(toks[k + 2][1])
                    k += 2
                if k + 1 < n and toks[k + 1] == ("op", "=") and not (k + 2 < n and toks[k + 2] == ("op", "=")):
                    try:
                        node, end = parse_expr(toks, k + 2)
                    except LuaSyntaxError:
                        j = k + 2
                        continue
                    name = ".".join(parts)
                    if isinstance(node, FuncExpr):
                        functions.setdefault(name, FunctionDef(name, node.params, node.body))
                    else:
                        assignments[name] = (k + 2, end)
                    j = end
                    continue
                j = k + 1
                continue
            if kind == "name" and value in _BLOCK_OPENERS:
                # a top-level block (rare): skip it entirely
                try:
                    j = block_end(toks, j + 1) + 1
                except LuaSyntaxError:
                    break
                continue
            j += 1
        return cls(toks, functions, assignments)

    def returns(self, fn: FunctionDef) -> list[Node]:
        return returns(self.tokens, fn.body)

    def function_at(self, pos: int) -> str | None:
        """Name of the top-level function whose body contains token ``pos``."""
        for start, end, name in self._ranges:
            if start <= pos <= end:
                return name
            if start > pos:
                break
        return None

    def top_level_assignments(self) -> dict[str, Node]:
        return {name: parse_expr(self.tokens, span[0])[0] for name, span in self._assign_spans.items()}

    def find_calls(self, names: frozenset[str] | set[str]) -> list[tuple[int, str, tuple]]:
        """``(pos, dotted_name, args)`` for every call ``a.b(...)`` whose dotted name is in ``names``."""
        toks = self.tokens
        n = len(toks)
        heads = {name.split(".", 1)[0] for name in names}
        out = []
        for j in range(n):
            kind, value = toks[j]
            if kind != "name" or value not in heads or (j and toks[j - 1][1] in (".", ":")):
                continue
            parts = [value]
            k = j
            while k + 2 < n and toks[k + 1] == ("op", ".") and toks[k + 2][0] == "name":
                parts.append(toks[k + 2][1])
                k += 2
            name = ".".join(parts)
            if name not in names or k + 1 >= n:
                continue
            nxt = toks[k + 1]
            if not ((nxt[0] == "op" and nxt[1] in ("(", "{")) or nxt[0] == "string"):
                continue
            try:
                args, _ = parse_args(toks, k + 1)
            except LuaSyntaxError:
                continue
            out.append((j, name, args))
        return out

    def find_method_calls(self, methods: frozenset[str] | set[str]) -> list[tuple[int, str, str, tuple]]:
        """``(pos, receiver, method, args)`` for ``<name>:<method>(...)`` with a plain-name receiver."""
        toks = self.tokens
        n = len(toks)
        out = []
        for j in range(1, n - 2):
            if toks[j] != ("op", ":"):
                continue
            meth = toks[j + 1]
            if meth[0] != "name" or meth[1] not in methods:
                continue
            recv = toks[j - 1]
            if recv[0] != "name" or (j >= 2 and toks[j - 2][1] in (".", ":")):
                continue
            try:
                args, _ = parse_args(toks, j + 2)
            except LuaSyntaxError:
                continue
            out.append((j - 1, recv[1], meth[1], args))
        return out
