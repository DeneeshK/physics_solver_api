"""
graph_builder/dim.py

Dependency-free dimensional algebra + a tiny arithmetic-expression parser.

This is the heart of the graph compiler's correctness guarantee. Every
equation's right-hand and left-hand sides are parsed into an expression tree,
each leaf symbol is replaced by its registered physical DIMENSION, and the tree
is evaluated under dimensional rules:

  - a + b / a - b  → the two operands MUST have identical dimension
  - a * b          → exponents add
  - a / b          → exponents subtract
  - a ** n         → n must be a dimensionless constant; base exponents × n
  - sqrt(a)        → base exponents × 1/2
  - sin/cos/tan/exp/log/...(a) → a MUST be dimensionless; result dimensionless
  - numbers, pi, e → dimensionless

If LHS-dimension != RHS-dimension, the equation is physically malformed and the
build FAILS. This is what would have rejected the real bugs we found:

    sound_beat_frequency : f = u - v     # frequency [T-1] vs velocity [LT-1]
    thermodynamics_first_law : Q = DeltaV + W   # if DeltaV is registered as a
                                                # volume, ML2T-2 != L3 → rejected

No SymPy, no third-party deps — runs anywhere, so the compiler is testable in
isolation. The runtime executor still consumes the emitted `Eq(...)` strings via
SymPy in the user's environment; this module only governs build-time validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import re

# Canonical base dimensions. 'Theta'→K and 'mol'→N are folded on parse so the
# registry may spell temperature/amount either way.
BASE_DIMS = ("M", "L", "T", "A", "K", "N")


class DimensionError(Exception):
    """Raised when an expression is dimensionally inconsistent."""


# ── Dimension vectors ─────────────────────────────────────────────────────────
# A dimension is a dict {base: Fraction exponent}. The empty dict is
# dimensionless. We use Fractions so sqrt (×1/2) stays exact.

def parse_dim_string(s: str) -> dict:
    """Parse a dimension string like 'MLT-2', 'ML2T-3A-2', 'L', '1', '' into a
    {base: Fraction} vector. Accepts the same notation the equation graph uses."""
    if s is None:
        return {}
    s = s.strip()
    if s in ("", "1", "dimensionless", "-"):
        return {}
    # Fold alternate spellings to the canonical single-letter bases.
    s = s.replace("Theta", "K").replace("theta", "K")
    s = s.replace("mol", "N")
    s = s.replace("^", "").replace("*", "").replace(" ", "")
    out: dict[str, Fraction] = {}
    # token = base letter followed by an optional signed integer exponent
    for m in re.finditer(r"(M|L|T|A|K|N)(-?\d+)?", s):
        base = m.group(1)
        exp = Fraction(int(m.group(2))) if m.group(2) else Fraction(1)
        out[base] = out.get(base, Fraction(0)) + exp
    return {k: v for k, v in out.items() if v != 0}


def dim_to_string(d: dict) -> str:
    """Render a dimension vector back to the graph's string notation."""
    if not d:
        return "1"
    parts = []
    for base in BASE_DIMS:
        if base in d and d[base] != 0:
            e = d[base]
            es = str(e.numerator) if e.denominator == 1 else f"{e.numerator}/{e.denominator}"
            parts.append(base if e == 1 else f"{base}{es}")
    return "".join(parts)


def _mul(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + v
    return {k: v for k, v in out.items() if v != 0}


def _scale(a: dict, n: Fraction) -> dict:
    return {k: v * n for k, v in a.items() if v * n != 0}


def _equal(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if v != 0} == {k: v for k, v in b.items() if v != 0}


# ── Expression parser (recursive descent) ─────────────────────────────────────
# Grammar:
#   expr   := term (('+'|'-') term)*
#   term   := unary (('*'|'/') unary)*
#   unary  := '-' unary | power
#   power  := atom ('**' unary)?          # right-assoc
#   atom   := number | name '(' args ')' | name | '(' expr ')'

_TOKEN = re.compile(r"""
    \s*(?:
      (?P<num>\d+\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?|\.\d+)
    | (?P<pow>\*\*)
    | (?P<op>[+\-*/(),])
    | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    )
""", re.VERBOSE)

# Functions and their dimensional behavior.
_DIMLESS_FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan",
                  "sinh", "cosh", "tanh", "exp", "log", "ln"}
_PASSTHROUGH_FUNCS = {"abs", "Abs"}            # keep dimension of the argument
# sqrt handled specially (×1/2). Numeric-only constants:
_CONSTANTS = {"pi": {}, "e": {}, "E": {}}


@dataclass
class _Node:
    kind: str            # 'num' | 'name' | 'binop' | 'unary' | 'pow' | 'call'
    value: object = None
    a: "object" = None
    b: "object" = None
    args: tuple = ()


def _tokenize(s: str):
    toks, i = [], 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m or m.end() == i:
            if s[i:].strip() == "":
                break
            raise ValueError(f"cannot tokenize near: {s[i:i+20]!r}")
        i = m.end()
        if m.lastgroup == "num":
            toks.append(("num", m.group("num")))
        elif m.lastgroup == "pow":
            toks.append(("pow", "**"))
        elif m.lastgroup == "op":
            toks.append(("op", m.group("op")))
        elif m.lastgroup == "name":
            toks.append(("name", m.group("name")))
    toks.append(("end", ""))
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def eat(self, kind=None, val=None):
        t = self.toks[self.i]
        if kind and t[0] != kind:
            raise ValueError(f"expected {kind}, got {t}")
        if val and t[1] != val:
            raise ValueError(f"expected {val!r}, got {t}")
        self.i += 1
        return t

    def parse(self):
        node = self.expr()
        if self.peek()[0] != "end":
            raise ValueError(f"trailing tokens: {self.toks[self.i:]}")
        return node

    def expr(self):
        node = self.term()
        while self.peek() == ("op", "+") or self.peek() == ("op", "-"):
            op = self.eat()[1]
            node = _Node("binop", op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() == ("op", "*") or self.peek() == ("op", "/"):
            op = self.eat()[1]
            node = _Node("binop", op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.eat()
            return _Node("unary", "-", self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek() == ("pow", "**"):
            self.eat()
            node = _Node("pow", "**", node, self.unary())
        return node

    def atom(self):
        t = self.peek()
        if t == ("op", "("):
            self.eat()
            node = self.expr()
            self.eat("op", ")")
            return node
        if t[0] == "num":
            self.eat()
            return _Node("num", float(t[1]))
        if t[0] == "name":
            self.eat()
            if self.peek() == ("op", "("):
                self.eat()
                args = [self.expr()]
                while self.peek() == ("op", ","):
                    self.eat()
                    args.append(self.expr())
                self.eat("op", ")")
                return _Node("call", t[1], args=tuple(args))
            return _Node("name", t[1])
        raise ValueError(f"unexpected token {t}")


def parse_expr(s: str) -> _Node:
    return _Parser(_tokenize(s)).parse()


# ── Numeric evaluation of constant (dimensionless) sub-expressions ────────────
# Used to resolve an exponent like (1/2) or 2 or gamma-when-numeric. If the
# sub-expression references any symbol, it's not a pure constant → error.

def _eval_const(node: _Node) -> Fraction:
    if node.kind == "num":
        f = Fraction(node.value).limit_denominator(10**6)
        return f
    if node.kind == "name":
        raise DimensionError(f"exponent must be a constant, got symbol {node.value!r}")
    if node.kind == "unary":
        return -_eval_const(node.a)
    if node.kind == "binop":
        x, y = _eval_const(node.a), _eval_const(node.b)
        return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[node.value]
    if node.kind == "pow":
        return _eval_const(node.a) ** int(_eval_const(node.b))
    raise DimensionError(f"non-constant exponent: {node.kind}")


# ── Dimension evaluation over the tree ────────────────────────────────────────

def eval_dim(node: _Node, var_dims: dict, *, where: str = "") -> dict:
    """Evaluate the dimension of an expression tree given {symbol: dim-vector}."""
    k = node.kind
    if k == "num":
        return {}
    if k == "name":
        nm = node.value
        if nm in _CONSTANTS:
            return {}
        if nm not in var_dims:
            raise DimensionError(f"{where}: symbol {nm!r} not in registry")
        return var_dims[nm]
    if k == "unary":
        return eval_dim(node.a, var_dims, where=where)
    if k == "binop":
        da = eval_dim(node.a, var_dims, where=where)
        db = eval_dim(node.b, var_dims, where=where)
        if node.value in "+-":
            if not _equal(da, db):
                raise DimensionError(
                    f"{where}: cannot add/subtract incompatible dimensions "
                    f"[{dim_to_string(da)}] {node.value} [{dim_to_string(db)}]")
            return da
        if node.value == "*":
            return _mul(da, db)
        if node.value == "/":
            return _mul(da, _scale(db, Fraction(-1)))
    if k == "pow":
        base = eval_dim(node.a, var_dims, where=where)
        exp = _eval_const(node.b)
        return _scale(base, exp)
    if k == "call":
        fn = node.value
        if fn == "sqrt":
            base = eval_dim(node.args[0], var_dims, where=where)
            return _scale(base, Fraction(1, 2))
        if fn in _PASSTHROUGH_FUNCS:
            return eval_dim(node.args[0], var_dims, where=where)
        if fn in _DIMLESS_FUNCS:
            for arg in node.args:
                d = eval_dim(arg, var_dims, where=where)
                if d:
                    raise DimensionError(
                        f"{where}: {fn}() requires a dimensionless argument, "
                        f"got [{dim_to_string(d)}]")
            return {}
        raise DimensionError(f"{where}: unknown function {fn!r}")
    raise DimensionError(f"{where}: cannot evaluate node {k}")


def symbols_in(node: _Node) -> set:
    """All identifier names used in an expression (excluding functions/consts)."""
    out = set()
    def walk(n):
        if n.kind == "name" and n.value not in _CONSTANTS:
            out.add(n.value)
        elif n.kind in ("unary",):
            walk(n.a)
        elif n.kind in ("binop", "pow"):
            walk(n.a); walk(n.b)
        elif n.kind == "call":
            for a in n.args:
                walk(a)
    walk(node)
    return out


def check_equation_dims(expr: str, var_dims: dict, *, eq_id: str = "") -> tuple[dict, dict]:
    """Parse 'LHS = RHS', verify both sides share a dimension. Returns (lhs_dim,
    rhs_dim). Raises DimensionError on mismatch."""
    if expr.count("=") != 1:
        raise DimensionError(f"{eq_id}: equation must contain exactly one '='")
    lhs_s, rhs_s = (p.strip() for p in expr.split("="))
    lhs = eval_dim(parse_expr(lhs_s), var_dims, where=f"{eq_id} LHS")
    rhs = eval_dim(parse_expr(rhs_s), var_dims, where=f"{eq_id} RHS")
    if not _equal(lhs, rhs):
        raise DimensionError(
            f"{eq_id}: LHS [{dim_to_string(lhs)}] != RHS [{dim_to_string(rhs)}]  "
            f"({lhs_s} = {rhs_s})")
    return lhs, rhs
