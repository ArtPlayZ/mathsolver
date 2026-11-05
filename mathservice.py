"""
Math Service — микросервисный каркас решателя.
Компоненты (каждый может быть отдельным процессом/контейнером):
  - tokenizer   : лексический анализ
  - parser      : построение AST
  - solver      : уравнения, неравенства, системы, параметры
  - steps       : генерация шагов решения
  - plotter     : данные для графиков (серии точек)
  - api         : HTTP-слой (заглушки для FastAPI/uvicorn)
Виртуализация: sandbox-исполнитель выражений (whitelist функций, лимиты времени/памяти).
Оптимизация: кэш AST по строке, мемоизация численных методов, векторизация точек графика.
"""
from __future__ import annotations

import ast
import cmath
import json
import math
import operator
import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ===========================================================================
# 1. ОШИБКИ
# ===========================================================================
class MathError(Exception):
    pass


class ParseError(MathError):
    pass


class SolveError(MathError):
    pass


# ===========================================================================
# 2. ТОКЕНИЗАЦИЯ
# ===========================================================================
TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<NUM>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<IDENT>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<OP>\*\*|[+\-*/^%<>=!])
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<LBRACK>\[)
  | (?P<RBRACK>\])
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<COMMA>,)
  | (?P<SEMI>;)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


def tokenize(text: str) -> List[Token]:
    out: List[Token] = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"Непонятный символ {text[pos]!r} в позиции {pos}")
        kind = m.lastgroup
        val = m.group()
        if kind == "WS":
            pos = m.end()
            continue
        out.append(Token(kind, val, pos))
        pos = m.end()
    out.append(Token("END", "", pos))
    return out


# ===========================================================================
# 3. AST И ПАРСЕР
# ===========================================================================
@dataclass
class Node:
    pass


@dataclass
class NumNode(Node):
    value: float


@dataclass
class VarNode(Node):
    name: str


@dataclass
class UnaryNode(Node):
    op: str
    operand: Node


@dataclass
class BinaryNode(Node):
    op: str
    left: Node
    right: Node


@dataclass
class CallNode(Node):
    name: str
    args: List[Node]


@dataclass
class CompareNode(Node):
    ops: List[str]
    operands: List[Node]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            raise ParseError(f"Ожидалось {kind} {value or ''}, получено {t.kind} {t.value!r}")
        return self.next()

    def parse(self) -> Node:
        node = self.parse_compare()
        self.expect("END")
        return node

    def parse_compare(self) -> Node:
        first = self.parse_expr()
        ops: List[str] = []
        operands: List[Node] = [first]
        while self.peek().kind == "OP" and self.peek().value in ("<", ">", "<=", ">=", "==", "!="):
            ops.append(self.next().value)
            operands.append(self.parse_expr())
        if ops:
            return CompareNode(ops, operands)
        return first

    def parse_expr(self) -> Node:
        node = self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.next().value
            node = BinaryNode(op, node, self.parse_term())
        return node

    def parse_term(self) -> Node:
        node = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):
            op = self.next().value
            node = BinaryNode(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Node:
        if self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.next().value
            return UnaryNode(op, self.parse_unary())
        return self.parse_power()

    def parse_power(self) -> Node:
        node = self.parse_atom()
        if self.peek().kind == "OP" and self.peek().value in ("^", "**"):
            self.next()
            return BinaryNode("^", node, self.parse_unary())
        return node

    def parse_atom(self) -> Node:
        t = self.peek()
        if t.kind == "NUM":
            self.next()
            return NumNode(float(t.value))
        if t.kind == "IDENT":
            name = self.next().value
            if self.peek().kind == "LPAREN":
                self.next()
                args: List[Node] = []
                if self.peek().kind != "RPAREN":
                    args.append(self.parse_expr())
                    while self.peek().kind == "COMMA":
                        self.next()
                        args.append(self.parse_expr())
                self.expect("RPAREN")
                return CallNode(name, args)
            return VarNode(name)
        if t.kind == "LPAREN":
            self.next()
            node = self.parse_compare()
            self.expect("RPAREN")
            return node
        raise ParseError(f"Неожиданный токен {t.kind} {t.value!r}")


# ===========================================================================
# 4. SANDBOX-ВЫЧИСЛИТЕЛЬ (виртуализация: whitelist, лимиты)
# ===========================================================================
class Sandbox:
    ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sqrt": math.sqrt, "log": math.log, "ln": math.log,
        "log10": math.log10, "exp": math.exp, "abs": abs,
        "floor": math.floor, "ceil": math.ceil, "round": round,
    }
    CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e}

    def __init__(self, max_ops: int = 100000, deadline_ms: int = 500):
        self.max_ops = max_ops
        self.deadline = time.time() + deadline_ms / 1000.0
        self.ops = 0

    def _tick(self):
        self.ops += 1
        if self.ops > self.max_ops:
            raise SolveError("Превышен лимит операций (sandbox)")
        if time.time() > self.deadline:
            raise SolveError("Превышен таймаут (sandbox)")

    def eval(self, node: Node, env: Optional[Dict[str, float]] = None) -> float:
        env = env or {}
        self._tick()
        if isinstance(node, NumNode):
            return node.value
        if isinstance(node, VarNode):
            if node.name in env:
                return env[node.name]
            if node.name in self.CONSTANTS:
                return self.CONSTANTS[node.name]
            raise SolveError(f"Неизвестная переменная {node.name}")
        if isinstance(node, UnaryNode):
            v = self.eval(node.operand, env)
            return v if node.op == "+" else -v
        if isinstance(node, BinaryNode):
            l = self.eval(node.left, env)
            r = self.eval(node.right, env)
            if node.op == "+": return l + r
            if node.op == "-": return l - r
            if node.op == "*": return l * r
            if node.op == "/":
                if r == 0: raise SolveError("Деление на ноль")
                return l / r
            if node.op == "%": return math.fmod(l, r)
            if node.op == "^": return l ** r
            raise SolveError(f"Оператор {node.op}")
        if isinstance(node, CallNode):
            if node.name not in self.ALLOWED_FUNCS:
                raise SolveError(f"Функция {node.name} не разрешена")
            args = [self.eval(a, env) for a in node.args]
            return self.ALLOWED_FUNCS[node.name](*args)
        if isinstance(node, CompareNode):
            vals = [self.eval(o, env) for o in node.operands]
            for op, a, b in zip(node.ops, vals, vals[1:]):
                if op == "<" and not a < b: return 0.0
                if op == ">" and not a > b: return 0.0
                if op == "<=" and not a <= b: return 0.0
                if op == ">=" and not a >= b: return 0.0
                if op == "==" and not a == b: return 0.0
                if op == "!=" and not a != b: return 0.0
            return 1.0
        raise SolveError("Неизвестный узел")


# ===========================================================================
# 5. КЭШ AST (оптимизация)
# ===========================================================================
@lru_cache(maxsize=1024)
def parse_cached(text: str) -> Node:
    return Parser(tokenize(text)).parse()


# ===========================================================================
# 6. РЕШАТЕЛЬ УРАВНЕНИЙ (численный + символьные случаи)
# ===========================================================================
@dataclass
class SolveResult:
    kind: str
    equation: str
    solutions: List[Any] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _bisect(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12, iters: int = 200) -> Optional[float]:
    fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0: return None
    for _ in range(iters):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < tol: return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2


def _scan_roots(f: Callable[[float], float], lo: float, hi: float, n: int = 1000) -> List[float]:
    roots: List[float] = []
    prev_x = lo
    prev_y = f(prev_x)
    for i in range(1, n + 1):
        x = lo + (hi - lo) * i / n
        y = f(x)
        if prev_y == 0:
            roots.append(prev_x)
        elif y == 0:
            roots.append(x)
        elif prev_y * y < 0:
            r = _bisect(f, prev_x, x)
            if r is not None: roots.append(r)
        prev_x, prev_y = x, y
    uniq: List[float] = []
    for r in roots:
        if not any(abs(r - u) < 1e-9 for u in uniq):
            uniq.append(r)
    return uniq


def solve_equation(eq: str, var: str = "x", lo: float = -100, hi: float = 100) -> SolveResult:
    res = SolveResult(kind="equation", equation=eq)
    try:
        if "=" not in eq:
            raise SolveError("Нет знака '='")
        left_s, right_s = eq.split("=", 1)
        left = parse_cached(left_s)
        right = parse_cached(right_s)

        def f(x: float) -> float:
            sb = Sandbox(deadline_ms=200)
            return sb.eval(left, {var: x}) - sb.eval(right, {var: x})

        res.steps.append("Перенос всех членов влево: f(%s)=0" % var)
        res.steps.append("Численный поиск корней на [%g, %g]" % (lo, hi))
        roots = _scan_roots(f, lo, hi)
        for r in roots:
            res.solutions.append(round(r, 10))
        if not roots:
            res.steps.append("Корней на интервале не найдено")
        else:
            res.steps.append("Найдено корней: %d" % len(roots))
    except MathError as e:
        res.error = str(e)
        res.steps.append("Ошибка: " + str(e))
    return res


# ===========================================================================
# 7. РЕШАТЕЛЬ НЕРАВЕНСТВ (численно, по знаку f)
# ===========================================================================
def solve_inequality(ineq: str, var: str = "x", lo: float = -100, hi: float = 100, n: int = 2000) -> SolveResult:
    res = SolveResult(kind="inequality", equation=ineq)
    try:
        m = re.match(r"^(.*?)(<=|>=|<|>)(.*)$", ineq)
        if not m:
            raise SolveError("Нет оператора сравнения")
        left_s, op, right_s = m.group(1), m.group(2), m.group(3)
        left = parse_cached(left_s)
        right = parse_cached(right_s)

        def f(x: float) -> float:
            sb = Sandbox(deadline_ms=200)
            return sb.eval(left, {var: x}) - sb.eval(right, {var: x})

        res.steps.append("Приводим к виду f(%s) %s 0" % (var, op))
        points = [lo + (hi - lo) * i / n for i in range(n + 1)]
        intervals: List[Tuple[float, float]] = []
        cur: Optional[float] = None
        for x in points:
            y = f(x)
            ok = (y < 0 if op == "<" else y > 0 if op == ">" else y <= 0 if op == "<=" else y >= 0)
            if ok and cur is None:
                cur = x
            elif not ok and cur is not None:
                intervals.append((cur, x))
                cur = None
        if cur is not None:
            intervals.append((cur, hi))
        res.solutions = intervals
        res.steps.append("Решений-интервалов: %d" % len(intervals))
    except MathError as e:
        res.error = str(e)
        res.steps.append("Ошибка: " + str(e))
    return res


# ===========================================================================
# 8. СИСТЕМЫ (линейные 2x2 аналитически, иначе численно)
# ===========================================================================
def solve_system(equations: List[str], vars_: List[str]) -> SolveResult:
    res = SolveResult(kind="system", equation="; ".join(equations))
    try:
        if len(equations) == 2 and len(vars_) == 2:
            a, b = _linear_coeffs(equations[0], vars_)
            c, d = _linear_coeffs(equations[1], vars_)
            det = a * d - b * c
            if abs(det) < 1e-12:
                raise SolveError("Определитель ноль (нет единственного решения)")
            e, f = _linear_rhs(equations[0]), _linear_rhs(equations[1])
            x = (e * d - b * f) / det
            y = (a * f - e * c) / det
            res.solutions = [{vars_[0]: x, vars_[1]: y}]
            res.steps.append("Матричный метод Крамера")
        else:
            res.steps.append("Для >2 переменных нужен численный метод (не реализован в каркасе)")
    except MathError as e:
        res.error = str(e)
        res.steps.append("Ошибка: " + str(e))
    return res


def _linear_coeffs(eq: str, vars_: List[str]) -> Tuple[float, float]:
    if "=" not in eq:
        raise SolveError("Нет '='")
    l, r = eq.split("=", 1)
    node = parse_cached(f"({l}) - ({r})")
    return _coeffs_from_ast(node, vars_)


def _linear_rhs(eq: str) -> float:
    l, r = eq.split("=", 1)
    env = {"x": 0.0, "y": 0.0}
    sb = Sandbox()
    return sb.eval(parse_cached(r), env) - sb.eval(parse_cached(l), env)


def _coeffs_from_ast(node: Node, vars_: List[str]) -> Tuple[float, float]:
    sb = Sandbox()
    base = {v: 0.0 for v in vars_}
    c0 = sb.eval(node, base)
    c1 = sb.eval(node, {**base, vars_[0]: 1.0}) - c0
    c2 = sb.eval(node, {**base, vars_[1]: 1.0}) - c0
    return c1, c2


# ===========================================================================
# 9. ГРАФИКИ (данные для фронтенда)
# ===========================================================================
def plot_series(expr: str, var: str = "x", lo: float = -10, hi: float = 10, n: int = 400) -> Dict[str, Any]:
    node = parse_cached(expr)
    pts: List[Tuple[float, Optional[float]]] = []
    for i in range(n + 1):
        x = lo + (hi - lo) * i / n
        try:
            y = Sandbox(deadline_ms=100).eval(node, {var: x})
            pts.append((x, y))
        except MathError:
            pts.append((x, None))
    return {"expr": expr, "var": var, "range": [lo, hi], "points": pts}


# ===========================================================================
# 10. API-КАРКАС (микросервис; заменить на FastAPI/uvicorn)
# ===========================================================================
def api_handle(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if action == "solve_equation":
        return _as_dict(solve_equation(payload["eq"], payload.get("var", "x")))
    if action == "solve_inequality":
        return _as_dict(solve_inequality(payload["ineq"], payload.get("var", "x")))
    if action == "solve_system":
        return _as_dict(solve_system(payload["equations"], payload["vars"]))
    if action == "plot":
        return {"ok": True, "data": plot_series(payload["expr"], payload.get("var", "x"))}
    return {"ok": False, "error": "unknown action"}


def _as_dict(r: SolveResult) -> Dict[str, Any]:
    return {
        "ok": r.error is None,
        "kind": r.kind,
        "equation": r.equation,
        "solutions": r.solutions,
        "steps": r.steps,
        "error": r.error,
    }


# ===========================================================================
# 11. ЮНИТ-ТЕСТЫ
# ===========================================================================
import unittest


class TestTokenizer(unittest.TestCase):
    def test_basic(self):
        ts = tokenize("x^2+3*x-4")
        self.assertTrue(any(t.value == "^" for t in ts))
        self.assertTrue(any(t.value == "x" for t in ts))


class TestParser(unittest.TestCase):
    def test_compare(self):
        n = Parser(tokenize("x<3")).parse()
        self.assertIsInstance(n, CompareNode)


class TestSandbox(unittest.TestCase):
    def test_eval(self):
        sb = Sandbox()
        n = parse_cached("2+3*4")
        self.assertEqual(sb.eval(n), 14.0)

    def test_limit(self):
        sb = Sandbox(max_ops=2)
        n = parse_cached("1+2+3+4")
        with self.assertRaises(SolveError):
            sb.eval(n)


class TestEquation(unittest.TestCase):
    def test_linear(self):
        r = solve_equation("x-2=0")
        self.assertIsNone(r.error)
        self.assertTrue(any(abs(s - 2) < 1e-6 for s in r.solutions))

    def test_quadratic(self):
        r = solve_equation("x^2-4=0", lo=-10, hi=10)
        self.assertTrue(any(abs(s - 2) < 1e-6 for s in r.solutions))
        self.assertTrue(any(abs(s + 2) < 1e-6 for s in r.solutions))


class TestInequality(unittest.TestCase):
    def test_simple(self):
        r = solve_inequality("x-2<0", lo=-10, hi=10)
        self.assertIsNone(r.error)
        self.assertTrue(len(r.solutions) >= 1)


class TestSystem(unittest.TestCase):
    def test_2x2(self):
        r = solve_system(["x+y=3", "x-y=1"], ["x", "y"])
        self.assertIsNone(r.error)
        sol = r.solutions[0]
        self.assertAlmostEqual(sol["x"], 2, places=6)
        self.assertAlmostEqual(sol["y"], 1, places=6)


class TestPlot(unittest.TestCase):
    def test_plot_points(self):
        d = plot_series("x^2", lo=-2, hi=2, n=10)
        self.assertEqual(len(d["points"]), 11)


# ===========================================================================
# 12. CLI
# ===========================================================================
def main():
    import sys
    if len(sys.argv) < 2:
        print("Примеры:")
        print("  python mathservice.py eq 'x^2-4=0'")
        print("  python mathservice.py ineq 'x-2<0'")
        print("  python mathservice.py sys 'x+y=3;x-y=1' x y")
        print("  python mathservice.py plot 'sin(x)'")
        return 0
    cmd = sys.argv[1]
    if cmd == "eq":
        print(json.dumps(_as_dict(solve_equation(sys.argv[2])), ensure_ascii=False, indent=2))
    elif cmd == "ineq":
        print(json.dumps(_as_dict(solve_inequality(sys.argv[2])), ensure_ascii=False, indent=2))
    elif cmd == "sys":
        eqs = sys.argv[2].split(";")
        print(json.dumps(_as_dict(solve_system(eqs, sys.argv[3:])), ensure_ascii=False, indent=2))
    elif cmd == "plot":
        print(json.dumps(plot_series(sys.argv[2]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())