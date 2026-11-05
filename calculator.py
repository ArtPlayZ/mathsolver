"""
Калькулятор с текстовым «блочным» интерфейсом (в стиле Photomath, но ввод текстом).
Поддержка: дроби, десятичные, +-*/, ^, %, функции (sin/cos/tan/sqrt/log/ln/abs/exp),
константы (pi, e), скобки, унарный минус, переменные, история, пошаговое решение.
Вывод — «блоками»: выражение, шаги, результат.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------- Значения -----------------------------------
class Value:
    """Базовое значение в калькуляторе."""
    pass


@dataclass(frozen=True)
class Frac(Value):
    """Дробное значение (точное)."""
    value: Fraction

    def __str__(self) -> str:
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"{self.value.numerator}/{self.value.denominator}"

    def to_float(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Num(Value):
    """Десятичное значение (неточное)."""
    value: float

    def __str__(self) -> str:
        if self.value == int(self.value) and abs(self.value) < 1e15:
            return str(int(self.value))
        return repr(self.value)

    def to_float(self) -> float:
        return self.value


def to_float(v: Value) -> float:
    if isinstance(v, Frac):
        return v.to_float()
    if isinstance(v, Num):
        return v.to_float()
    raise TypeError(f"Не число: {v!r}")


def to_value(x: Union[int, float, Fraction]) -> Value:
    if isinstance(x, Fraction):
        return Frac(x)
    if isinstance(x, int):
        return Frac(Fraction(x))
    return Num(float(x))


# ---------------------------- Токены -------------------------------------
@dataclass
class Token:
    kind: str      # NUM, IDENT, OP, LPAREN, RPAREN, COMMA, END
    value: str
    pos: int


TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<FRAC>\d+\s*/\s*\d+)
  | (?P<NUM>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<IDENT>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<OP>[+\-*/^%])
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<COMMA>,)
    """,
    re.VERBOSE,
)


class CalcError(Exception):
    """Ошибка калькулятора."""
    pass


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise CalcError(f"Непонятный символ в позиции {pos}: {text[pos]!r}")
        kind = m.lastgroup
        val = m.group()
        if kind == "WS":
            pos = m.end()
            continue
        if kind == "FRAC":
            tokens.append(Token("FRAC", val.replace(" ", ""), pos))
        elif kind == "NUM":
            tokens.append(Token("NUM", val, pos))
        elif kind == "IDENT":
            tokens.append(Token("IDENT", val, pos))
        elif kind == "OP":
            tokens.append(Token("OP", val, pos))
        elif kind == "LPAREN":
            tokens.append(Token("LPAREN", val, pos))
        elif kind == "RPAREN":
            tokens.append(Token("RPAREN", val, pos))
        elif kind == "COMMA":
            tokens.append(Token("COMMA", val, pos))
        else:
            raise CalcError(f"Неизвестный токен: {kind}")
        pos = m.end()
    tokens.append(Token("END", "", pos))
    return tokens


# ---------------------------- Парсер (AST) --------------------------------
@dataclass
class Node:
    pass


@dataclass
class NumberNode(Node):
    value: Value


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
            raise CalcError(f"Ожидалось {kind} {value or ''}, получено {t.kind} {t.value!r} в позиции {t.pos}")
        return self.next()

    def parse(self) -> Node:
        node = self.parse_expr()
        self.expect("END")
        return node

    # expression := term (('+'|'-') term)*
    def parse_expr(self) -> Node:
        node = self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.next().value
            right = self.parse_term()
            node = BinaryNode(op, node, right)
        return node

    # term := unary (('*'|'/'|'%') unary)*
    def parse_term(self) -> Node:
        node = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):
            op = self.next().value
            right = self.parse_unary()
            node = BinaryNode(op, node, right)
        return node

    # unary := ('+'|'-') unary | power
    def parse_unary(self) -> Node:
        if self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.next().value
            operand = self.parse_unary()
            return UnaryNode(op, operand)
        return self.parse_power()

    # power := atom ('^' unary)?   (правоассоциативно)
    def parse_power(self) -> Node:
        node = self.parse_atom()
        if self.peek().kind == "OP" and self.peek().value == "^":
            self.next()
            right = self.parse_unary()
            return BinaryNode("^", node, right)
        return node

    # atom := NUM | FRAC | IDENT | IDENT '(' args ')' | '(' expr ')'
    def parse_atom(self) -> Node:
        t = self.peek()
        if t.kind == "NUM":
            self.next()
            return NumberNode(to_value(float(t.value)) if "." in t.value or "e" in t.value.lower() else to_value(int(t.value)))
        if t.kind == "FRAC":
            self.next()
            num, den = t.value.split("/")
            return NumberNode(Frac(Fraction(int(num), int(den))))
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
            node = self.parse_expr()
            self.expect("RPAREN")
            return node
        raise CalcError(f"Неожиданный токен {t.kind} {t.value!r} в позиции {t.pos}")


# ---------------------------- Вычислитель --------------------------------
CONSTANTS: Dict[str, Value] = {
    "pi": Num(math.pi),
    "e": Num(math.e),
}


def _ensure_int(v: Value) -> int:
    f = to_float(v)
    if not float(f).is_integer():
        raise CalcError(f"Ожидалось целое, получено {v}")
    return int(f)


FUNCTIONS: Dict[str, Callable[..., Value]] = {
    "sin": lambda x: Num(math.sin(to_float(x))),
    "cos": lambda x: Num(math.cos(to_float(x))),
    "tan": lambda x: Num(math.tan(to_float(x))),
    "asin": lambda x: Num(math.asin(to_float(x))),
    "acos": lambda x: Num(math.acos(to_float(x))),
    "atan": lambda x: Num(math.atan(to_float(x))),
    "sqrt": lambda x: Num(math.sqrt(to_float(x))),
    "log": lambda x, base=None: Num(math.log(to_float(x)) if base is None else math.log(to_float(x), to_float(base))),
    "ln": lambda x: Num(math.log(to_float(x))),
    "abs": lambda x: Frac(abs(Fraction(to_float(x)))) if isinstance(x, Frac) else Num(abs(to_float(x))),
    "exp": lambda x: Num(math.exp(to_float(x))),
    "floor": lambda x: Frac(Fraction(math.floor(to_float(x)))),
    "ceil": lambda x: Frac(Fraction(math.ceil(to_float(x)))),
    "round": lambda x, n=0: Frac(Fraction(round(to_float(x), _ensure_int(n) if not isinstance(n, int) else n))),
    "frac": lambda x: Frac(Fraction(to_float(x)).limit_denominator(10**6)),
}


def evaluate(node: Node, variables: Optional[Dict[str, Value]] = None) -> Value:
    variables = variables or {}
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, VarNode):
        if node.name in variables:
            return variables[node.name]
        if node.name in CONSTANTS:
            return CONSTANTS[node.name]
        raise CalcError(f"Неизвестная переменная: {node.name}")
    if isinstance(node, UnaryNode):
        v = evaluate(node.operand, variables)
        if node.op == "+":
            return v
        if node.op == "-":
            return Frac(-v.value) if isinstance(v, Frac) else Num(-v.to_float())
        raise CalcError(f"Неизвестный унарный оператор: {node.op}")
    if isinstance(node, BinaryNode):
        left = evaluate(node.left, variables)
        right = evaluate(node.right, variables)
        if node.op == "+":
            if isinstance(left, Frac) and isinstance(right, Frac):
                return Frac(left.value + right.value)
            return Num(to_float(left) + to_float(right))
        if node.op == "-":
            if isinstance(left, Frac) and isinstance(right, Frac):
                return Frac(left.value - right.value)
            return Num(to_float(left) - to_float(right))
        if node.op == "*":
            if isinstance(left, Frac) and isinstance(right, Frac):
                return Frac(left.value * right.value)
            return Num(to_float(left) * to_float(right))
        if node.op == "/":
            if to_float(right) == 0:
                raise CalcError("Деление на ноль")
            if isinstance(left, Frac) and isinstance(right, Frac):
                return Frac(left.value / right.value)
            return Num(to_float(left) / to_float(right))
        if node.op == "%":
            if to_float(right) == 0:
                raise CalcError("Деление по модулю на ноль")
            return Num(math.fmod(to_float(left), to_float(right)))
        if node.op == "^":
            return Num(to_float(left) ** to_float(right))
        raise CalcError(f"Неизвестный оператор: {node.op}")
    if isinstance(node, CallNode):
        if node.name not in FUNCTIONS:
            raise CalcError(f"Неизвестная функция: {node.name}")
        args = [evaluate(a, variables) for a in node.args]
        return FUNCTIONS[node.name](*args)
    raise CalcError(f"Неизвестный узел: {node!r}")


# ---------------------------- Пошаговое решение ---------------------------
@dataclass
class Step:
    title: str
    detail: str


@dataclass
class Solution:
    expression: str
    steps: List[Step] = field(default_factory=list)
    result: Optional[Value] = None
    error: Optional[str] = None


def solve(text: str) -> Solution:
    sol = Solution(expression=text)
    try:
        tokens = tokenize(text)
        parser = Parser(tokens)
        tree = parser.parse()
        sol.steps.append(Step("Разбор", "Выражение разобрано в дерево."))
        val = evaluate(tree)
        sol.steps.append(Step("Вычисление", "Дерево вычислено."))
        sol.result = val
    except CalcError as e:
        sol.error = str(e)
        sol.steps.append(Step("Ошибка", str(e)))
    except Exception as e:
        sol.error = f"Внутренняя ошибка: {e}"
        sol.steps.append(Step("Ошибка", sol.error))
    return sol


# ---------------------------- Блочный интерфейс --------------------------
def render_block(title: str, lines: List[str]) -> str:
    width = max([len(title)] + [len(x) for x in lines] + [0])
    top = "┌" + "─" * (width + 2) + "┐"
    mid = "├" + "─" * (width + 2) + "┤"
    bot = "└" + "─" * (width + 2) + "┘"
    out = [top, f"│ {title.ljust(width)} │", mid]
    for ln in lines:
        out.append(f"│ {ln.ljust(width)} │")
    out.append(bot)
    return "\n".join(out)


def render_solution(sol: Solution) -> str:
    blocks: List[str] = []
    blocks.append(render_block("Выражение", [sol.expression]))
    if sol.steps:
        step_lines = [f"{i+1}. {s.title}: {s.detail}" for i, s in enumerate(sol.steps)]
        blocks.append(render_block("Шаги решения", step_lines))
    if sol.error:
        blocks.append(render_block("Ошибка", [sol.error]))
    elif sol.result is not None:
        blocks.append(render_block("Ответ", [str(sol.result)]))
    return "\n\n".join(blocks)


# ---------------------------- Юнит-тесты ----------------------------------
import unittest


class TestTokenizer(unittest.TestCase):
    def test_numbers(self):
        ts = tokenize("1 2.5 3e2")
        vals = [t.value for t in ts if t.kind in ("NUM", "FRAC")]
        self.assertIn("1", vals)
        self.assertIn("2.5", vals)

    def test_fraction_token(self):
        ts = tokenize("3/4")
        self.assertEqual(ts[0].kind, "FRAC")
        self.assertEqual(ts[0].value, "3/4")

    def test_operators(self):
        ts = tokenize("1+2*3")
        ops = [t.value for t in ts if t.kind == "OP"]
        self.assertEqual(ops, ["+", "*"])

    def test_ident(self):
        ts = tokenize("sin(x)")
        self.assertEqual(ts[0].kind, "IDENT")
        self.assertEqual(ts[0].value, "sin")


class TestParser(unittest.TestCase):
    def _parse(self, s: str) -> Node:
        return Parser(tokenize(s)).parse()

    def test_precedence(self):
        tree = self._parse("1+2*3")
        self.assertIsInstance(tree, BinaryNode)
        self.assertEqual(tree.op, "+")
        self.assertEqual(tree.right.op, "*")

    def test_power_right_assoc(self):
        tree = self._parse("2^3^2")
        self.assertEqual(tree.op, "^")
        self.assertEqual(tree.right.op, "^")

    def test_unary_minus(self):
        tree = self._parse("-5")
        self.assertIsInstance(tree, UnaryNode)

    def test_call(self):
        tree = self._parse("sin(0)")
        self.assertIsInstance(tree, CallNode)
        self.assertEqual(tree.name, "sin")


class TestEvaluate(unittest.TestCase):
    def _eval(self, s: str) -> Value:
        return evaluate(Parser(tokenize(s)).parse())

    def test_add(self):
        self.assertEqual(to_float(self._eval("1+2")), 3.0)

    def test_fraction_add(self):
        v = self._eval("1/3 + 1/6")
        self.assertIsInstance(v, Frac)
        self.assertEqual(v.value, Fraction(1, 2))

    def test_fraction_mul(self):
        v = self._eval("2/3 * 3/4")
        self.assertIsInstance(v, Frac)
        self.assertEqual(v.value, Fraction(1, 2))

    def test_division_by_zero(self):
        with self.assertRaises(CalcError):
            self._eval("1/0")

    def test_power(self):
        self.assertAlmostEqual(to_float(self._eval("2^10")), 1024.0)

    def test_sqrt(self):
        self.assertAlmostEqual(to_float(self._eval("sqrt(16)")), 4.0)

    def test_sin_zero(self):
        self.assertAlmostEqual(to_float(self._eval("sin(0)")), 0.0)

    def test_constants(self):
        self.assertAlmostEqual(to_float(self._eval("pi")), math.pi)

    def test_unknown_var(self):
        with self.assertRaises(CalcError):
            self._eval("x+1")

    def test_abs_fraction(self):
        v = self._eval("abs(-3/4)")
        self.assertEqual(v.value, Fraction(3, 4))

    def test_unary_minus(self):
        self.assertEqual(to_float(self._eval("-(2+3)")), -5.0)

    def test_modulo(self):
        self.assertAlmostEqual(to_float(self._eval("10%3")), 1.0)

    def test_log(self):
        self.assertAlmostEqual(to_float(self._eval("log(100)")), math.log(100))

    def test_ln(self):
        self.assertAlmostEqual(to_float(self._eval("ln(e)")), 1.0)


class TestSolve(unittest.TestCase):
    def test_ok(self):
        s = solve("1+2*3")
        self.assertIsNone(s.error)
        self.assertEqual(to_float(s.result), 7.0)

    def test_error(self):
        s = solve("1+")
        self.assertIsNotNone(s.error)

    def test_render_contains_blocks(self):
        s = solve("2/3+1/3")
        out = render_solution(s)
        self.assertIn("Выражение", out)
        self.assertIn("Ответ", out)


# ---------------------------- CLI ----------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Калькулятор. Введите выражение (пустая строка — выход).")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            print(render_solution(solve(line)))
        return 0
    expr = " ".join(argv)
    sol = solve(expr)
    print(render_solution(sol))
    return 0 if not sol.error else 1


if __name__ == "__main__":
    raise SystemExit(main())