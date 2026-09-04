"""Bounded executors and answer-directed analysis. No eval/exec or generated code.

A stable answer is a certificate about this IR, not about a model's interpretation
of natural language. Tracking alternatives are independent (or an overapproximation
of correlated alternatives); the queried output is one position, not a joint state.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import math
import re
import time
from typing import Any


class Rejected(ValueError):
    """Unsupported, malformed, or resource-exhausting input; callers must fall back."""


@dataclass(frozen=True)
class Limits:
    max_chars: int = 32000
    max_nodes: int = 256
    max_bits: int = 1024
    max_entities: int = 64
    max_steps: int = 256
    max_alternatives: int = 4
    max_order_entities: int = 7
    max_states: int = 10000
    seconds: float = 0.25

    def __post_init__(self) -> None:
        for key, value in self.__dict__.items():
            if key == "seconds":
                if not math.isfinite(value) or not 0 < value <= 10:
                    raise ValueError("invalid executor time limit")
            elif type(value) is not int or value < 1:
                raise ValueError("executor count limits must be positive integers")


@dataclass(frozen=True)
class Analysis:
    answers: tuple[str, ...]
    relevant_steps: tuple[int, ...] = ()
    states: int = 0

    @property
    def stable(self) -> bool:
        return len(self.answers) == 1


def digest_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atom(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise Rejected("expected nonempty bounded string")
    return value


def _bounded(value: Fraction, limits: Limits) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > limits.max_bits:
        raise Rejected("arithmetic magnitude limit")
    return value


def expression(text: str, variables: dict[str, int] | None = None,
               limits: Limits = Limits()) -> str:
    """Exact rational arithmetic OR Boolean expressions, never mixed/coerced.

    Supported: integers, + - * / // %, bounded **; True/False/not/and/or.
    Names may only be explicit integer template slots. Floats, calls, attributes,
    containers, subscripts, comparisons and Python's bool-as-int coercion are rejected.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > min(limits.max_chars, 4096):
        raise Rejected("expression length")
    if variables is None:
        variables = {}
    if any(type(v) is not int for v in variables.values()):
        raise Rejected("template values must be integers")
    try:
        tree = ast.parse(text.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > limits.max_nodes:
            raise Rejected("expression node limit")

        def visit(node: ast.AST, depth: int = 0) -> bool | Fraction:
            if depth > 32:
                raise Rejected("expression depth limit")
            if isinstance(node, ast.Constant):
                if type(node.value) is bool:
                    return node.value
                if type(node.value) is int:
                    return _bounded(Fraction(node.value), limits)
                raise Rejected("only integer and Boolean literals")
            if isinstance(node, ast.Name) and node.id in variables:
                return _bounded(Fraction(variables[node.id]), limits)
            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand, depth + 1)
                if isinstance(node.op, ast.Not) and type(value) is bool:
                    return not value
                if isinstance(value, Fraction) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    return value if isinstance(node.op, ast.UAdd) else -value
                raise Rejected("invalid unary operation")
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                values = [visit(n, depth + 1) for n in node.values]
                if any(type(v) is not bool for v in values):
                    raise Rejected("Boolean operands required")
                return all(values) if isinstance(node.op, ast.And) else any(values)
            if isinstance(node, ast.BinOp):
                a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
                if not isinstance(a, Fraction) or not isinstance(b, Fraction):
                    raise Rejected("numeric operands required")
                if isinstance(node.op, ast.Add):
                    value = a + b
                elif isinstance(node.op, ast.Sub):
                    value = a - b
                elif isinstance(node.op, ast.Mult):
                    value = a * b
                elif isinstance(node.op, ast.Div):
                    value = a / b
                elif isinstance(node.op, ast.FloorDiv):
                    value = Fraction(a // b)
                elif isinstance(node.op, ast.Mod):
                    value = a % b
                elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12:
                    value = a ** int(b)
                else:
                    raise Rejected("unsupported arithmetic operation")
                return _bounded(value, limits)
            raise Rejected("unsupported expression syntax")

        result = visit(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise Rejected(str(exc)) from exc
    if type(result) is bool:
        return str(result)
    if result.denominator == 1:
        return str(result.numerator)
    denominator, twos, fives = result.denominator, 0, 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator == 1:
        digits = max(twos, fives)
        integer = abs(result.numerator) * 2 ** (digits - twos) * 5 ** (digits - fives)
        text = str(integer).zfill(digits + 1)
        return ("-" if result < 0 else "") + text[:-digits] + "." + text[-digits:]
    return f"{result.numerator}/{result.denominator}"


def exact_input(question: str, limits: Limits = Limits()) -> str | None:
    """Full-input recognizer, intentionally narrow; never extract a convenient substring."""
    if len(question) > 4096:
        return None
    text = question.strip()
    text = re.sub(r"^(?:Calculate|Evaluate|What is)\s+", "", text, count=1)
    text = re.sub(r"\s*(?:\bis|=|\?)\s*$", "", text, count=1).strip()
    if not re.fullmatch(r"[\d\s()+*/%.\-]+|[\s()TrueFalsnotd]+", text):
        return None
    try:
        return expression(text, limits=limits)
    except Rejected:
        return None


def validate_ir(raw: Any, question: str, limits: Limits = Limits()) -> dict:
    """Validate a source-grounded JSON IR. 'complete' is a model claim, not a proof."""
    if not isinstance(raw, dict) or raw.get("complete") is not True:
        raise Rejected("compiler did not claim a complete supported interpretation")
    if len(question) > limits.max_chars:
        raise Rejected("question too long")
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise Rejected("IR kind must be a string")
    if kind == "expression":
        if set(raw) != {"kind", "complete", "expression", "source"}:
            raise Rejected("expression schema")
        if not isinstance(raw["source"], str) or not raw["source"] or raw["source"] not in question:
            raise Rejected("ungrounded source quote")
        expression(raw["expression"], limits=limits)
        return raw
    if kind not in {"tracking", "ordering"}:
        raise Rejected("unsupported IR kind")
    expected = {"kind", "complete", "initial", "query", "steps"} if kind == "tracking" else {
        "kind", "complete", "entities", "query", "steps"}
    if set(raw) != expected:
        raise Rejected("unexpected/missing IR fields")
    if kind == "tracking":
        initial = raw["initial"]
        if not isinstance(initial, dict):
            raise Rejected("initial must map positions to values")
        entities = list(initial)
        for value in initial.values():
            _atom(value)
        if not isinstance(raw["query"], str) or raw["query"] not in initial:
            raise Rejected("unknown queried position")
    else:
        entities = raw["entities"]
        if not isinstance(entities, list) or len(entities) > limits.max_order_entities:
            raise Rejected("ordering entity limit")
        if type(raw["query"]) is not int or not 0 <= raw["query"] < len(entities):
            raise Rejected("ordering query must be a zero-based rank")
    if not 1 <= len(entities) <= limits.max_entities:
        raise Rejected("entity count")
    for entity in entities:
        _atom(entity)
    if len(set(entities)) != len(entities):
        raise Rejected("duplicate entities")
    steps = raw["steps"]
    if not isinstance(steps, list) or len(steps) > limits.max_steps:
        raise Rejected("step limit")
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"source", "alternatives"}:
            raise Rejected("step schema")
        if not isinstance(step["source"], str) or not step["source"] or step["source"] not in question:
            raise Rejected("ungrounded source quote")
        alts = step["alternatives"]
        if not isinstance(alts, list) or not 1 <= len(alts) <= limits.max_alternatives:
            raise Rejected("alternative count")
        for pair in alts:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(v, str) for v in pair):
                raise Rejected("an alternative must be a pair of entity names")
            if pair[0] == pair[1] or any(v not in entities for v in pair):
                raise Rejected("invalid relation/swap")
    return raw


def analyze(ir: dict, limits: Limits = Limits(), deadline: float | None = None) -> Analysis:
    """IR must have passed validate_ir. Deadline exhaustion never yields a certificate."""
    deadline = min(deadline or math.inf, time.monotonic() + limits.seconds)

    def check() -> None:
        if time.monotonic() >= deadline:
            raise Rejected("analysis deadline exceeded")

    check()
    if ir["kind"] == "expression":
        return Analysis((expression(ir["expression"], limits=limits),), states=1)
    steps = ir["steps"]
    if ir["kind"] == "tracking":
        positions = {ir["query"]}
        relevant = []
        work = 0
        for i in range(len(steps) - 1, -1, -1):
            check()
            alternatives = steps[i]["alternatives"]
            if any(a in positions or b in positions for a, b in alternatives):
                relevant.append(i)
            previous = set()
            for a, b in alternatives:
                previous.update(b if p == a else a if p == b else p for p in positions)
            work += len(positions) * len(alternatives)
            if work > limits.max_states:
                raise Rejected("tracking work limit")
            positions = previous
        return Analysis(tuple(sorted({ir["initial"][p] for p in positions})),
                        tuple(sorted(relevant)), work)
    answers = set()
    states = 0
    for order in permutations(ir["entities"]):
        check()
        states += 1
        if states > limits.max_states:
            raise Rejected("ordering enumeration limit")
        ranks = {entity: rank for rank, entity in enumerate(order)}
        if all(any(ranks[a] < ranks[b] for a, b in step["alternatives"]) for step in steps):
            answers.add(order[ir["query"]])
    return Analysis(tuple(sorted(answers)), tuple(range(len(steps))), states)


def resolve(ir: dict, step: int, choice: int) -> dict:
    import copy
    if type(step) is not int or type(choice) is not int or not 0 <= step < len(ir.get("steps", [])):
        raise Rejected("invalid resolution index")
    alternatives = ir["steps"][step]["alternatives"]
    if not 0 <= choice < len(alternatives):
        raise Rejected("invalid resolution choice")
    result = copy.deepcopy(ir)
    result["steps"][step]["alternatives"] = [alternatives[choice]]
    return result


def clarification(ir: dict, analysis: Analysis, limits: Limits = Limits()) -> int | None:
    """Maximize worst-case answer-set reduction; abstain on no positive single-step gain.

    This greedy policy can miss synergistic ambiguities. A shared deadline bounds
    the whole selection, not just one counterfactual analysis.
    """
    deadline = time.monotonic() + limits.seconds
    best, best_gain = None, 0
    for i in analysis.relevant_steps:
        alts = ir["steps"][i]["alternatives"]
        if len(alts) < 2:
            continue
        sizes = [len(analyze(resolve(ir, i, j), limits, deadline).answers) for j in range(len(alts))]
        gain = len(analysis.answers) - max(sizes)
        if gain > best_gain:
            best, best_gain = i, gain
    return best
