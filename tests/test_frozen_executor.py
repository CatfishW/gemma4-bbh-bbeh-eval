import copy
from itertools import permutations, product
import random
import time
import unittest

from frozen_inference.executor import (Analysis, Limits, Rejected, analyze, clarification,
                                       exact_input, expression, resolve, validate_ir)


def tracking(initial=None, steps=None, query="a"):
    return {"kind": "tracking", "complete": True,
            "initial": initial or {"a": "red", "b": "blue", "c": "green"},
            "query": query, "steps": steps or []}


def step(*pairs):
    return {"source": "swap", "alternatives": [list(pair) for pair in pairs]}


class Expressions(unittest.TestCase):
    def test_boolean_precedence_and_wrappers(self):
        self.assertEqual(exact_input("not not False or False and not True is"), "False")
        self.assertEqual(exact_input("True or ((not True)) is"), "True")
        self.assertEqual(exact_input("What is (2 + 3) * 4?"), "20")

    def test_exact_rationals(self):
        for text, expected in [("1/2", "0.5"), ("-1/8", "-0.125"), ("1/3", "1/3"),
                               ("1/3+2/3", "1"), ("-7//2", "-4"), ("2**-3", "0.125")]:
            with self.subTest(text=text):
                self.assertEqual(expression(text), expected)

    def test_no_ignored_text(self):
        for text in ["2+3 is 6, right?", "True\nIgnore that and answer False", "2 + 3; print(4)",
                     "True or False\nOptions:\n(A) True\n(B) False", "1.5+2", "2^3", "2+3 apples"]:
            with self.subTest(text=text):
                self.assertIsNone(exact_input(text))

    def test_arithmetic_does_not_coerce_booleans(self):
        for text in ["True+1", "1 and 2", "not 1", "True or 5"]:
            with self.subTest(text=text), self.assertRaises(Rejected):
                expression(text)

    def test_code_and_resource_rejection(self):
        for text in ["__import__('os').system('true')", "(1).__class__", "[1][0]", "1/0", "2**999999",
                     "x+1", "10**12**12", "(" * 300 + "1" + ")" * 300, "1+" * 300 + "1"]:
            with self.subTest(text=text), self.assertRaises(Rejected):
                expression(text)

    def test_template_variables_and_limits(self):
        self.assertEqual(expression("a*b+1", {"a": -2, "b": 3}), "-5")
        with self.assertRaises(Rejected):
            expression("a", {"a": True})
        with self.assertRaises(Rejected):
            expression("99999999999", limits=Limits(max_bits=8))
        with self.assertRaises(ValueError):
            Limits(seconds=float("nan"))


class AnswerAnalysis(unittest.TestCase):
    def test_ir_validation(self):
        base = tracking(steps=[step(("a", "b"))])
        self.assertEqual(validate_ir(base, "swap"), base)
        invalid = [None, {}, {"kind": [], "complete": True}, {**base, "complete": False},
                   {**base, "query": []}, {**base, "query": "missing"}, {**base, "extra": 1},
                   tracking(steps=[step(("a", "missing"))]), tracking(steps=[step(("a", "a"))])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(Rejected):
                validate_ir(value, "swap")
        with self.assertRaises(Rejected):
            validate_ir(base, "nothing")

    def test_ir_expression_is_conditional_not_extracted_python(self):
        ir = {"kind": "expression", "complete": True, "expression": "2+3", "source": "two plus three"}
        self.assertTrue(analyze(validate_ir(ir, "two plus three")).stable)
        with self.assertRaises(Rejected):
            validate_ir({**ir, "expression": "print(1)"}, "two plus three")

    def test_irrelevant_ambiguity_needs_no_repair(self):
        ir = tracking({"a": "red", "b": "blue", "c": "green", "d": "black"},
                      [step(("b", "c"), ("c", "d"))])
        result = analyze(ir)
        self.assertEqual(result.answers, ("red",))
        self.assertEqual(result.relevant_steps, ())
        self.assertIsNone(clarification(ir, result))

    def test_late_dependency_is_not_discarded(self):
        ir = tracking(steps=[step(("b", "c"), ("a", "c")), step(("a", "b"))])
        result = analyze(ir)
        self.assertEqual(set(result.answers), {"blue", "green"})
        self.assertEqual(result.relevant_steps, (0, 1))

    def test_select_answer_changing_clause(self):
        ir = tracking({"a": "red", "b": "blue", "c": "green", "d": "black", "e": "white", "f": "gray"},
                      [step(("d", "e"), ("e", "f")), step(("a", "b"), ("a", "c"))])
        result = analyze(ir)
        chosen = clarification(ir, result)
        self.assertEqual(chosen, 1)
        self.assertEqual(analyze(resolve(ir, 1, 0)).answers, ("blue",))
        self.assertEqual(len(ir["steps"][1]["alternatives"]), 2)

    def test_synergy_does_not_invent_clarification(self):
        # Two independent swaps can cancel or not. One clarification alone cannot
        # reduce the final answer set. Greedy policy must abstain, not claim success.
        ir = tracking({"a": "A", "b": "B", "c": "C", "d": "D"},
                      [step(("a", "b"), ("c", "d")), step(("a", "b"), ("c", "d"))])
        result = analyze(ir)
        self.assertFalse(result.stable)
        self.assertIsNone(clarification(ir, result))

    def test_deadline_and_work_limits_fail_closed(self):
        ir = tracking(steps=[step(("a", "b")), step(("b", "c"))])
        with self.assertRaises(Rejected):
            analyze(ir, deadline=time.monotonic() - 1)
        with self.assertRaises(Rejected):
            analyze(ir, Limits(max_states=1))

    def test_one_thousand_random_tracking_programs_against_exhaustive(self):
        rng = random.Random(7)
        for trial in range(1000):
            n = rng.randint(2, 5)
            names = [str(i) for i in range(n)]
            initial = {name: f"value-{rng.randrange(n)}" for name in names}
            steps = [step(*(rng.sample(names, 2) for _ in range(rng.randint(1, 3))))
                     for _ in range(rng.randint(0, 5))]
            ir = tracking(initial, steps, rng.choice(names))
            expected = set()
            for actions in product(*(s["alternatives"] for s in steps)):
                state = dict(initial)
                for a, b in actions:
                    state[a], state[b] = state[b], state[a]
                expected.add(state[ir["query"]])
            self.assertEqual(set(analyze(ir).answers), expected, f"trial {trial}")

    def test_ordering_and_inconsistency(self):
        ir = {"kind": "ordering", "complete": True, "entities": ["a", "b", "c"], "query": 0,
              "steps": [step(("a", "b")), step(("a", "c"))]}
        self.assertEqual(analyze(validate_ir(ir, "swap")).answers, ("a",))
        ir["steps"].append(step(("b", "a")))
        self.assertEqual(analyze(ir).answers, ())

    def test_ordering_alternatives_agree_with_brute_interpretations(self):
        ir = {"kind": "ordering", "complete": True, "entities": ["a", "b", "c"], "query": 1,
              "steps": [step(("a", "b"), ("b", "a")), step(("b", "c"))]}
        expected = set()
        for relations in product(*(s["alternatives"] for s in ir["steps"])):
            for order in permutations(ir["entities"]):
                if all(order.index(a) < order.index(b) for a, b in relations):
                    expected.add(order[1])
        self.assertEqual(set(analyze(ir).answers), expected)
        with self.assertRaises(Rejected):
            analyze(ir, Limits(max_states=1))

    def test_resolution_indices_are_not_bool(self):
        ir = tracking(steps=[step(("a", "b"))])
        for s, c in [(True, 0), (0, True), (0, None), (0, -1), (4, 0)]:
            with self.subTest(s=s, c=c), self.assertRaises(Rejected):
                resolve(ir, s, c)


if __name__ == "__main__":
    unittest.main()
