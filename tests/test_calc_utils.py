"""
Unit tests for calc_utils.py (Python-computed arithmetic for `eval.py --calculator`).

Run with:
    python3 tests/test_calc_utils.py -v
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from calc_utils import correct_rhs, format_value, lhs_expression  # noqa: E402


class TestLhsExpression(unittest.TestCase):
    CASES = [
        ("(50 / 4) + 2 =", "(50 / 4) + 2"),
        ("- (50 - 4) * 2 =", "(50 - 4) * 2"),          # bullet, not negation
        ("2. **(50 + 4) / 2 =", "**(50 + 4) / 2"),     # list marker + bold
        ("<answer>(50 + 4) / 2</answer>\n=", "(50 + 4) / 2"),
        ("= (54) / 2\n=", "(54) / 2"),                 # continuation line
        ("Calculation: 50 - 8 =", "50 - 8"),
        (r"\(15 * \frac{2}{3} * 1 =", "15 * (2)/(3) * 1"),
        ("= 2,584 - 3,465\n=", "2584 - 3465"),
        ("= 2.444... - 49\n=", "2.444 - 49"),
        ("target 56 =", None),                         # no operator
        ("x ==", None),
    ]

    def test_cases(self):
        for text, want in self.CASES:
            self.assertEqual(lhs_expression(text), want, text)


class TestCorrectRhs(unittest.TestCase):
    def check(self, lhs, seg, want, changed):
        self.assertEqual(correct_rhs(lhs, seg), (want, changed), (lhs, seg))

    def test_correct_steps_untouched(self):
        for lhs, seg in [("(1*5)*20 + 8 - 43", " 100 + 8 - 43\n="),
                         ("100 + 8 - 43", " 65 (Too low)\n\nAttempt 2: 5 ="),
                         ("(50 + 4) / 2", " (54) / 2\n="),
                         ("8 / 3", " 2.67 (approx)")]:
            self.check(lhs, seg, seg, False)

    def test_intermediate_step_fixed(self):
        self.check("(1*5)*20 + 8 - 43", " 65 + 8 - 43\n#### next", " 100 + 8 - 43", True)
        self.check("50 / 4 + 2", " 12 + 2 = ", " 12.5 + 2", True)

    def test_final_value_fixed_and_continuation_dropped(self):
        self.check("100 + 8 - 43", " 30 (Too low)\n\nAttempt 2: 5 =", " 65", True)

    def test_grouping_in_flat_chain(self):
        self.check("2 + 3 + 4", " 2 + 8 =", " 2 + 7", True)     # groups 3 + 4
        self.check("10 - 3 + 2", " 10 - 5 = ", " 10 - 1", True)  # "- (3 - 2)"

    def test_not_a_reduction_falls_back_to_value(self):
        self.check("2 * (3 + 4)", " 2 * 3 + 2 * 4", " 14", True)
        self.check("4 * 2", "\nSo", " 8", True)                 # no RHS written

    def test_bold(self):
        self.check("(1 * 5 + 20 - 8) + 43", " **60** + 43 =", " 17 + 43", True)
        self.check("(1 * 5 + 20 - 8) + 43", " **17** + 43 =", " **17** + 43 =", False)

    def test_rounding(self):
        self.check("8 / 3", " 2.666 (approx)", " 2.67", True)
        self.assertEqual(format_value(12.5), "12.5")
        self.assertEqual(format_value(-2 / 3), "-0.67")
        self.assertEqual(format_value(-0.001), "0")


if __name__ == "__main__":
    unittest.main()
