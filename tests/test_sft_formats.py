"""
Unit tests for sft_formats.py (shared SFT-target / few-shot-exemplar formats).

Run with:
    source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
    conda activate verl
    python3 tests/test_sft_formats.py -v
"""
import contextlib
import io
import os
import random
import sys
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from grader_utils import compute_score, evaluate_equation, validate_equation  # noqa: E402
from sft_formats import (FORMATS, eval_steps, plug_in_nums, render_response,  # noqa: E402
                         sample_distractors)

BASE_PROMPT = ("A conversation between User and Assistant.\n"
               "Assistant: Let me solve this step by step.\n<think>")
SIMPLE_PROMPT = "Solve the Countdown problem.\nUsing the numbers [14, 27, 9] ...\n\nSolution:"


def _rows(dataset, split, n):
    path = os.path.join(REPO_ROOT, "data", dataset, f"{split}.parquet")
    df = pd.read_parquet(path)
    idx = random.Random(0).sample(range(len(df)), n)
    for i in idx:
        row = df.iloc[i]
        ei = row["extra_info"]
        yield (str(row["prompt"][0]["content"]),
               [int(x) for x in ei["numbers"]],
               int(row["reward_model"]["ground_truth"]),
               ei["canonical_pattern"],
               dict(ei))


def _score(response, target, extra_info):
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_score("countdown", response, target, extra_info=extra_info)


class TestRenderResponse(unittest.TestCase):
    def test_every_format_scores_one_on_real_rows(self):
        """The graded answer is the ground truth for all formats, both prompt families."""
        for dataset in ("balanced-simple", "balanced"):
            for prompt, nums, target, pattern, ei in _rows(dataset, "train", 25):
                for fmt in FORMATS:
                    r = render_response(fmt, prompt, nums, target, pattern,
                                        rng=random.Random(0))
                    self.assertEqual(_score(r, target, ei), 1.0,
                                     f"{dataset}/{fmt}: {r!r}")

    def test_open_think_is_closed_not_reopened(self):
        """template_type='base' prompts end inside <think>; the response must close it."""
        for fmt in FORMATS:
            r = render_response(fmt, BASE_PROMPT, [14, 27, 9], 50, "A+B+C",
                                rng=random.Random(0))
            self.assertNotIn("<think>", r, fmt)
            self.assertIn("</think>", r, fmt)

    def test_simple_prompt_opens_its_own_think(self):
        for fmt in ("think_empty", "verify", "search"):
            r = render_response(fmt, SIMPLE_PROMPT, [14, 27, 9], 50, "A+B+C",
                                rng=random.Random(0))
            self.assertEqual(r.count("<think>"), 1, fmt)
            self.assertEqual(r.count("</think>"), 1, fmt)
        self.assertNotIn("think", render_response("answer", SIMPLE_PROMPT, [14, 27, 9],
                                                  50, "A+B+C"))

    def test_deterministic_given_seed(self):
        kw = dict(prompt=SIMPLE_PROMPT, nums=[25, 18, 12, 60], target=90,
                  canonical_pattern="A*B*C/D")
        a = render_response("search", rng=random.Random(3), **kw)
        b = render_response("search", rng=random.Random(3), **kw)
        c = render_response("search", rng=random.Random(4), **kw)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            render_response("nope", SIMPLE_PROMPT, [1, 2, 3], 6, "A+B+C")


class TestDistractors(unittest.TestCase):
    def test_distractors_are_legal_wrong_expressions(self):
        """Every rejected attempt uses each number exactly once, and misses the target."""
        for prompt, nums, target, pattern, ei in _rows("balanced-simple", "train", 30):
            ds = sample_distractors(nums, target, 3, random.Random(1))
            self.assertTrue(ds, f"no distractors for {nums}")
            values = set()
            for eq, v in ds:
                self.assertTrue(validate_equation(eq, nums), f"{eq!r} vs {nums}")
                self.assertEqual(evaluate_equation(eq), v)
                self.assertNotEqual(v, target, f"{eq!r} is a correct solution")
                self.assertNotIn(v, values, "two attempts with the same value")
                values.add(v)

    def test_attempts_narrow_toward_target(self):
        ds = sample_distractors([25, 18, 12, 60], 90, 3, random.Random(0))
        gaps = [abs(v - 90) for _, v in ds]
        self.assertEqual(gaps, sorted(gaps, reverse=True))

    def test_k_zero_and_strategies(self):
        self.assertEqual(sample_distractors([1, 2, 3], 6, 0, random.Random(0)), [])
        for strategy in ("near", "random", "mixed"):
            ds = sample_distractors([25, 18, 12, 60], 90, 2, random.Random(0),
                                    strategy=strategy)
            self.assertEqual(len(ds), 2, strategy)

    def test_search_trace_lines_are_self_consistent(self):
        """Each 'Try <expr> = <value>' line states the expression's true value."""
        import re
        for prompt, nums, target, pattern, ei in _rows("balanced-simple", "train", 20):
            r = render_response("search", prompt, nums, target, pattern,
                                rng=random.Random(2), n_attempts=3)
            lines = [l for l in r.splitlines() if l.startswith("Try ")]
            self.assertGreaterEqual(len(lines), 2)
            for line in lines:
                eq, val = re.match(r"Try (.+) = (-?\d+);", line).groups()
                self.assertEqual(evaluate_equation(eq), int(val), line)


class TestEvalSteps(unittest.TestCase):
    def test_steps_end_at_the_target(self):
        for prompt, nums, target, pattern, ei in _rows("balanced-simple", "train", 30):
            steps = eval_steps(plug_in_nums(pattern, nums))
            self.assertEqual(len(steps), len(nums) - 1)
            self.assertEqual(steps[-1][1], target)


if __name__ == "__main__":
    unittest.main()
