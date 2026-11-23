"""Response formats shared by GT-SFT data generation and few-shot eval prompts.

One definition of "what a solution looks like" for both sides:

    scripts/generate_gt_sft_data.py --format <fmt>   -> SFT targets
    eval.py --n_shot K --shot_format <fmt>           -> few-shot exemplars

so a k-shot base-model eval measures exactly the format an SFT run would install.

Formats
-------
answer       answer tag only (historical GT-SFT target; no reasoning invented)
think_empty  vacuous <think></think> scaffold, then the answer
verify       <think> evaluates the ground-truth expression step by step
search       <think> tries k wrong-but-legal expressions, rejects each on its
             value, then lands on the ground-truth one

`search` exists because answer-only SFT installs a ~15-token response, which
leaves GRPO no room to sample a second attempt: the "try -> check -> retry"
event is outside the support of the rollout distribution, so RL cannot amplify
it (see research_logs/2026-08-25: post-GRPO pass@32 collapses onto pass@1).

Distractors are legal Countdown expressions for the *same* puzzle -- a pattern
from annotated_expressions.json with the puzzle's own numbers plugged into a
permutation -- keeping only integer values != target. They therefore use each
number exactly once and would validate; they are simply wrong. Rejected
attempts live strictly inside <think>: grader_utils.extract_solution takes the
*last* <answer> tag, so the graded answer is unaffected.
"""

import ast
import itertools
import json
import os
import random
import re

from grader_utils import evaluate_equation

FORMATS = ("answer", "think_empty", "verify", "search")
DISTRACTOR_STRATEGIES = ("mixed", "near", "random")

_BANK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "annotated_expressions.json")
_BANK = None

# Sampling budget per puzzle. n=3 has 32 patterns x 6 permutations = 192 total
# candidates, so the budget is exhaustive there; n=5/6 (3584 x 120, 43008 x 720)
# must be sampled, and this keeps the cost per row flat across puzzle sizes.
MAX_CANDIDATE_TRIES = 600
# Values beyond this are dropped: legal, but a 12-digit "too high" reads as
# noise rather than as a search step.
MAX_ABS_VALUE = 100_000


def _bank(size):
    """Expression patterns (all parenthesizations) for a puzzle size, as strings."""
    global _BANK
    if _BANK is None:
        with open(_BANK_PATH) as f:
            raw = json.load(f)
        _BANK = {k: [d["expression"] for d in v] for k, v in raw.items()}
    return _BANK[str(size)]


def plug_in_nums(pattern, nums):
    """Substitute numbers (in order) into a letter pattern and space the operators
    like the prompt's example: '(A-B)*C' + [53,22,3] -> '(53 - 22) * 3'."""
    eq = pattern
    for letter, n in zip("ABCDEF", nums):
        eq = eq.replace(letter, str(n))
    return re.sub(r"\s*([+\-*/])\s*", r" \1 ", eq)


# The name this substitution has in scripts/generate_gt_sft_data.py and
# scripts/diagnose_gt_ntp.py, which each carried their own copy.
build_gt_equation = plug_in_nums


def _fmt_value(v):
    """Render an evaluated value: 93 not 93.0, 3.43 not 3.4285714285714284."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def eval_steps(equation):
    """Ground-truth expression -> [(sub-expression, value), ...] in evaluation order.

    Parsed with `ast` rather than re-derived from the pattern string, so it works
    for both the canonical form ('A*B+C') and a fully parenthesized one.
    """
    ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def walk(node):
        if isinstance(node, ast.Constant):
            return node.value, None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v, _ = walk(node.operand)
            return -v, None
        if not isinstance(node, ast.BinOp) or type(node.op) not in ops:
            raise ValueError(f"unsupported node {ast.dump(node)}")
        lv, _ = walk(node.left)
        rv, _ = walk(node.right)
        v = {"+": lv + rv, "-": lv - rv, "*": lv * rv,
             "/": (lv / rv if rv else float("nan"))}[ops[type(node.op)]]
        steps.append((f"{_fmt_value(lv)} {ops[type(node.op)]} {_fmt_value(rv)}", v))
        return v, None

    steps = []
    walk(ast.parse(equation, mode="eval").body)
    return steps


def sample_distractors(nums, target, k, rng, strategy="mixed",
                       max_tries=MAX_CANDIDATE_TRIES, near_band=10):
    """Up to k (expression, value) pairs: legal expressions over `nums` that miss.

    Candidates are (pattern, permutation) pairs drawn from the puzzle size's bank.
    Kept only if the value is an integer, |value| <= MAX_ABS_VALUE, and != target
    -- an alternative *correct* expression must never appear as a rejected
    attempt. Distinct expressions and distinct values, so no attempt repeats a
    number already ruled out.

    strategy:
        near    the k closest misses           (a tight, already-warm search)
        random  k uniform draws                (no narrowing)
        mixed   half random, half near, ordered by decreasing |value - target|
                (default: reads like a search closing in on the target)
    """
    if k <= 0:
        return []
    patterns = _bank(len(nums))
    perms = list(itertools.permutations(nums))
    space = len(patterns) * len(perms)

    if space <= max_tries:          # small puzzle sizes: enumerate exhaustively
        pairs = [(p, q) for p in patterns for q in perms]
        rng.shuffle(pairs)
    else:                           # large: sample with replacement, dedup below
        pairs = [(rng.choice(patterns), rng.choice(perms)) for _ in range(max_tries)]

    by_expr, seen_values = {}, {target}
    for pattern, perm in pairs:
        eq = plug_in_nums(pattern, perm)
        if eq in by_expr:
            continue
        v = evaluate_equation(eq)
        if v is None or not float(v).is_integer():
            continue
        v = int(v)
        if abs(v) > MAX_ABS_VALUE or v in seen_values:
            continue
        by_expr[eq] = v
        seen_values.add(v)

    cands = sorted(by_expr.items(), key=lambda ev: abs(ev[1] - target))
    if not cands:
        return []

    if strategy == "near":
        chosen = cands[:k]
    elif strategy == "random":
        chosen = rng.sample(cands, min(k, len(cands)))
    elif strategy == "mixed":
        near = [c for c in cands if abs(c[1] - target) <= near_band] or cands[:k]
        far = [c for c in cands if c not in near]
        n_near = max(1, k // 2)
        chosen = rng.sample(near, min(n_near, len(near)))
        n_far = k - len(chosen)
        pool = [c for c in far if c not in chosen]
        chosen += rng.sample(pool, min(n_far, len(pool)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # Order the trace so it narrows toward the target.
    return sorted(chosen, key=lambda ev: -abs(ev[1] - target))


def _think_body(fmt, equation, nums, target, rng, n_attempts, strategy):
    """The text that goes between <think> and </think>; '' for a vacuous scaffold."""
    if fmt == "think_empty":
        return ""
    if fmt == "verify":
        steps = eval_steps(equation)
        return " " + ", ".join(f"{lhs} = {_fmt_value(v)}" for lhs, v in steps) + \
               ", which is the target. "
    if fmt == "search":
        k = rng.randint(*n_attempts) if isinstance(n_attempts, tuple) else n_attempts
        lines = []
        for eq, v in sample_distractors(nums, target, k, rng, strategy=strategy):
            side = "too high" if v > target else "too low"
            lines.append(f"Try {eq} = {v}; {side}.")
        lines.append(f"Try {equation} = {target}; that is the target.")
        return "\n" + "\n".join(lines) + "\n"
    raise ValueError(f"unknown format {fmt!r}")


def render_response(fmt, prompt, nums, target, canonical_pattern, equation=None,
                    rng=None, n_attempts=(1, 3), strategy="mixed"):
    """The continuation a model should emit after `prompt`.

    A prompt ending inside an open <think> (the canonical template_type='base'
    text) must close it rather than open a second one; a prompt ending at
    'Solution:' (template_type='base-simple') opens its own.

    n_attempts: int, or an inclusive (lo, hi) range sampled per row so `search`
    traces are not a constant length.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    rng = rng or random.Random(0)
    eq = equation or plug_in_nums(canonical_pattern, nums)
    inside_think = prompt.rstrip().endswith("<think>")

    if fmt == "answer":
        body = f"<answer> {eq} </answer>"
        return ("</think>\n" + body) if inside_think else " " + body

    think = _think_body(fmt, eq, nums, target, rng, n_attempts, strategy)
    answer = f"<answer> {eq} </answer>"
    if inside_think:
        return f"{think}</think>\n{answer}"
    return f" <think>{think}</think>\n{answer}"
