"""Calculator intervention for eval: every arithmetic step after "=" is Python's.

Used by `eval.py --calculator`. Generation stops at every "="; when the text
before it ends in an arithmetic expression E, the model's right-hand side R is
checked against E once it has been written (at the next stop):

    (1*5)*20 + 8 - 43 = 100 + 8 - 43 = 65
                        ^^^ step 1 aligned to E: 100 := py((1*5)*20)
                                       ^^ step 2 aligned to step 1: 65 := py(100 + 8 - 43)

R is aligned to E as a partial reduction (`correct_rhs`): each number in R that
stands for a sub-expression of E (a single term/factor, or a contiguous run of
terms/factors of a flattened +- or */ chain) is replaced by Python's value of
that part, rounded to 2 decimals. If R is not a partial reduction of E
(reordered, expanded, unparsable) it is replaced by Python's value of E. When R
changes, everything the model wrote after it is discarded and regenerated, since
it was conditioned on the wrong number. The model still chooses which
expressions to try, how to reduce them, and what to put in <answer>.
"""
import ast
import operator
import re

# Characters an inline arithmetic expression can be built from (after _NORMALIZE).
# Unicode ×/÷ are accepted and mapped to */ in safe_eval.
_EXPR_CHARS = set("0123456789.+-*/()×÷ \t")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}
# Markup that may sit between an expression and its "=": **bold**, </answer> tag.
_TRAILING_MARKUP = re.compile(r"(\*\*|</answer>|\s)+$")
_MAX_ABS = 1e12  # refuse to print absurd magnitudes (e.g. 99*99*99*99*99*99*99)
# Rewrites applied to the text before the "=" so LaTeX / prose number styles
# become plain Python arithmetic.
_NORMALIZE = [
    (re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}"), r"(\1)/(\2)"),
    (re.compile(r"\\(times|cdot)"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\[()\[\]]|\$"), ""),
    (re.compile(r"(\d)(\.\.\.+|…)"), r"\1"),   # 0.555... -> 0.555
    (re.compile(r"(?<=\d),(?=\d{3}(?!\d))"), ""),  # 2,584 -> 2584
]


def _strip_bold(expr):
    """Drop markdown bold. "**" is never exponentiation here: Countdown has only
    +-*/, and "2**60" towers would hang eval (see grader_utils)."""
    return expr.replace("**", "")


def _eval_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval_node(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("not arithmetic")


def _has_binop(node):
    return any(isinstance(n, ast.BinOp) for n in ast.walk(node))


def safe_eval(expr):
    """Value of a +-*/ expression, or None if it isn't one (or divides by zero).

    Requires at least one binary operator, so a bare number ("56 =") or a negated
    number is not treated as an equation to compute.
    """
    expr = _strip_bold(expr).replace("×", "*").replace("÷", "/").strip()
    if not expr or "//" in expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval").body
        if not _has_binop(tree):
            return None
        v = _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, TypeError):
        return None
    if abs(v) > _MAX_ABS:
        return None
    return v


def format_value(v):
    """Round to 2 decimals; drop trailing zeros (42, 12.5, 3.33, -0.67)."""
    s = f"{round(v, 2):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def lhs_expression(text):
    """The arithmetic expression ending just before a trailing "=", or None.

    `text` is the response so far, ending in "=". Takes the maximal run of
    expression characters before the "=" -- after normalising LaTeX \\frac/\\times,
    "0.555..." and "2,584", and stripping **bold** / </answer> markup and
    whitespace, so "= ..." continuation lines and expressions inside an <answer>
    tag on the previous line are picked up -- then the LONGEST suffix
    of that run that parses -- which drops list markers ("2. (50-4)*2"), a
    leading "**", or an unmatched "(".
    """
    if not text.endswith("="):
        return None
    body = text[:-1]
    if body.endswith(("=", "<", ">", "!")):  # ==, <=, >=, !=
        return None
    body = body[-400:]
    for pat, rep in _NORMALIZE:
        body = pat.sub(rep, body)
    body = _TRAILING_MARKUP.sub("", body)
    if body.endswith("</answer>"):
        body = body[: -len("</answer>")]
    i = len(body)
    while i > 0 and body[i - 1] in _EXPR_CHARS:
        i -= 1
    run = body[i:]
    # At the start of a line, "- " is a markdown bullet, not a negation:
    # "- (50 - 4) * 2 =" must compute 92, not -92.
    if i == 0 or body[i - 1] == "\n":
        run = re.sub(r"^\s*-\s+", "", run)
    for j in range(len(run)):
        cand = run[j:]
        if not cand.strip() or cand[0] in ")*/":
            continue
        if safe_eval(cand) is not None:
            return cand.strip()
    return None


# --- aligning a model-written step R against the expression E it reduces ------

def _parse(expr):
    """AST of a +-*/ expression (a bare number allowed), or None."""
    expr = expr.replace("×", "*").replace("÷", "/")  # same length: offsets survive
    if not expr.strip() or "**" in expr or "//" in expr:  # callers strip bold first
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
        _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, TypeError):
        return None
    return tree


def _flat(node):
    """('num', node) | ('sum', [(sign, node)]) | ('prod', [(op, node)]).

    Left-associative +- (and */) chains are flattened so "2 + 3 + 4 = 2 + 7"
    can group the last two terms, which the binary AST ((2+3)+4) cannot.
    """
    if isinstance(node, ast.Constant):
        return ("num", node)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.operand, ast.Constant):
            return ("num", node)  # a negative literal is one leaf
        return ("sum", [("+" if isinstance(node.op, ast.UAdd) else "-", node.operand)])
    kind = "sum" if isinstance(node.op, (ast.Add, ast.Sub)) else "prod"
    op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
    left = _flat(node.left)
    items = left[1] if left[0] == kind else [("+" if kind == "sum" else "*", node.left)]
    return (kind, items + [(op, node.right)])


def _contribution(kind, items):
    """Value a run of chain items adds (sum) or multiplies in (prod)."""
    if kind == "sum":
        return sum(_eval_node(n) if s == "+" else -_eval_node(n) for s, n in items)
    v = 1.0
    for op, n in items:
        v = v * _eval_node(n) if op == "*" else v / _eval_node(n)
    return v


def _same(a, b):
    return abs(a - b) < 1e-9


def _align(e, r):
    """(n changed leaves, {R leaf: new value}) if R is a partial reduction of E."""
    rf = _flat(r)
    if rf[0] == "num":
        target = float(format_value(_eval_node(e)))
        return (0, {}) if _same(_eval_node(r), target) else (1, {r: target})
    ef = _flat(e)
    if ef[0] != rf[0]:
        return None
    kind, eitems, ritems = ef[0], ef[1], rf[1]
    best = {}

    def rec(i, k):
        # Cheapest alignment of E items i.. with R items k..
        if (i, k) in best:
            return best[(i, k)]
        if k == len(ritems):
            res = (0, {}) if i == len(eitems) else None
            best[(i, k)] = res
            return res
        rop, rnode = ritems[k]
        res = None
        for j in range(i + 1, len(eitems) - (len(ritems) - k - 1) + 1):
            group = eitems[i:j]
            sub = None
            if j - i == 1 and group[0][0] == rop:
                sub = _align(group[0][1], rnode)
            elif _flat(rnode)[0] == "num":
                c = _contribution(kind, group)
                if rop in "-/":
                    if kind == "prod" and c == 0:
                        continue
                    c = -c if kind == "sum" else 1 / c
                c = float(format_value(c))
                sub = (0, {}) if _same(_eval_node(rnode), c) else (1, {rnode: c})
            if sub is None:
                continue
            rest = rec(j, k + 1)
            if rest is None:
                continue
            cand = (sub[0] + rest[0], {**sub[1], **rest[1]})
            if res is None or cand[0] < res[0]:
                res = cand
        best[(i, k)] = res
        return res

    try:
        return rec(0, 0)
    except ZeroDivisionError:
        return None


def rhs_span(segment):
    """(start, end) of the arithmetic expression the segment opens with, or None.

    `segment` is the text right after an "=". Leading spaces and a "**" bold
    marker are skipped (not newlines: "=\n" means the model gave no value);
    the expression is the longest prefix of the expression-character run that
    parses, so "65 (Too low)" -> "65" and "14.5." -> "14.5".
    """
    m = re.match(r"[ \t]*(\*\*)?", segment)
    start = m.end()
    end = start
    while end < len(segment) and segment[end] in _EXPR_CHARS:
        end += 1
    for stop in range(end, start, -1):
        cand = segment[start:stop]
        if cand.strip() and cand.strip()[0] not in ")*/" and _parse(_strip_bold(cand)) is not None:
            return start, start + len(cand.rstrip())
    return None


def correct_rhs(lhs, segment):
    """Rewrite of the text after "=" so every arithmetic result in it is Python's.

    `lhs` is the expression before the "=" (from `lhs_expression`), `segment`
    the model's text after it. Returns (new_text, changed). When the model's
    right-hand side is already right, the segment is returned untouched;
    otherwise the corrected right-hand side alone (the model's continuation is
    dropped). With no parsable right-hand side, Python's value of `lhs` is used.
    """
    e = _parse(_strip_bold(lhs))
    span = rhs_span(segment)
    if span is None:
        return " " + format_value(safe_eval(lhs)), True
    start, end = span
    rhs = segment[start:end]
    body = _strip_bold(rhs.strip())  # a corrected RHS loses its bold; harmless
    r = _parse(body)
    aligned = _align(e, r)
    if aligned is None:
        new_rhs = format_value(_eval_node(e))
    elif aligned[0] == 0:
        return segment, False
    else:
        # Substitute changed leaves by their column offsets in `body`.
        new_rhs, last = [], 0
        for node, v in sorted(aligned[1].items(), key=lambda kv: kv[0].col_offset):
            s = format_value(v)
            if v < 0 and node.col_offset > 0:
                s = f"({s})"
            new_rhs += [body[last:node.col_offset], s]
            last = node.end_col_offset
        new_rhs = "".join(new_rhs + [body[last:]])
    ws = len(segment) - len(segment.lstrip(" \t"))  # keep spacing, drop a leading "**"
    return segment[:ws] + new_rhs, True
