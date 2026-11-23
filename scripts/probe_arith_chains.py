"""Behavioral arithmetic-fidelity probe for free-form `expr = ... = value` chains.

Generalizes scripts/probe_arith_fidelity.py (which needs the fixed
`Try <expr> = <value>; too high` lines of search-format SFT) to models that were
never taught a trace format, e.g. the Qwen3 base GRPO runs, which write

    (83 - 62 + 32) - 30
    = 53 + 32 - 30
    = 85 - 30
    = 55

Every `=` (or `≈`) between two arithmetic expressions is one *claim* ("link"):
the right side should evaluate to the same value as the left. A maximal run of
links is a *chain*; its first expression is what the model set out to evaluate
and its last segment is the value it settled on.

Per link:  correct if eval(left) == eval(right) (tolerance below).
Per chain: correct if eval(first) == eval(last). A chain can be correct with a
           wrong intermediate link (errors that cancel) or wrong with every link
           correct... not possible -- equality is transitive, so a wrong chain
           always has >=1 wrong link; the chain rate is the "did the attempt end
           at the right number" view.
Also per chain whose true value != target: does the model nonetheless claim the
target ("false hit")? And per chain whose true value == target: does it fail to
claim it ("missed hit")?

Tolerance: exact (1e-9 relative) when every number on both sides is an integer
and the link is `=`; otherwise (decimals, `...`, `≈`) the right side is compared
at its printed precision: |true - claimed| <= 10^-d, where d = max decimals on
the right, with a 1% relative floor for `≈` (models round loosely there).

Text is normalized first: ×/\\times/\\cdot -> *, ÷/\\div -> /, unicode minus,
LaTeX \\( \\) \\[ \\], markdown ** and <answer> tags stripped (some
runs wrap every candidate in its own <answer> tag), bullets / list numbers at line
start dropped, and lines that start with `=`/`≈` joined onto the previous line
(the multi-line layout above). Segments that do not parse (words, `1.` list
numbers, unbalanced parens, x/0, `**`) are not claims and are skipped, so this
undercounts claims rather than inventing errors.

Self-test: python3 scripts/probe_arith_chains.py --self-test
"""
import argparse
import re
import sys

ARITH = re.compile(r"[\d\.\s\+\-\*/\(\)]+")
NUM = re.compile(r"\d+(?:\.\d+)?")
SEP = re.compile(r"(=|≈)")


def normalize(text):
    t = text
    for a, b in (("\\times", "*"), ("×", "*"), ("\\cdot", "*"), ("·", "*"), ("÷", "/"),
                 ("\\div", "/"), ("−", "-"), ("–", "-"), ("\\(", " "), ("\\)", " "),
                 ("\\[", " "), ("\\]", " "), ("$", " "), ("**", " "), ("\\approx", "≈"),
                 ("==", "="), ("<answer>", " "), ("</answer>", " ")):
        t = t.replace(a, b)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)  # 1,000 -> 1000
    lines = []
    for ln in t.split("\n"):
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", ln)  # bullets / "1. " list markers
        if lines and re.match(r"^\s*[=≈]", ln):
            lines[-1] += " " + ln.strip()
        else:
            lines.append(ln)
    return "\n".join(lines)


def _value(seg):
    s = seg.strip()
    if not s or not re.search(r"\d", s) or "**" in s or "//" in s:
        return None
    s = s.replace("...", "")
    if re.search(r"\d\s+\d", s) or re.search(r"\d\.(?!\d)", s):
        return None  # "1 2" or a list number like "1."
    if s.count("(") != s.count(")"):
        return None
    try:
        v = eval(s, {"__builtins__": None}, {})
    except Exception:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _decimals(seg):
    return max((len(m.split(".")[1]) for m in NUM.findall(seg) if "." in m), default=0)


def link_ok(left, right, op, lv, rv):
    approx = op == "≈" or "..." in left or "..." in right
    d = _decimals(right)
    if not approx and d == 0 and _decimals(left) == 0:
        return abs(lv - rv) <= 1e-9 * max(1.0, abs(lv))
    tol = 10 ** (-d) if d else 0.5
    if d and re.search(r"\d\s*[-+*/]", right):
        # rounded decimals INSIDE an expression ("13 * 0.714" for 13 * 5/7):
        # the rounding error is scaled by the other operands -> relative tol
        tol *= max(1.0, abs(lv))
    if approx:
        tol = max(tol, 0.01 * abs(lv))
    return abs(lv - rv) <= tol + 1e-9


def chains(text):
    """Yield chains as lists of (segment_text, value, op_before) from normalized text.

    A response that does not end in </answer> was most likely cut at the token
    cap, and its last line can stop mid-number ("= 61 + 18" for 184), which
    would read as a miscomputation -- so that line is dropped."""
    lines = normalize(text).split("\n")
    if not text.rstrip().endswith("</answer>"):
        lines = lines[:-1]
    for line in lines:
        parts = SEP.split(line)
        if len(parts) < 3:
            continue
        # parts = [chunk0, op1, chunk1, op2, chunk2, ...]; the left expr of the
        # first link is the arithmetic SUFFIX of chunk0, each later segment is
        # the arithmetic PREFIX of its chunk; a chunk with trailing words ends
        # the chain (its prefix is the last segment).
        cur = []
        for i in range(0, len(parts), 2):
            chunk = parts[i]
            op = parts[i - 1] if i else None
            if i == 0:
                m = re.search(r"[\d\.\s\+\-\*/\(\)]*$", chunk)
                seg = m.group(0)
                # drop an unmatched leading ')' run or opening-paren-only junk
                seg = seg.lstrip(" )")
            else:
                m = ARITH.match(chunk)
                seg = m.group(0) if m else ""
            # a prefix like "29 (" (value then a parenthetical comment):
            # cut back to before the last unmatched '('
            if i:
                while seg.count("(") > seg.count(")"):
                    seg = seg[:seg.rfind("(")]
            # balance: trim unmatched trailing ')' from a prefix
            while seg.count(")") > seg.count("(") and seg.rstrip().endswith(")"):
                seg = seg.rstrip()[:-1]
            # trim unmatched leading '(' from a suffix
            while seg.count("(") > seg.count(")") and seg.lstrip().startswith("("):
                seg = seg.lstrip()[1:]
            v = _value(seg)
            if v is None:
                if len(cur) >= 2:
                    yield cur
                cur = []
                # a bad segment can still start the next link if it is a prefix
                continue
            cur.append((seg.strip(), v, op))
            rest = chunk[len(m.group(0)):] if i else ""
            if i and rest.strip() and i + 1 < len(parts):
                # words between this value and the next '=' break the chain;
                # the next link's left side is this chunk's arithmetic suffix
                if len(cur) >= 2:
                    yield cur
                tail = re.search(r"[\d\.\s\+\-\*/\(\)]*$", chunk).group(0).lstrip(" )")
                tv = _value(tail)
                cur = [(tail.strip(), tv, None)] if tv is not None else []
        if len(cur) >= 2:
            yield cur


def probe_text(text, target):
    """Counts for one response."""
    out = dict(links=0, link_err=0, chains=0, chain_err=0, false_hit=0, wrong_val_chains=0,
               missed_hit=0, target_val_chains=0)
    for ch in chains(text):
        out["chains"] += 1
        for (ls, lv, _), (rs, rv, op) in zip(ch, ch[1:]):
            out["links"] += 1
            out["link_err"] += not link_ok(ls, rs, op, lv, rv)
        first_s, first_v, _ = ch[0]
        last_s, last_v, last_op = ch[-1]
        ok = link_ok(first_s, last_s, last_op, first_v, last_v)
        out["chain_err"] += not ok
        if abs(first_v - target) > 1e-6:
            out["wrong_val_chains"] += 1
            out["false_hit"] += abs(last_v - target) < 1e-6
        else:
            out["target_val_chains"] += 1
            out["missed_hit"] += abs(last_v - target) > 1e-6
    return out


def self_test():
    cases = [
        ("(83 - 62 + 32) - 30\n= 53 + 32 - 30\n= 85 - 30\n= 55", 82, dict(links=3, link_err=1, chains=1, chain_err=1)),
        ("(59 * 2) - (2 * 92) = 118 - 184 = -66, which is not 22.", 22, dict(links=2, link_err=0, chains=1, chain_err=0)),
        ("(59 + 2) / (2 * 92) = (61) / (184) ≈ 0.33, which is not 22.", 22, dict(links=2, link_err=0)),
        ("(83 - 62) * (32 / 30)\n= (21) * (1.0666...)\n= 22.399...", 82, dict(links=2, link_err=0)),
        ("1. 45 + 3 = 48\n- 7 × 8 = 56\n\\( 10 \\div 4 = 2.5 \\)", 0, dict(links=3, link_err=0, chains=3)),
        ("So 6 * 7 = 42 and 42 - 2 = 40, close.", 40, dict(links=2, link_err=0, chains=2)),
        ("Attempt 1: 12 + 5 = 18", 17, dict(links=1, link_err=1, chain_err=1)),
        ("(45 - 24) / (25 - 26) = 21 / -1 = -21", 4, dict(links=2, link_err=0)),
        ("8 * 3 = 24 = target", 24, dict(links=1, link_err=0)),
        ("<answer>(50 / 2) + 4</answer>\n= 25 + 4\n= 29 (no)", 56, dict(links=2, link_err=0, chains=1)),
        ("(83 - 62) * (32 - 30) = 21 * 2 = 42 (This is not equal to 82)", 82, dict(links=2, link_err=0)),
        ("(41 - 28) * (5 / 7) = 13 * 0.714 = 9.28", 0, dict(links=2, link_err=0)),
        ("2548 / 112 ≈ 23.643", 0, dict(links=1, link_err=1)),
        ("12 + 5 = 17\n(61 + 2) * 3 = 61 + 18", 17, dict(links=1, link_err=0)),  # truncated tail
        ("50 + 49 = 100", 100, dict(links=1, link_err=1, false_hit=1)),
    ]
    bad = 0
    for text, tgt, want in cases:
        if "truncated" not in str(want) and not text.endswith("61 + 18"):
            text += "\n<answer>x</answer>"
        got = probe_text(text, tgt)
        diff = {k: (got[k], v) for k, v in want.items() if got[k] != v}
        if diff:
            bad += 1
            print(f"FAIL {text!r}: (got, want) {diff}  chains={list(chains(text))}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} passed")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
