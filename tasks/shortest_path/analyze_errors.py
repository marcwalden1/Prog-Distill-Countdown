"""Error taxonomy for shortest-path generations.

    # GRPO training rollouts (verl dumps <step>.jsonl in the exp dir)
    python3 tasks/shortest_path/analyze_errors.py --rollouts <exp_dir> --steps 1-10 225-234

    # eval.py outputs (one group per --group; files in a group are pooled, e.g. d=2-4 + d=5 + d=6)
    python3 tasks/shortest_path/analyze_errors.py \
        --group step50 results/.../global_step_50/max512/*.json \
        --group step234 results/.../global_step_234/max512/*.json

Each sample is put in exactly one class:
    correct          a shortest path (reward 1.0)
    no_answer        no <answer>...</answer>            (truncated at the cap or never closed)
    malformed        <answer> content isn't "X -> Y -> ..."
    wrong_endpoints  doesn't start at source / end at target (edges may also be bad)
    reversed_edge    uses a -> b where only b -> a exists
    absent_node      uses a label that is not in the graph
    missing_edge     uses a -> b between real nodes with no edge either way
    too_long         a valid source->target walk, but longer than the shortest distance

For paths with endpoints right, "first_bad_hop" is the 1-based hop where the path
first leaves every shortest path (the edge is missing, or it doesn't move one
step closer to the target).
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphs import bfs  # noqa: E402
from reward import extract_answer, parse_path  # noqa: E402

CLASSES = ["correct", "no_answer", "malformed", "wrong_endpoints", "reversed_edge",
           "absent_node", "missing_edge", "too_long"]


def classify(text, gt):
    ans = extract_answer(text)
    if ans is None:
        return "no_answer", None
    path = parse_path(ans)
    if path is None:
        return "malformed", None
    src, dst = [str(x) for x in gt["edges_src"]], [str(x) for x in gt["edges_dst"]]
    edges = set(zip(src, dst))
    nodes = set(src) | set(dst)
    s, t, d = str(gt["source"]), str(gt["target"]), int(gt["distance"])
    if path[0] != s or path[-1] != t:
        return "wrong_endpoints", None
    # Distance to target via BFS on the reversed graph.
    radj = collections.defaultdict(list)
    for a, b in edges:
        radj[b].append(a)
    to_t, _ = bfs(radj, t)
    first_bad = None
    for i, (a, b) in enumerate(zip(path, path[1:]), 1):
        if (a, b) not in edges or to_t.get(b, 1e9) != to_t.get(a, 1e9) - 1:
            first_bad = i
            break
    bad = [(a, b) for a, b in zip(path, path[1:]) if (a, b) not in edges]
    if not bad:
        return ("correct" if len(path) == d + 1 else "too_long"), first_bad
    if any(a not in nodes or b not in nodes for a, b in bad):
        return "absent_node", first_bad
    if any((b, a) in edges for a, b in bad):
        return "reversed_edge", first_bad
    return "missing_edge", first_bad


def parse_steps(specs):
    out = set()
    for s in specs:
        a, _, b = s.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return sorted(out)


def load_rollouts(exp_dir, steps):
    for st in steps:
        p = os.path.join(exp_dir, f"{st}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                yield st, r["output"], r["gts"]


def load_results(paths):
    """eval.py JSONs, aligned with data/<ds>/test.parquet by position (as score_passk.py)."""
    import pandas as pd
    for path in paths:
        ds = os.path.basename(path).split("_temp")[0]
        df = pd.read_parquet(f"data/{ds}/test.parquet")
        with open(path) as f:
            recs = json.load(f)
        assert len(recs) == len(df), path
        for rec, gt in zip(recs, df["reward_model"].map(lambda r: r["ground_truth"])):
            for o in rec["outputs"]:
                yield ds, o, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", help="exp dir holding <step>.jsonl")
    ap.add_argument("--steps", nargs="+", default=[], help="e.g. 1-10 225-234 (each group reported separately)")
    ap.add_argument("--group", nargs="+", action="append", default=[], metavar=("LABEL", "JSON"),
                    help="eval.py result files to pool under LABEL; repeatable")
    ap.add_argument("--examples", type=int, default=2, help="examples to print per error class (last group)")
    args = ap.parse_args()

    groups = [(spec, lambda spec=spec: load_rollouts(args.rollouts, parse_steps([spec]))) for spec in args.steps]
    groups += [(g[0], lambda g=g: load_results(g[1:])) for g in args.group]
    last = groups[-1][0] if groups else None
    for spec, loader in groups:
        cnt = collections.Counter()
        by_d = collections.defaultdict(collections.Counter)
        hop = collections.defaultdict(collections.Counter)
        lens, tail, think = [], 0, 0
        examples = collections.defaultdict(list)
        n = 0
        for st, text, gt in loader():
            c, fb = classify(text, gt)
            d = int(gt["distance"])
            cnt[c] += 1
            by_d[d][c] += 1
            if fb is not None and c != "correct":
                hop[d][fb] += 1
            lens.append(len(text))
            think += "<think>" in text
            i = text.rfind("</answer>")
            tail += i >= 0 and bool(text[i + 9:].strip())
            if len(examples[c]) < args.examples:
                examples[c].append((st, gt, text))
            n += 1
        if not n:
            print(f"{spec}: no samples found")
            continue
        print(f"\n=== {spec}: {n} samples | mean chars {sum(lens)/n:.0f} | "
              f"uses <think> {think/n:.1%} | text after </answer> {tail/n:.1%}")
        print(f"{'class':16s} {'all':>7s} " + " ".join(f"{'d=' + str(d):>7s}" for d in sorted(by_d)))
        for c in CLASSES:
            row = f"{c:16s} {cnt[c]/n:7.1%} "
            row += " ".join(f"{by_d[d][c]/max(1, sum(by_d[d].values())):7.1%}" for d in sorted(by_d))
            print(row)
        print("first bad hop (errors with correct endpoints), per d:")
        for d in sorted(hop):
            tot = sum(hop[d].values())
            print(f"  d={d}: " + "  ".join(f"hop{h}:{v/tot:.0%}" for h, v in sorted(hop[d].items())) + f"  (n={tot})")
        if spec == last and args.examples:
            for c in CLASSES[1:]:
                for st, gt, text in examples[c]:
                    edges = ", ".join(f"{a}->{b}" for a, b in zip(gt["edges_src"], gt["edges_dst"]))
                    print(f"\n--- {c} ({st}, d={gt['distance']}, {gt['source']}->{gt['target']})\n"
                          f"edges: {edges}\noutput: {text[:400]!r}")


if __name__ == "__main__":
    main()
