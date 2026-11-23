#!/usr/bin/env python3
"""Write `.passk_multi` caches next to eval JSONs: the correct-count histogram.

`compute_passk.py` caches only pass@1 and pass@N, because it short-circuits at
the first correct sample. Any *other* k needs to know how many of a prompt's N
samples were correct, so this script grades all N and caches the histogram

    hist[c] = number of prompts with exactly c of N samples exactly correct

from which pass@k for EVERY k <= N follows with no re-grading (see
`scripts/plot/plot_passk_curve.py:pass_at_k` for the estimator). 33 integers per
eval file, so the cache is ~1 KB regardless of prompt count.

Cache format (JSON):
    {"step": int, "n_prompts": int, "n_samples": int, "hist": [int] * (N+1)}

Written alongside, not instead of, `.passk` — existing readers are untouched.

    python3 scripts/compute_passk_multi.py --model OLMo-2-0425-1B-pt4001B \
        --exp balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1 [--force]

`--temp` selects which eval files to score (default 0.6); the temperature is
part of the eval filename, so caches for different temperatures never collide.
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from grader_utils import compute_score  # noqa: E402


def _n_correct(args):
    """(n_samples, n_correct) for one prompt. No short-circuit: every sample is
    graded, which is the whole point of this cache."""
    outputs, target, nums = args
    if not outputs:
        return None
    extra = {"numbers": nums}
    c = sum(compute_score(None, o, target, extra, verbose=False) == 1.0 for o in outputs)
    return (len(outputs), c)


def _step(path):
    """Same convention as compute_passk.py / plot_results._extract_eval_step."""
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"round_(\d+)", path)
    if m:
        return int(m.group(1))
    if "/sft_final/" in path:
        return 1600
    return None


def build(path, force=False, workers=8):
    cache = path + ".passk_multi"
    if not force:
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(path):
                with open(cache) as f:
                    return json.load(f), True
        except Exception:
            pass
    step = _step(path)
    if step is None:
        return None, False
    with open(path) as f:
        data = json.load(f)
    items = [(d["outputs"], d["target"], d["nums"]) for d in data if d.get("outputs")]
    if not items:
        return None, False
    with Pool(workers) as pool:
        counts = [c for c in pool.map(_n_correct, items) if c is not None]
    if not counts:
        return None, False
    # Prompts whose sample count differs from the mode are dropped rather than
    # silently folded into the wrong histogram: a ragged file would make pass@k
    # an average over different N, which the estimator does not define.
    n_samples = max(set(n for n, _ in counts), key=[n for n, _ in counts].count)
    ragged = sum(1 for n, _ in counts if n != n_samples)
    hist = [0] * (n_samples + 1)
    for n, c in counts:
        if n == n_samples:
            hist[c] += 1
    rec = {"step": step, "n_prompts": sum(hist), "n_samples": n_samples, "hist": hist}
    if ragged:
        rec["dropped_ragged"] = ragged
    try:
        with open(cache, "w") as f:
            json.dump(rec, f)
    except Exception:
        pass
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--temp", default="0.6", help="eval temperature to score")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pats = [os.path.join(args.result_dir, args.model, args.exp, sub,
                         f"*_temp{args.temp}_n*_max*.json")
            for sub in ("global_step_*", "round_*", "sft_final")]
    files = sorted(f for p in pats for f in glob.glob(p)
                   if not f.endswith((".scores", ".lengths", ".passk", ".passk_multi")))
    if not files:
        raise SystemExit(f"ERROR: no temp{args.temp} eval JSONs under "
                         f"{args.result_dir}/{args.model}/{args.exp}")

    print(f"{args.model}/{args.exp}: {len(files)} eval files at temp {args.temp}")
    for f in files:
        rec, hit = build(f, force=args.force, workers=args.workers)
        if rec is None:
            print(f"  SKIP {os.path.relpath(f, args.result_dir)}")
            continue
        h, n = rec["hist"], rec["n_prompts"]
        N = rec["n_samples"]
        tag = "cached" if hit else "computed"
        print(f"  {tag:>8}  step {rec['step']:>5}  n={n:<5} N={N:<4} "
              f"never-solved {h[0] / n:.3f}  always-solved {h[N] / n:.3f}  "
              f"{os.path.basename(f).split('_temp')[0]}", flush=True)


if __name__ == "__main__":
    main()
