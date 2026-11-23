#!/usr/bin/env python3
"""Write `.passk` caches next to eval JSONs: [step, pass@1, pass@32, n_prompts].

Distinct from the `.scores` caches that eval.sh/plot_results.py produce:

  mean@32  (in .scores) — average raw compute_score() over all 32 samples, where
                          scores are 0.0 / 0.1 / 1.0. Mixes format credit with
                          correctness, so it is NOT a pass rate.
  pass@32  (here)       — fraction of prompts where AT LEAST ONE of the 32 samples
                          is exactly correct (score == 1.0). A coverage metric.

pass@1 is the fraction whose FIRST sample is exactly correct — numerically
identical to mean@1, and recomputed here so both axes come from one pass.

Usage:
    python3 scripts/compute_passk.py --model OLMo-2-0425-1B-pt4001B \
        --exp balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1 [--force]
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader_utils import compute_score  # noqa: E402


def _passk_item(args):
    """(first_correct, any_correct) for one prompt's samples."""
    outputs, target, nums = args
    if not outputs:
        return None
    extra = {"numbers": nums}
    first = compute_score(None, outputs[0], target, extra, verbose=False) == 1.0
    if first:
        return (1.0, 1.0)  # short-circuit: already covered
    for o in outputs[1:]:
        if compute_score(None, o, target, extra, verbose=False) == 1.0:
            return (0.0, 1.0)
    return (0.0, 0.0)


def _step(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"round_(\d+)", path)
    if m:
        return int(m.group(1))
    if "/sft_final/" in path:
        return 1600           # same convention as plot_results._extract_eval_step
    return None


def build(path, force=False, workers=8):
    cache = path + ".passk"
    if not force:
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(path):
                with open(cache) as f:
                    return tuple(json.load(f)), True
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
        scored = [s for s in pool.map(_passk_item, items) if s is not None]
    n = len(scored)
    rec = (step, sum(s[0] for s in scored) / n, sum(s[1] for s in scored) / n, n)
    try:
        with open(cache, "w") as f:
            json.dump(list(rec), f)
    except Exception:
        pass
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pats = [os.path.join(args.result_dir, args.model, args.exp, sub, "*_temp*_n32_max*.json")
            for sub in ("global_step_*", "round_*", "sft_final")]
    files = sorted(f for p in pats for f in glob.glob(p) if not f.endswith((".scores", ".passk")))
    if not files:
        raise SystemExit(f"ERROR: no eval JSONs under {args.result_dir}/{args.model}/{args.exp}")

    print(f"{args.model}/{args.exp}: {len(files)} eval files")
    for f in files:
        rec, hit = build(f, force=args.force, workers=args.workers)
        if rec is None:
            print(f"  SKIP {os.path.relpath(f, args.result_dir)}")
            continue
        tag = "cached" if hit else "computed"
        print(f"  {tag:>8}  step {rec[0]:>5}  pass@1 {rec[1]:.4f}  pass@32 {rec[2]:.4f}  "
              f"n={rec[3]}  {os.path.basename(f).split('_temp')[0]}", flush=True)


if __name__ == "__main__":
    main()
