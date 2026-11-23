#!/usr/bin/env python3
"""
Export full-set ID/OOD eval scores for one or more runs into a tidy CSV.

Motivation: the runs we want to compare can live in different result trees, and
some trees still carry `.scores` caches written before the MAX_PROMPTS=1000 cap
was removed (see research_logs/2026-07-13_fix-maxprompts-cap-fullset-regen.md).
Those capped caches over-state n=5/n=6 accuracy by up to ~12 points, so mixing
them with full-set caches distorts a cross-run comparison.

This script re-scores the FULL prompt set of every eval JSON it touches and
caches per-file results in its own cache dir (keyed by path + mtime + size), so
it never needs write access to the source tree and never trusts a foreign
`.scores` file.

Metrics per (run, dataset, step) — all four from one scoring pass:

    pass@1 / pass@32   unbiased pass@k estimator (Chen et al. 2021), matching
                       scripts/plot/plot_passk.py:  mean over prompts of
                       1 - C(n-c, k) / C(n, k), with c = correct samples (score
                       1.0) out of n. pass@1 reduces to mean c/n; pass@32 (=n)
                       reduces to the fraction of prompts with >=1 correct sample.
    mean@1 / mean@32   the older repo metrics, kept for continuity: mean@1 is
                       "outputs[0] exactly correct", mean@32 averages the raw
                       compute_score (0 / 0.1 / 1.0, i.e. partial credit for
                       valid-but-wrong format).

Usage:
    python3 scripts/export_idood_scores.py \
        --run "teacher GRPO=/path/to/results::Qwen2.5-1.5B::balanced-grpo-lr3e-6-kl3e-3-seed1" \
        --run "student GRPO=results::Qwen2.5-0.5B::balanced-grpo-seed1" \
        --out analysis_data/idood_scores.csv

Output columns: run, model, exp_name, dataset, step,
                pass_at_1, pass_at_32, mean_at_1, mean_at_32, n_prompts, n_samples
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
from math import comb
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader_utils import compute_score

DATASETS = ["balanced", "balanced5", "balanced6"]
EVAL_TEMP = "0.6"
EVAL_N = "32"
CACHE_VERSION = "v2-passk"  # bump when the per-file record shape changes


def _score_item(args):
    """Score one prompt's n outputs -> (mean@1 term, mean@32 term, #correct, n)."""
    outputs, target, nums = args
    if not outputs:
        return None
    extra = {"numbers": nums}
    scores = [compute_score(None, out, target, extra, verbose=False) for out in outputs]
    mean1 = float(scores[0] == 1.0)
    n_correct = sum(s == 1.0 for s in scores)
    return mean1, sum(scores) / len(scores), n_correct, len(scores)


def pass_at_k(counts, n, k):
    """Unbiased pass@k averaged over prompts (Chen et al. 2021)."""
    k = min(k, n)
    return sum(1.0 - comb(n - c, k) / comb(n, k) for c in counts) / len(counts)


def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"round_(\d+)", path)
    if m:
        return int(m.group(1))
    return 1600 if f"{os.sep}sft_final{os.sep}" in path else None


def _cache_key(path):
    st = os.stat(path)
    raw = f"{CACHE_VERSION}|{os.path.realpath(path)}|{int(st.st_mtime)}|{st.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()


def score_file(path, cache_dir, nproc):
    """Metrics dict over the FULL prompt set, with local cache."""
    cache_path = os.path.join(cache_dir, _cache_key(path) + ".json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            rec = json.load(f)
        print(f"    [cache hit] n={rec['n_prompts']}", flush=True)
        return rec

    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    print(f"    json load {time.time()-t0:.0f}s ({len(data)} prompts)", flush=True)

    items = [(it["outputs"], it["target"], it["nums"]) for it in data if it.get("outputs")]
    if not items:
        return None
    t0 = time.time()
    with Pool(nproc) as pool:
        scored = [s for s in pool.map(_score_item, items) if s is not None]
    print(f"    scored {len(scored)} prompts x{EVAL_N} in {time.time()-t0:.0f}s", flush=True)
    if not scored:
        return None

    n_prompts = len(scored)
    # Prompts can in principle carry different sample counts; pass@k needs a single
    # n, so use the minimum across prompts (same convention as plot_passk.py).
    n_samples = min(s[3] for s in scored)
    counts = [min(s[2], n_samples) for s in scored]
    rec = {"pass_at_1": pass_at_k(counts, n_samples, 1),
           "pass_at_32": pass_at_k(counts, n_samples, 32),
           "mean_at_1": sum(s[0] for s in scored) / n_prompts,
           "mean_at_32": sum(s[1] for s in scored) / n_prompts,
           "n_prompts": n_prompts,
           "n_samples": n_samples}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(rec, f)
    return rec


def parse_run(spec):
    """'label=result_dir::model::exp' -> (label, result_dir, model, exp)"""
    label, _, rest = spec.partition("=")
    parts = rest.split("::")
    if not label or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"bad --run {spec!r}; expected 'label=result_dir::model::exp'")
    return (label, *parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_run, required=True,
                    help="label=result_dir::model::exp (repeatable)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default="analysis_data/idood_scores.csv")
    ap.add_argument("--cache-dir", default="analysis_data/.fullset_score_cache")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--max-tokens", action="append", default=[], metavar="DATASET=N",
                    help="read DATASET's evals from the max<N>/ subdir that eval.py "
                         "writes for a non-1024 generation cap (repeatable)")
    args = ap.parse_args()
    max_tokens = dict(spec.split("=", 1) for spec in args.max_tokens)

    rows = []
    for label, result_dir, model, exp in args.run:
        for dataset in args.datasets:
            if dataset in max_tokens:
                mt = max_tokens[dataset]
                pattern = os.path.join(result_dir, model, exp, "*", f"max{mt}",
                                       f"{dataset}_temp{EVAL_TEMP}_n{EVAL_N}_max{mt}.json")
            else:
                pattern = os.path.join(result_dir, model, exp, "*",
                                       f"{dataset}_temp{EVAL_TEMP}_n{EVAL_N}_*.json")
            paths = [p for p in glob.glob(pattern) if _step_from_path(p) is not None]
            paths.sort(key=_step_from_path)
            print(f"[{label}] {dataset}: {len(paths)} eval file(s)", flush=True)
            for p in paths:
                step = _step_from_path(p)
                print(f"  step {step}: {p}", flush=True)
                rec = score_file(p, args.cache_dir, args.nproc)
                if rec is None:
                    print("    [skip] no scorable prompts", flush=True)
                    continue
                print(f"    pass@1={rec['pass_at_1']:.4f} pass@32={rec['pass_at_32']:.4f}"
                      f"  mean@1={rec['mean_at_1']:.4f} mean@32={rec['mean_at_32']:.4f}"
                      f"  (n_prompts={rec['n_prompts']}, n_samples={rec['n_samples']})",
                      flush=True)
                rows.append({"run": label, "model": model, "exp_name": exp,
                             "dataset": dataset, "step": step, **rec})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "model", "exp_name", "dataset", "step",
                                          "pass_at_1", "pass_at_32",
                                          "mean_at_1", "mean_at_32",
                                          "n_prompts", "n_samples"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
