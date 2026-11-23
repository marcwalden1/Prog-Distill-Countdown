#!/usr/bin/env python3
"""Score the few-shot base-model evals written by eval.py --n_shot.

Those JSONs live in results/baselines/<model>/<K>shot-<format>/, which the
global_step_*/round_*/sft_final globs in scripts/compute_passk.py and
plot_results.py deliberately do not match (a few-shot point is not a training
checkpoint). This walks that layout instead and adds the metrics that decide a
format choice:

  pass@1    fraction of prompts whose FIRST sample is exactly correct
  pass@32   fraction with at least one exactly-correct sample (coverage)
  fmt       fraction of samples carrying a parseable <answer> tag
  attempts  mean number of 'Try ... = v' lines per sample, and the fraction of
            samples making >= 2 attempts -- i.e. did the model actually adopt
            the multi-attempt format, or just the answer tag?
  chars     mean response length

Usage:
    python3 scripts/score_fewshot_evals.py                       # every model
    python3 scripts/score_fewshot_evals.py --model OLMo-2-0425-1B-pt4001B
    python3 scripts/score_fewshot_evals.py --csv analysis_data/fewshot.csv
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader_utils import compute_score, extract_solution  # noqa: E402

TRY_RE = re.compile(r"^\s*Try .+ = -?\d+", re.M)


def truncate_at_answer(text):
    """Cut at the first </answer>.

    A base model few-shot-prompted with K solved exemplars answers the real
    puzzle and then *continues the pattern*: it invents further puzzles and
    answers those too. `extract_solution` takes the LAST <answer> tag, so the
    graded answer would come from a hallucinated puzzle and pass@1 collapses to
    ~0 for mechanical reasons. Truncating here yields exactly the text a
    generation-time stop sequence on '</answer>' would have produced (the prefix
    is unchanged by what follows), so JSONs written before and after eval.py
    gained --stop_after_answer score identically.
    """
    i = text.find("</answer>")
    return text if i < 0 else text[:i + len("</answer>")]


def _score_prompt(args):
    outputs, target, nums = args
    outputs = [truncate_at_answer(o) for o in outputs]
    extra = {"numbers": nums}
    scores = [compute_score(None, o, target, extra, verbose=False) for o in outputs]
    n_try = [len(TRY_RE.findall(o)) for o in outputs]
    return {
        "pass1": 1.0 if scores[0] == 1.0 else 0.0,
        "pass32": 1.0 if any(s == 1.0 for s in scores) else 0.0,
        "fmt": sum(extract_solution(o) is not None for o in outputs) / len(outputs),
        "tries": sum(n_try) / len(n_try),
        "multi": sum(t >= 2 for t in n_try) / len(n_try),
        "chars": sum(len(o) for o in outputs) / len(outputs),
    }


def build(path, force=False, workers=8):
    cache = path + ".fsscores"
    if not force:
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(path):
                with open(cache) as f:
                    return json.load(f), True
        except Exception:
            pass
    with open(path) as f:
        data = json.load(f)
    items = [(d["outputs"], d["target"], d["nums"]) for d in data if d.get("outputs")]
    if not items:
        return None, False
    with Pool(workers) as pool:
        rows = pool.map(_score_prompt, items)
    rec = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    rec["n_prompts"] = len(rows)
    rec["n_samples"] = len(items[0][0])
    try:
        with open(cache, "w") as f:
            json.dump(rec, f)
    except Exception:
        pass
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--model", default="*")
    ap.add_argument("--dataset", default="*", help="eval dataset name filter")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pat = os.path.join(args.result_dir, "baselines", args.model, "*shot-*",
                       f"{args.dataset}_temp*_n*_max*.json")
    files = sorted(f for f in glob.glob(pat) if not f.endswith((".scores", ".fsscores")))
    if not files:
        raise SystemExit(f"ERROR: no few-shot eval JSONs matching {pat}")

    print(f"{'model':24s} {'shots':>6} {'format':8s} {'dataset':18s} "
          f"{'pass@1':>7} {'pass@32':>8} {'fmt':>6} {'tries':>6} {'multi':>6} {'chars':>7}")
    rows = []
    for f in files:
        parts = f.split(os.sep)
        model, shot_dir = parts[-3], parts[-2]
        k, fmt = shot_dir.split("shot-", 1)
        dataset = os.path.basename(f).split("_temp")[0]
        rec, hit = build(f, force=args.force, workers=args.workers)
        if rec is None:
            print(f"  SKIP (empty) {f}")
            continue
        print(f"{model:24s} {k:>6} {fmt:8s} {dataset:18s} "
              f"{rec['pass1']:7.4f} {rec['pass32']:8.4f} {rec['fmt']:6.3f} "
              f"{rec['tries']:6.2f} {rec['multi']:6.3f} {rec['chars']:7.0f}"
              f"{'' if not hit else '   [cached]'}", flush=True)
        rows.append(dict(model=model, shots=int(k), format=fmt, dataset=dataset, **rec))

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
