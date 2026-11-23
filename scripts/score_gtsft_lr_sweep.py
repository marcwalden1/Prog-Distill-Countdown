#!/usr/bin/env python3
"""Score a GT-SFT LR sweep and print the LR-selection table (pass@1 / pass@32).

Builds the `.passk` caches via scripts/compute_passk.py's own `build()` — the same
code path used for the GRPO curves — so the sweep table and the trajectory plots
cannot disagree. (An earlier ad-hoc scorer for the 1B sweep capped OOD at 1000
prompts and got some mean@1 cells wrong; going through one shared implementation is
the point of this script.)

pass@1  = fraction of prompts whose FIRST sample is exactly correct (== mean@1)
pass@32 = fraction with >=1 exactly-correct sample of 32 (coverage)

Usage:
    python3 scripts/score_gtsft_lr_sweep.py --model-prefix OLMo-2-1124-7B \
        --arms pt101B,pt252B,pt500B,pt1007B --lrs 1e-5,3e-5,1e-4,3e-4
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_passk import build  # noqa: E402

DATASETS = [("balanced-simple", "ID n=3,4"),
            ("balanced5-simple", "n=5"),
            ("balanced6-simple", "n=6")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-prefix", default="OLMo-2-1124-7B")
    ap.add_argument("--arms", default="pt101B,pt252B,pt500B,pt1007B")
    ap.add_argument("--lrs", default="1e-5,3e-5,1e-4,3e-4")
    ap.add_argument("--exp-template", default="balanced-simple-gtsft{steps}-lr{lr}-seed{seed}")
    ap.add_argument("--steps", default="200")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arms = args.arms.split(",")
    lrs = args.lrs.split(",")

    rows, missing = {}, []
    for arm in arms:
        model = f"{args.model_prefix}-{arm}"
        for lr in lrs:
            exp = args.exp_template.format(steps=args.steps, lr=lr, seed=args.seed)
            cells = []
            for ds, _ in DATASETS:
                g = glob.glob(os.path.join(args.result_dir, model, exp, "sft_final",
                                           f"{ds}_temp*_n32_max*.json"))
                g = [x for x in g if not x.endswith((".scores", ".passk"))]
                if not g:
                    cells.append(None)
                    missing.append(f"{arm}/lr{lr}/{ds}")
                    continue
                rec, _ = build(g[0], force=args.force)
                cells.append((rec[1], rec[2]) if rec else None)
            rows[(arm, lr)] = cells

    w = max(len(a) for a in arms) + 2
    hdr = f"{'arm':<{w}}{'sft_lr':<8}" + "".join(f"{l:>24}" for _, l in DATASETS)
    print("\n" + hdr); print("-" * len(hdr))
    for arm in arms:
        for lr in lrs:
            cs = rows[(arm, lr)]
            line = f"{arm:<{w}}{lr:<8}"
            for c in cs:
                line += f"{'--':>24}" if c is None else f"{c[0]:>11.4f} /{c[1]:>10.4f}"
            print(line)
        print()
    print("(each cell: pass@1 / pass@32)")

    # Best LR per arm on ID pass@1, plus the best single LR shared across all arms —
    # the matched choice is what keeps a ladder comparison clean.
    print("\nBest LR per arm (ID pass@1):")
    for arm in arms:
        avail = [(lr, rows[(arm, lr)][0]) for lr in lrs if rows[(arm, lr)][0]]
        if avail:
            lr, c = max(avail, key=lambda t: t[1][0])
            print(f"  {arm:<{w}} {lr}   pass@1 {c[0]:.4f}")
    print("\nMatched LR (mean ID pass@1 across arms with data):")
    scored = []
    for lr in lrs:
        vals = [rows[(a, lr)][0][0] for a in arms if rows[(a, lr)][0]]
        if vals:
            scored.append((sum(vals) / len(vals), lr, len(vals)))
    for m, lr, n in sorted(scored, reverse=True):
        print(f"  {lr:<8} mean {m:.4f}  (over {n}/{len(arms)} arms)")
    if scored:
        print(f"  -> use lr={sorted(scored, reverse=True)[0][1]} for a matched ladder")

    if missing:
        print(f"\nWARNING: {len(missing)} eval file(s) missing; those cells show '--'")
        for m in missing[:12]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
