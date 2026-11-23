#!/usr/bin/env python3
"""Plot a GT-SFT LR sweep: pass@1 and pass@32 vs pretraining budget, one line per LR.

Reads only the `.passk` caches (built by scripts/compute_passk.py, same code path as
the trajectory plots), so the figures and scripts/score_gtsft_lr_sweep.py's table
cannot disagree.

Deliberately no mean@32: with scores in {0, 0.1, 1.0} and near-perfect format
compliance it reduces to 0.1 + 0.9*mean@1, so it adds nothing over pass@1 while
inflating the numbers (see research_logs/2026-08-25_olmo2-pretrain-ladder-grpo.md).

Usage:
  python3 scripts/plot/plot_gtsft_lr_sweep.py --model-prefix OLMo-2-1124-7B \
      --arms pt101B,pt252B,pt500B,pt1007B --lrs 1e-5,3e-5,1e-4,3e-4 \
      --datasets balanced-simple --tag 7b
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LR_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
DS_LABEL = {"balanced-simple": "in-distribution (n=3,4)",
            "balanced5-simple": "OOD n=5", "balanced6-simple": "OOD n=6"}


def tokens_of(tag):
    m = re.search(r"pt(\d+)B", tag)
    return int(m.group(1)) if m else None


def load(result_dir, model, exp, ds):
    pat = os.path.join(result_dir, model, exp, "sft_final", f"{ds}_temp*.json.passk")
    for p in sorted(glob.glob(pat)):
        try:
            with open(p) as f:
                rec = json.load(f)
            if isinstance(rec, list) and len(rec) >= 3:
                return rec[1], rec[2]
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-prefix", default="OLMo-2-1124-7B")
    ap.add_argument("--arms", default="pt101B,pt252B,pt500B,pt1007B")
    ap.add_argument("--lrs", default="1e-5,3e-5,1e-4,3e-4")
    ap.add_argument("--datasets", default="balanced-simple")
    ap.add_argument("--exp-template", default="balanced-simple-gtsft{steps}-lr{lr}-seed{seed}")
    ap.add_argument("--steps", default="200")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--figures-dir", default="figures/gtsft-lr-sweep")
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    arms = args.arms.split(",")
    lrs = args.lrs.split(",")
    datasets = args.datasets.split(",")
    xs = [tokens_of(a) for a in arms]

    for ds in datasets:
        data = {}
        for lr in lrs:
            pts = []
            for arm, x in zip(arms, xs):
                model = f"{args.model_prefix}-{arm}"
                exp = args.exp_template.format(steps=args.steps, lr=lr, seed=args.seed)
                v = load(args.result_dir, model, exp, ds)
                if v:
                    pts.append((x, v[0], v[1]))
            data[lr] = pts
        if not any(data.values()):
            print(f"  no .passk data for {ds}; skipping")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
        for mi, (name, ax) in enumerate(zip(["pass@1", "pass@32"], axes)):
            for ci, lr in enumerate(lrs):
                pts = data[lr]
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1 + mi] for p in pts], "-o",
                        color=LR_COLORS[ci % len(LR_COLORS)], ms=6, lw=1.6,
                        label=f"SFT lr={lr}")
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("pretraining tokens (B, log scale)")
            ax.set_ylabel(name)
            ax.set_title(f"{name} — {DS_LABEL.get(ds, ds)}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        title = args.title or f"{args.model_prefix}: GT-SFT {args.steps} steps, LR sweep"
        fig.suptitle(f"{title}\n{DS_LABEL.get(ds, ds)}", fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
        os.makedirs(args.figures_dir, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        out = os.path.join(args.figures_dir, f"lr_sweep_{ds}{suffix}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
