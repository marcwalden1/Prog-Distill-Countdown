#!/usr/bin/env python3
"""
ID-vs-OOD trajectory plot for a GRPO KL sweep (same student, same lr, different KL).

x-axis: in-distribution accuracy (balanced, n=3,4)
y-axis: OOD accuracy (balanced5 = n=5, balanced6 = n=6)
One trajectory per KL arm; each point is a training checkpoint.
Grid: rows = metric (mean@1, mean@32), cols = OOD dataset (n=5, n=6).

Reads ONLY the tiny `.scores` caches ([step, mean@1, mean@32]) that
plot_results.py leaves next to each result JSON.

Usage:
    python3 scripts/plot/plot_id_vs_ood_kl_sweep.py \
        --model gemma-3-270m \
        --arm "kl=3e-3:balanced-grpo-from-progdistill-mixture-sftlr3e-4-lr3e-6-kl3e-3-seed1" \
        --arm "kl=1e-3:balanced-grpo-from-progdistill-mixture-sftlr3e-4-lr3e-6-kl1e-3-seed1" \
        --out figures/gemma-3-270m/pareto/id_vs_ood_kl_sweep.png
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_DATASET = "balanced"
OOD_DATASETS = ["balanced5", "balanced6"]
OOD_LABEL = {"balanced5": "n=5 (OOD)", "balanced6": "n=6 (far OOD)"}
METRICS = [("mean@1", 1), ("mean@32", 2)]
ARM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"round_(\d+)", path)
    return int(m.group(1)) if m else None


def load_scores(result_dir, model, exp, dataset, metric_idx):
    """{step: metric} from the .scores caches for one (exp, dataset)."""
    pattern = os.path.join(result_dir, model, exp, "global_step_*",
                           f"{dataset}_temp*.json.scores")
    out = {}
    for sp in glob.glob(pattern):
        step = _step_from_path(sp)
        if step is None:
            continue
        try:
            with open(sp) as f:
                rec = json.load(f)            # [step, mean@1, mean@32]
            out[step] = float(rec[metric_idx])
        except Exception:
            continue
    return out


def trajectory(id_by_step, ood_by_step):
    steps = sorted(set(id_by_step) & set(ood_by_step))
    return steps, [id_by_step[s] for s in steps], [ood_by_step[s] for s in steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--model", default="gemma-3-270m")
    ap.add_argument("--arm", action="append", required=True,
                    help='"label:exp_name", repeatable (one per KL arm)')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = []
    for a in args.arm:
        label, exp = a.split(":", 1)
        arms.append((label, exp))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "figures", args.model, "pareto", "id_vs_ood_kl_sweep.png")

    fig, axes = plt.subplots(len(METRICS), len(OOD_DATASETS),
                             figsize=(6.4 * len(OOD_DATASETS), 5.8 * len(METRICS)),
                             squeeze=False)

    for r, (mname, midx) in enumerate(METRICS):
        for c, ood in enumerate(OOD_DATASETS):
            ax = axes[r][c]
            for i, (label, exp) in enumerate(arms):
                id_by = load_scores(args.result_dir, args.model, exp, ID_DATASET, midx)
                ood_by = load_scores(args.result_dir, args.model, exp, ood, midx)
                steps, xs, ys = trajectory(id_by, ood_by)
                if not steps:
                    print(f"[{label}] {mname} {ood}: no matched checkpoints")
                    continue
                col = ARM_COLORS[i % len(ARM_COLORS)]
                ax.plot(xs, ys, "-", color=col, alpha=0.45, zorder=1)
                ax.scatter(xs, ys, s=40, color=col, label=label, zorder=2)
                ax.scatter([xs[0]], [ys[0]], s=120, facecolors="none",
                           edgecolors=col, linewidths=1.8, zorder=3)
                ax.scatter([xs[-1]], [ys[-1]], s=180, marker="*", color=col,
                           edgecolors="k", linewidths=0.5, zorder=4)
                for s, x, y in zip(steps, xs, ys):
                    ax.annotate(str(s), (x, y), fontsize=6, alpha=0.65,
                                xytext=(3, 3), textcoords="offset points")
                print(f"[{label}] {mname} {ood}: {len(steps)} checkpoints")
            ax.set_xlabel(f"In-dist ({ID_DATASET}, n=3,4)  {mname}")
            ax.set_ylabel(f"OOD ({OOD_LABEL[ood]})  {mname}")
            ax.set_title(f"{OOD_LABEL[ood]}  —  {mname}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"ID vs OOD trajectory — {args.model} GRPO KL sweep\n"
                 "○ = first ckpt   ★ = last ckpt (1600)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
