#!/usr/bin/env python3
"""
Combined 2D Pareto plot: overlays ALL distill->GRPO (red shades) and
progdistill->GRPO (blue shades) trajectories for a given model on one figure
per OOD dataset (balanced5 = n5, balanced6 = n6).

This is the "everything in one plot" view across the several hyperparameter
combinations that were previously plotted separately by plot_pareto_2d.py.

Usage:
    python3 scripts/plot_pareto_2d_combined.py --model Qwen2.5-0.5B
    python3 scripts/plot_pareto_2d_combined.py --model gemma-3-270m
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pareto_2d import (ID_DATASET, OOD_DATASETS, OOD_LABEL,
                             load_mean1_by_step, trajectory)

# Each entry: (exp_name, role, label). role in {"distill", "progdistill"}.
CURVES = {
    "Qwen2.5-0.5B": [
        ("balanced-distill-cross-shlokbest-n4-sft3e-5-frozen-pre-grpo-kl3e-4-lr1e-6-seed1",
         "distill", "distill (sftlr3e-5, kl3e-4)", "sftlr=3e-5"),
        ("balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1",
         "distill", "distill (sftlr1e-5, kl3e-4)", "sftlr=1e-5"),
        ("balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-3-seed1",
         "distill", "distill (sftlr1e-5, kl3e-3)", "sftlr=1e-5,kl=3e-3"),
        ("balanced-progdistill-sftlr1e-6-grpo-lr1e-6-kl3e-4-seed1",
         "progdistill", "progdistill (sftlr1e-6, kl3e-4)", "sftlr=1e-6"),
        ("balanced-progdistill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1",
         "progdistill", "progdistill (sftlr3e-5, kl3e-4)", "sftlr=3e-5"),
        ("balanced-progdistill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1",
         "progdistill", "progdistill (sftlr1e-5, kl3e-4)", "sftlr=1e-5"),
    ],
    "gemma-3-270m": [
        ("balanced-grpo-from-distill-lr1e-6-kl3e-3-seed1",
         "distill", "distill (kl3e-3)", "kl=3e-3"),
        ("balanced-grpo-from-distill-lr1e-6-kl3e-4-seed1",
         "distill", "distill (kl3e-4)", "kl=3e-4"),
        ("balanced-grpo-from-progdistill-lr1e-6-kl3e-3-seed1",
         "progdistill", "progdistill (kl3e-3)", "kl=3e-3"),
        ("balanced-grpo-from-progdistill-lr1e-6-kl3e-4-seed1",
         "progdistill", "progdistill (kl3e-4)", "kl=3e-4"),
    ],
}

DISTILL_COLOR = "#d62728"
PROGDISTILL_COLOR = "#1f77b4"


def plot_combined(ood_dataset, model, curves_data, out_path, x_min=None):
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    for exp, role, label, short_tag, steps, xs, ys in curves_data:
        if not steps:
            continue
        c = DISTILL_COLOR if role == "distill" else PROGDISTILL_COLOR
        # filter by minimum in-dist accuracy
        if x_min is not None:
            pts = [(s, x, y) for s, x, y in zip(steps, xs, ys) if x >= x_min]
            if not pts:
                continue
            steps, xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=c, alpha=0.5, zorder=1)
        ax.scatter(xs, ys, s=36, color=c, label=label, zorder=2)
        ax.scatter([xs[0]], [ys[0]], s=110, facecolors="none",
                   edgecolors=c, linewidths=1.6, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=160, marker="*", color=c,
                   edgecolors="k", linewidths=0.5, zorder=4)
        for s, x, y in zip(steps, xs, ys):
            ax.annotate(str(s), (x, y), fontsize=6.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.annotate(short_tag, (xs[-1], ys[-1]), fontsize=7.5, color=c,
                    fontweight="bold", xytext=(5, -10), textcoords="offset points")
    ax.set_xlabel(f"In-dist accuracy  ({ID_DATASET}, n=3,4)  mean@1")
    ax.set_ylabel(f"OOD accuracy  ({OOD_LABEL[ood_dataset]})  mean@1")
    ax.set_title(f"Pareto (all combos): distill→GRPO (red) vs progdistill→GRPO (blue)\n"
                 f"{model}  —  {OOD_LABEL[ood_dataset]}\n"
                 "○ = first ckpt (150)   ★ = last ckpt (1600)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/n/netscratch/kdbrantley_lab/Lab/mwalden/"
                                             "rl-skill-comp-results/results")
    ap.add_argument("--model", required=True, choices=list(CURVES.keys()))
    ap.add_argument("--x-min", type=float, default=None,
                    help="drop checkpoints with in-dist mean@1 below this threshold")
    ap.add_argument("--figures-dir", default=None,
                    help="default: figures/<model>/pareto")
    args = ap.parse_args()

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", args.model, "pareto")

    id_curves = {}
    for exp, role, label, short_tag in CURVES[args.model]:
        c = load_mean1_by_step(args.result_dir, args.model, exp, ID_DATASET)
        id_curves[exp] = c
        print(f"[{label}] {ID_DATASET}: {len(c)} checkpoints with scores")

    for ood in OOD_DATASETS:
        curves_data = []
        for exp, role, label, short_tag in CURVES[args.model]:
            ood_curve = load_mean1_by_step(args.result_dir, args.model, exp, ood)
            steps, xs, ys = trajectory(id_curves[exp], ood_curve)
            print(f"[{label}] {ood}: {len(ood_curve)} ckpts -> {len(steps)} matched with ID")
            curves_data.append((exp, role, label, short_tag, steps, xs, ys))
        out_path = os.path.join(figures_dir, f"pareto_{ood}_all_combined.png")
        plot_combined(ood, args.model, curves_data, out_path, x_min=args.x_min)


if __name__ == "__main__":
    main()
