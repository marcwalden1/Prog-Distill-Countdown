#!/usr/bin/env python3
"""Plot the best completed canonical Gemma GRPO trajectories from the CSV."""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EXPS = {
    "distill->GRPO": "balanced-grpo-from-distill-lr1e-6-kl3e-3-seed1",
    "progdistill->GRPO": "balanced-grpo-from-progdistill-lr1e-6-kl3e-3-seed1",
}
COLORS = {"distill->GRPO": "#d62728", "progdistill->GRPO": "#1f77b4"}


def load_trajectory(csv_path, exp_name, ood_dataset):
    values = {"balanced": {}, ood_dataset: {}}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "gemma-3-270m" or row["exp_name"] != exp_name:
                continue
            if row["dataset"] in values:
                values[row["dataset"]][int(row["step"])] = float(row["mean_at_1"])

    steps = sorted(set(values["balanced"]) & set(values[ood_dataset]))
    return (
        steps,
        [values["balanced"][step] for step in steps],
        [values[ood_dataset][step] for step in steps],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="analysis_data/pareto_eval_progression.csv"
    )
    parser.add_argument("--ood-dataset", default="balanced5")
    parser.add_argument(
        "--output",
        default=(
            "figures/gemma-3-270m/pareto/"
            "pareto_balanced5_best-completed-canonical.png"
        ),
    )
    args = parser.parse_args()

    ood_label = "n=5 (OOD)" if args.ood_dataset == "balanced5" else "n=6 (far OOD)"
    fig, ax = plt.subplots(figsize=(8.4, 7.2))

    for label, exp_name in DEFAULT_EXPS.items():
        steps, xs, ys = load_trajectory(args.csv, exp_name, args.ood_dataset)
        color = COLORS[label]
        ax.plot(xs, ys, "-", color=color, alpha=0.48, linewidth=2, zorder=1)
        ax.scatter(xs, ys, s=62, color=color, label=label, zorder=2)
        ax.scatter(
            [xs[0]], [ys[0]], s=180, facecolors="none",
            edgecolors=color, linewidths=2.2, zorder=3,
        )
        ax.scatter(
            [xs[-1]], [ys[-1]], s=260, marker="*", color=color,
            edgecolors="black", linewidths=0.7, zorder=4,
        )
        for step, x, y in zip(steps, xs, ys):
            ax.annotate(
                str(step), (x, y), fontsize=8, alpha=0.72,
                xytext=(4, 4), textcoords="offset points",
            )

    ax.set_xlabel("In-dist accuracy (balanced, n=3,4) mean@1", fontsize=11)
    ax.set_ylabel(f"OOD accuracy ({ood_label}) mean@1", fontsize=11)
    ax.set_title(
        f"Best completed canonical Gemma trajectories - {ood_label}\n"
        "SFT LR=1e-4, GRPO LR=1e-6, KL=3e-3\n"
        "o = first checkpoint   * = last checkpoint",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.28)
    fig.text(
        0.5,
        0.012,
        "Selection: highest n=5 Pareto hypervolume within the four completed "
        "canonical runs in analysis_data/pareto_eval_progression.csv",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
