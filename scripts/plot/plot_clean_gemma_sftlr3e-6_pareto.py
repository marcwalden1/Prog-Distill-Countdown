#!/usr/bin/env python3
"""Plot the audited Gemma sftlr=3e-6 Pareto comparison from local score caches."""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_ROOT = "results/gemma-3-270m"
OUTPUT_ROOT = "figures/wandb_pareto_sftlr"
EXPERIMENTS = {
    "distill→GRPO": (
        "gemma-270m-balanced-distill-teacher-Qwen1.5B-"
        "sftlr3e-6-grpo-lr1e-6-kl3e-3-seed1"
    ),
    "progdistill→GRPO": (
        "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-"
        "sftlr3e-6-grpo-lr1e-6-kl3e-3-seed1"
    ),
}
DATASETS = {
    "balanced5": "n=5 (OOD)",
    "balanced6": "n=6 (far OOD)",
}
COLORS = {"distill→GRPO": "#d62728", "progdistill→GRPO": "#1f77b4"}


def load_scores(experiment, dataset):
    pattern = os.path.join(
        RESULT_ROOT,
        experiment,
        "global_step_*",
        f"{dataset}_temp0.6_n32_max1024.json.scores",
    )
    values = {}
    for path in glob.glob(pattern):
        step = int(re.search(r"global_step_(\d+)", path).group(1))
        with open(path) as handle:
            values[step] = float(json.load(handle)[1])
    return values


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    id_scores = {
        arm: load_scores(experiment, "balanced")
        for arm, experiment in EXPERIMENTS.items()
    }

    for dataset, dataset_label in DATASETS.items():
        ood_scores = {
            arm: load_scores(experiment, dataset)
            for arm, experiment in EXPERIMENTS.items()
        }
        common_steps = sorted(
            set(id_scores["distill→GRPO"])
            & set(id_scores["progdistill→GRPO"])
            & set(ood_scores["distill→GRPO"])
            & set(ood_scores["progdistill→GRPO"])
        )

        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        for arm in EXPERIMENTS:
            xs = [id_scores[arm][step] for step in common_steps]
            ys = [ood_scores[arm][step] for step in common_steps]
            color = COLORS[arm]
            ax.plot(xs, ys, "-", color=color, alpha=0.48, linewidth=2)
            ax.scatter(xs, ys, s=62, color=color, label=arm, zorder=2)
            ax.scatter(
                [xs[0]], [ys[0]], s=180, facecolors="none",
                edgecolors=color, linewidths=2.2, zorder=3,
            )
            ax.scatter(
                [xs[-1]], [ys[-1]], s=260, marker="*", color=color,
                edgecolors="black", linewidths=0.7, zorder=4,
            )
            for step, x, y in zip(common_steps, xs, ys):
                ax.annotate(
                    str(step), (x, y), fontsize=8, alpha=0.72,
                    xytext=(4, 4), textcoords="offset points",
                )

        ax.set_xlabel("In-dist accuracy (balanced, n=3,4) mean@1", fontsize=11)
        ax.set_ylabel(f"OOD accuracy ({dataset_label}) mean@1", fontsize=11)
        ax.set_title(
            "Gemma-3-270M — student SFT LR=3e-6\n"
            f"Pareto: distill→GRPO vs progdistill→GRPO — {dataset_label}\n"
            "○ = step 150   ★ = last shared checkpoint (1200)",
            fontsize=12,
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.28)
        fig.text(
            0.5,
            0.012,
            "Audited local .scores | temp=0.6, n=32, max_tokens=1024 | "
            "GRPO lr=1e-6, KL=3e-3, seed=1",
            ha="center",
            fontsize=8,
            style="italic",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        output = os.path.join(
            OUTPUT_ROOT, f"gemma_sftlr3e-6_kl3e-3_{dataset}.png"
        )
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(f"{output}: shared steps={common_steps}")


if __name__ == "__main__":
    main()
