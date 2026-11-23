#!/usr/bin/env python3
"""Pareto frontier (n=5 OOD vs ID) for each method's BEST stable GRPO setting.

Same visual style as plot_gemma_selected_best_comparison.py. Each arm is shown
at its own OOD-n5-maximizing, non-collapsing hyperparameters:
  distill (KD):     SFT 1e-4, GRPO 1e-5, KL 3e-3
  progdistill (PD): SFT 1e-4, GRPO 3e-6, KL 1e-3
Metric: mean@1 (binary) read from cached .scores. Single seed.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_ROOT = "results/gemma-3-270m"
DISTILL_EXP = (
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-"
    "grpo-lr1e-5-kl3e-3-seed1"
)
PROGDISTILL_EXP = (
    "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-"
    "grpo-lr3e-6-kl1e-3-seed1"
)
OUTPUT = "figures/gemma-3-270m/pareto/pareto_balanced5_bestpermethod.png"


def step_from_path(path):
    return int(re.search(r"global_step_(\d+)", path).group(1))


def load_arm(exp):
    curves = {}
    for dataset in ("balanced", "balanced5"):
        pattern = os.path.join(
            RESULT_ROOT, exp, "global_step_*", f"{dataset}_temp*.json.scores"
        )
        values = {}
        for path in glob.glob(pattern):
            with open(path) as handle:
                _, mean_at_1, _ = json.load(handle)
            values[step_from_path(path)] = float(mean_at_1)
        curves[dataset] = values
    return curves


def trajectory(curves):
    steps = sorted(set(curves["balanced"]) & set(curves["balanced5"]))
    return (
        steps,
        [curves["balanced"][s] for s in steps],
        [curves["balanced5"][s] for s in steps],
    )


def plot(arms, output):
    colors = {
        "distill + GRPO": "#d62f27",
        "progdistill + GRPO": "#247db3",
    }
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for label, (steps, xs, ys) in arms.items():
        color = colors[label]
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=2, zorder=1)
        ax.scatter(xs, ys, color=color, s=62, label=label, zorder=2)
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
                str(step), (x, y), fontsize=8, alpha=0.75,
                xytext=(4, 4), textcoords="offset points",
            )

    ax.set_xlabel("In-dist accuracy (balanced, n=3,4) mean@1", fontsize=11)
    ax.set_ylabel("OOD accuracy (balanced5, n=5) mean@1", fontsize=11)
    ax.set_title(
        "Strongest observed Gemma trajectories, each method's best setting - n=5 (OOD)\n"
        "distill: SFT 1e-4, GRPO 1e-5, KL 3e-3  |  "
        "progdistill: SFT 1e-4, GRPO 3e-6, KL 1e-3\n"
        "open circle = first observed checkpoint   star = last observed checkpoint",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.28)
    footer = (
        "Each arm at its own OOD-n5-maximizing non-collapsing setting; single seed. "
        "Progdistill balanced5 step 300 is missing.\n"
        "Source: central .scores caches (mean@1, binary)."
    )
    fig.text(0.5, 0.014, footer, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.replace(".png", ".pdf"))
    plt.close(fig)
    print(output)


def main():
    arms = {
        "distill + GRPO": trajectory(load_arm(DISTILL_EXP)),
        "progdistill + GRPO": trajectory(load_arm(PROGDISTILL_EXP)),
    }
    plot(arms, OUTPUT)


if __name__ == "__main__":
    main()
