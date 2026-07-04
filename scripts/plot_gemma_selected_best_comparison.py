#!/usr/bin/env python3
"""Plot the selected strongest observed Gemma distill/progdistill trajectories."""

import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISTILL_EXP = (
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-"
    "grpo-lr1e-5-kl1e-3-seed1"
)
PROGDISTILL_EXP = "balanced-grpo-from-progdistill-lr1e-6-kl3e-3-seed1"
CSV_PATH = "analysis_data/pareto_eval_progression.csv"
RESULT_ROOT = "results/gemma-3-270m"
OUTPUT = (
    "figures/gemma-3-270m/pareto/"
    "pareto_balanced5_selected-best-observed.png"
)
ZOOM_OUTPUT = (
    "figures/gemma-3-270m/pareto/"
    "pareto_balanced5_selected-best-observed-filtered.png"
)


def step_from_path(path):
    return int(re.search(r"global_step_(\d+)", path).group(1))


def load_distill():
    curves = {}
    for dataset in ("balanced", "balanced5"):
        pattern = os.path.join(
            RESULT_ROOT,
            DISTILL_EXP,
            "global_step_*",
            f"{dataset}_temp*.json.scores",
        )
        values = {}
        for path in glob.glob(pattern):
            with open(path) as handle:
                _, mean_at_1, _ = json.load(handle)
            values[step_from_path(path)] = float(mean_at_1)
        curves[dataset] = values
    return curves


def load_progdistill():
    curves = {"balanced": {}, "balanced5": {}}
    with open(CSV_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["model"] == "gemma-3-270m"
                and row["exp_name"] == PROGDISTILL_EXP
                and row["dataset"] in curves
            ):
                curves[row["dataset"]][int(row["step"])] = float(
                    row["mean_at_1"]
                )
    return curves


def trajectory(curves):
    steps = sorted(set(curves["balanced"]) & set(curves["balanced5"]))
    return (
        steps,
        [curves["balanced"][step] for step in steps],
        [curves["balanced5"][step] for step in steps],
    )


def plot(arms, output, filtered=False):
    colors = {
        "distill + GRPO": "#d62f27",
        "progdistill + GRPO": "#247db3",
    }

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for label, (steps, xs, ys) in arms.items():
        if filtered and label == "distill + GRPO":
            kept = [
                (step, x, y)
                for step, x, y in zip(steps, xs, ys)
                if step <= 1050
            ]
            steps, xs, ys = map(list, zip(*kept))
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
        "Selected strongest observed Gemma trajectories - n=5 (OOD)\n"
        "distill: SFT 1e-4, GRPO 1e-5, KL 1e-3  |  "
        "progdistill: SFT 1e-4, GRPO 1e-6, KL 3e-3\n"
        "open circle = first observed checkpoint   star = "
        + ("last presented checkpoint" if filtered else "last observed checkpoint"),
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.28)
    if filtered:
        ax.set_xlim(0.73, 0.90)
        ax.set_ylim(0.125, 0.32)
    footer = (
        "Distill steps 1200, 1400, and 1600 omitted after performance collapse; "
        "step 300 balanced5 is missing.\n"
        if filtered
        else
        "All observed points retained. Distill balanced5 step 300 is missing.\n"
    )
    footer += (
        "Sources: central .scores caches (distill) and curated CSV "
        "(progdistill)."
    )
    fig.text(
        0.5,
        0.014,
        footer,
        ha="center",
        fontsize=7.5,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def main():
    arms = {
        "distill + GRPO": trajectory(load_distill()),
        "progdistill + GRPO": trajectory(load_progdistill()),
    }
    plot(arms, OUTPUT)
    plot(arms, ZOOM_OUTPUT, filtered=True)


if __name__ == "__main__":
    main()
