#!/usr/bin/env python3
"""Generate checkpoint-level Pareto plots directly from W&B eval histories."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


PROJECT = "progressive_distill/prog_distill"
METRICS = {
    "balanced5": ("eval/n3_4/mean_at_1", "eval/n5/mean_at_1", "n=5 (OOD)"),
    "balanced6": ("eval/n3_4/mean_at_1", "eval/n6/mean_at_1", "n=6 (far OOD)"),
}
PAIRS = [
    {
        "slug": "gemma_sftlr3e-6_kl3e-3",
        "model": "Gemma-3-270M",
        "sft_lr": "3e-6",
        "grpo": "GRPO lr=1e-6, KL=3e-3",
        "distill": "c9c84d55",
        "progdistill": "ead0b10d",
    },
    {
        "slug": "qwen_sftlr3e-6_kl3e-4",
        "model": "Qwen2.5-0.5B",
        "sft_lr": "3e-6",
        "grpo": "GRPO lr=1e-6, KL=3e-4",
        "distill": "e1782dc4",
        "progdistill": "7cfa74ab",
    },
    {
        "slug": "qwen_sftlr1e-5_kl3e-4",
        "model": "Qwen2.5-0.5B",
        "sft_lr": "1e-5",
        "grpo": "GRPO lr=1e-6, KL=3e-4",
        "distill": "5970d980",
        "progdistill": "fe04c1f0",
    },
    {
        "slug": "qwen_sftlr3e-5_kl3e-4",
        "model": "Qwen2.5-0.5B",
        "sft_lr": "3e-5",
        "grpo": "GRPO lr=1e-6, KL=3e-4",
        "distill": "e8762b2c",
        "progdistill": "31759ccd",
    },
]


def load_run(api, run_id):
    run = api.run(f"{PROJECT}/{run_id}")
    history = run.history(samples=10000, pandas=True)
    return run, history


def trajectory(history, x_metric, y_metric):
    rows = history[["_step", x_metric, y_metric]].dropna()
    rows = rows.sort_values("_step").drop_duplicates("_step", keep="last")
    return (
        [int(step) for step in rows["_step"]],
        rows[x_metric].astype(float).tolist(),
        rows[y_metric].astype(float).tolist(),
    )


def plot_pair(pair, dataset, histories, output_dir):
    x_metric, y_metric, ood_label = METRICS[dataset]
    arms = {
        "distill→GRPO": trajectory(histories["distill"], x_metric, y_metric),
        "progdistill→GRPO": trajectory(histories["progdistill"], x_metric, y_metric),
    }
    colors = {"distill→GRPO": "#d62728", "progdistill→GRPO": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    for name, (steps, xs, ys) in arms.items():
        color = colors[name]
        ax.plot(xs, ys, "-", color=color, alpha=0.48, linewidth=2, zorder=1)
        ax.scatter(xs, ys, s=62, color=color, label=name, zorder=2)
        if steps:
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
        f"{pair['model']} — student SFT LR={pair['sft_lr']}\n"
        f"Pareto: distill→GRPO vs progdistill→GRPO — {ood_label}\n"
        "○ = first available checkpoint   ★ = last available checkpoint",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.28)
    fig.text(
        0.5, 0.012,
        f"{pair['grpo']} | W&B runs: {pair['distill']} / {pair['progdistill']}",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(output_dir, f"{pair['slug']}_{dataset}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(
        f"{path}: distill={len(arms['distill→GRPO'][0])}, "
        f"progdistill={len(arms['progdistill→GRPO'][0])}"
    )
    return path


def make_contact_sheet(paths, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13.5))
    for ax, path in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    output_dir = os.path.join("figures", "wandb_pareto_sftlr")
    os.makedirs(output_dir, exist_ok=True)
    api = wandb.Api(timeout=60)

    sheets = {"balanced5": [], "balanced6": []}
    for pair in PAIRS:
        histories = {}
        for arm in ("distill", "progdistill"):
            run, history = load_run(api, pair[arm])
            histories[arm] = history
            print(f"{pair['slug']} {arm}: {run.name}")
        for dataset in METRICS:
            sheets[dataset].append(
                plot_pair(pair, dataset, histories, output_dir)
            )

    for dataset, paths in sheets.items():
        make_contact_sheet(
            paths, os.path.join(output_dir, f"all_sftlr_{dataset}.png")
        )


if __name__ == "__main__":
    main()
