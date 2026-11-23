#!/usr/bin/env python3
"""
ID-vs-OOD Pareto trajectories for an arbitrary set of runs, from the tidy CSV
written by scripts/export_idood_scores.py (full-set scores — no 1000-prompt cap).

x = in-distribution accuracy (balanced, n=3,4)
y = OOD accuracy (balanced5 -> n=5, balanced6 -> n=6)
One marker per evaluated checkpoint, connected in step order; open circle = first
checkpoint, star = last.

Output — one subfolder per plot set (figures/<out-dir>/id_vs_ood_<tag>/):
    grid.png                 2x2 grid: rows = metric, cols = OOD set
    <metric>_<ood>.png       the four panels standalone

Usage:
    python3 scripts/plot/plot_id_vs_ood_runs.py \
        --csv analysis_data/idood_teacher1p5b_vs_student05b.csv \
        --tag teacher1p5b_vs_student05b --out-dir comparison
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from colors import COLORS

ID_DATASET = "balanced"
OOD_LABEL = {"balanced5": "n=5", "balanced6": "n=6"}
METRIC_LABEL = {"pass_at_1": "pass@1", "pass_at_32": "pass@32",
                "mean_at_1": "mean@1", "mean_at_32": "mean@32"}
SURFACE = "#fcfcfb"
INK = "#1c1b19"
INK_MUTED = "#6b6a66"
# fallbacks for runs whose label isn't in the repo's semantic map (colors.py)
FALLBACK = ["#2a78d6", "#1baf7a", "#d6772a", "#9b59b6"]


def run_color(label, i):
    if label in COLORS:
        return COLORS[label]
    # tolerate a parenthetical model suffix, e.g. "teacher GRPO (Qwen2.5-1.5B)"
    base = label.split(" (")[0].strip()
    return COLORS.get(base, FALLBACK[i % len(FALLBACK)])


def draw_panel(ax, df, runs, ood, metric, annotate=True):
    for i, label in enumerate(runs):
        color = run_color(label, i)
        sub = df[df["run"] == label]
        id_by_step = dict(zip(sub[sub.dataset == ID_DATASET].step,
                              sub[sub.dataset == ID_DATASET][metric]))
        ood_by_step = dict(zip(sub[sub.dataset == ood].step,
                               sub[sub.dataset == ood][metric]))
        steps = sorted(set(id_by_step) & set(ood_by_step))
        if not steps:
            print(f"  [skip] {label}: no checkpoint has both {ID_DATASET} and {ood}")
            continue
        xs = [id_by_step[s] for s in steps]
        ys = [ood_by_step[s] for s in steps]

        ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=2, zorder=1)
        ax.scatter(xs, ys, s=44, color=color, edgecolors=SURFACE, linewidths=1.2,
                   label=f"{label}  ({len(steps)} ckpts)", zorder=3)
        ax.scatter([xs[0]], [ys[0]], s=115, facecolors="none", edgecolors=color,
                   linewidths=1.8, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=185, marker="*", color=color,
                   edgecolors=INK, linewidths=0.6, zorder=5)
        if annotate:
            # Alternate the offset by index: consecutive checkpoints are the ones
            # that bunch up, and flipping every other label apart keeps them legible.
            for j, (s, x, y) in enumerate(zip(steps, xs, ys)):
                dx, dy = (5, 4) if j % 2 == 0 else (5, -9)
                ax.annotate(str(s), (x, y), fontsize=6.5, color=INK_MUTED,
                            textcoords="offset points", xytext=(dx, dy), zorder=6)

    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcec9")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel(f"in-distribution (n=3,4)  {METRIC_LABEL[metric]}",
                  fontsize=10, color=INK)
    ax.set_ylabel(f"OOD {OOD_LABEL[ood]}  {METRIC_LABEL[metric]}",
                  fontsize=10, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="comparison",
                    help="subdir under figures/")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--oods", nargs="+", default=["balanced5", "balanced6"])
    ap.add_argument("--metrics", nargs="+", default=["pass_at_1", "pass_at_32"],
                    choices=list(METRIC_LABEL))
    ap.add_argument("--no-annotate", action="store_true",
                    help="drop the per-point step labels")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # keep CSV order of first appearance so color assignment is stable
    runs = list(dict.fromkeys(df["run"]))
    print(f"runs: {runs}")

    # one subfolder per plot set: figures/<out-dir>/id_vs_ood_<tag>/{grid,<panel>}.png
    out_root = os.path.join(args.figures_dir, args.out_dir, f"id_vs_ood_{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    annotate = not args.no_annotate

    # --- combined grid: rows = metric, cols = OOD set ---
    fig, axes = plt.subplots(len(args.metrics), len(args.oods),
                             figsize=(6.0 * len(args.oods), 5.2 * len(args.metrics)),
                             facecolor=SURFACE, squeeze=False)
    for r, metric in enumerate(args.metrics):
        for c, ood in enumerate(args.oods):
            print(f"[grid] {METRIC_LABEL[metric]} / {OOD_LABEL[ood]}")
            draw_panel(axes[r][c], df, runs, ood, metric, annotate)
            axes[r][c].set_title(f"{METRIC_LABEL[metric]} — ID (n=3,4) vs OOD {OOD_LABEL[ood]}",
                                 fontsize=11, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9.5,
               title="open circle = first ckpt, star = last ckpt (labels = training step)",
               title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    grid_path = os.path.join(out_root, "grid.png")
    fig.savefig(grid_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {grid_path}")

    # --- standalone panels ---
    for metric in args.metrics:
        for ood in args.oods:
            f, ax = plt.subplots(figsize=(6.6, 5.8), facecolor=SURFACE)
            draw_panel(ax, df, runs, ood, metric, annotate)
            ax.set_title(f"{METRIC_LABEL[metric]} — ID (n=3,4) vs OOD {OOD_LABEL[ood]}",
                         fontsize=11.5, color=INK)
            ax.legend(loc="upper left", frameon=False, fontsize=9)
            f.tight_layout()
            p = os.path.join(out_root,
                             f"{METRIC_LABEL[metric].replace('@','at')}"
                             f"_{OOD_LABEL[ood].replace('=','')}.png")
            f.savefig(p, facecolor=SURFACE, bbox_inches="tight")
            plt.close(f)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
