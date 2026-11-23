#!/usr/bin/env python3
"""GRPO on OLMo-2 GT-SFT students: 7B vs 1B, pass@1 vs pass@32.

The finding this figure exists to show: GRPO sharpens (pass@1 up) while it
collapses coverage (pass@32 down), and it does so identically at 1B and 7B —
every arm converges to roughly the same endpoint regardless of model size or
pretraining budget.

Reads analysis_data/olmo_7b_vs_1b_grpo.csv (written by the inline exporter in
the same commit; regenerate by re-running compute_passk.py then that snippet).

    python3 scripts/plot/plot_olmo_7b_vs_1b_grpo.py

Writes figures/olmo2_ladder/7b_vs_1b_grpo_{trajectories,scatter}.png
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "analysis_data/olmo_7b_vs_1b_grpo.csv"
# One folder per comparison set; these two figures live alongside the
# plot_olmo2_ladder_pareto.py figures for the same 7B-vs-1B comparison.
OUTDIR = os.environ.get("OUTDIR", "figures/olmo2-ladder/7b-vs-1b-sftgrpo-lr3e-6kl3e-3")

# Categorical slots 1-2 of the validated default palette (light mode).
# Colour encodes model size — the comparison the figure is about; line style
# encodes the pretraining checkpoint within a size. Two hues keeps the palette
# inside the all-pairs floors (CVD dE 24.7, normal-vision 33.6).
C_1B = "#2a78d6"
C_7B = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

# (size, tokens) -> (colour, linestyle, label)
SERIES = {
    ("1B", 1007): (C_1B, "--", "1B · 1007B tok"),
    ("1B", 4001): (C_1B, "-", "1B · 4001B tok"),
    ("7B", 126): (C_7B, "--", "7B · 126B tok"),
    ("7B", 135): (C_7B, "-", "7B · 135B tok"),
}
ORDER = [("1B", 1007), ("1B", 4001), ("7B", 126), ("7B", 135)]
DATASETS = [("n34", "In-dist  n=3,4"), ("n5", "OOD  n=5"), ("n6", "OOD  n=6")]
METRICS = [("pass1", "pass@1"), ("pass32", "pass@32")]


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["step"] = int(r["step"])
        r["tokens"] = int(r["tokens"])
        r["pass1"] = float(r["pass1"])
        r["pass32"] = float(r["pass32"])
    return rows


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def trajectories(rows):
    """Six panels: metric x dataset, one line per model, SFT start marked."""
    sft = {(r["size"], r["tokens"], r["dataset"]): r for r in rows if r["kind"] == "sft"}
    grpo = [r for r in rows if r["kind"] == "grpo" and r["lr"] == "3e-6" and r["kl"] == "3e-3"]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ri, (ds, ds_label) in enumerate(DATASETS):
        for ci, (metric, m_label) in enumerate(METRICS):
            ax = axes[ri][ci]
            style_axis(ax)
            for key in ORDER:
                colour, ls, label = SERIES[key]
                pts = sorted(
                    [r for r in grpo if (r["size"], r["tokens"]) == key and r["dataset"] == ds],
                    key=lambda r: r["step"],
                )
                if not pts:
                    continue
                xs = [0] + [p["step"] for p in pts]
                base = sft.get((key[0], key[1], ds))
                ys = [base[metric] if base else pts[0][metric]] + [p[metric] for p in pts]
                ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=2, label=label,
                        marker="o", markersize=3.5, markeredgewidth=0, zorder=3)
                # SFT start: hollow marker with a surface ring so overlaps stay legible
                ax.plot([0], [ys[0]], marker="o", markersize=8, color=colour,
                        markerfacecolor="#fcfcfb", markeredgewidth=2, zorder=4)

            if ri == 0:
                ax.set_title(m_label, color=INK, fontsize=12, fontweight="bold", pad=10)
            if ci == 0:
                ax.set_ylabel(ds_label, color=INK, fontsize=10, fontweight="bold")
            if ri == 2:
                ax.set_xlabel("GRPO step   (0 = SFT student)", color=INK_2, fontsize=9)
            ax.set_xlim(-60, 1680)

    # Per-row shared y so the pass@1 / pass@32 panels are directly comparable.
    for ri in range(3):
        lo = min(a.get_ylim()[0] for a in axes[ri])
        hi = max(a.get_ylim()[1] for a in axes[ri])
        for a in axes[ri]:
            a.set_ylim(lo, hi)

    handles = [Line2D([], [], color=SERIES[k][0], linestyle=SERIES[k][1], linewidth=2,
                      marker="o", markersize=4, label=SERIES[k][2]) for k in ORDER]
    handles.append(Line2D([], [], color=INK_2, marker="o", markersize=8, linestyle="none",
                          markerfacecolor="#fcfcfb", markeredgewidth=2, label="SFT student (step 0)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.928), fontsize=9, labelcolor=INK_2)

    fig.suptitle("GRPO sharpens and collapses in equal measure — at 1B and 7B alike",
                 color=INK, fontsize=14, fontweight="bold", y=0.988)
    fig.text(0.5, 0.955, "OLMo-2 GT-SFT students, GRPO lr 3e-6 / KL 3e-3, seed 1  ·  "
                         "pass@1 roughly doubles while pass@32 roughly halves, on every split",
             ha="center", color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/7b_vs_1b_grpo_trajectories.png"
    fig.savefig(p, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return p


def scatter(rows):
    """pass@1 vs pass@32: the sharpening/coverage trade-off as one path per model."""
    sft = {(r["size"], r["tokens"], r["dataset"]): r for r in rows if r["kind"] == "sft"}
    grpo = [r for r in rows if r["kind"] == "grpo" and r["lr"] == "3e-6" and r["kl"] == "3e-3"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    fig.patch.set_facecolor("#fcfcfb")

    for ci, (ds, ds_label) in enumerate(DATASETS):
        ax = axes[ci]
        style_axis(ax)
        for key in ORDER:
            colour, ls, label = SERIES[key]
            pts = sorted(
                [r for r in grpo if (r["size"], r["tokens"]) == key and r["dataset"] == ds],
                key=lambda r: r["step"],
            )
            if not pts:
                continue
            base = sft.get((key[0], key[1], ds))
            xs = ([base["pass1"]] if base else []) + [p["pass1"] for p in pts]
            ys = ([base["pass32"]] if base else []) + [p["pass32"] for p in pts]
            ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.6, alpha=0.85,
                    marker="o", markersize=3.5, markeredgewidth=0, zorder=3)
            if base:
                ax.plot([xs[0]], [ys[0]], marker="o", markersize=9, color=colour,
                        markerfacecolor="#fcfcfb", markeredgewidth=2, zorder=5)
            ax.plot([xs[-1]], [ys[-1]], marker="s", markersize=7, color=colour,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=5)

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color=GRID, linewidth=1, linestyle=":", zorder=1)
        ax.set_title(ds_label, color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("pass@1", color=INK_2, fontsize=9.5)
        if ci == 0:
            ax.set_ylabel("pass@32", color=INK_2, fontsize=9.5)

    handles = [Line2D([], [], color=SERIES[k][0], linestyle=SERIES[k][1], linewidth=2, label=SERIES[k][2])
               for k in ORDER]
    handles += [
        Line2D([], [], color=INK_2, marker="o", markersize=9, linestyle="none",
               markerfacecolor="#fcfcfb", markeredgewidth=2, label="SFT student"),
        Line2D([], [], color=INK_2, marker="s", markersize=7, linestyle="none", label="step 1600"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=9, labelcolor=INK_2)
    fig.suptitle("Every arm walks the same way: right and down, off the SFT frontier",
                 color=INK, fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p = f"{OUTDIR}/7b_vs_1b_grpo_scatter.png"
    fig.savefig(p, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return p


if __name__ == "__main__":
    rows = load()
    for p in (trajectories(rows), scatter(rows)):
        print("wrote", p)
