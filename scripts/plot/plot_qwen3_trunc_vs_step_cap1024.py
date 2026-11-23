#!/usr/bin/env python3
"""
Qwen3 base GRPO, n=3,4 (balanced-simple), generation cap 1024: accuracy and
truncation rate vs training step.

  pass_vs_step_n34_cap1024.png   2x2: rows = all | non-truncated responses,
                                 columns = pass@1 | pass@32
  trunc_vs_step_n34_cap1024.png  fraction of responses that hit the cap

Reads analysis_data/qwen3_n34_trunc_passk.csv (written by
plot_qwen3_trunc_pass1_vs_pass32.py; metric definitions are in its docstring).
CPU only, seconds.

Usage:
    python3 scripts/plot/plot_qwen3_trunc_vs_step_cap1024.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_qwen3_size_comparison as base  # styling + run list
import plot_qwen3_trunc_pass1_vs_pass32 as tr

CAP = 1024
OUT_DIR = os.path.join(tr.REPO, "figures", "Qwen3-base-grpo")
OUT_PASS = os.path.join(OUT_DIR, f"pass_vs_step_n34_cap{CAP}.png")
OUT_TRUNC = os.path.join(OUT_DIR, f"trunc_vs_step_n34_cap{CAP}.png")


def plot_runs(ax, df, runs, col):
    drawn = set()
    for model, exp, label, color, marker, ls in runs:
        d = df[(df.model == model) & (df.exp_name == exp)].sort_values("step")
        if d.empty:
            continue
        ax.plot(d.step, d[col], color=color, linestyle=ls, linewidth=2,
                marker=marker, markersize=5.5, markeredgecolor=base.SURFACE,
                markeredgewidth=1, label=label, zorder=3)
        drawn.add(label)
    ax.axvline(tr.EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    return drawn


def epoch_label(ax):
    ax.text(tr.EPOCH_STEPS, 0.02, " 1 epoch", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=base.INK_MUTED, va="bottom")


def legend(fig, runs, drawn, y):
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, y),
               handlelength=3)


def fig_pass(df, runs):
    cols = [("pass_at_1", "pass@1"), ("pass_at_32", "pass@32")]
    rows = [("", "all responses"), ("nt_", "non-truncated responses only")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6), sharex=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for r, (pre, rname) in enumerate(rows):
        for c, (metric, mname) in enumerate(cols):
            ax = axes[r][c]
            drawn |= plot_runs(ax, df, runs, pre + metric)
            if r == 0 and c == 0:
                epoch_label(ax)
            note = (f"responses truncated: {df.trunc_rate.min():.0%}-{df.trunc_rate.max():.0%}"
                    if r == 0 else
                    f"prompts with >=1 finished response: "
                    f"{(df.nt_n_prompts / df.n_prompts).min():.0%}-"
                    f"{(df.nt_n_prompts / df.n_prompts).max():.0%}")
            ytxt = mname if r == 0 or metric == "pass_at_1" else "any correct among finished"
            base.style_axes(ax, f"{ytxt}  ·  {rname}", "training step" if r == 1 else "",
                            ytxt, note)
    # shared y within each metric column, so the truncation filter reads as a vertical offset
    for c in range(2):
        lo = min(axes[r][c].get_ylim()[0] for r in range(2))
        hi = max(axes[r][c].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r][c].set_ylim(lo, min(hi, 1.005))
    legend(fig, runs, drawn, 0.005)
    fig.suptitle(f"Qwen3 base GRPO, n=3,4 (balanced-simple), cap {CAP}: accuracy vs training step",
                 fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.075, "checkpoints every 200 steps   ·   columns share y   ·   "
             "non-truncated: prompts with no finished response dropped",
             ha="center", fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.09, 1, 0.965])
    fig.savefig(OUT_PASS, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_PASS}")


def fig_trunc(df, runs):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.patch.set_facecolor(base.SURFACE)
    drawn = plot_runs(ax, df, runs, "trunc_rate")
    epoch_label(ax)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    base.style_axes(ax, f"Qwen3 base GRPO, n=3,4, cap {CAP}: truncation rate vs training step",
                    "training step", "responses hitting the cap",
                    f"truncated = response >= {CAP - 2} tokens  ·  997 prompts x 32 samples")
    legend(fig, runs, drawn, 0.005)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(OUT_TRUNC, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_TRUNC}")


def main():
    runs = base.PRESETS["all"][0]
    df = pd.read_csv(tr.OUT_CSV)
    df = df[df.cap == CAP]
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_pass(df, runs)
    fig_trunc(df, runs)


if __name__ == "__main__":
    main()
