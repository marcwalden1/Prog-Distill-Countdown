#!/usr/bin/env python3
"""
pass@1 (x) vs pass@32 (y) trajectories for the base-model GRPO runs listed in
results/grpo_Qwen_0.5B_vs_1.5B.md.

One curve per run, one marker per evaluated checkpoint, connected in step order;
open ring = first checkpoint, star = last. Three panels: n=3,4 (`balanced`),
n=5 (`balanced5`), n=6 (`balanced6`).

Reading the plot: pass@32 >= pass@1 always, so every point sits above the dashed
y=x line. Distance from that line is the sampling headroom (how much 32 samples
buy over 1); movement right along it is a policy getting sharper.

Scores come from the tidy full-set CSVs written by scripts/export_idood_scores.py
(no 1000-prompt cap, unbiased pass@k). Every analysis_data/idood_*.csv is read and
de-duplicated, so a run is picked up from whichever export happens to carry it.

Output — one subfolder per plot set (figures/<out-dir>/pass1_vs_pass32_<tag>/):
    grid.png        2x3: row 1 all runs, row 2 the 0.5B runs zoomed
    <dataset>.png   each column standalone (full panel + zoomed panel)

Usage:
    python3 scripts/plot/plot_pass1_vs_pass32.py
    python3 scripts/plot/plot_pass1_vs_pass32.py --annotate   # + per-point step labels
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# --- the runs, in the order they appear in results/grpo_Qwen_0.5B_vs_1.5B.md -------
# color: the seven 0.5B runs take slots 1-7 of the validated categorical order (worst
# adjacent CVD dE 9.1, normal-vision 19.6). The three 1.5B runs are a small reference
# family: slot 8 (red) plus two neutrals, which separate by lightness rather than hue
# (checked: normal-vision dE 18.7, all >= 3:1 on the light surface). Neutrals sit
# outside the categorical palette on purpose — that is what keeps the hue count at 8
# instead of inventing a 9th and 10th slot. Model size is also carried by line style
# (0.5B solid, 1.5B dashed) and marker shape, so identity never rests on color alone.
RUNS = [
    # (model, exp_name, short label, color, marker, linestyle)
    ("Qwen2.5-0.5B", "balanced-grpo-seed1",
     "0.5B lr1e-6 kl1e-3 (n=4, 1ep)", "#2a78d6", "o", "-"),
    ("Qwen2.5-0.5B", "balanced-grpo-warmup0.03-beta2_0.95-seed1",
     "0.5B lr1e-6 kl1e-3 +warmup/β₂", "#eb6834", "s", "-"),
    ("Qwen2.5-0.5B", "balanced-grpo-dynbsz-n8-lr1e-6-kl3e-3-ep2-seed1",
     "0.5B lr1e-6 kl3e-3 (n=8, 2ep)", "#1baf7a", "^", "-"),
    ("Qwen2.5-0.5B", "balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1",
     "0.5B lr3e-6 kl3e-3 (n=8, 2ep) s1", "#eda100", "D", "-"),
    ("Qwen2.5-0.5B", "balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2",
     "0.5B lr3e-6 kl3e-3 (n=8, 2ep) s2", "#e87ba4", "v", "-"),
    ("Qwen2.5-0.5B", "balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-tp0.5-ep2-seed1",
     "0.5B lr3e-6 kl3e-3 +tp0.5", "#008300", "P", "-"),
    ("Qwen2.5-0.5B", "balanced-simple-raw-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1",
     "0.5B lr3e-6 kl3e-3, raw-simple prompt", "#4a3aa7", "X", "-"),
    # the 1.5B runs from the .md's Qwen2.5-1.5B section, in increasing OOD strength
    ("Qwen2.5-1.5B", "balanced-grpo-lr3e-6-kl1e-3-seed1",
     "1.5B lr3e-6 kl1e-3 (mine)", "#8a8880", "s", "--"),
    ("Qwen2.5-1.5B", "balanced-grpo-lr6e-6-kl3e-3-seed1",
     "1.5B lr6e-6 kl3e-3 (mine)", "#e34948", "D", "--"),
    ("Qwen2.5-1.5B", "balanced-grpo-lr3e-6-kl3e-3-seed1",
     "1.5B lr3e-6 kl3e-3 (Shlok's)", "#2f2e2b", "h", "--"),
]

# runs in the .md with no eval JSONs at all — reported, not silently dropped
NO_DATA_NOTE = [
    ("Qwen2.5-0.5B", "balanced-simple-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1"),
]

PANELS = [("balanced", "n=3,4  (in-distribution)"),
          ("balanced5", "n=5  (OOD)"),
          ("balanced6", "n=6  (OOD)")]

SURFACE = "#fcfcfb"
INK = "#1c1b19"
INK_MUTED = "#6b6a66"
GRID = "#e6e5e1"


def load_scores(analysis_dir):
    """All idood_*.csv rows, de-duplicated on (model, exp, dataset, step)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(analysis_dir, "idood_*.csv"))):
        df = pd.read_csv(path)
        need = {"model", "exp_name", "dataset", "step", "pass_at_1", "pass_at_32"}
        if not need.issubset(df.columns):
            continue
        df["_src"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        sys.exit(f"no idood_*.csv found under {analysis_dir}")
    df = pd.concat(frames, ignore_index=True)
    # the raw-simple run was evaluated on the `*-simple` prompt variants and its
    # export writes both names for the same JSONs; fold onto the base names so it
    # lands in the same panels as everything else.
    df["dataset"] = df["dataset"].str.replace("-simple", "", regex=False)
    df = df.drop_duplicates(subset=["model", "exp_name", "dataset", "step"])
    return df


def series(df, model, exp, dataset):
    sub = df[(df.model == model) & (df.exp_name == exp) &
             (df.dataset == dataset)].sort_values("step")
    if sub.empty:
        return None
    return list(sub.pass_at_1), list(sub.pass_at_32), list(sub.step)


def draw_curves(ax, df, dataset, runs, annotate, scale=1.0, legend=False):
    plotted = 0
    for model, exp, label, color, marker, linestyle in runs:
        s = series(df, model, exp, dataset)
        if s is None:
            continue
        xs, ys, steps = s
        plotted += 1
        kw = dict(label=f"{label}  ({len(steps)} ckpts)") if legend else {}

        ax.plot(xs, ys, linestyle, color=color, alpha=0.55, linewidth=2 * scale,
                zorder=2)
        ax.scatter(xs, ys, s=34 * scale**2, marker=marker, color=color,
                   edgecolors=SURFACE, linewidths=1.0 * scale, zorder=3, **kw)
        ax.scatter([xs[0]], [ys[0]], s=110 * scale**2, marker="o", facecolors="none",
                   edgecolors=color, linewidths=1.8 * scale, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=210 * scale**2, marker="*", color=color,
                   edgecolors=INK, linewidths=0.6 * scale, zorder=5)
        if annotate:
            for j, (st, x, y) in enumerate(zip(steps, xs, ys)):
                dx, dy = (5, 4) if j % 2 == 0 else (5, -9)
                ax.annotate(str(st), (x, y), fontsize=5.8 * scale, color=INK_MUTED,
                            textcoords="offset points", xytext=(dx, dy), zorder=6)
    return plotted


SMALL = [r for r in RUNS if r[0] == "Qwen2.5-0.5B"]


def style_axes(ax, title, label_size=9, title_size=11):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcec9")
    ax.tick_params(colors=INK_MUTED, labelsize=label_size)
    ax.set_xlabel("pass@1", fontsize=10, color=INK)
    ax.set_ylabel("pass@32", fontsize=10, color=INK)
    ax.set_title(title, fontsize=title_size, color=INK)


def diagonal(ax):
    """pass@32 >= pass@1 by construction; the diagonal is the no-headroom bound.

    The line has to span up to the y-range to stay a true y=x, but pass@32 >> pass@1
    here, so letting it autoscale would stretch x far past the data and squash every
    curve into the left half. Snapshot the data-driven limits and restore them after
    drawing; matplotlib clips the line to the axes box.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo, hi = min(x0, y0), max(x1, y1)
    ax.plot([lo, hi], [lo, hi], "--", color="#c9c8c3", linewidth=1, zorder=1)
    ax.set_xlim(max(0.0, x0), x1)
    ax.set_ylim(max(0.0, y0), y1)


def draw_panel(ax, df, dataset, title, annotate, legend=True):
    """Top row: every run, full range."""
    plotted = draw_curves(ax, df, dataset, RUNS, annotate, legend=legend)
    diagonal(ax)
    style_axes(ax, title)
    return plotted


def draw_zoom_panel(ax, df, dataset, title, annotate, legend=False):
    """Bottom row: the 0.5B runs only, on their own scale.

    Shlok's 1.5B run sits 3-5x out and compresses the 0.5B curves into the corner
    of the top row, so they get a row of their own rather than an inset.
    """
    plotted = draw_curves(ax, df, dataset, SMALL, annotate, legend=legend)
    pts = [p for p in (series(df, m, e, dataset) for m, e, _, _, _, _ in SMALL) if p]
    xs = [v for p in pts for v in p[0]]
    ys = [v for p in pts for v in p[1]]
    padx, pady = 0.06 * (max(xs) - min(xs)), 0.06 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - padx, max(xs) + padx)
    ax.set_ylim(min(ys) - pady, max(ys) + pady)
    diagonal(ax)
    style_axes(ax, title, title_size=10.5)
    return plotted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=os.path.join(REPO, "analysis_data"))
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "figures"))
    ap.add_argument("--out-dir", default="comparison", help="subdir under figures/")
    ap.add_argument("--tag", default="qwen_base_grpo")
    ap.add_argument("--annotate", action="store_true",
                    help="label every checkpoint with its training step")
    args = ap.parse_args()

    df = load_scores(args.analysis_dir)

    have = {(m, e) for m, e in zip(df.model, df.exp_name)}
    for model, exp, label, _, _, _ in RUNS:
        if (model, exp) not in have:
            print(f"  [missing] no pass@k rows for {model}/{exp} — curve omitted")
    for model, exp in NO_DATA_NOTE:
        print(f"  [no evals] {model}/{exp} — never evaluated, cannot be plotted")

    # one subfolder per plot set: figures/<out-dir>/pass1_vs_pass32_<tag>/{grid,<dataset>}.png
    out_root = os.path.join(args.figures_dir, args.out_dir,
                            f"pass1_vs_pass32_{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # row 1 = all runs at full range; row 2 = the 0.5B runs zoomed to their own scale
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.2), facecolor=SURFACE)
    for c, (dataset, title) in enumerate(PANELS):
        n = draw_panel(axes[0][c], df, dataset, title, args.annotate)
        z = draw_zoom_panel(axes[1][c], df, dataset,
                            f"{title} — 0.5B runs, zoomed", args.annotate)
        print(f"[grid] {dataset}: {n} runs (top), {z} runs (zoom)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9,
               title="GRPO from the base model — solid = 0.5B, dashed = 1.5B; "
                     "open ring = first ckpt, star = last ckpt; "
                     "grey diagonal = pass@32 = pass@1",
               title_fontsize=8.5)
    fig.suptitle("pass@1 vs pass@32 across training — Qwen2.5-0.5B and "
                 "Qwen2.5-1.5B base-model GRPO",
                 fontsize=13, color=INK, y=0.997)
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    grid_path = os.path.join(out_root, "grid.png")
    fig.savefig(grid_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {grid_path}")

    for dataset, title in PANELS:
        fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(8.0, 12.4),
                                              facecolor=SURFACE)
        draw_panel(ax_all, df, dataset, title, args.annotate)
        draw_zoom_panel(ax_zoom, df, dataset, f"{title} — 0.5B runs, zoomed",
                        args.annotate)
        handles, labels = ax_all.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   fontsize=7.6)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        path = os.path.join(out_root, f"{dataset}.png")
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # Table view of the plotted endpoints — the numbers behind the marks, and the
    # accessible fallback for readers who can't separate the categorical hues.
    rows = []
    for model, exp, label, _, _, _ in RUNS:
        row = {"run": label, "model": model, "exp_name": exp}
        for dataset, _ in PANELS:
            s = series(df, model, exp, dataset)
            if s:
                row["final_step"] = s[2][-1]
                row[f"{dataset}_pass@1"] = round(s[0][-1], 4)
                row[f"{dataset}_pass@32"] = round(s[1][-1], 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(args.analysis_dir, f"pass1_vs_pass32_{args.tag}_final.csv")
    table.to_csv(csv_path, index=False)
    print(f"\nfinal checkpoint of each curve:\n{table.to_string(index=False)}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
