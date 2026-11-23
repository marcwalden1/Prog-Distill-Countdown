#!/usr/bin/env python3
"""OLMo-2-7B pt135B vs pt3896B: GT-SFT answer-format vs search-format, across GRPO.

Two figures, four curves each. A curve is one GRPO run (lr 3e-6, kl 3e-3) plotted
from its SFT student outward: step 0 is the pre-GRPO `sft_final` checkpoint, then
the eight GRPO checkpoints 200..1600.

    id_vs_ood.png        x = in-dist (n=3,4), y = OOD (n=5 / n=6); one panel per
                         (metric, OOD set) -> 2x2
    pass1_vs_pass32.png  x = pass@1, y = pass@32; one panel per dataset -> 1x3

Encoding: colour = model (pretraining budget), linestyle+marker = SFT format.
Two hues only, so the palette stays inside the CVD floors
(`validate_palette.js "#2a78d6,#eb6834"`: worst adjacent dE 24.7 protan / 33.6
normal, all six checks PASS). Format is carried by a second, non-colour channel
and by direct endpoint labels, so identity never rests on colour alone.

Arm selection: GRPO config is matched across all four arms (lr 3e-6, kl 3e-3);
where several SFT LRs were swept the arm shown is the one with the highest
in-distribution pass@1 at step 1600. That criterion is not neutral for the
search arms -- the losing LR (1e-5 in both) has *better* OOD coverage. See the
--all-lrs flag to see both.

Reads the `.passk` caches written by scripts/compute_passk.py (pass@32 is
coverage: >=1 of 32 samples exactly correct -- NOT the mean@32 in `.scores`).

    python3 scripts/plot/plot_olmo7b_format_arms.py [--annotate] [--all-lrs]
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Categorical slots 1-2 of the validated default palette (light mode).
C_135B = "#2a78d6"
C_3896B = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

# (model, format) -> (grpo exp, sft-origin exp, sft_lr)
# GRPO is lr3e-6/kl3e-3 for every arm; only the SFT stage differs.
ARMS = [
    ("OLMo-2-1124-7B-pt135B", "answer", "3e-5",
     "balanced-simple-gtsft200lr3e-5-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-lr3e-5-seed1"),
    ("OLMo-2-1124-7B-pt135B", "search", "1e-4",
     "balanced-simple-gtsft200-search-sftlr1e-4-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-search-lr1e-4-seed1"),
    ("OLMo-2-1124-7B-pt3896B", "answer", "1e-5",
     "balanced-simple-gtsft200lr1e-5-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-lr1e-5-seed1"),
    ("OLMo-2-1124-7B-pt3896B", "search", "3e-5",
     "balanced-simple-gtsft200-search-sftlr3e-5-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-search-lr3e-5-seed1"),
]
# The SFT LRs that lost the ID-pass@1 tiebreak, kept for --all-lrs.
RUNNERS_UP = [
    ("OLMo-2-1124-7B-pt135B", "search", "1e-5",
     "balanced-simple-gtsft200-search-sftlr1e-5-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-search-lr1e-5-seed1"),
    ("OLMo-2-1124-7B-pt3896B", "search", "1e-5",
     "balanced-simple-gtsft200-search-sftlr1e-5-grpo-lr3e-6-kl3e-3-seed1",
     "balanced-simple-gtsft200-search-lr1e-5-seed1"),
]

MODEL_COLOR = {"OLMo-2-1124-7B-pt135B": C_135B, "OLMo-2-1124-7B-pt3896B": C_3896B}
MODEL_LABEL = {"OLMo-2-1124-7B-pt135B": "7B · 135B tok",
               "OLMo-2-1124-7B-pt3896B": "7B · 3896B tok"}
FORMAT_STYLE = {"answer": ("-", "o"), "search": ("--", "s")}

DATASETS = [("balanced-simple", "in-dist  n=3,4"),
            ("balanced5-simple", "OOD  n=5"),
            ("balanced6-simple", "OOD  n=6")]
DS_SHORT = {"balanced-simple": "n34", "balanced5-simple": "n5", "balanced6-simple": "n6"}
METRICS = [("pass1", "pass@1"), ("pass32", "pass@32")]
# ID-vs-OOD panels, in grid order (rows = OOD set, cols = metric)
COMBOS = [(m, ds) for ds in ("balanced5-simple", "balanced6-simple") for m, _ in METRICS]


def read_passk(model, exp, sub, dataset):
    p = os.path.join(REPO, "results", model, exp, sub,
                     f"{dataset}_temp0.6_n32_max1024.json.passk")
    if not os.path.exists(p):
        return None
    step, pass1, pass32, n = json.load(open(p))
    return {"pass1": pass1, "pass32": pass32, "n": n}


def load_arm(model, fmt, sft_lr, grpo_exp, sft_exp):
    """[{step, dataset, pass1, pass32, n}] with step 0 = the SFT student."""
    rows = []
    for dataset, _ in DATASETS:
        r = read_passk(model, sft_exp, "sft_final", dataset)
        if r:
            rows.append(dict(r, step=0, dataset=dataset, kind="sft"))
        for step in range(200, 1601, 200):
            r = read_passk(model, grpo_exp, f"global_step_{step}", dataset)
            if r:
                rows.append(dict(r, step=step, dataset=dataset, kind="grpo"))
    for r in rows:
        r.update(model=model, format=fmt, sft_lr=sft_lr, grpo_exp=grpo_exp)
    return rows


def series(rows, dataset, metric):
    pts = sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: r["step"])
    return [r["step"] for r in pts], [r[metric] for r in pts]


def style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


LABEL_OFFSET = {"answer": (9, 5), "search": (9, -10)}


def draw_curve(ax, xs, ys, steps, colour, ls, marker, annotate, label=None):
    """Trajectory with an open ring at the SFT student and a star at step 1600."""
    ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=2, marker=marker,
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    ax.plot([xs[0]], [ys[0]], marker="o", markersize=10, color=colour,
            markerfacecolor=SURFACE, markeredgewidth=2.2, zorder=4)
    ax.plot([xs[-1]], [ys[-1]], marker="*", markersize=15, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=5)
    if label:
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=LABEL_OFFSET.get(label, (9, 4)), fontsize=7.5,
                    color=INK_2, zorder=6)
    if annotate:
        for x, y, s in zip(xs, ys, steps):
            ax.annotate(str(s), (x, y), textcoords="offset points", xytext=(3, -8),
                        fontsize=5.5, color=INK_2, zorder=6)


def legend_handles(arms):
    h = [Line2D([], [], color=MODEL_COLOR[m], linestyle=FORMAT_STYLE[f][0],
                linewidth=2, marker=FORMAT_STYLE[f][1], markersize=5,
                label=f"{MODEL_LABEL[m]}  ·  {f} SFT (lr {lr})")
         for m, f, lr, _, _ in arms]
    h += [Line2D([], [], color=INK_2, marker="o", markersize=10, linestyle="none",
                 markerfacecolor=SURFACE, markeredgewidth=2.2, label="SFT student (GRPO step 0)"),
          Line2D([], [], color=INK_2, marker="*", markersize=15, linestyle="none",
                 label="GRPO step 1600")]
    return h


def fig_id_vs_ood(data, arms, annotate, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4))
    fig.patch.set_facecolor(SURFACE)
    for i, (metric, ood) in enumerate(COMBOS):
        ax = axes[i // 2][i % 2]
        mlabel = dict(METRICS)[metric]
        oodlabel = dict(DATASETS)[ood]
        for m, f, lr, gexp, _ in arms:
            rows = data[(m, f, lr)]
            # keep only steps scored on BOTH datasets, so no point is mismatched
            id_by = dict(zip(*series(rows, "balanced-simple", metric)))
            steps, ys = series(rows, ood, metric)
            keep = [(id_by[s], y, s) for s, y in zip(steps, ys) if s in id_by]
            if not keep:
                continue
            ls, marker = FORMAT_STYLE[f]
            draw_curve(ax, [k[0] for k in keep], [k[1] for k in keep],
                       [k[2] for k in keep], MODEL_COLOR[m], ls, marker, annotate,
                       label=f)
        style_axis(ax, f"{mlabel} — ID vs {oodlabel}",
                   f"in-distribution (n=3,4)  {mlabel}", f"{oodlabel}  {mlabel}")
    fig.legend(handles=legend_handles(arms), loc="lower center", ncol=3,
               frameon=False, fontsize=8.5)
    fig.suptitle("ID vs OOD across GRPO — OLMo-2-7B, GT-SFT answer vs search format\n"
                 "GRPO lr 3e-6 / kl 3e-3 for every arm; best SFT LR by ID pass@1; "
                 "up-and-right is better", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    path = os.path.join(outdir, "id_vs_ood.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def fig_pass1_vs_pass32(data, arms, annotate, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    fig.patch.set_facecolor(SURFACE)
    for c, (ds, dslabel) in enumerate(DATASETS):
        ax = axes[c]
        for m, f, lr, gexp, _ in arms:
            rows = data[(m, f, lr)]
            steps, xs = series(rows, ds, "pass1")
            _, ys = series(rows, ds, "pass32")
            if not steps:
                continue
            ls, marker = FORMAT_STYLE[f]
            draw_curve(ax, xs, ys, steps, MODEL_COLOR[m], ls, marker, annotate,
                       label=f)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, linestyle=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        # equal aspect so the y=x reference really is the 45-degree line and the
        # vertical gap to it reads directly as remaining coverage headroom
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax, dslabel, "pass@1", "pass@32  (coverage)")
    fig.legend(handles=legend_handles(arms), loc="lower center", ncol=3,
               frameon=False, fontsize=8.5)
    fig.suptitle("pass@1 vs pass@32 across GRPO — OLMo-2-7B, GT-SFT answer vs search "
                 "format\nGRPO lr 3e-6 / kl 3e-3; dotted line is pass@1 = pass@32 "
                 "(no headroom left); moving down-right is sharpening at the cost of "
                 "coverage", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))
    path = os.path.join(outdir, "pass1_vs_pass32.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true", help="label every point with its step")
    ap.add_argument("--all-lrs", action="store_true",
                    help="also draw the runner-up search SFT LRs")
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO, "figures", "olmo2-ladder",
                                         "7b-135B-vs-3896B-formats"))
    args = ap.parse_args()

    arms = ARMS + (RUNNERS_UP if args.all_lrs else [])
    data, flat = {}, []
    for m, f, lr, gexp, sexp in arms:
        rows = load_arm(m, f, lr, gexp, sexp)
        data[(m, f, lr)] = rows
        flat += rows
        n_steps = len({r["step"] for r in rows})
        print(f"{MODEL_LABEL[m]:16s} {f:7s} sftlr {lr:5s}: {n_steps} steps, {len(rows)} rows")

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig_id_vs_ood(data, arms, args.annotate, args.out_dir)
    fig_pass1_vs_pass32(data, arms, args.annotate, args.out_dir)

    csv_path = os.path.join(REPO, "analysis_data", "olmo7b_format_arms_passk.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "format", "sft_lr", "grpo_exp",
                                           "kind", "step", "dataset", "pass1",
                                           "pass32", "n"])
        w.writeheader()
        for r in sorted(flat, key=lambda r: (r["model"], r["format"], r["dataset"], r["step"])):
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {csv_path}  ({len(flat)} rows)")


if __name__ == "__main__":
    main()
