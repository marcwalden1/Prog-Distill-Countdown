#!/usr/bin/env python3
"""
Qwen3 base GRPO across model size -- 0.6B vs 1.7B vs 4B -- on `balanced-simple`
(lr=3e-6, kl=3e-3, rollout.n=4, seed1). Two figures:

  pass1_vs_pass32.png   x = pass@1, y = pass@32, one panel per dataset
                        (n=3,4 / n=5 / n=6). Distance above the dashed y=x line
                        is sampling headroom: how much 32 samples buy over 1.

  id_vs_ood.png         x = in-distribution (n=3,4), y = OOD, 2x2:
                        rows = metric (pass@1, pass@32), cols = OOD set (n=5, n=6).
                        Up-and-right is strictly better; a curve that moves right
                        while dropping is trading OOD for in-dist.

One marker per evaluated checkpoint, connected in step order; open ring = first
checkpoint, star = last (direct-labeled with the model).

Metrics are the unbiased pass@k estimator over the FULL prompt set, read from the
tidy CSV written by scripts/export_idood_scores.py -- NOT the `.scores` caches,
whose `mean@32` averages raw compute_score (0 / 0.1 / 1.0) and so mixes format
credit into what looks like a pass rate.

Identity never rests on color alone: each model also carries a marker shape and a
line style, and the final point is direct-labeled. Palette is blue/orange/purple
from the repo's validated categorical order -- all six checks pass with no
contrast warning (worst adjacent CVD dE 24.7 protan / 32.7 tritan, normal-vision
33.6, all >= 3:1 on the light surface). The obvious green third slot (#1baf7a)
was rejected: it only WARNs at 2.74:1 contrast, and every darker green that fixes
that drops CVD separation against the orange to dE 3-6 (a hard FAIL).

Usage:
    python3 scripts/export_idood_scores.py \
        --run "0.6B=results::Qwen3-0.6B-Base::balanced-simple-grpo-lr3e-6-kl3e-3-seed1" \
        --run "1.7B=results::Qwen3-1.7B-Base::balanced-simple-grpo-lr3e-6-kl3e-3-seed1" \
        --run "4B=results::Qwen3-4B-Base::balanced-simple-grpo-lr3e-6-kl3e-3-seed1" \
        --datasets balanced-simple balanced5-simple balanced6-simple \
        --out analysis_data/idood_qwen3_base_grpo.csv
    python3 scripts/plot/plot_qwen3_size_comparison.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EXP34 = "balanced-simple-grpo-lr3e-6-kl3e-3-seed1"
_EXP1E6 = "balanced-simple-grpo-lr1e-6-kl3e-3-seed1"

# preset -> (runs, figure subdir, what the arms vary).
# A run is (model, exp_name, label, color, marker, linestyle). Colors are the
# validated categorical order (blue/orange/purple); shape + dash carry the same
# distinction so every figure survives greyscale and CVD.
PRESETS = {
    # Best lr per model, decided from the per-size `lr-*` ablations at a
    # 1633-step budget -- NOT a matched-hyperparameter comparison:
    #   0.6B -> 3e-6 (1e-6 is still climbing at 1600, i.e. undertrained here)
    #   1.7B -> 1e-6 (3e-6 collapses OOD after step 600)
    #   4B   -> 1e-6 (3e-6 peaks OOD by step 500-800 then gives half of it back)
    # The lr is in every series label and in the suptitle so the figure cannot
    # be misread as lr-matched.
    "size": (
        [("Qwen3-0.6B-Base", _EXP34,  "Qwen3-0.6B  lr=3e-6", "#2a78d6", "o", "-"),
         ("Qwen3-1.7B-Base", _EXP1E6, "Qwen3-1.7B  lr=1e-6", "#eb6834", "s", "--"),
         ("Qwen3-4B-Base",   _EXP1E6, "Qwen3-4B  lr=1e-6",   "#4a3aa7", "^", ":")],
        "Qwen3-base-grpo",
        "kl=3e-3, n=4, best lr per model",
    ),
    "rolloutn": (
        [("Qwen3-0.6B-Base", _EXP34, "0.6B  rollout.n=4", "#2a78d6", "o", "-"),
         ("Qwen3-0.6B-Base", "balanced-simple-grpo-lr3e-6-kl3e-3-n8-seed1",
          "0.6B  rollout.n=8", "#eb6834", "s", "--")],
        "Qwen3-0.6B-rolloutn",
        "Qwen3-0.6B, lr=3e-6, kl=3e-3",
    ),
}

# One lr-ablation preset per model size: lr=3e-6 vs lr=1e-6, everything else
# matched. Output goes under that model's own figures dir.
for _size in ("0.6B", "1.7B", "4B"):
    _m = f"Qwen3-{_size}-Base"
    PRESETS[f"lr-{_size}"] = (
        [(_m, _EXP34, f"{_size}  lr=3e-6", "#2a78d6", "o", "-"),
         (_m, _EXP1E6, f"{_size}  lr=1e-6", "#eb6834", "s", "--")],
        os.path.join(_m, f"Qwen3-{_size}-lr"),
        f"Qwen3-{_size}, kl=3e-3, n=4",
    )

# 0.6B lr=1e-6, 1 epoch (1633 steps) vs 2 epochs (3266 steps). Separate runs,
# not a continuation, so the 2-epoch curve's first 1600 steps are their own
# trajectory, not a retrace of the 1-epoch one.
PRESETS["ep-0.6B"] = (
    [("Qwen3-0.6B-Base", _EXP1E6, "0.6B  lr=1e-6, 1 ep", "#2a78d6", "o", "-"),
     ("Qwen3-0.6B-Base", "balanced-simple-grpo-lr1e-6-kl3e-3-ep2-seed1",
      "0.6B  lr=1e-6, 2 ep", "#eb6834", "s", "--")],
    os.path.join("Qwen3-0.6B-Base", "Qwen3-0.6B-epochs"),
    "Qwen3-0.6B, lr=1e-6, kl=3e-3, n=4",
)

# Every lr arm of every size in one figure. Hue family = model size (blue /
# orange / violet, as in "size"); within a family each arm gets its own shade,
# plus its own marker + line style, so the lr comparison reads within a family
# and the size comparison across families. lr=1e-6 keeps the family's base
# shade. Every cross-family pair passes the dataviz validator (--pairs all,
# light surface); within-family shades are intentionally close (normal-vision
# dE 9-14) and lean on marker + dash. #5aa6ef WARNs on contrast (2.5:1).
# 0.6B lr=1e-6 also gets its 2-epoch run (3266 steps) -- a separate run, not a
# continuation of the 1-epoch one, so its first 1600 steps do not retrace the
# 1-epoch curve.
_ARMS = {"lr3e-6": ("o", "-"), "lr1e-6": ("s", "--"), "lr1e-6-ep2": ("D", ":")}
_SHADES = {"0.6B": {"lr1e-6": "#2a78d6", "lr3e-6": "#5aa6ef", "lr1e-6-ep2": "#0b8fb5"},
           "1.7B": {"lr1e-6": "#eb6834", "lr3e-6": "#b8410f"},
           "4B":   {"lr1e-6": "#4a3aa7", "lr3e-6": "#80409f"}}
_all_runs = []
for _size, _shade in _SHADES.items():
    _m = f"Qwen3-{_size}-Base"
    _all_runs.append((_m, _EXP34, f"{_size}  lr=3e-6", _shade["lr3e-6"], *_ARMS["lr3e-6"]))
    _all_runs.append((_m, _EXP1E6, f"{_size}  lr=1e-6", _shade["lr1e-6"], *_ARMS["lr1e-6"]))
    if "lr1e-6-ep2" in _shade:
        _all_runs.append((_m, "balanced-simple-grpo-lr1e-6-kl3e-3-ep2-seed1",
                          f"{_size}  lr=1e-6 (2 ep)", _shade["lr1e-6-ep2"],
                          *_ARMS["lr1e-6-ep2"]))
PRESETS["all"] = (_all_runs, "Qwen3-base-grpo", "kl=3e-3, n=4, lr in {1e-6, 3e-6}")
# 0.6B GRPO from the text-only distilled student (Qwen3-4B-Base teacher, NTP SFT
# lr=1e-5, 1 epoch over 4.18M teacher samples) vs the three base-model GRPO runs
# of the "size" preset. Same GRPO recipe as the 0.6B base run (lr=3e-6, kl=3e-3,
# n=4, 1 epoch). Fourth hue #c2185b: validator passes all checks against the
# blue/orange/purple trio (--pairs all, light surface).
PRESETS["distill"] = (
    PRESETS["size"][0] + [
        ("Qwen3-0.6B-Base",
         "balanced-simple-grpo-from-distill-ntp-t4b-base-sftlr1e-5-lr3e-6-kl3e-3-seed1",
         "Qwen3-0.6B  distilled from 4B, lr=3e-6", "#c2185b", "D", "-.")],
    "Qwen3-distill-grpo",
    "kl=3e-3, n=4; base models vs 0.6B distilled from 4B",
)

# Presets that share a figures dir with another preset get a filename suffix so
# they don't overwrite it ("all" lives next to "size").
FILE_SUFFIX = {"all": "_all"}

# dataset key in the CSV -> panel title
PANELS = [("balanced-simple", "n=3,4  (in-distribution)"),
          ("balanced5-simple", "n=5  (OOD)"),
          ("balanced6-simple", "n=6  (OOD)")]

ID_DS = "balanced-simple"
OOD = [("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
METRICS = [("pass_at_1", "pass@1"), ("pass_at_32", "pass@32")]

SURFACE = "#fcfcfb"
INK = "#1c1b19"
INK_MUTED = "#6b6a66"
GRID = "#e6e5e1"
DIAG = "#c9c8c3"


def load(csv_paths):
    need = {"model", "exp_name", "dataset", "step", "pass_at_1", "pass_at_32"}
    frames = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        d = pd.read_csv(csv_path)
        missing = need - set(d.columns)
        if missing:
            sys.exit(f"{csv_path}: missing columns {sorted(missing)}")
        frames.append(d)
    if not frames:
        sys.exit(f"none of {csv_paths} exist")
    df = pd.concat(frames, ignore_index=True)
    # NB: de-dup on (model, exp_name, dataset, step) -- the rolloutn preset draws
    # two arms of the SAME model, so exp_name has to be part of the key.
    return df.drop_duplicates(subset=["model", "exp_name", "dataset", "step"])


def series(df, model, exp, dataset, cols):
    """Rows for one (model, exp, dataset) in step order -> (col0, col1, steps)."""
    sub = df[(df.model == model) & (df.exp_name == exp) &
             (df.dataset == dataset)].sort_values("step")
    sub = sub.dropna(subset=list(cols))
    if sub.empty:
        return None
    return list(sub[cols[0]]), list(sub[cols[1]]), list(sub.step)


def style_axes(ax, title, xlabel, ylabel, subtitle=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcec9")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    # Eval coverage rides under the title in muted ink: with partial evals the
    # first/last marker is not necessarily step 100/1600, so the panel has to say
    # which steps it actually covers.
    ax.set_title(title, fontsize=11, color=INK, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color=INK_MUTED)


def draw_curves(ax, df, runs, dataset_or_pair, cols, annotate, label_last, direct_labels=True):
    """One trajectory per model. `dataset_or_pair` is a dataset name (both cols
    from it) or an (x_dataset, y_dataset) pair (one col from each).

    Returns [(label, steps)] for the models actually drawn, so the caller can put
    the real coverage in the panel title -- with partial evals the first/last
    marker is NOT necessarily step 100/1600."""
    drawn = []
    for model, exp, label, color, marker, ls in runs:
        if isinstance(dataset_or_pair, tuple):
            xd, yd = dataset_or_pair
            sx = series(df, model, exp, xd, (cols[0], cols[0]))
            sy = series(df, model, exp, yd, (cols[1], cols[1]))
            if sx is None or sy is None:
                continue
            # only steps evaluated on BOTH sides can be paired
            xmap = dict(zip(sx[2], sx[0]))
            ymap = dict(zip(sy[2], sy[0]))
            steps = sorted(set(xmap) & set(ymap))
            if not steps:
                continue
            xs = [xmap[s] for s in steps]
            ys = [ymap[s] for s in steps]
        else:
            s = series(df, model, exp, dataset_or_pair, cols)
            if s is None:
                continue
            xs, ys, steps = s

        drawn.append((label, steps))
        ax.plot(xs, ys, ls, color=color, alpha=0.55, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=60, marker=marker, color=color, edgecolors=SURFACE,
                   linewidths=1.4, zorder=3, label=label)
        ax.scatter([xs[0]], [ys[0]], s=150, marker="o", facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=4)
        if label_last:
            ax.scatter([xs[-1]], [ys[-1]], s=240, marker="*", color=color,
                       edgecolors=INK, linewidths=0.6, zorder=5)
        else:
            # Too many curves to direct-label without collisions: the final point
            # keeps the run's own marker (enlarged, ink edge) so arms that share
            # a color stay distinguishable, and the legend carries identity.
            ax.scatter([xs[-1]], [ys[-1]], s=150, marker=marker, color=color,
                       edgecolors=INK, linewidths=1.0, zorder=5)
        if label_last and direct_labels:
            ax.annotate(label, (xs[-1], ys[-1]), fontsize=8.5, color=INK_MUTED,
                        textcoords="offset points", xytext=(9, -3), zorder=6)
        if annotate:
            for j, (st, x, y) in enumerate(zip(steps, xs, ys)):
                dy = 5 if j % 2 == 0 else -10
                ax.annotate(str(st), (x, y), fontsize=6, color=INK_MUTED,
                            textcoords="offset points", xytext=(5, dy), zorder=6)
    return drawn


def coverage_subtitle(drawn):
    """Per-model eval coverage, collapsed when every model has the same steps.

    Spelling it out per model overflows the panel once there are three or more
    runs, and in the common case (a finished sweep) the line is the same text
    repeated N times."""
    if not drawn:
        return ""
    shapes = {(len(st), st[0], st[-1]) for _, st in drawn}
    if len(shapes) == 1:
        n, first, last = shapes.pop()
        who = f"all {len(drawn)} runs" if len(drawn) > 2 else " + ".join(l for l, _ in drawn)
        return f"{who}:  {n} ckpts ({first}–{last})"
    if len(drawn) > 3:
        # many curves: one entry per distinct coverage, naming lone outliers
        groups = {}
        for lab, st in drawn:
            groups.setdefault((len(st), st[0], st[-1]), []).append(lab)
        short = lambda lab: lab.replace("Qwen3-", "").split(",")[0].strip()
        return "   ·   ".join(
            f"{short(labs[0]) if len(labs) == 1 else f'{len(labs)} runs'}: "
            f"{n} ckpts ({first}–{last})"
            for (n, first, last), labs in sorted(groups.items()))
    # mixed coverage: compact form, model label without the 'Qwen3-' prefix
    return "   ·   ".join(
        f"{lab.replace('Qwen3-', '')} {len(st)} ({st[0]}–{st[-1]})"
        for lab, st in drawn)


def pad_x(ax, frac=0.10):
    """Room on the right for the direct label at the final point."""
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * frac)


def run_legend(runs, drawn_labels):
    """Line + marker proxies, so the legend shows the line style as well as the
    marker -- needed when marker/dash (not color) is what separates two runs."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for _, _, label, color, marker, ls in runs:
        if label in drawn_labels:
            handles.append(Line2D([], [], color=color, linestyle=ls, linewidth=2,
                                  marker=marker, markersize=7,
                                  markeredgecolor=SURFACE))
            labels.append(label)
    return handles, labels


def collect_legend(axes):
    """Handles from whichever panels actually drew something, de-duplicated."""
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    return handles, labels


def diagonal(ax):
    """pass@32 >= pass@1 by construction, so every point sits above y=x. Snapshot
    the data-driven limits and restore them; otherwise a true y=x stretches x far
    past the data and squashes the curves into the left edge."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo, hi = min(x0, y0), max(x1, y1)
    ax.plot([lo, hi], [lo, hi], "--", color=DIAG, linewidth=1, zorder=1)
    ax.set_xlim(max(0.0, x0), x1)
    ax.set_ylim(max(0.0, y0), y1)


def fig_pass1_vs_pass32(df, runs, arm_desc, out_dir, annotate, suffix="", panels=None):
    panels = panels or PANELS
    fig, axes = plt.subplots(1, len(panels), figsize=(15.5 if len(panels) == 3 else 5.2 * len(panels) + 2.6, 5.0 if len(panels) > 1 else 6.4),
                             squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)
    any_plotted = 0
    for ax, (ds, title) in zip(axes, panels):
        # one panel: final points of several runs bunch up at the top right, so
        # direct labels collide -- the legend below carries identity instead
        drawn = draw_curves(ax, df, runs, ds, ("pass_at_1", "pass_at_32"),
                            annotate, label_last=True,
                            direct_labels=len(panels) > 1 and len(runs) <= 3)
        any_plotted += len(drawn)
        if drawn:
            diagonal(ax)
            if len(panels) > 1:
                pad_x(ax)
        else:
            ax.text(0.5, 0.5, "no evals yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color=INK_MUTED)
        style_axes(ax, title, "pass@1", "pass@32", coverage_subtitle(drawn))
    handles, labels = collect_legend(axes)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 4 if len(panels) > 1 else 2),
                   frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
                   bbox_to_anchor=(0.5, -0.005))
    sep = ":\n" if len(panels) == 1 else ": "
    fig.suptitle(f"Qwen3 GRPO (balanced-simple, {arm_desc}){sep}"
                 "sampling headroom over training", fontsize=12.5, color=INK, y=0.99)
    fig.text(0.5, 0.085 if len(panels) > 1 else 0.115, "open ring = first evaluated checkpoint   ·   star = last"
             "   ·   dashed = y=x (no headroom)", ha="center", fontsize=8.5,
             color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.11, 1, 0.96] if len(panels) > 1 else [0, 0.14, 1, 0.97])
    out = os.path.join(out_dir, f"pass1_vs_pass32{suffix}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out, any_plotted


def fig_id_vs_ood(df, runs, arm_desc, out_dir, annotate, suffix=""):
    dense = len(runs) > 4
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.6 if dense else 10.0))
    fig.patch.set_facecolor(SURFACE)
    any_plotted = 0
    for r, (mcol, mname) in enumerate(METRICS):
        for c, (ood_ds, ood_name) in enumerate(OOD):
            ax = axes[r][c]
            drawn = draw_curves(ax, df, runs, (ID_DS, ood_ds), (mcol, mcol),
                                annotate, label_last=not dense)
            any_plotted += len(drawn)
            if drawn and not dense:
                pad_x(ax)
            elif not drawn:
                ax.text(0.5, 0.5, "no paired evals yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color=INK_MUTED)
            style_axes(ax, f"{mname}:  in-dist vs {ood_name}",
                       f"in-dist n=3,4   {mname}", f"OOD {ood_name}   {mname}",
                       coverage_subtitle(drawn))
    if dense:
        drawn_labels = {l for a in axes.flat for l in a.get_legend_handles_labels()[1]}
        handles, labels = run_legend(runs, drawn_labels)
    else:
        handles, labels = collect_legend([a for row in axes for a in row])
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4),
                   frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
                   bbox_to_anchor=(0.5, 0.005), handlelength=3)
    fig.suptitle(f"Qwen3 base GRPO (balanced-simple, {arm_desc}): "
                 "in-distribution vs OOD", fontsize=12.5, color=INK, y=0.99)
    last_mark = "ink-edged marker" if dense else "star"
    fig.text(0.5, 0.075 if dense else 0.045,
             f"open ring = first evaluated checkpoint   ·   {last_mark} = last"
             "   ·   up-and-right is strictly better", ha="center", fontsize=8.5,
             color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.095 if dense else 0.07, 1, 0.965])
    out = os.path.join(out_dir, f"id_vs_ood{suffix}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out, any_plotted


def coverage(df, runs):
    rows = []
    for model, exp, label, *_ in runs:
        for ds, _ in PANELS:
            sub = df[(df.model == model) & (df.exp_name == exp) &
                     (df.dataset == ds)]
            rows.append((label, ds, len(sub),
                         ",".join(str(s) for s in sorted(sub.step)) or "-"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",
                    default=[os.path.join(REPO, "analysis_data", f)
                             for f in ("idood_qwen3_base_grpo.csv",
                                       "idood_qwen3_distill_grpo.csv")])
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of panels, e.g. balanced-simple for n=3,4 only "
                         "(writes pass1_vs_pass32<suffix>_<ds>.png; id_vs_ood is "
                         "skipped unless both n=5 and n=6 are included)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="size",
                    help="size: 0.6B vs 1.7B vs 4B at n=4.  "
                         "rolloutn: 0.6B rollout.n=4 vs n=8.  "
                         "all: every size x lr in {1e-6, 3e-6} (+ 0.6B 2-epoch).  "
                         "ep-0.6B: 0.6B lr=1e-6, 1 vs 2 epochs.")
    ap.add_argument("--out-dir", default=None,
                    help="default: figures/<preset's subdir>")
    ap.add_argument("--out-name", default=None,
                    help="filename for the pass@1-vs-pass@32 figure (overrides the "
                         "suffix-derived name; only with --datasets)")
    ap.add_argument("--annotate", action="store_true",
                    help="label every point with its training step")
    args = ap.parse_args()

    runs, subdir, arm_desc = PRESETS[args.preset]
    out_dir = args.out_dir or os.path.join(REPO, "figures", subdir)

    df = load(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preset: {args.preset}  ({arm_desc})")
    print("Eval coverage (steps present in the CSV):")
    for label, ds, n, steps in coverage(df, runs):
        print(f"  {label:24s} {ds:18s} {n:2d}  {steps}")
    print()

    suffix = FILE_SUFFIX.get(args.preset, "")
    if args.datasets:
        panels = [p for p in PANELS if p[0] in args.datasets]
        figs = [fig_pass1_vs_pass32(df, runs, arm_desc, out_dir, args.annotate,
                                    suffix + "_" + "_".join(args.datasets), panels)]
        if args.out_name:
            dst = os.path.join(out_dir, args.out_name)
            os.replace(figs[0][0], dst)
            figs = [(dst, figs[0][1])]
    else:
        figs = [fig_pass1_vs_pass32(df, runs, arm_desc, out_dir, args.annotate, suffix),
                fig_id_vs_ood(df, runs, arm_desc, out_dir, args.annotate, suffix)]
    for out, n in figs:
        print(f"wrote {out}  ({n} curves)")


if __name__ == "__main__":
    main()
