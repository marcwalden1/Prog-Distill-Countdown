#!/usr/bin/env python3
"""
Shared plotting grammar for the Qwen2.5-0.5B SFT-then-GRPO run set
(`scripts/qwen_sft_grpo_registry.py`), used by both

    scripts/plot/plot_pass1_vs_pass32_sft_grpo.py
    scripts/plot/plot_id_vs_ood_sft_grpo.py

so the two figure families are decodable with one legend key.

19 student curves is far past the 8-slot categorical cap, so identity is
composite rather than one-hue-per-run, and every channel carries a real
variable:

    color      SFT learning rate, categorical hues in fixed slot order
               (1e-6 blue, 3e-6 orange, 1e-5 aqua, 3e-5 violet). Validated
               all-pairs on the light surface: worst CVD dE 9.2, worst
               normal-vision dE 16.3. See the note by SFT_LR_COLOR for why this
               is categorical rather than the ordinal ramp it started as.
    linestyle  GRPO KL coefficient (3e-4 solid, 1e-3 dotted, 3e-3 dashed)
    marker     GRPO seed (circle = 1, triangle = 21)
    facet      SFT method (distill / progdistill / progdistill-mixture)

so a panel never shows more than four hues. The 1.5B teacher is a neutral
dash-dot reference; the pre-GRPO SFT student is an open square.

The one exception is the cross-method comparison panel, where METHOD is the
question being asked and therefore takes the color channel — categorical slots
1-3, the three that clear the all-pairs CVD gate. Those are the same three hues
SFT LR uses elsewhere, so hue means something different in that one figure; its
legend header says so, and its endpoint labels spell the SFT LR out in text.

It also carries the score loading both scripts need, so the `run`-label lookup
(including the teacher aliases) is defined once.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from qwen_sft_grpo_registry import SFT_ONLY, TEACHERS  # noqa: E402

ANALYSIS_DIR = os.path.join(REPO, "analysis_data")
ID_DATASET = "balanced"
OOD_LABEL = {"balanced5": "n=5", "balanced6": "n=6"}
METRIC_COL = {"pass@1": "pass_at_1", "pass@32": "pass_at_32"}

# Four 1.5B runs were already exported by earlier work under their own `run`
# labels; the registry's label is the display name, so resolve through this map
# rather than re-scoring 120 files to rename them.
LABEL_ALIASES = {label: have for label, _t, _e, _lr, _kl, _o, have in TEACHERS if have}


def load_scores(analysis_dir=None):
    """All idood_*.csv rows, de-duplicated on (run, dataset, step).

    Curves are keyed on the `run` label, NOT (model, exp_name):
    `balanced-grpo-lr3e-6-kl1e-3-seed1` exists as two different 1.5B runs owned
    by two different people.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(analysis_dir or ANALYSIS_DIR,
                                              "idood_*.csv"))):
        df = pd.read_csv(path)
        if not {"run", "dataset", "step", "pass_at_1", "pass_at_32"}.issubset(df.columns):
            continue
        frames.append(df)
    if not frames:
        sys.exit(f"no idood_*.csv found under {analysis_dir or ANALYSIS_DIR}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["run", "dataset", "step"])


def series(df, label, dataset, metric="pass_at_1"):
    """(xs, ys, steps) for one run on one dataset, or None."""
    names = [label] + ([LABEL_ALIASES[label]] if label in LABEL_ALIASES else [])
    sub = df[df.run.isin(names) & (df.dataset == dataset)].sort_values("step")
    if sub.empty:
        return None
    sub = sub.drop_duplicates(subset="step")
    return list(sub.pass_at_1), list(sub.pass_at_32), list(sub.step)


def id_ood_series(df, label, ood, metric):
    """(ID, OOD, steps) at the checkpoints that have BOTH datasets scored.

    Coverage differs per dataset for a few runs (e.g. 10 `balanced` vs 9
    `balanced5`), so the intersection is what can honestly be paired.
    """
    names = [label] + ([LABEL_ALIASES[label]] if label in LABEL_ALIASES else [])
    sub = df[df.run.isin(names)]
    idm = dict(zip(sub[sub.dataset == ID_DATASET].step,
                   sub[sub.dataset == ID_DATASET][metric]))
    oodm = dict(zip(sub[sub.dataset == ood].step, sub[sub.dataset == ood][metric]))
    steps = sorted(set(idm) & set(oodm))
    if not steps:
        return None
    return [idm[s] for s in steps], [oodm[s] for s in steps], steps

# --- encoding ---------------------------------------------------------------
# SFT LR is ordinal, and a one-hue light->dark ramp is the textbook encoding for
# that — but it was unreadable here. An ordinal ramp is validated on ADJACENT
# pairs, on the assumption that neighbouring steps sit next to each other and are
# compared locally. In these panels all four levels are overlaid as crossing
# lines, so every pair is on screen at once: an all-pairs problem, where four
# steps of one hue are four shades of blue nobody can tell apart.
#
# So SFT LR takes categorical hues, assigned in fixed slot order (1e-6 -> slot 1,
# 3e-6 -> 2, 1e-5 -> 3, 3e-5 -> 7). Validated all-pairs on the light surface:
# worst CVD dE 9.2 (aqua<->orange, deutan), worst normal-vision dE 16.3
# (violet<->blue) — both clear. Slots 4-6 and 8 were rejected: red/orange and
# green/orange fail the all-pairs floors together.
#
# Cost of the change: hue no longer reads as "more/less LR". The legend is
# ordered by LR and the .md carries the full score table, which is the
# non-colour route to the same information — also the relief the validator's
# contrast WARN requires for aqua (2.74:1 on this surface).
#
# NB: exactly one hex per SFT LR. An earlier version mapped 1e-4 onto the 3e-5
# colour, which made two different LRs identical in the mixture panels. Any LR
# not listed falls back to grey, visibly.
SFT_LR_COLOR = {"1e-6": "#2a78d6", "3e-6": "#eb6834",
                "1e-5": "#1baf7a", "3e-5": "#4a3aa7"}
SFT_LR_ORDER = ["1e-6", "3e-6", "1e-5", "3e-5"]
KL_STYLE = {"3e-4": "-", "1e-3": ":", "3e-3": "--"}
SEED_MARKER = {1: "o", 21: "^"}

# the 1.5B runs are a neutral reference family, deliberately outside the ramp
TEACHER_INK = "#2f2e2b"
# lr ramp for the 1.5B-only panels: same grammar, its own lr levels
GRPO_LR_COLOR = {"3e-7": "#86b6ef", "1e-6": "#3987e5",
                 "3e-6": "#1c5cab", "6e-6": "#0d366b"}
GRPO_LR_ORDER = ["3e-7", "1e-6", "3e-6", "6e-6"]

# categorical slots 1-3 — the three that clear the all-pairs CVD floor, used
# ONLY where method is the variable under comparison
METHOD_COLOR = {"distill": "#2a78d6", "progdistill": "#eb6834",
                "mixture": "#1baf7a"}

METHODS = [("distill", "vanilla distill → GRPO"),
           ("progdistill", "progressive distill → GRPO"),
           ("mixture", "progdistill-mixture → GRPO")]
METHOD_SHORT = {"distill": "vanilla distill", "progdistill": "progressive distill",
                "mixture": "progdistill-mixture"}

CANONICAL_TEACHER = "1.5B lr3e-6 kl3e-3 (teacher)"

SURFACE = "#fcfcfb"
INK = "#1c1b19"
INK_MUTED = "#6b6a66"
GRID = "#e6e5e1"

LEGEND_NCOL = 4


def student_style(entry):
    """(color, linestyle, marker) for a STUDENTS registry row."""
    _label, _tree, _exp, _method, sft_lr, kl, seed, _owner, _note = entry
    return (SFT_LR_COLOR[sft_lr], KL_STYLE.get(kl, "-"), SEED_MARKER.get(seed, "s"))


def draw_curve(ax, xs, ys, color, linestyle, marker, scale=1.0, label=None):
    """One run: line through checkpoints, open ring on the first, star on the last."""
    ax.plot(xs, ys, linestyle=linestyle, color=color, alpha=0.75,
            linewidth=2 * scale, zorder=2)
    kw = {"label": label} if label else {}
    ax.scatter(xs, ys, s=34 * scale ** 2, marker=marker, color=color,
               edgecolors=SURFACE, linewidths=1.0 * scale, zorder=3, **kw)
    ax.scatter([xs[0]], [ys[0]], s=110 * scale ** 2, marker="o", facecolors="none",
               edgecolors=color, linewidths=1.8 * scale, zorder=4)
    ax.scatter([xs[-1]], [ys[-1]], s=210 * scale ** 2, marker="*", color=color,
               edgecolors=INK, linewidths=0.6 * scale, zorder=5)


def annotate_steps(ax, xs, ys, steps, scale=1.0):
    """Alternate the offset: consecutive checkpoints bunch up, flipping keeps them legible."""
    for j, (st, x, y) in enumerate(zip(steps, xs, ys)):
        dx, dy = (5, 4) if j % 2 == 0 else (5, -9)
        ax.annotate(str(st), (x, y), fontsize=5.8 * scale, color=INK_MUTED,
                    textcoords="offset points", xytext=(dx, dy), zorder=6)


def style_axes(ax, title, xlabel, ylabel, title_size=11):
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
    ax.set_title(title, fontsize=title_size, color=INK)


def _blank():
    return Line2D([], [], linestyle="none", label=" ")


def legend_handles(lr_color, lr_order, lr_title, students=True):
    """Compound legend: one block per encoding channel, not one entry per run.

    19 runs would make a 19-row legend nobody reads; the channels are only a few
    rows each and let a reader decode any curve. matplotlib fills a legend
    column-major, so each block is padded to the same length — that puts one
    channel per column with its header on the top row, instead of headers and
    items interleaving.
    """
    blocks = [
        [Line2D([], [], linestyle="none", label=lr_title)] +
        [Line2D([], [], color=lr_color[lr], linewidth=3, label=f"   {lr}")
         for lr in lr_order if lr in lr_color],
        [Line2D([], [], linestyle="none", label="GRPO KL")] +
        [Line2D([], [], color=INK_MUTED, linestyle=ls, linewidth=2, label=f"   {kl}")
         for kl, ls in KL_STYLE.items()],
        [Line2D([], [], linestyle="none", label="GRPO seed")] +
        [Line2D([], [], color=INK_MUTED, marker=mk, linestyle="none",
                label=f"   seed {seed}")
         for seed, mk in SEED_MARKER.items() if students or seed == 1],
        # the 1.5B-only panels have no SFT origin and no teacher overlay, so those
        # two rows would be legend entries with nothing on the canvas
        ([Line2D([], [], linestyle="none", label="reference"),
          Line2D([], [], color=INK_MUTED, marker="s", markerfacecolor="none",
                 linestyle="none", label="   SFT student (pre-GRPO)"),
          Line2D([], [], color=TEACHER_INK, marker="h", linestyle="-.",
                 linewidth=2, label="   1.5B teacher (lr3e-6 kl3e-3)")]
         if students else []),
    ]
    blocks = [b for b in blocks if b]
    rows = max(len(b) for b in blocks)
    return [h for b in blocks for h in b + [_blank()] * (rows - len(b))]


def method_legend_handles():
    """Legend for the cross-method panel, where color = method.

    Padded per block for the same reason as legend_handles: matplotlib fills
    column-major, so equal-length blocks keep one channel per column.
    """
    blocks = [
        [Line2D([], [], linestyle="none", label="SFT method (GRPO kl=3e-3 only)")] +
        [Line2D([], [], color=METHOD_COLOR[m], marker="o", linewidth=2,
                label=f"   {METHOD_SHORT[m]}") for m, _t in METHODS],
        [Line2D([], [], linestyle="none", label="reference"),
         Line2D([], [], color=INK_MUTED, marker="s", markerfacecolor="none",
                linestyle="none", label="   SFT student (pre-GRPO)"),
         Line2D([], [], color=TEACHER_INK, marker="h", linestyle="-.",
                linewidth=2, label="   1.5B teacher (lr3e-6 kl3e-3)")],
    ]
    rows = max(len(b) for b in blocks)
    return [h for b in blocks for h in b + [_blank()] * (rows - len(b))]


# Zoomed variants drop this many leading checkpoints from every 1.5B curve. The
# 1.5B runs start far below where the students live (the teacher's step-150
# checkpoint is at ID pass@1 0.337 vs the students' 0.92), so including them
# stretches the axes ~2.5-3.7x and squashes the interesting cluster.
TEACHER_SKIP = 2

# Two 1.5B runs (`lr3e-6 kl1e-3` and `lr3e-6 kl3e-4`, both in Shlok's tree) never
# clear ID pass@1 0.01 at ANY checkpoint — they are flat at the origin, so a
# leading-checkpoint trim does nothing for them and they alone pin the 1.5B
# panel's axes to 0. The zoomed 1.5B panel drops any run whose peak ID pass@1 is
# below this floor, for the same "performance there is too poor" reason.
DEAD_RUN_FLOOR = 0.05


def is_dead(df, label, floor=DEAD_RUN_FLOOR):
    """True if a run never clears `floor` in-distribution pass@1."""
    q = series(df, label, ID_DATASET)
    return q is None or max(q[0]) < floor


def trim(p, n):
    """Drop the first n checkpoints of a (xs, ys, steps) triple, or None."""
    if p is None:
        return None
    xs, ys, steps = p
    if len(steps) <= n:
        return None
    return xs[n:], ys[n:], steps[n:]


def draw_origins(ax, df, entries, fetch, color_of, scale=1.0):
    """Open squares for the pre-GRPO SFT students the plotted curves start from.

    Only students whose (method, SFT LR) matches a run actually drawn in this
    panel are shown. Drawing every SFT student regardless — as an earlier version
    did — puts squares on the canvas that are the origin of no trajectory in
    view, which reads as several unexplained boxes per colour. The full
    SFT-student set is tabulated in results/sft_grpo_Qwen_0.5B_vs_1.5B.md.
    """
    shown = {(e[3], e[4]) for e in entries}
    for label, _tree, _exp, method, sft_lr, _owner in SFT_ONLY:
        if (method, sft_lr) not in shown:
            continue
        p = fetch(label)
        if not p:
            continue
        ax.scatter(p[0], p[1], s=95 * scale ** 2, marker="s", facecolors="none",
                   edgecolors=color_of(method, sft_lr), linewidths=1.8 * scale,
                   zorder=6)


def label_endpoint(ax, x, y, text, color, scale=1.0):
    """Direct-label a curve's last checkpoint.

    Used on the cross-method panel: there color carries METHOD, so the spread
    *within* a method — which is driven by SFT LR — would otherwise be
    unreadable. Eight curves is past the direct-label-everything threshold, so
    only the endpoint is labeled.
    """
    ax.annotate(text, (x, y), fontsize=6.8 * scale, color=color, zorder=7,
                textcoords="offset points", xytext=(7, -3),
                path_effects=None)


def save(fig, out_root, name):
    path = os.path.join(out_root, name)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path
