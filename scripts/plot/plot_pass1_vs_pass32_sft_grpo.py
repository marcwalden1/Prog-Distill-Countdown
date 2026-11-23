#!/usr/bin/env python3
"""
pass@1 (x) vs pass@32 (y) trajectories for the Qwen2.5-0.5B students that were
SFT'd on the Qwen2.5-1.5B teacher and then GRPO'd — the runs listed in
results/sft_grpo_Qwen_0.5B_vs_1.5B.md.

Sibling of scripts/plot/plot_pass1_vs_pass32.py, which covers GRPO started from
the *base* model. Same reading: pass@32 >= pass@1 by construction, so every point
sits above the grey y=x line; vertical distance from it is the sampling headroom
32 samples buy, and movement right along it is a policy getting sharper.

Encoding — 19 student curves is far past the 8-slot categorical cap, so identity
is composite rather than one-hue-per-run, and every channel carries a real
variable:

    color      SFT learning rate, ordinal one-hue blue ramp (light = 1e-6 ->
               dark = 3e-5). Validated with the skill's --ordinal gate: monotone
               L, adjacent dL >= 0.06, light end 2.06:1 on the light surface.
    linestyle  GRPO KL coefficient (3e-4 solid, 1e-3 dotted, 3e-3 dashed)
    marker     GRPO seed (circle = 1, triangle = 21)
    row facet  SFT method (distill / progdistill / progdistill-mixture)

so a panel never shows more than four hues. The 1.5B teacher is a neutral
dash-dot reference repeated in every panel; the pre-GRPO SFT student, where it
was evaluated, is an open square — the point each GRPO curve starts from.

Axis limits are shared down a dataset column so the three method rows are
directly comparable.

Scores come from the tidy full-set CSVs written by scripts/export_idood_scores.py
(no 1000-prompt cap, unbiased pass@k). Curves are keyed on the `run` label, NOT
(model, exp_name): `balanced-grpo-lr3e-6-kl1e-3-seed1` exists as two different
1.5B runs owned by two different people.

Output — figures/<out-dir>/pass1_vs_pass32_<tag>/:
    grid.png            3 methods x 3 datasets
    <dataset>.png       one dataset, the three method panels standalone
    qwen15b_grpo.png    the 1.5B GRPO runs on their own, 3 datasets

Usage:
    python3 scripts/plot/plot_pass1_vs_pass32_sft_grpo.py
    python3 scripts/plot/plot_pass1_vs_pass32_sft_grpo.py --annotate
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from qwen_sft_grpo_registry import STUDENTS, SFT_ONLY, TEACHERS  # noqa: E402
import qwen_sft_grpo_style as S  # noqa: E402

PANELS = [("balanced", "n=3,4  (in-distribution)"),
          ("balanced5", "n=5  (OOD)"),
          ("balanced6", "n=6  (OOD)")]


def draw_method_panel(ax, df, method, dataset, annotate, scale=1.0, keep=None,
                      teacher_drop=0):
    """One (method, dataset) panel: its students + SFT origins + the teacher.

    `keep`, when given, is the set of run labels to draw; the SFT-student origins
    follow it, so a filtered panel never shows the origin of a curve it dropped.
    `teacher_drop` trims that many leading checkpoints off the 1.5B reference,
    whose early steps sit far left of where the students end up.
    """
    plotted = 0
    # the 1.5B teacher first, so student curves draw over it
    t = S.trim(S.series(df, S.CANONICAL_TEACHER, dataset), teacher_drop)
    if t:
        S.draw_curve(ax, t[0], t[1], S.TEACHER_INK, "-.", "h", scale)
    entries = [e for e in STUDENTS
               if e[3] == method and (keep is None or e[0] in keep)]
    # pre-GRPO SFT students: one open square per curve origin in this panel
    S.draw_origins(ax, df, entries, lambda l: S.series(df, l, dataset),
                   lambda m, lr: S.SFT_LR_COLOR.get(lr, S.INK_MUTED), scale)
    for entry in entries:
        p = S.series(df, entry[0], dataset)
        if not p:
            continue
        plotted += 1
        color, linestyle, marker = S.student_style(entry)
        S.draw_curve(ax, p[0], p[1], color, linestyle, marker, scale)
        if annotate:
            S.annotate_steps(ax, p[0], p[1], p[2], scale)
    return plotted


def surviving_origins(keep):
    """SFT students whose (method, SFT LR) still has a kept run."""
    live = {(e[3], e[4]) for e in STUDENTS if keep is None or e[0] in keep}
    return [s[0] for s in SFT_ONLY if (s[3], s[4]) in live]


def column_limits(df, dataset, pad=0.06, keep=None, teacher_drop=0):
    """Limits shared by the three method rows of one dataset column."""
    xs, ys = [], []
    labels = ([e[0] for e in STUDENTS if keep is None or e[0] in keep] +
              surviving_origins(keep))
    for label in labels + [S.CANONICAL_TEACHER]:
        p = S.series(df, label, dataset)
        if label == S.CANONICAL_TEACHER:
            p = S.trim(p, teacher_drop)
        if p:
            xs += p[0]
            ys += p[1]
    if not xs:
        return None
    dx = pad * (max(xs) - min(xs)) or 0.01
    dy = pad * (max(ys) - min(ys)) or 0.01
    return (max(0.0, min(xs) - dx), max(xs) + dx,
            max(0.0, min(ys) - dy), max(ys) + dy)


def diagonal(ax):
    """pass@32 >= pass@1 by construction; the diagonal is the no-headroom bound.

    Drawn after the limits are fixed and restored afterwards, so the true y=x
    line never stretches the x-range past the data.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo, hi = min(x0, y0), max(x1, y1)
    ax.plot([lo, hi], [lo, hi], "--", color="#c9c8c3", linewidth=1, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=os.path.join(REPO, "analysis_data"))
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "figures"))
    ap.add_argument("--out-dir", default="comparison", help="subdir under figures/")
    ap.add_argument("--tag", default="qwen_sft_grpo")
    ap.add_argument("--annotate", action="store_true",
                    help="label every checkpoint with its training step")
    ap.add_argument("--pass1-floor", type=float, default=0.6,
                    help="grid_filtered.png keeps only runs whose LAST checkpoint "
                         "exceeds this `balanced` (n=3,4) pass@1")
    args = ap.parse_args()

    df = S.load_scores(args.analysis_dir)
    have = set(df.run.unique())
    for entry in STUDENTS:
        if entry[0] not in have:
            print(f"  [missing] no pass@k rows for {entry[0]} — curve omitted")
    for s in SFT_ONLY:
        if s[0] not in have:
            print(f"  [missing] no pass@k rows for {s[0]} — SFT origin omitted")

    out_root = os.path.join(args.figures_dir, args.out_dir,
                            f"pass1_vs_pass32_{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # --- grid: 3 methods x 3 datasets ---------------------------------------
    def grid_figure(keep=None, name="grid.png", note="", teacher_drop=0):
        lim = {ds: column_limits(df, ds, keep=keep, teacher_drop=teacher_drop)
               for ds, _ in PANELS}
        fig, axes = plt.subplots(3, 3, figsize=(18.0, 16.0), facecolor=S.SURFACE)
        for r, (method, mtitle) in enumerate(S.METHODS):
            for c, (dataset, dtitle) in enumerate(PANELS):
                ax = axes[r][c]
                n = draw_method_panel(ax, df, method, dataset, args.annotate,
                                      keep=keep, teacher_drop=teacher_drop)
                if lim[dataset]:
                    ax.set_xlim(*lim[dataset][:2])
                    ax.set_ylim(*lim[dataset][2:])
                diagonal(ax)
                S.style_axes(ax, f"{mtitle}\n{dtitle}", "pass@1", "pass@32",
                             title_size=10.5)
                print(f"[{name}] {method} / {dataset}: {n} runs")
        fig.legend(handles=S.legend_handles(S.SFT_LR_COLOR, S.SFT_LR_ORDER, "SFT LR"),
                   loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=9,
                   title="open ring = first ckpt, star = last ckpt; "
                         "grey diagonal = pass@32 = pass@1; "
                         "axis limits shared down each column",
                   title_fontsize=8.5)
        fig.suptitle("pass@1 vs pass@32 across GRPO — Qwen2.5-0.5B students SFT'd on "
                     f"the Qwen2.5-1.5B teacher{note}",
                     fontsize=13, color=S.INK, y=0.997)
        fig.tight_layout(rect=(0, 0.09, 1, 0.98))
        S.save(fig, out_root, name)

    grid_figure()
    keep = {e[0] for e in STUDENTS
            if (q := S.series(df, e[0], "balanced")) and q[0][-1] > args.pass1_floor}
    print(f"[grid_filtered] {len(keep)}/{len(STUDENTS)} runs clear final "
          f"`balanced` pass@1 > {args.pass1_floor}")
    grid_figure(keep, "grid_filtered.png",
                f"\nonly runs whose LAST checkpoint exceeds pass@1 "
                f"{args.pass1_floor} on n=3,4 ({len(keep)} of {len(STUDENTS)}); "
                f"first {S.TEACHER_SKIP} checkpoints of the 1.5B teacher dropped",
                teacher_drop=S.TEACHER_SKIP)

    limits = {ds: column_limits(df, ds) for ds, _ in PANELS}

    # --- one file per dataset ------------------------------------------------
    for dataset, dtitle in PANELS:
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), facecolor=S.SURFACE)
        for c, (method, mtitle) in enumerate(S.METHODS):
            ax = axes[c]
            draw_method_panel(ax, df, method, dataset, args.annotate)
            if limits[dataset]:
                ax.set_xlim(*limits[dataset][:2])
                ax.set_ylim(*limits[dataset][2:])
            diagonal(ax)
            S.style_axes(ax, mtitle, "pass@1", "pass@32", title_size=11)
        fig.legend(handles=S.legend_handles(S.SFT_LR_COLOR, S.SFT_LR_ORDER, "SFT LR"),
                   loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=8.2)
        fig.suptitle(f"pass@1 vs pass@32 — {dtitle}", fontsize=13, color=S.INK)
        fig.tight_layout(rect=(0, 0.17, 1, 0.96))
        path = os.path.join(out_root, f"{dataset}.png")
        fig.savefig(path, facecolor=S.SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # --- the 1.5B GRPO runs on their own ------------------------------------
    def teachers_figure(drop):
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), facecolor=S.SURFACE)
        for c, (dataset, dtitle) in enumerate(PANELS):
            ax = axes[c]
            n = 0
            for label, _tree, _exp, lr, kl, _owner, _have in TEACHERS:
                if drop and S.is_dead(df, label):
                    continue
                p = S.trim(S.series(df, label, dataset), drop)
                if not p:
                    continue
                n += 1
                S.draw_curve(ax, p[0], p[1], S.GRPO_LR_COLOR.get(lr, S.TEACHER_INK),
                             S.KL_STYLE.get(kl, "-"), "h")
                if args.annotate:
                    S.annotate_steps(ax, p[0], p[1], p[2])
            diagonal(ax)
            S.style_axes(ax, dtitle, "pass@1", "pass@32")
            if not drop:
                print(f"[1.5B] {dataset}: {n} runs")
        fig.legend(handles=S.legend_handles(S.GRPO_LR_COLOR, S.GRPO_LR_ORDER,
                                            "GRPO LR", students=False),
                   loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=8.2)
        zoom = (f"\nZOOM — first {drop} checkpoints dropped, and the 2 runs that never "
                f"clear ID pass@1 {S.DEAD_RUN_FLOOR} omitted" if drop else "")
        fig.suptitle("pass@1 vs pass@32 across GRPO — Qwen2.5-1.5B, GRPO from the "
                     f"base model{zoom}", fontsize=13, color=S.INK)
        fig.tight_layout(rect=(0, 0.17, 1, 0.94))
        S.save(fig, out_root,
               "qwen15b_grpo_zoom.png" if drop else "qwen15b_grpo.png")

    teachers_figure(0)
    teachers_figure(S.TEACHER_SKIP)

    # --- endpoint table: the numbers behind the marks, and the non-color read -
    # Several runs collapse late (in-dist pass@1 0.90 at step 1200 -> 0.00 at
    # 1600), so the final checkpoint alone misreports them. Carry the in-dist
    # peak and the step it happened at alongside it.
    def score_row(row, label):
        for dataset, _ in PANELS:
            p = S.series(df, label, dataset)
            if not p:
                continue
            row["final_step"] = p[2][-1]
            row[f"{dataset}_pass@1"] = round(p[0][-1], 4)
            row[f"{dataset}_pass@32"] = round(p[1][-1], 4)
            if dataset == "balanced":
                best = max(range(len(p[0])), key=lambda i: p[0][i])
                row["balanced_peak_pass@1"] = round(p[0][best], 4)
                row["peak_step"] = p[2][best]
                row["collapsed"] = bool(p[0][best] - p[0][-1] > 0.10)
        return row

    rows = []
    for label, tree, exp, method, sft_lr, kl, seed, owner, _note in STUDENTS:
        rows.append(score_row(
            {"run": label, "model": "Qwen2.5-0.5B", "exp_name": exp,
             "method": method, "sft_lr": sft_lr, "grpo_kl": kl, "seed": seed,
             "owner": owner, "tree": tree}, label))
    for label, tree, exp, method, sft_lr, owner in SFT_ONLY:
        rows.append(score_row(
            {"run": label, "model": "Qwen2.5-0.5B", "exp_name": exp,
             "method": method, "sft_lr": sft_lr, "grpo_kl": "", "seed": "",
             "owner": owner, "tree": tree}, label))
    for label, tree, exp, lr, kl, owner, _have in TEACHERS:
        rows.append(score_row(
            {"run": label, "model": "Qwen2.5-1.5B", "exp_name": exp,
             "method": "base GRPO", "sft_lr": "", "grpo_kl": kl, "seed": 1,
             "owner": owner, "tree": tree, "grpo_lr": lr}, label))
    table = pd.DataFrame(rows)
    csv_path = os.path.join(args.analysis_dir,
                            f"pass1_vs_pass32_{args.tag}_final.csv")
    table.to_csv(csv_path, index=False)
    pd.set_option("display.width", 250)
    print(f"\nfinal checkpoint of each curve:\n{table.to_string(index=False)}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
