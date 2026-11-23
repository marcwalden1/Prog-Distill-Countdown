#!/usr/bin/env python3
"""
ID-vs-OOD Pareto trajectories for the Qwen2.5-0.5B students that were SFT'd on
the Qwen2.5-1.5B teacher and then GRPO'd — the runs listed in
results/sft_grpo_Qwen_0.5B_vs_1.5B.md.

    x = in-distribution accuracy (`balanced`, n=3,4)
    y = OOD accuracy (`balanced5` -> n=5, `balanced6` -> n=6)

One marker per evaluated checkpoint, connected in step order; open ring = first
checkpoint, star = last. A point is plotted only where the SAME checkpoint has
both datasets scored, so a run with 10 `balanced` and 9 `balanced5` files
contributes 9 points, not a mismatched 10.

Reading the plot: up-and-right is strictly better, so one arm Pareto-dominates
another only if its curve sits above and to the right. A curve that climbs while
moving left is buying OOD by giving up in-distribution accuracy.

Why not scripts/plot/plot_id_vs_ood_runs.py: that one assigns one color per run
from colors.py plus four fallbacks, which cycles after four curves. This run set
has 19 students, so it uses the composite encoding in
scripts/plot/qwen_sft_grpo_style.py (color = SFT LR, linestyle = GRPO KL,
marker = seed, facet = SFT method), shared with
scripts/plot/plot_pass1_vs_pass32_sft_grpo.py so one legend key decodes both.

Output — figures/<out-dir>/id_vs_ood_<tag>/:
    grid.png              3 methods x (2 metrics x 2 OOD sets)
    <metric>_<ood>.png    one metric+OOD pair, the three method panels standalone
    methods_kl3e-3.png    cross-method comparison at matched KL — the panel that
                          answers "does one SFT method Pareto-dominate another?",
                          where color carries METHOD instead of SFT LR
    qwen15b_grpo.png      the 1.5B GRPO runs on their own

Usage:
    python3 scripts/plot/plot_id_vs_ood_sft_grpo.py
    python3 scripts/plot/plot_id_vs_ood_sft_grpo.py --annotate
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

# the four (metric, OOD set) panels, in the order they appear as grid columns
COMBOS = [("pass@1", "balanced5"), ("pass@1", "balanced6"),
          ("pass@32", "balanced5"), ("pass@32", "balanced6")]


def panel_title(metric, ood):
    return f"{metric} — ID (n=3,4) vs OOD {S.OOD_LABEL[ood]}"


def axis_labels(metric, ood):
    return (f"in-distribution (n=3,4)  {metric}",
            f"OOD {S.OOD_LABEL[ood]}  {metric}")


def draw_method_panel(ax, df, method, metric, ood, annotate, scale=1.0):
    """One (method, metric, OOD) panel: its students + SFT origins + the teacher."""
    col = S.METRIC_COL[metric]
    plotted = 0
    # the teacher first, so student curves draw over it
    t = S.id_ood_series(df, S.CANONICAL_TEACHER, ood, col)
    if t:
        S.draw_curve(ax, t[0], t[1], S.TEACHER_INK, "-.", "h", scale)
    entries = [e for e in STUDENTS if e[3] == method]
    # pre-GRPO SFT students: one open square per curve origin in this panel
    S.draw_origins(ax, df, entries, lambda l: S.id_ood_series(df, l, ood, col),
                   lambda m, lr: S.SFT_LR_COLOR.get(lr, S.INK_MUTED), scale)
    for entry in entries:
        p = S.id_ood_series(df, entry[0], ood, col)
        if not p:
            continue
        plotted += 1
        color, linestyle, marker = S.student_style(entry)
        S.draw_curve(ax, p[0], p[1], color, linestyle, marker, scale)
        if annotate:
            S.annotate_steps(ax, p[0], p[1], p[2], scale)
    return plotted


def combo_limits(df, metric, ood, pad=0.06):
    """Limits shared by the three method rows of one (metric, OOD) column."""
    col = S.METRIC_COL[metric]
    xs, ys = [], []
    labels = ([e[0] for e in STUDENTS] + [s[0] for s in SFT_ONLY] +
              [S.CANONICAL_TEACHER])
    for label in labels:
        p = S.id_ood_series(df, label, ood, col)
        if p:
            xs += p[0]
            ys += p[1]
    if not xs:
        return None
    dx = pad * (max(xs) - min(xs)) or 0.01
    dy = pad * (max(ys) - min(ys)) or 0.01
    return (max(0.0, min(xs) - dx), max(xs) + dx,
            max(0.0, min(ys) - dy), max(ys) + dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=S.ANALYSIS_DIR)
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "figures"))
    ap.add_argument("--out-dir", default="comparison", help="subdir under figures/")
    ap.add_argument("--tag", default="qwen_sft_grpo")
    ap.add_argument("--annotate", action="store_true",
                    help="label every checkpoint with its training step")
    args = ap.parse_args()

    df = S.load_scores(args.analysis_dir)
    have = set(df.run.unique())
    for entry in STUDENTS:
        if entry[0] not in have:
            print(f"  [missing] no pass@k rows for {entry[0]} — curve omitted")

    out_root = os.path.join(args.figures_dir, args.out_dir, f"id_vs_ood_{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    limits = {c: combo_limits(df, *c) for c in COMBOS}

    # --- grid: 3 methods x 4 (metric, OOD) combos ---------------------------
    fig, axes = plt.subplots(3, 4, figsize=(23.0, 15.5), facecolor=S.SURFACE)
    for r, (method, mtitle) in enumerate(S.METHODS):
        for c, (metric, ood) in enumerate(COMBOS):
            ax = axes[r][c]
            n = draw_method_panel(ax, df, method, metric, ood, args.annotate)
            if limits[(metric, ood)]:
                ax.set_xlim(*limits[(metric, ood)][:2])
                ax.set_ylim(*limits[(metric, ood)][2:])
            S.style_axes(ax, f"{mtitle}\n{panel_title(metric, ood)}",
                         *axis_labels(metric, ood), title_size=10)
            print(f"[grid] {method} / {metric} / {ood}: {n} runs")
    fig.legend(handles=S.legend_handles(S.SFT_LR_COLOR, S.SFT_LR_ORDER, "SFT LR"),
               loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=8.5,
               title="open ring = first ckpt, star = last ckpt; up-and-right is "
                     "better; axis limits shared down each column",
               title_fontsize=8.5)
    fig.suptitle("ID vs OOD across GRPO — Qwen2.5-0.5B students SFT'd on the "
                 "Qwen2.5-1.5B teacher", fontsize=13, color=S.INK, y=0.997)
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    S.save(fig, out_root, "grid.png")

    # --- one file per (metric, OOD) combo ------------------------------------
    for metric, ood in COMBOS:
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), facecolor=S.SURFACE)
        for c, (method, mtitle) in enumerate(S.METHODS):
            ax = axes[c]
            draw_method_panel(ax, df, method, metric, ood, args.annotate)
            if limits[(metric, ood)]:
                ax.set_xlim(*limits[(metric, ood)][:2])
                ax.set_ylim(*limits[(metric, ood)][2:])
            S.style_axes(ax, mtitle, *axis_labels(metric, ood), title_size=11)
        fig.legend(handles=S.legend_handles(S.SFT_LR_COLOR, S.SFT_LR_ORDER, "SFT LR"),
                   loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=8.2)
        fig.suptitle(panel_title(metric, ood), fontsize=13, color=S.INK)
        fig.tight_layout(rect=(0, 0.17, 1, 0.96))
        S.save(fig, out_root,
               f"{metric.replace('@', 'at')}_{S.OOD_LABEL[ood].replace('=', '')}.png")

    # --- cross-method comparison at matched KL -------------------------------
    # The KL=3e-3 runs are the ones that don't collapse, so they are the only
    # apples-to-apples set for "does one SFT method dominate another?". Here
    # method IS the question, so it takes the color channel.
    kl_runs = [e for e in STUDENTS if e[5] == "3e-3"]

    def methods_figure(drop):
        fig, axes = plt.subplots(1, 4, figsize=(23.0, 5.9), facecolor=S.SURFACE)
        for c, (metric, ood) in enumerate(COMBOS):
            ax = axes[c]
            col = S.METRIC_COL[metric]
            t = S.trim(S.id_ood_series(df, S.CANONICAL_TEACHER, ood, col), drop)
            if t:
                S.draw_curve(ax, t[0], t[1], S.TEACHER_INK, "-.", "h")
            S.draw_origins(ax, df, kl_runs, lambda l: S.id_ood_series(df, l, ood, col),
                           lambda m, lr: S.METHOD_COLOR[m])
            for entry in kl_runs:
                p = S.id_ood_series(df, entry[0], ood, col)
                if not p:
                    continue
                S.draw_curve(ax, p[0], p[1], S.METHOD_COLOR[entry[3]], "-",
                             S.SEED_MARKER.get(entry[6], "o"))
                # color is method here, so name the SFT LR at the endpoint — the
                # within-method spread below is almost entirely an SFT-LR effect
                S.label_endpoint(ax, p[0][-1], p[1][-1], entry[4],
                                 S.METHOD_COLOR[entry[3]])
                if args.annotate:
                    S.annotate_steps(ax, p[0], p[1], p[2])
            S.style_axes(ax, panel_title(metric, ood), *axis_labels(metric, ood))
        fig.legend(handles=S.method_legend_handles(), loc="lower center", ncol=2,
                   frameon=False, fontsize=9)
        zoom = (f"\nZOOM — first {drop} checkpoints of the 1.5B teacher curve dropped, "
                "so the axes fit where the students actually sit" if drop else "")
        fig.suptitle("ID vs OOD by SFT method — Qwen2.5-0.5B, GRPO lr=1e-6 kl=3e-3 "
                     "(the non-collapsing runs)\nsquares = the pre-GRPO SFT students; "
                     f"endpoint labels = SFT LR{zoom}", fontsize=12, color=S.INK)
        fig.tight_layout(rect=(0, 0.15, 1, 0.93))
        S.save(fig, out_root,
               "methods_kl3e-3_zoom.png" if drop else "methods_kl3e-3.png")

    print(f"[methods] {len(kl_runs)} runs x {len(COMBOS)} panels")
    methods_figure(0)
    methods_figure(S.TEACHER_SKIP)

    # --- the 1.5B GRPO runs on their own -------------------------------------
    def teachers_figure(drop):
        fig, axes = plt.subplots(1, 4, figsize=(23.0, 5.9), facecolor=S.SURFACE)
        for c, (metric, ood) in enumerate(COMBOS):
            ax = axes[c]
            n = 0
            for label, _tree, _exp, lr, kl, _owner, _have in TEACHERS:
                if drop and S.is_dead(df, label):
                    continue
                p = S.trim(S.id_ood_series(df, label, ood, S.METRIC_COL[metric]), drop)
                if not p:
                    continue
                n += 1
                S.draw_curve(ax, p[0], p[1], S.GRPO_LR_COLOR.get(lr, S.TEACHER_INK),
                             S.KL_STYLE.get(kl, "-"), "h")
                if args.annotate:
                    S.annotate_steps(ax, p[0], p[1], p[2])
            S.style_axes(ax, panel_title(metric, ood), *axis_labels(metric, ood))
            if not drop:
                print(f"[1.5B] {metric} / {ood}: {n} runs")
        fig.legend(handles=S.legend_handles(S.GRPO_LR_COLOR, S.GRPO_LR_ORDER,
                                            "GRPO LR", students=False),
                   loc="lower center", ncol=S.LEGEND_NCOL, frameon=False, fontsize=8.2)
        zoom = (f"\nZOOM — first {drop} checkpoints dropped, and the 2 runs that never "
                f"clear ID pass@1 {S.DEAD_RUN_FLOOR} omitted" if drop else "")
        fig.suptitle("ID vs OOD across GRPO — Qwen2.5-1.5B, GRPO from the base "
                     f"model{zoom}", fontsize=13, color=S.INK)
        fig.tight_layout(rect=(0, 0.15, 1, 0.95))
        S.save(fig, out_root,
               "qwen15b_grpo_zoom.png" if drop else "qwen15b_grpo.png")

    teachers_figure(0)
    teachers_figure(S.TEACHER_SKIP)

    # --- endpoint table ------------------------------------------------------
    rows = []
    for label, _tree, exp, method, sft_lr, kl, seed, owner, _note in STUDENTS:
        row = {"run": label, "model": "Qwen2.5-0.5B", "method": method,
               "sft_lr": sft_lr, "grpo_kl": kl, "seed": seed, "owner": owner}
        for metric, ood in COMBOS:
            p = S.id_ood_series(df, label, ood, S.METRIC_COL[metric])
            if p:
                row["final_step"] = p[2][-1]
                row[f"ID_{metric}"] = round(p[0][-1], 4)
                row[f"{S.OOD_LABEL[ood]}_{metric}"] = round(p[1][-1], 4)
        rows.append(row)
    for label, _tree, exp, lr, kl, owner, _have in TEACHERS:
        row = {"run": label, "model": "Qwen2.5-1.5B", "method": "base GRPO",
               "sft_lr": "", "grpo_kl": kl, "seed": 1, "owner": owner}
        for metric, ood in COMBOS:
            p = S.id_ood_series(df, label, ood, S.METRIC_COL[metric])
            if p:
                row["final_step"] = p[2][-1]
                row[f"ID_{metric}"] = round(p[0][-1], 4)
                row[f"{S.OOD_LABEL[ood]}_{metric}"] = round(p[1][-1], 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(args.analysis_dir, f"id_vs_ood_{args.tag}_final.csv")
    table.to_csv(csv_path, index=False)
    pd.set_option("display.width", 250)
    print(f"\nfinal checkpoint of each curve:\n{table.to_string(index=False)}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
