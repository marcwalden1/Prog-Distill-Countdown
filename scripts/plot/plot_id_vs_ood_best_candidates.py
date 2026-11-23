#!/usr/bin/env python3
"""ID-vs-OOD comparison of the hand-picked best-candidate Gemma-270M GRPO runs.

The runs are the ones listed in
figures/gemma-3-270m/id_vs_ood_draft/best_candidates.md (7 runs across the KD,
PD, and mixture arms). Each run is drawn as an ID-vs-OOD accuracy trajectory:
in-distribution (balanced, n=3,4) on x vs OOD (n=5 or n=6) on y, one marker per
evaluated checkpoint, connected in step order (open circle = first step, star =
last). 2x2 grid: rows = metric (mean@1, mean@32), cols = OOD set (n=5, n=6).

Scoring reuses plot_id_vs_ood_gemma_groups.load_scores, which scores the FULL
prompt set (not the buggy first-1k) and memoizes into the same
.score_cache_full.json, so this is instant after the group plots have been run.

Usage: python3 scripts/plot/plot_id_vs_ood_best_candidates.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_id_vs_ood_gemma_groups import (
    ID_DATASET, METRICS, OOD, OUT_DIR, REPO,
    _load_cache, _save_cache, load_scores, resolve_score_dir,
)

RS = os.path.join(REPO, "results_shlok", "gemma-3-270m")
RSN = os.path.join(REPO, "results_shlok", "results", "gemma-3-270m")
RES = os.path.join(REPO, "results", "gemma-3-270m")

# (label, run dir, color) — grouped by arm, one distinct color each.
RUNS = [
    # KD (distill) + GRPO, all sft 1e-4
    ("KD lr3e-6 kl3e-4",
     f"{RS}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr3e-6-kl3e-4-seed1",
     "#1c5cab"),
    ("KD lr3e-6 kl1e-3",
     f"{RSN}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr3e-6-kl1e-3-seed1",
     "#4a90d9"),
    ("KD lr3e-6 kl3e-3",
     f"{RS}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr3e-6-kl3e-3-seed1",
     "#8fc1e8"),
    # PD (progdistill) + GRPO, all sft 1e-4
    ("PD lr3e-6 kl1e-3",
     f"{RS}/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr3e-6-kl1e-3-seed1",
     "#b45309"),
    # mixture (mine) + GRPO, sft 3e-4
    ("mixture sft3e-4 lr3e-6 kl1e-3",
     f"{RES}/balanced-grpo-from-progdistill-mixture-sftlr3e-4-lr3e-6-kl1e-3-seed1",
     "#15803d"),
]

OUT = os.path.join(OUT_DIR, "best_candidates.png")
plt.rcParams.update({"font.size": 12})


def main():
    _load_cache()
    wanted = [ID_DATASET] + [ds for ds, _ in OOD]
    data = {}
    for lbl, d, _c in RUNS:
        rd = resolve_score_dir(d)
        if rd is None:
            print(f"  [skip] no eval JSONs under {d}")
            continue
        data[lbl] = {ds: load_scores(rd, ds) for ds in wanted}
    _save_cache()

    fig, axes = plt.subplots(len(METRICS), len(OOD), figsize=(13, 11))
    for r, (metric_name, mi) in enumerate(METRICS):
        idx = mi - 1
        for c, (ood_ds, ood_label) in enumerate(OOD):
            ax = axes[r][c]
            for lbl, _d, color in RUNS:
                if lbl not in data:
                    continue
                id_by, ood_by = data[lbl][ID_DATASET], data[lbl][ood_ds]
                steps = sorted(set(id_by) & set(ood_by))
                if not steps:
                    continue
                xs = [id_by[s][idx] for s in steps]
                ys = [ood_by[s][idx] for s in steps]
                ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=2, zorder=1)
                ax.scatter(xs, ys, s=34, color=color, label=lbl, zorder=2)
                ax.scatter([xs[0]], [ys[0]], s=100, facecolors="none",
                           edgecolors=color, linewidths=1.8, zorder=3)
                ax.scatter([xs[-1]], [ys[-1]], s=170, marker="*", color=color,
                           edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
            ax.set_xlabel(f"ID {metric_name} (n=3,4)")
            ax.set_ylabel(f"OOD {metric_name} ({ood_label})")
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, frameon=False,
               loc="center left", bbox_to_anchor=(0.87, 0.5),
               title="○ first step   ★ last step", title_fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}  ({len(data)} runs)")
    for lbl in (l for l, *_ in RUNS):
        if lbl in data:
            print(f"  {lbl:24s} checkpoints {sorted(data[lbl][ID_DATASET])}")

    # Single-panel version = top-left subplot only: mean@1, OOD n=5 (balanced5).
    metric_name, mi = METRICS[0]          # ("mean@1", 1)
    idx = mi - 1
    ood_ds, ood_label = OOD[0]            # ("balanced5", "n=5")
    fig1, ax = plt.subplots(figsize=(7.2, 6.2))
    for lbl, _d, color in RUNS:
        if lbl not in data:
            continue
        id_by, ood_by = data[lbl][ID_DATASET], data[lbl][ood_ds]
        steps = sorted(set(id_by) & set(ood_by))
        if not steps:
            continue
        xs = [id_by[s][idx] for s in steps]
        ys = [ood_by[s][idx] for s in steps]
        ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=2, zorder=1)
        ax.scatter(xs, ys, s=34, color=color, label=lbl, zorder=2)
        ax.scatter([xs[0]], [ys[0]], s=100, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=170, marker="*", color=color,
                   edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
    ax.set_xlabel(f"ID {metric_name} (n=3,4)")
    ax.set_ylabel(f"OOD {metric_name} ({ood_label})")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="best",
              title="○ first step   ★ last step", title_fontsize=9)
    fig1.tight_layout()
    out1 = os.path.join(OUT_DIR, "best_candidates_mean1_n5.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"wrote {out1}  ({len(data)} runs)")


if __name__ == "__main__":
    main()
