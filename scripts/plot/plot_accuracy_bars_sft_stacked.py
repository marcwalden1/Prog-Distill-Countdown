"""Stacked accuracy bar plot for the three Qwen2.5-0.5B SFT checkpoints
listed in sft-checkpoints/best_models.md (distill / progdistill / PD mixture).

Same encoding as plot_accuracy_bars.py: each bar stacks
  - solid       = pass@1  (accuracy of the first sample)
  - striped top = pass@32 - pass@1  (extra prompts solved by >=1 of 32 samples)
so full bar height is pass@32.

Scores are computed directly from each eval JSON (full prompt set, no cap),
NOT from the .scores/.passk cache files: an earlier bug left some cached n=5/n=6
values wrong, and the caches are trusted whenever newer than the JSON, so they
must be bypassed here. Scoring reuses compute_score via plot_passk._count_correct
and plot_passk.pass_at_k so the metric definition matches every repo table.

Three bars per run: `balanced` (n=3,4), `balanced5` (n=5), `balanced6` (n=6).
Unlike plot_accuracy_bars.py, run dirs point directly at the folder holding the
{ds}_temp0.6_n32_max1024.json files (SFT checkpoints have no global_step_N dir).

Usage: python3 scripts/plot/plot_accuracy_bars_sft_stacked.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_passk import _count_correct, pass_at_k

# Local copy of sdholakia's eval results (full-set re-scored) — see results_sdholakia/.
_SDH = "results_sdholakia/Qwen2.5-0.5B"
# Each entry: (figures_dir, [(run label, dir directly containing the
# {ds}_temp0.6_n32_max1024.json files), ...]). The best SFT checkpoints per
# model are listed in sft-checkpoints/best_models.md.
FIGURES = [
    ("figures/Qwen2.5-0.5B", [
        ("distill",
         f"{_SDH}/balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr3e-5-seed1/sft_final"),
        ("progdistill",
         f"{_SDH}/balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr3e-5-seed1/round_1600"),
        ("PD mixture",
         "results/Qwen2.5-0.5B/"
         "balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr3e-5-steps1600-seed1/sft_final"),
    ]),
    ("figures/Qwen2.5-40M-deep", [
        ("distill",
         "results/Qwen2.5-40M-deep/balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
        ("PD mixture",
         "results/Qwen2.5-40M-deep/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
    ]),
    ("figures/Qwen2.5-40M-wide", [
        ("distill",
         "results/Qwen2.5-40M-wide/balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
        ("PD mixture",
         "results/Qwen2.5-40M-wide/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
    ]),
    # wide + deep merged into one figure (mirrors figures/Qwen2.5-40M layout).
    ("figures/Qwen2.5-40M", [
        ("40M-wide\ndistill",
         "results/Qwen2.5-40M-wide/balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
        ("40M-wide\nPD mixture",
         "results/Qwen2.5-40M-wide/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
        ("40M-deep\ndistill",
         "results/Qwen2.5-40M-deep/balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
        ("40M-deep\nPD mixture",
         "results/Qwen2.5-40M-deep/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-seed1/sft_final"),
    ]),
]
DATASETS = [("balanced", "n=3,4"), ("balanced5", "n=5"), ("balanced6", "n=6")]
# ordinal single-hue blue ramp: datasets ordered by difficulty
DS_COLORS = ["#86b6ef", "#3987e5", "#1c5cab"]
plt.rcParams.update({"font.size": 16})


def _score_json(path):
    """(pass1, pass32) computed directly from the eval JSON — no cache read.

    pass1 = fraction of prompts whose first sample is correct.
    pass32 = unbiased pass@k (k=n samples/prompt) averaged over prompts.
    """
    with open(path) as f:
        data = json.load(f)
    data = [item for item in data if item.get("outputs")]
    if not data:
        return None, None
    n = min(len(item["outputs"]) for item in data)
    items = [(item["outputs"][:n], item["target"], item["nums"]) for item in data]

    from multiprocessing import Pool
    with Pool(8) as pool:
        counts = pool.map(_count_correct, items)

    # _count_correct scores outputs[0] first, so pass@1 = fraction with a
    # correct first sample. Recompute pass@1 from the first-sample check.
    from grader_utils import compute_score
    first_correct = [
        compute_score(None, item["outputs"][0], item["target"],
                      {"numbers": item["nums"]}, verbose=False) == 1.0
        for item in data
    ]
    p1 = sum(first_correct) / len(first_correct)
    p32 = pass_at_k(n, counts, 32)
    return p1, p32


def scores_for_run(run_dir):
    """Return {ds: (pass1, pass32)} for one run, None where the JSON is missing."""
    out = {}
    for ds, _ in DATASETS:
        path = os.path.join(run_dir, f"{ds}_temp0.6_n32_max1024.json")
        out[ds] = _score_json(path) if os.path.isfile(path) else None
    return out


def plot_figure(figures_dir, runs):
    os.makedirs(figures_dir, exist_ok=True)
    labels = [lbl for lbl, _ in runs]
    data = [scores_for_run(d) for _, d in runs]

    fig, ax = plt.subplots(figsize=(2.4 * len(runs) + 2, 5.5))
    n_ds = len(DATASETS)
    width = 0.8 / n_ds
    for j, ((ds, ds_label), color) in enumerate(zip(DATASETS, DS_COLORS)):
        for i in range(len(runs)):
            cell = data[i][ds]
            x = i + (j - (n_ds - 1) / 2) * width
            if cell is None or cell[0] is None or cell[1] is None:
                continue
            p1, p32 = cell
            gap = max(p32 - p1, 0.0)
            # solid = pass@1
            ax.bar(x, p1, width * 0.92, color=color,
                   label=ds_label if i == 0 else None)
            # striped top = pass@32 - pass@1 (white hatch over same color)
            ax.bar(x, gap, width * 0.92, bottom=p1, color=color,
                   hatch="////", edgecolor="white", linewidth=0.0)
            # value labels: pass@32 above bar; pass@1 at the solid boundary
            ax.text(x, p32 + 0.012, f"{p32:.2f}", ha="center", va="bottom",
                    fontsize=12)
            if gap > 0.03:
                ax.text(x, p1 - 0.012, f"{p1:.2f}", ha="center", va="top",
                        fontsize=11, color="white")

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="test set", fontsize=16, title_fontsize=16, frameon=False,
              loc="upper right", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out = os.path.join(figures_dir, "accuracy_bars_sft_stacked.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for lbl, cells in zip(labels, data):
        row = "  ".join(
            f"{ds}:{'--' if cells[ds] is None else f'{cells[ds][0]:.3f}/{cells[ds][1]:.3f}'}"
            for ds, _ in DATASETS)
        print(f"    {lbl:14s} {row}")


def main():
    for figures_dir, runs in FIGURES:
        print(f"\n=== {figures_dir} ===")
        plot_figure(figures_dir, runs)


if __name__ == "__main__":
    main()
