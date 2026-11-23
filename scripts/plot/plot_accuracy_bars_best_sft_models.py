"""Grouped bar plot: final-checkpoint accuracy for the sft-checkpoints/best_models.md runs.

Four groups (the 1.5B GRPO teacher + the three Qwen2.5-0.5B students listed in
sft-checkpoints/best_models.md), three bars per group: the `balanced` (n=3,4),
`balanced5` (n=5) and `balanced6` (n=6) test sets. mean@1 comes from
plot_results.py's _score_checkpoint; pass@32 (fraction of prompts with >=1
correct among 32 samples) from plot_passk.py's unbiased estimator at k=n=32.
Both use the same .scores/.passk caching + n=5,6 prompt cap, so numbers match
every other repo table. Writes one figure for mean@1 and one for pass@32.

Usage: python3 scripts/plot/plot_accuracy_bars_best_sft_models.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "plot"))
from plot_results import _score_checkpoint
from plot_passk import correct_counts, pass_at_k

# Local copy of sdholakia's eval results (full-set re-scored) — see results_sdholakia/.
_SDH = "results_sdholakia/Qwen2.5-0.5B"
RUNS = [  # (short label, dir directly containing the {ds}_temp0.6_n32_max1024.json files)
    ("Qwen2.5-1.5B\nGRPO (teacher)",
     "results/Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1/global_step_1600"),
    ("Qwen2.5-0.5B\ndistill",
     f"{_SDH}/balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr3e-5-seed1/sft_final"),
    ("Qwen2.5-0.5B\nprogdistill",
     f"{_SDH}/balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr3e-5-seed1/round_1600"),
    ("Qwen2.5-0.5B\nPD mixture",
     "results/Qwen2.5-0.5B/"
     "balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr3e-5-steps1600-seed1/sft_final"),
]
DATASETS = [("balanced", "n=3,4"), ("balanced5", "n=5"), ("balanced6", "n=6")]
# ordinal single-hue ramp (blue 250/400/550): datasets are ordered by difficulty
DS_COLORS = ["#86b6ef", "#3987e5", "#1c5cab"]
FIGURES_DIR = "figures/Qwen2.5-0.5B"
plt.rcParams.update({"font.size": 14})


def plot_bars(values, ylabel, out_path):
    """values[run_idx][ds_idx] -> float or None."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_ds = len(DATASETS)
    width = 0.8 / n_ds
    for j, ((_, ds_label), color) in enumerate(zip(DATASETS, DS_COLORS)):
        xs = [i + (j - (n_ds - 1) / 2) * width for i in range(len(RUNS))]
        ys = [values[i][j] for i in range(len(RUNS))]
        bars = ax.bar(xs, ys, width * 0.92, color=color, label=ds_label)
        ax.bar_label(bars, fmt="%.2f", fontsize=14, padding=2)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([label for label, _ in RUNS], fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    mean1, pass32 = [], []
    for label, run_dir in RUNS:
        row1, rowp = [], []
        for ds, _ in DATASETS:
            path = os.path.join(run_dir, f"{ds}_temp0.6_n32_max1024.json")
            if os.path.isfile(path):
                res = _score_checkpoint(path)
                m1 = res[1] if res else None
                n, counts = correct_counts(path)
                pk = pass_at_k(n, counts, 32) if counts else None
            else:
                m1 = pk = None
            row1.append(m1)
            rowp.append(pk)
            print(f"{label.replace(chr(10), ' ')}  {ds}: "
                  f"{'missing' if m1 is None else f'mean@1 {m1:.4f}  pass@32 {pk:.4f}'}")
        mean1.append(row1)
        pass32.append(rowp)

    plot_bars(mean1, "Accuracy (mean@1)",
              os.path.join(FIGURES_DIR, "accuracy_bars_best_sft_models_mean1.png"))
    plot_bars(pass32, "pass@32 (>=1 correct of 32 samples)",
              os.path.join(FIGURES_DIR, "accuracy_bars_best_sft_models_pass32.png"))


if __name__ == "__main__":
    main()
