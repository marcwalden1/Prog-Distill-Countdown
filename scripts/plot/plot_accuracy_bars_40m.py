"""Grouped bar plot: final-checkpoint (step 1600) accuracy for the ~40M students.

Four groups (40M-wide / 40M-deep × KD+GRPO / PD-mixture+GRPO, sftlr=3e-4,
grpo lr=1e-6 kl=3e-3, seed1), three bars per group: the `balanced` (n=3,4),
`balanced5` (n=5) and `balanced6` (n=6) test sets. Scores come from
plot_results.py's _score_checkpoint (with .scores caching, n=5,6 capped at
1000 prompts), so numbers match every other repo table. Writes one figure for
mean@1 (accuracy of first sample) and one for mean@32 (mean compute_score over
32 samples; includes 0.1 format credit). Modeled on scripts/plot/plot_accuracy_bars.py.

Usage: python3 scripts/plot/plot_accuracy_bars_40m.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plot_results import _score_checkpoint

STEP = 1600
CKPT_BASE = ("/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/"
             "rl-checkpoints/checkpoints")
RUNS = [  # (short label, run results dir) — results/ckpts.md order
    ("40M-wide\nKD + GRPO",
     "results/Qwen2.5-40M-wide/"
     "balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1"),
    ("40M-wide\nPD mixture + GRPO",
     "results/Qwen2.5-40M-wide/"
     "balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1"),
    ("40M-deep\nKD + GRPO",
     "results/Qwen2.5-40M-deep/"
     "balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1"),
    ("40M-deep\nPD mixture + GRPO",
     "results/Qwen2.5-40M-deep/"
     "balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1"),
]
DATASETS = [("balanced", "n=3,4"), ("balanced5", "n=5"), ("balanced6", "n=6")]
# ordinal single-hue ramp (blue 250/400/550): datasets are ordered by difficulty
DS_COLORS = ["#86b6ef", "#3987e5", "#1c5cab"]
FIGURES_DIR = "figures/40M-comparison"
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
    ax.set_xticklabels([label for label, _ in RUNS], fontsize=13)
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
    mean1, mean32 = [], []
    for label, run_dir in RUNS:
        row1, row32 = [], []
        for ds, _ in DATASETS:
            path = os.path.join(run_dir, f"global_step_{STEP}",
                                f"{ds}_temp0.6_n32_max1024.json")
            res = _score_checkpoint(path) if os.path.isfile(path) else None
            row1.append(res[1] if res else None)
            row32.append(res[2] if res else None)
            print(f"{label.replace(chr(10), ' ')}  {ds}: "
                  f"{'missing' if res is None else f'mean@1 {res[1]:.4f}  mean@32 {res[2]:.4f}'}")
        mean1.append(row1)
        mean32.append(row32)

    plot_bars(mean1, "Accuracy (mean@1), step 1600",
              os.path.join(FIGURES_DIR, "accuracy_bars_mean1.png"))
    plot_bars(mean32, "Mean score @32 (incl. 0.1 format credit), step 1600",
              os.path.join(FIGURES_DIR, "accuracy_bars_mean32.png"))


if __name__ == "__main__":
    main()
