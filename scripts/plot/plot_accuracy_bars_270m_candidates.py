"""Stacked accuracy bars comparing gemma-3-270m SFT candidates.

Three arms — distill, progdistill (both Shlok's), and progdistill_mixture
(mine) — each ordered by increasing sftlr. Only sftlr >= 3e-5 is plotted.
Three bars per group for the `balanced` (n=3,4), `balanced5` (n=5), `balanced6`
(n=6) test sets. Same encoding as plot_accuracy_bars_sft_stacked.py:
solid = pass@1, striped top = pass@32 - pass@1, full height = pass@32. Scores
are read directly from the eval JSONs (no .scores/.passk cache; see that
script's docstring for why).

Candidates + checkpoint paths are catalogued in sft-checkpoints/270M_candidates.md.
Only runs whose final-checkpoint eval JSONs exist are plotted.

Usage: python3 scripts/plot/plot_accuracy_bars_270m_candidates.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_accuracy_bars_sft_stacked import scores_for_run, DATASETS, DS_COLORS

# Shlok's gemma distill/progdistill SFT eval results.
_SHLOK = "results_shlok/gemma-3-270m"
# My gemma progdistill_mixture SFT eval results (local results/ tree).
_MINE = "results/gemma-3-270m"


def _distill(lr):
    return ("distill", lr,
            f"{_SHLOK}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr{lr}-seed1/sft_final")


def _progdistill(lr):
    return ("progdistill", lr,
            f"{_SHLOK}/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr{lr}-seed1/round_1600")


def _mixture(lr):
    return ("PD mixture", lr,
            f"{_MINE}/balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr{lr}-steps1600-seed1/sft_final")


# (arm, sftlr label, dir holding {ds}_temp0.6_n32_max1024.json), ordered by lr
# within each arm. Only sftlr >= 3e-5. distill 1e-3 and all PD-mixture runs are
# mine (freshly evaluated); the rest are Shlok's. progdistill 2e-4 is omitted
# (no eval on disk); see sft-checkpoints/270M_candidates.md.
RUNS = [
    _distill("3e-5"), _distill("5e-5"), _distill("1e-4"), _distill("2e-4"),
    _distill("3e-4"), _distill("1e-3"),
    _progdistill("3e-5"), _progdistill("5e-5"), _progdistill("1e-4"),
    _progdistill("3e-4"),
    _mixture("1e-4"), _mixture("3e-4"), _mixture("1e-3"),
]
# distill 1e-3 is my run, so it lives in the local results/ tree, not Shlok's.
RUNS[5] = ("distill", "1e-3",
           f"{_MINE}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-3-seed1/sft_final")
FIGURES_DIR = "figures/gemma-3-270m"
plt.rcParams.update({"font.size": 14})


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    labels = [f"{arm}\n{lr}" for arm, lr, _ in RUNS]
    data = [scores_for_run(d) for _, _, d in RUNS]

    fig, ax = plt.subplots(figsize=(1.35 * len(RUNS) + 2, 5.5))
    n_ds = len(DATASETS)
    width = 0.8 / n_ds
    for j, ((ds, ds_label), color) in enumerate(zip(DATASETS, DS_COLORS)):
        for i in range(len(RUNS)):
            cell = data[i][ds]
            x = i + (j - (n_ds - 1) / 2) * width
            if cell is None or cell[0] is None or cell[1] is None:
                continue
            p1, p32 = cell
            gap = max(p32 - p1, 0.0)
            ax.bar(x, p1, width * 0.92, color=color,
                   label=ds_label if i == 0 else None)
            ax.bar(x, gap, width * 0.92, bottom=p1, color=color,
                   hatch="////", edgecolor="white", linewidth=0.0)
            ax.text(x, p32 + 0.012, f"{p32:.2f}", ha="center", va="bottom",
                    fontsize=9)

    # dashed separators between arms (at each arm boundary)
    for i in range(1, len(RUNS)):
        if RUNS[i][0] != RUNS[i - 1][0]:
            ax.axvline(i - 0.5, color="0.6", linewidth=0.8, linestyle="--")

    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("SFT run (arm / sftlr)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="test set", fontsize=13, title_fontsize=13, frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "accuracy_bars_sft_candidates.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for (arm, lr, _), cells in zip(RUNS, data):
        row = "  ".join(
            f"{ds}:{'--' if cells[ds] is None else f'{cells[ds][0]:.3f}/{cells[ds][1]:.3f}'}"
            for ds, _ in DATASETS)
        print(f"    {arm:12s} {lr:5s} {row}")


if __name__ == "__main__":
    main()
