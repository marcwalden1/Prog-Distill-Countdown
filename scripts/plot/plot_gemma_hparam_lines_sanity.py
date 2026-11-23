"""DEPRECATED (2026-07-13): reads a precomputed CSV built from the OLD capped
(first-1000-prompt) .scores, so its n=5/n=6 numbers are biased and cannot be
regenerated here. Output PNG kept as *_deprecated.png. Use the JSON-based
plot_gemma_hparam_lines.py (full-set re-scored) instead.

Sanity version of scripts/plot/plot_gemma_hparam_lines.py.

Same 2x3 layout (mean@1 / mean@32 rows x n=3,4 / n=5 / n=6 cols), one line per
run (color = method, linestyle = KL), but reads the precomputed metrics straight
from the analysis CSV instead of re-scoring the eval JSONs. Use it to sanity-check
that the CSV progression matches the JSON-derived hparam_lines plot.

Note: the CSV stores mean@1 / mean@32 (raw compute_score averages), whereas the
original plot computes pass@1 / pass@32 (Chen et al. estimator) -- expect the same
shapes/orderings, not identical y-values.

Usage: python3 scripts/plot/plot_gemma_hparam_lines_sanity.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = ("/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/repo-artifacts/"
       "RL-skill-comp/analysis_data/gemma270m_pareto_eval_progression.csv")

# (label, method, kl, color, linestyle) -- matches the original script.
RUNS = [
    ("distill kl3e-3",     "distill",     "3e-3", "#1c5cab", "-"),
    ("distill kl3e-4",     "distill",     "3e-4", "#86b6ef", "--"),
    ("progdistill kl3e-3", "progdistill", "3e-3", "#b45309", "-"),
    ("progdistill kl3e-4", "progdistill", "3e-4", "#f0a04b", "--"),
]
DATASETS = [("balanced", "n=3,4 (in-dist)"), ("balanced5", "n=5 (OOD)"),
            ("balanced6", "n=6 (OOD)")]
FIGURES_DIR = "figures/gemma-3-270"
plt.rcParams.update({"font.size": 12})


def load_rows():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def collect(rows, method, kl):
    """Return {ds: ([steps], [mean@1], [mean@32])} for a run, sorted by step."""
    out = {}
    for ds, _ in DATASETS:
        pts = [(int(r["step"]), float(r["mean_at_1"]), float(r["mean_at_32"]))
               for r in rows
               if r["method"] == method and r["grpo_kl"] == kl
               and r["dataset"] == ds]
        pts.sort()
        xs = [p[0] for p in pts]
        m1 = [p[1] for p in pts]
        m32 = [p[2] for p in pts]
        out[ds] = (xs, m1, m32)
    return out


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = load_rows()
    data = {lbl: collect(rows, m, kl) for lbl, m, kl, _, _ in RUNS}

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16, 8), sharex=True)
    metrics = [("mean@1", 1), ("mean@32", 2)]
    for r, (metric_name, idx) in enumerate(metrics):
        for c, (ds, ds_label) in enumerate(DATASETS):
            ax = axes[r][c]
            for lbl, _m, _kl, color, ls in RUNS:
                xs, m1, m32 = data[lbl][ds]
                ys = m1 if idx == 1 else m32
                ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                        markersize=4, linewidth=1.8, label=lbl)
            ax.set_title(f"{metric_name} — {ds_label}")
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 1:
                ax.set_xlabel("GRPO step")
            if c == 0:
                ax.set_ylabel(metric_name)
    axes[0][-1].legend(fontsize=10, frameon=False)
    fig.suptitle("Gemma-270M (sanity, from CSV): distill vs progdistill x GRPO KL (seed 1)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGURES_DIR, "hparam_lines_distill_progdistill_sanity.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    print("\nfinal-step (1600) mean@1 / mean@32:")
    for lbl, m, kl, _, _ in RUNS:
        cells = []
        for ds, _ in DATASETS:
            xs, m1, m32 = data[lbl][ds]
            if xs and xs[-1] == 1600:
                cells.append(f"{ds}: {m1[-1]:.3f}/{m32[-1]:.3f}")
            else:
                cells.append(f"{ds}: n/a")
        print(f"  {lbl:22s}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
