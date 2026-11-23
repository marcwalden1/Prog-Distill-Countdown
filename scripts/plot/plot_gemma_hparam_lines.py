"""Line plot: the four Gemma-270M distill/progdistill x KL runs vs training step.

Purpose: pick the better GRPO KL for each method (distill, progdistill) before
building the final bar plot. Reads the raw 32-sample eval JSONs (repo-artifacts)
and computes pass@1 (= accuracy of first sample) and pass@32 (>=1 of 32 correct,
Chen et al. estimator) via the shared plot_results / plot_passk infra, so the
numbers match every other repo table. n=5,6 are capped at 1000 prompts (same as
plot_results). Caches land next to the JSONs (.scores / .passk).

Layout: 2 rows (pass@1, pass@32) x 3 cols (n=3,4 / n=5 / n=6). One line per run;
color = method, linestyle = KL.

Usage: python3 scripts/plot/plot_gemma_hparam_lines.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plot_results import _score_checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_passk import correct_counts, pass_at_k

# Local copy of sdholakia's gemma eval results (full-set re-scored) — see results_sdholakia/.
RA = "results_sdholakia/gemma-3-270m"

def run_dir(method, kl):
    return (f"{RA}/gemma-270m-balanced-{method}-teacher-Qwen1.5B-"
            f"sftlr1e-4-grpo-lr1e-6-kl{kl}-seed1")

# (label, method, kl, color, linestyle)
RUNS = [
    ("distill kl3e-3",     "distill",     "3e-3", "#1c5cab", "-"),
    ("distill kl3e-4",     "distill",     "3e-4", "#86b6ef", "--"),
    ("progdistill kl3e-3", "progdistill", "3e-3", "#b45309", "-"),
    ("progdistill kl3e-4", "progdistill", "3e-4", "#f0a04b", "--"),
]
DATASETS = [("balanced", "n=3,4 (in-dist)"), ("balanced5", "n=5 (OOD)"),
            ("balanced6", "n=6 (OOD)")]
STEPS = [150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600]
FIGURES_DIR = "figures/gemma-3-270m"
plt.rcParams.update({"font.size": 12})


def collect(method, kl):
    """Return {ds: ([steps], [pass1], [pass32])} for a run, skipping missing."""
    out = {}
    for ds, _ in DATASETS:
        xs, p1s, p32s = [], [], []
        for step in STEPS:
            path = (f"{run_dir(method, kl)}/global_step_{step}/"
                    f"{ds}_temp0.6_n32_max1024.json")
            if not os.path.isfile(path):
                continue
            res = _score_checkpoint(path)
            n, counts = correct_counts(path)
            if res is None or not counts:
                continue
            xs.append(step)
            p1s.append(res[1])
            p32s.append(pass_at_k(n, counts, 32))
        out[ds] = (xs, p1s, p32s)
    return out


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    data = {lbl: collect(m, kl) for lbl, m, kl, _, _ in RUNS}

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16, 8), sharex=True)
    metrics = [("pass@1", 1), ("pass@32", 2)]
    for r, (metric_name, idx) in enumerate(metrics):
        for c, (ds, ds_label) in enumerate(DATASETS):
            ax = axes[r][c]
            for lbl, _m, _kl, color, ls in RUNS:
                xs, p1s, p32s = data[lbl][ds]
                ys = p1s if idx == 1 else p32s
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
    fig.suptitle("Gemma-270M: distill vs progdistill x GRPO KL (seed 1)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGURES_DIR, "hparam_lines_distill_progdistill.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # Also print the final-step (1600) table for quick hyperparameter picking.
    print("\nfinal-step (1600) pass@1 / pass@32:")
    for lbl, m, kl, _, _ in RUNS:
        cells = []
        for ds, _ in DATASETS:
            xs, p1s, p32s = data[lbl][ds]
            if xs and xs[-1] == 1600:
                cells.append(f"{ds}: {p1s[-1]:.3f}/{p32s[-1]:.3f}")
            else:
                cells.append(f"{ds}: n/a")
        print(f"  {lbl:22s}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
