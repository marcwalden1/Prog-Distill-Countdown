"""Gemma-270M ID-vs-OOD trajectories for the three kl=3e-4 runs.

Overlays distill+GRPO, progdistill+GRPO, and progdistill-mixture+GRPO (all GRPO
kl=3e-4) as accuracy trajectories: in-distribution (balanced, n=3,4) on x vs OOD
(n=5 or n=6) on y, one marker per evaluated checkpoint, connected in step order
(open circle = first step, star = last). The progdistill-mixture run collapsed
after ~step 900, so it is capped at step 900.

2x2 grid: rows = metric (pass@1, pass@32), cols = OOD set (n=5, n=6). Metric and
dataset are encoded in the axis labels, so there is no figure/plot title.
pass@1/pass@32 come from plot_results._score_checkpoint + plot_passk.pass_at_k
(shared caches; n=5,6 capped at 1000 prompts), matching every other repo table.

Usage: python3 scripts/plot/plot_id_vs_ood_gemma_kl3e4.py
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
MINE = ("results/gemma-3-270m/"
        "balanced-progdistill-mixture-t1p5b-sftlr1e-4-steps1600-grpo-lr3e-6-kl3e-4-seed1")

STEPS = [150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600]
# (label, results_dir, color, max_step)
RUNS = [
    ("distill+GRPO",
     f"{RA}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1",
     "#1c5cab", 1600),
    ("progdistill+GRPO",
     f"{RA}/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1",
     "#b45309", 1600),
    ("progdistill-mixture+GRPO",
     MINE, "#15803d", 900),   # collapses after ~900
]
# OOD columns: (dataset, axis label)
OOD = [("balanced5", "n=5"), ("balanced6", "n=6")]
METRICS = [("pass@1", 0), ("pass@32", 1)]  # (name, tuple index into (pass1,pass32))
FIGURES_DIR = "figures/gemma-3-270m"
plt.rcParams.update({"font.size": 13})


def scores(run_dir, step, ds):
    path = os.path.join(run_dir, f"global_step_{step}",
                        f"{ds}_temp0.6_n32_max1024.json")
    if not os.path.isfile(path):
        return None
    res = _score_checkpoint(path)
    n, counts = correct_counts(path)
    if res is None or not counts:
        return None
    return (res[1], pass_at_k(n, counts, 32))


def collect(run_dir, max_step):
    """{ds: {step: (pass1, pass32)}} for the ID set + both OOD sets."""
    wanted = ["balanced"] + [ds for ds, _ in OOD]
    out = {ds: {} for ds in wanted}
    for step in STEPS:
        if step > max_step:
            continue
        for ds in wanted:
            s = scores(run_dir, step, ds)
            if s is not None:
                out[ds][step] = s
    return out


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    data = {lbl: collect(d, ms) for lbl, d, _, ms in RUNS}

    fig, axes = plt.subplots(len(METRICS), len(OOD), figsize=(13, 11))
    for r, (metric_name, mi) in enumerate(METRICS):
        for c, (ood_ds, ood_label) in enumerate(OOD):
            ax = axes[r][c]
            for lbl, _d, color, _ms in RUNS:
                id_by, ood_by = data[lbl]["balanced"], data[lbl][ood_ds]
                steps = sorted(set(id_by) & set(ood_by))
                if not steps:
                    continue
                xs = [id_by[s][mi] for s in steps]
                ys = [ood_by[s][mi] for s in steps]
                ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=2,
                        zorder=1)
                ax.scatter(xs, ys, s=40, color=color, label=lbl, zorder=2)
                ax.scatter([xs[0]], [ys[0]], s=120, facecolors="none",
                           edgecolors=color, linewidths=1.8, zorder=3)
                ax.scatter([xs[-1]], [ys[-1]], s=190, marker="*", color=color,
                           edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
                for s, x, y in zip(steps, xs, ys):
                    ax.annotate(str(s), (x, y), fontsize=6.5, color="#52514e",
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(f"ID {metric_name} (n=3,4)")
            ax.set_ylabel(f"OOD {metric_name} ({ood_label})")
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    # single shared legend (○ = first step, ★ = last step)
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(handles, labels, fontsize=11, frameon=False,
                      loc="best", title="○ first step   ★ last step",
                      title_fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "id_vs_ood_kl3e-4.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    for lbl, _d, _c, ms in RUNS:
        got = sorted(set(data[lbl]["balanced"]))
        print(f"  {lbl:26s} (cap {ms}): checkpoints {got}")


if __name__ == "__main__":
    main()
