"""Stacked accuracy bar plots (one figure per model family).

Each bar stacks two segments for a (run, dataset) cell:
  - solid       = pass@1  (accuracy of the first sample)
  - striped top = pass@32 - pass@1  (the extra prompts solved by >=1 of 32 samples)
so the full bar height is pass@32 (>= pass@1 by construction). pass@1/pass@32
come from plot_results._score_checkpoint + plot_passk.pass_at_k (shared .scores /
.passk caches, n=5,6 capped at 1000 prompts), so numbers match every repo table.

Three bars per run: `balanced` (n=3,4), `balanced5` (n=5), `balanced6` (n=6).

Usage: python3 scripts/plot/plot_accuracy_bars.py
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

TEACHER = ("results/Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1")
# Local copies of sdholakia's eval results (full-set re-scored). See
# results_sdholakia/ (copied from sdholakia's results + repo-artifacts trees).
SD = "results_sdholakia"
RA = "results_sdholakia/gemma-3-270m"
MINE = ("results/gemma-3-270m/"
        "balanced-progdistill-mixture-t1p5b-sftlr1e-4-steps1600-grpo-lr3e-6-kl3e-4-seed1")

# Each figure: (figures_dir, title, [(run_label, results_dir, step), ...]).
# Run label may contain \n for a two-line x-tick.
FIGURES = [
    ("figures/gemma-3-270m", "Gemma-270M students (vs Qwen2.5-1.5B teacher)", [
        ("Qwen2.5-1.5B\nteacher",       TEACHER, 1600),
        ("distill\n+GRPO",              f"{RA}/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1", 1600),
        ("progdistill\n+GRPO",          f"{RA}/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1", 1600),
        ("PD-mixture\n+GRPO (step 900)", MINE, 900),
    ]),
    # Qwen-0.5B: bars reach the top-right corner, so push the legend outside
    # the axes (loc upper-left anchored just past the right spine).
    ("figures/Qwen2.5-0.5B", "Qwen2.5-0.5B students (vs Qwen2.5-1.5B teacher)", [
        ("Qwen2.5-1.5B\nteacher",      TEACHER, 1600),
        ("Qwen2.5-0.5B\nGRPO",         "results/Qwen2.5-0.5B/balanced-grpo-seed1", 1600),
        ("distill\n+GRPO",             f"{SD}/Qwen2.5-0.5B/balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1", 1600),
        ("PD-mixture\n+GRPO",          "results/Qwen2.5-0.5B/balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl3e-3-seed1", 1600),
    ], {"loc": "upper left", "bbox_to_anchor": (1.0, 1.0)}),
    # 40M wide + deep merged into one figure: teacher + {wide,deep} x {distill, PD-mixture},
    # plus offline OPD on the same wide-distill SFT student as the wide distill+GRPO bar
    # (so those two bars isolate GRPO vs OPD from an identical starting point).
    ("figures/Qwen2.5-40M", "Qwen2.5-40M students (vs Qwen2.5-1.5B teacher)", [
        ("Qwen2.5-1.5B\nteacher",   TEACHER, 1600),
        ("40M-wide\ndistill+GRPO",  "results/Qwen2.5-40M-wide/balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1", 1600),
        # OPD ran 6250 steps (10 epochs); 6250 is its final checkpoint, so the
        # label needs no step annotation (unlike the step-900 gemma bar above).
        ("40M-wide\ndistill+OPD",   "results/Qwen2.5-40M-wide/balanced-opd-from-distill-t1p5b-sftlr3e-4-opdlr5e-5-ep10-seed1", 6250),
        ("40M-wide\nPD-mix+GRPO",   "results/Qwen2.5-40M-wide/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1", 1600),
        ("40M-deep\ndistill+GRPO",  "results/Qwen2.5-40M-deep/balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1", 1600),
        ("40M-deep\nPD-mix+GRPO",   "results/Qwen2.5-40M-deep/balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1", 1600),
    ]),
]
DATASETS = [("balanced", "n=3,4"), ("balanced5", "n=5"), ("balanced6", "n=6")]
# ordinal single-hue blue ramp: datasets ordered by difficulty
DS_COLORS = ["#86b6ef", "#3987e5", "#1c5cab"]
plt.rcParams.update({"font.size": 16})


def scores_for_run(run_dir, step):
    """Return {ds: (pass1, pass32)} for one run, None where the JSON is missing."""
    out = {}
    for ds, _ in DATASETS:
        path = os.path.join(run_dir, f"global_step_{step}",
                            f"{ds}_temp0.6_n32_max1024.json")
        if not os.path.isfile(path):
            out[ds] = None
            continue
        res = _score_checkpoint(path)
        n, counts = correct_counts(path)
        p1 = res[1] if res else None
        p32 = pass_at_k(n, counts, 32) if counts else None
        out[ds] = (p1, p32)
    return out


def plot_figure(figures_dir, title, runs, legend_kwargs=None):
    os.makedirs(figures_dir, exist_ok=True)
    labels = [lbl for lbl, _, _ in runs]
    data = [scores_for_run(d, s) for _, d, s in runs]

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

    # Dataset-color legend only (the solid/striped pass@1 vs pass@32-pass@1
    # meaning is described in the caption instead, to avoid blocking bars).
    lkw = {"loc": "upper right", "bbox_to_anchor": (1.0, 1.0)}
    if legend_kwargs:
        lkw.update(legend_kwargs)
    ax.legend(title="test set", fontsize=16, title_fontsize=16,
              frameon=False, **lkw)

    fig.tight_layout()
    out = os.path.join(figures_dir, "accuracy_bars_stacked.png")
    # bbox_inches='tight' so an outside-the-axes legend isn't clipped.
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for lbl, cells in zip(labels, data):
        row = "  ".join(
            f"{ds}:{'--' if cells[ds] is None else f'{cells[ds][0]:.3f}/{cells[ds][1]:.3f}'}"
            for ds, _ in DATASETS)
        print(f"    {lbl.replace(chr(10),' '):28s} {row}")


def main():
    # Optional argv filters: substrings matched against the figures_dir, so
    # `plot_accuracy_bars.py Qwen2.5-40M` refreshes only that figure instead of
    # rewriting all three (avoids touching figures whose data didn't change).
    only = sys.argv[1:]
    for entry in FIGURES:
        figures_dir, title, runs = entry[0], entry[1], entry[2]
        if only and not any(pat in figures_dir for pat in only):
            continue
        legend_kwargs = entry[3] if len(entry) > 3 else None
        print(f"\n=== {title} ===")
        plot_figure(figures_dir, title, runs, legend_kwargs)


if __name__ == "__main__":
    main()
