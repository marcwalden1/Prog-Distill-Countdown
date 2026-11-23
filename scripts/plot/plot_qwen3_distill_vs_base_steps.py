#!/usr/bin/env python3
"""
Qwen3-0.6B GRPO from the base model vs from the 4B-distilled (NTP SFT, lr=1e-5)
student, both GRPO lr=3e-6 kl=3e-3 n=4: pass@1 and pass@32 vs training step on
n=3,4 (balanced-simple). One panel per metric.

Reads the same unbiased pass@k CSVs as plot_qwen3_size_comparison.py.

    python3 scripts/plot/plot_qwen3_distill_vs_base_steps.py [--dataset balanced-simple]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_qwen3_size_comparison import (  # noqa: E402
    INK, INK_MUTED, PANELS, REPO, SURFACE, load, plt, series, style_axes)

RUNS = [
    ("Qwen3-0.6B-Base", "balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
     "0.6B base  lr=3e-6", "#2a78d6", "o", "-"),
    ("Qwen3-0.6B-Base",
     "balanced-simple-grpo-from-distill-ntp-t4b-base-sftlr1e-5-lr3e-6-kl3e-3-seed1",
     "0.6B distilled from 4B  lr=3e-6", "#c2185b", "D", "-."),
    ("Qwen3-1.7B-Base", "balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
     "1.7B base  lr=1e-6", "#eb6834", "s", "--"),
    ("Qwen3-4B-Base", "balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
     "4B base  lr=1e-6", "#4a3aa7", "^", ":"),
]
METRICS = [("pass_at_1", "pass@1"), ("pass_at_32", "pass@32")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",
                    default=[os.path.join(REPO, "analysis_data", f)
                             for f in ("idood_qwen3_base_grpo.csv",
                                       "idood_qwen3_distill_grpo.csv")])
    ap.add_argument("--dataset", default="balanced-simple")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "figures", "Qwen3-distill-grpo"))
    args = ap.parse_args()

    df = load(args.csv)
    ds_title = dict(PANELS).get(args.dataset, args.dataset)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, (col, name) in zip(axes, METRICS):
        ends = []
        for model, exp, label, color, marker, ls in RUNS:
            s = series(df, model, exp, args.dataset, (col, col))
            if s is None:
                print(f"no {args.dataset} rows for {exp}")
                continue
            ys, _, steps = s
            ax.plot(steps, ys, ls, color=color, linewidth=2, zorder=2)
            ax.scatter(steps, ys, s=50, marker=marker, color=color,
                       edgecolors=SURFACE, linewidths=1.2, zorder=3, label=label)
            ends.append((ys[-1], steps[-1]))
        # final-value labels, nudged apart so close finishes don't overlap
        y0, y1 = ax.get_ylim()
        gap = 0.045 * (y1 - y0)
        placed = []
        for y, st in sorted(ends):
            ly = max(y, placed[-1] + gap) if placed else y
            placed.append(ly)
            ax.annotate(f"{y:.3f}", (st, y), xytext=(st + 35, ly), fontsize=8.5,
                        color=INK_MUTED, va="center", zorder=4,
                        textcoords="data", annotation_clip=False)
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + (x1 - x0) * 0.08)
        style_axes(ax, name, "GRPO training step", name)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Qwen3 GRPO (kl=3e-3, n=4) on {ds_title}: "
                 "0.6B base vs 0.6B distilled from 4B, with 1.7B / 4B base", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"pass_vs_step_distill_vs_sizes_{args.dataset}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
