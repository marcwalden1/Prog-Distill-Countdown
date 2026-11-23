#!/usr/bin/env python3
"""
ID-vs-OOD comparison for the from-scratch ~40M students (wide / deep).

For a given model, plot in-distribution accuracy (balanced, n=3,4) on x against
OOD accuracy (balanced5, n=5) on y, one dot per evaluated checkpoint connected
in step order. The two GRPO arms — KD+GRPO (distill) and PD-mixture+GRPO
(progdistill-mixture), sftlr=3e-4, grpo lr=1e-6 kl=3e-3 — share one panel.

Reads the per-run `.scores` caches ([step, mean@1, mean@32]) written by
plot_results.py. Modeled on scripts/plot/plot_id_vs_ood_comparison.py.

Metric: mean@1 by default.

Output: figures/<model>/id_vs_ood.png
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# shared semantic color map lives at repo root (colors.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from colors import COLORS

ID_DATASET = "balanced"      # n=3,4 in-distribution
OOD_DATASET = "balanced5"    # n=5 OOD

# the two GRPO arms per 40M model (sftlr=3e-4, grpo lr=1e-6 kl=3e-3, seed1)
ARMS = [
    ("KD + GRPO",
     "balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1",
     COLORS["student KD + GRPO"]),
    ("PD mixture + GRPO",
     "balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1",
     COLORS["student PD mixture + GRPO"]),
]


def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    return int(m.group(1)) if m else None


def load_from_scores(result_dir, model, exp, dataset, metric, temp="0.6"):
    """{step: value} from the .scores caches ([step, mean@1, mean@32])."""
    idx = {"mean@1": 1, "mean@32": 2}[metric]
    out = {}
    pattern = os.path.join(result_dir, model, exp, "global_step_*",
                           f"{dataset}_temp{temp}_*.json.scores")
    for sp in glob.glob(pattern):
        step = _step_from_path(sp)
        if step is None:
            continue
        try:
            with open(sp) as f:
                rec = json.load(f)
            out[step] = float(rec[idx])
        except Exception:
            continue
    return out


def plot_model(result_dir, model, metric, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    n_plotted = 0
    for label, exp, color in ARMS:
        id_by = load_from_scores(result_dir, model, exp, ID_DATASET, metric)
        ood_by = load_from_scores(result_dir, model, exp, OOD_DATASET, metric)
        steps = sorted(set(id_by) & set(ood_by))
        if not steps:
            print(f"  [skip] {model} {label}: no step has both "
                  f"{ID_DATASET} and {OOD_DATASET} (id={len(id_by)}, ood={len(ood_by)})")
            continue
        xs = [id_by[s] for s in steps]
        ys = [ood_by[s] for s in steps]
        ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=1.8, zorder=1)
        ax.scatter(xs, ys, s=30, color=color, zorder=2, label=label)
        # start = open circle, end = star
        ax.scatter([xs[0]], [ys[0]], s=95, facecolors="none",
                   edgecolors=color, linewidths=1.6, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=150, marker="*", color=color,
                   edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
        for s in (steps[0], steps[-1]):
            ax.annotate(str(s), (id_by[s], ood_by[s]), fontsize=6,
                        color="#52514e", xytext=(3, 3), textcoords="offset points")
        n_plotted += 1

    ax.set_xlabel(f"In-dist accuracy ({ID_DATASET}, n=3,4) — {metric}", color="#0b0b0b")
    ax.set_ylabel(f"OOD accuracy ({OOD_DATASET}, n=5) — {metric}", color="#0b0b0b")
    ax.set_title(f"{model}   ID vs OOD across checkpoints\n"
                 f"sftlr=3e-4, grpo_lr=1e-6, kl=3e-3, seed1   "
                 f"○ = first step   ★ = last step   (temp=0.6)",
                 fontsize=9)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors="#898781")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path} ({n_plotted} arms)")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mean@1", choices=["mean@1", "mean@32"])
    ap.add_argument("--result-dir", default=os.path.join(repo, "results"))
    ap.add_argument("--figures-dir", default=os.path.join(repo, "figures"))
    ap.add_argument("--models", nargs="+",
                    default=["Qwen2.5-40M-wide", "Qwen2.5-40M-deep"])
    args = ap.parse_args()

    for model in args.models:
        out = os.path.join(args.figures_dir, model, "id_vs_ood.png")
        plot_model(args.result_dir, model, args.metric, out)


if __name__ == "__main__":
    main()
