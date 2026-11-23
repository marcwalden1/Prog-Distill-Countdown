#!/usr/bin/env python3
"""
Per-experiment ID-vs-OOD trajectory plot: for every experiment that has a
directory under figures/<model>/, plot in-distribution accuracy (balanced,
n=3,4) on x against OOD accuracy (balanced5, n=5) on y. One dot per evaluated
checkpoint, connected in checkpoint order. Output: figures/<model>/<exp>/ID_vs_OOD.png

Reads only the tiny `.scores` caches ([step, mean@1, mean@32]) next to each
eval JSON, so it is safe to run locally (like scripts/plot/plot_pareto_2d.py).

Usage:
    python3 scripts/plot/plot_id_vs_ood.py [--model Qwen2.5-0.5B] [--exp EXP ...]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_DATASET = "balanced"     # n=3,4 in-distribution
OOD_DATASET = "balanced5"   # n=5 OOD

# metric -> (index into the .scores record, color)
METRICS = {"mean@1": (1, "#2a78d6"), "mean@32": (2, "#1baf7a")}


def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    if os.sep + "sft_final" + os.sep in path:
        return 1600
    m = re.search(r"round_(\d+)", path)
    return int(m.group(1)) if m else None


def load_scores_by_step(result_dir, model, exp, dataset, temp):
    """{step: (mean@1, mean@32)} from the .scores caches for one (exp, dataset)."""
    out = {}
    for ckpt_glob in ("global_step_*", "sft_final", "round_*"):
        pattern = os.path.join(result_dir, model, exp, ckpt_glob,
                               f"{dataset}_temp{temp}_*.json.scores")
        for sp in glob.glob(pattern):
            step = _step_from_path(sp)
            if step is None:
                continue
            try:
                with open(sp) as f:
                    rec = json.load(f)  # [step, mean@1, mean@32]
                out[step] = (float(rec[1]), float(rec[2]))
            except Exception:
                continue
    return out


def plot_exp(exp, id_by_step, ood_by_step, out_path, temp):
    steps = sorted(set(id_by_step) & set(ood_by_step))
    if not steps:
        print(f"  [skip] {exp}: no checkpoint has both {ID_DATASET} and {OOD_DATASET} scores")
        return False

    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for metric, (idx, color) in METRICS.items():
        xs = [id_by_step[s][idx - 1] for s in steps]
        ys = [ood_by_step[s][idx - 1] for s in steps]
        ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=2, zorder=1)
        ax.scatter(xs, ys, s=42, color=color, label=metric, zorder=2)
        # start = open circle, end = star
        ax.scatter([xs[0]], [ys[0]], s=110, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=170, marker="*", color=color,
                   edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
        for s, x, y in zip(steps, xs, ys):
            ax.annotate(str(s), (x, y), fontsize=6.5, color="#52514e",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"In-dist accuracy ({ID_DATASET}, n=3,4)", color="#0b0b0b")
    ax.set_ylabel(f"OOD accuracy ({OOD_DATASET}, n=5)", color="#0b0b0b")
    n = len(steps)
    subtitle = (f"single checkpoint   (temp={temp})" if n == 1
                else f"{n} checkpoints   ○ = step {steps[0]}   ★ = step {steps[-1]}   (temp={temp})")
    ax.set_title(f"{exp}\n{subtitle}", fontsize=9)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors="#898781")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path} ({n} checkpoints)")
    return True


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-0.5B")
    ap.add_argument("--result-dir", default=os.path.join(repo, "results"))
    ap.add_argument("--figures-dir", default=os.path.join(repo, "figures"))
    ap.add_argument("--temp", default="0.6",
                    help="eval sampling temperature to read scores for")
    ap.add_argument("--exp", nargs="*", default=None,
                    help="default: every subdirectory of figures/<model>/ that "
                         "looks like an experiment (skips 'pareto' etc. without scores)")
    args = ap.parse_args()

    model_fig_dir = os.path.join(args.figures_dir, args.model)
    exps = args.exp or sorted(
        d for d in os.listdir(model_fig_dir)
        if os.path.isdir(os.path.join(model_fig_dir, d)))

    for exp in exps:
        id_scores = load_scores_by_step(args.result_dir, args.model, exp, ID_DATASET, args.temp)
        ood_scores = load_scores_by_step(args.result_dir, args.model, exp, OOD_DATASET, args.temp)
        out_path = os.path.join(model_fig_dir, exp, "ID_vs_OOD.png")
        plot_exp(exp, id_scores, ood_scores, out_path, args.temp)


if __name__ == "__main__":
    main()
