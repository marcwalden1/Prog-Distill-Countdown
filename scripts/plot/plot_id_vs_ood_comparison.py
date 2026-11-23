#!/usr/bin/env python3
"""
Data note (2026-07-14): the default CSV (analysis_data/qwen05b_fig7_pareto_fullset.csv)
is rebuilt by scripts/rebuild_qwen05b_fig7_csv_fullset.py with FULL-set mean@1/mean@32
(cap removed). It covers only the 6 runs whose full eval JSONs are recoverable
(downloaded from HF marcwalden/rl-skill-comp-evals). The 4 `balanced-grpo-from-*`
runs in the original CSV have no eval data on this cluster or HF and are dropped.
See research_logs/2026-07-13_fix-maxprompts-cap-fullset-regen.md.

ID-vs-OOD comparison across hyperparameter setups, one plot per method.

For every experiment of a given method (distill / progdistill), plot
in-distribution accuracy (balanced, n=3,4) on x against OOD accuracy
(balanced5, n=5) on y, one dot per evaluated checkpoint connected in step
order. All hyperparameter setups for the method share one panel, coloured by
(sft_lr, kl). Reads the tidy long-format CSV `pareto_eval_progression.csv`
(cols: method, sft_lr, grpo_lr, grpo_kl, exp_name, dataset, step, mean_at_1,
mean_at_32) rather than the per-run `.scores` caches.

Metric: mean@1 by default (matches the "merge into the mean@1 curve" request).

Output: figures/Qwen2.5-0.5B/comparison/id_vs_ood_<method>.png
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
import pandas as pd

# shared semantic color map lives at repo root (colors.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from colors import COLORS

ID_DATASET = "balanced"      # n=3,4 in-distribution
OOD_DATASET = "balanced5"    # n=5 OOD

DEFAULT_CSV = "analysis_data/qwen05b_fig7_pareto_fullset.csv"

# distinct, colour-blind-friendly-ish palette (enough for 5 runs/panel)
PALETTE = ["#2a78d6", "#1baf7a", "#d6772a", "#9b59b6", "#c0392b",
           "#16a085", "#e67e22", "#34495e"]


def run_label(sft_lr, kl):
    return f"sft={sft_lr}, kl={kl}"


# ---- loaders returning {step: mean-metric} for a given (dataset, metric) ----

def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"round_(\d+)", path)
    return int(m.group(1)) if m else None


def load_from_scores(result_dir, model, exp, dataset, metric, temp="0.6"):
    """{step: value} from the .scores caches ([step, mean@1, mean@32])."""
    idx = {"mean@1": 1, "mean@32": 2}[metric]
    out = {}
    for ckpt_glob in ("global_step_*", "round_*"):
        pattern = os.path.join(result_dir, model, exp, ckpt_glob,
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


def load_from_csv(df, exp, dataset, metric):
    """{step: value} for one (exp, dataset) from the tidy CSV."""
    col = {"mean@1": "mean_at_1", "mean@32": "mean_at_32"}[metric]
    e = df[(df.exp_name == exp) & (df.dataset == dataset)]
    return {int(s): float(v) for s, v in zip(e.step, e[col])}


def plot_curves(curves, metric, out_path, title):
    """curves: list of dicts {label, id_by_step, ood_by_step}."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for i, c in enumerate(curves):
        id_by, ood_by = c["id_by_step"], c["ood_by_step"]
        steps = sorted(set(id_by) & set(ood_by))
        if not steps:
            print(f"  [skip] {c['label']}: no step has both datasets")
            continue
        xs = [id_by[s] for s in steps]
        ys = [ood_by[s] for s in steps]
        color = c.get("color") or PALETTE[i % len(PALETTE)]
        linestyle = c.get("linestyle", "-")
        ax.plot(xs, ys, linestyle=linestyle, color=color, alpha=0.8,
                linewidth=1.8, zorder=1, label=c["label"])
        ax.scatter(xs, ys, s=30, color=color, zorder=2)
        ax.scatter([xs[0]], [ys[0]], s=95, facecolors="none",
                   edgecolors=color, linewidths=1.6, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=150, marker="*", color=color,
                   edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
        for s in (steps[0], steps[-1]):
            ax.annotate(str(s), (id_by[s], ood_by[s]), fontsize=6,
                        color="#52514e", xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"In-dist accuracy ({ID_DATASET}, n=3,4) — {metric}", color="#0b0b0b")
    ax.set_ylabel(f"OOD accuracy ({OOD_DATASET}, n=5) — {metric}", color="#0b0b0b")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors="#898781")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path} ({len(curves)} curves)")


def plot_method(df, method, metric, out_path):
    col = {"mean@1": "mean_at_1", "mean@32": "mean_at_32"}[metric]
    sub = df[df["method"] == method]
    exps = sorted(sub["exp_name"].unique(),
                  key=lambda e: (sub[sub.exp_name == e].sft_lr.iloc[0],
                                 sub[sub.exp_name == e].grpo_kl.iloc[0]))
    if not exps:
        print(f"  [skip] no runs for method={method}")
        return False

    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for i, exp in enumerate(exps):
        e = sub[sub.exp_name == exp]
        id_by = e[e.dataset == ID_DATASET].set_index("step")[col]
        ood_by = e[e.dataset == OOD_DATASET].set_index("step")[col]
        steps = sorted(set(id_by.index) & set(ood_by.index))
        if not steps:
            print(f"  [skip] {exp}: no step has both {ID_DATASET} and {OOD_DATASET}")
            continue
        xs = [id_by[s] for s in steps]
        ys = [ood_by[s] for s in steps]
        color = PALETTE[i % len(PALETTE)]
        sft_lr = e.sft_lr.iloc[0]
        kl = e.grpo_kl.iloc[0]
        ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=1.8, zorder=1)
        ax.scatter(xs, ys, s=30, color=color, zorder=2,
                   label=run_label(sft_lr, kl))
        # start = open circle, end = star
        ax.scatter([xs[0]], [ys[0]], s=95, facecolors="none",
                   edgecolors=color, linewidths=1.6, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=150, marker="*", color=color,
                   edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
        # annotate first/last step only, to avoid clutter across 5 runs
        for s in (steps[0], steps[-1]):
            ax.annotate(str(s), (id_by[s], ood_by[s]), fontsize=6,
                        color="#52514e", xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"In-dist accuracy ({ID_DATASET}, n=3,4) — {metric}", color="#0b0b0b")
    ax.set_ylabel(f"OOD accuracy ({OOD_DATASET}, n=5) — {metric}", color="#0b0b0b")
    ax.set_title(f"Qwen2.5-0.5B  {method}→GRPO   ID vs OOD across checkpoints\n"
                 f"grpo_lr=1e-6, seed1   ○ = first step   ★ = last step   (temp=0.6)",
                 fontsize=9)
    ax.legend(loc="best", fontsize=8, title="hyperparameters")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors="#898781")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    n_runs = sub["exp_name"].nunique()
    print(f"  wrote {out_path} ({n_runs} runs)")
    return True


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--metric", default="mean@1", choices=["mean@1", "mean@32"])
    ap.add_argument("--out-dir",
                    default=os.path.join(repo, "figures", "Qwen2.5-0.5B", "comparison"))
    ap.add_argument("--result-dir", default=os.path.join(repo, "results"))
    ap.add_argument("--model", default="Qwen2.5-0.5B")
    ap.add_argument("--selected", action="store_true",
                    help="plot the 4 hand-picked curves in one panel instead of "
                         "the per-method comparison figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"sft_lr": str, "grpo_lr": str, "grpo_kl": str})

    if args.selected:
        # (label, source, key, color, linestyle) — source 'csv' -> exp_name in
        # CSV; 'scores' -> exp dir under results/. Colors from repo-root
        # colors.py: distill=KD, progdistill=PD (both share darkorange,
        # distinguished by solid (sft=1e-5) vs dashed (sft=3e-5)), mixture=PD mixture.
        SELECTED = [
            ("distill sft=3e-5, kl=3e-4", "csv",
             "balanced-distill-cross-shlokbest-n4-sft3e-5-frozen-pre-grpo-kl3e-4-lr1e-6-seed1",
             COLORS["student KD + GRPO"], "-"),
            ("progdistill sft=1e-5, kl=3e-4 (solid)", "csv",
             "balanced-progdistill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1",
             COLORS["student PD + GRPO"], "-"),
            ("progdistill sft=3e-5, kl=3e-4 (dashed)", "csv",
             "balanced-progdistill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1",
             COLORS["student PD + GRPO"], "--"),
            ("progdistill_mixture sft=3e-5, kl=3e-3", "scores",
             "balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl3e-3-seed1",
             COLORS["student PD mixture + GRPO"], "-"),
        ]
        curves = []
        for label, src, key, color, linestyle in SELECTED:
            if src == "csv":
                id_by = load_from_csv(df, key, ID_DATASET, args.metric)
                ood_by = load_from_csv(df, key, OOD_DATASET, args.metric)
            else:
                id_by = load_from_scores(args.result_dir, args.model, key, ID_DATASET, args.metric)
                ood_by = load_from_scores(args.result_dir, args.model, key, OOD_DATASET, args.metric)
            if not id_by or not ood_by:
                print(f"  [warn] {label}: missing data (id={len(id_by)}, ood={len(ood_by)})")
            curves.append({"label": label, "id_by_step": id_by, "ood_by_step": ood_by,
                           "color": color, "linestyle": linestyle})
        suffix = "" if args.metric == "mean@1" else f"_{args.metric.replace('@', '')}"
        out = os.path.join(args.out_dir, f"id_vs_ood_combined{suffix}.png")
        plot_curves(curves, args.metric, out, title="")
        return

    for method in ("distill", "progdistill"):
        out = os.path.join(args.out_dir, f"id_vs_ood_{method}.png")
        plot_method(df, method, args.metric, out)


if __name__ == "__main__":
    main()
