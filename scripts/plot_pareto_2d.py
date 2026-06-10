#!/usr/bin/env python3
"""
2D Pareto plots: in-distribution (n=3,4) mean@1 on x, OOD mean@1 on y, for two
GRPO-on-top arms (distill->GRPO vs progdistill->GRPO) across their intermediate
checkpoints. One figure per OOD dataset (balanced5 = n5, balanced6 = n6).

Reads ONLY the tiny `.scores` cache files ([step, mean@1, mean@32]) that
plot_results.py / upload_evals_to_wandb.py leave next to each result JSON, so it
is cheap and safe to run locally (it never loads the 240 MB rollout JSONs).

Usage:
    python3 scripts/plot_pareto_2d.py \
        --result-dir /n/netscratch/kdbrantley_lab/Lab/mwalden/rl-skill-comp-results/results \
        --model Qwen2.5-0.5B \
        --distill-exp     balanced-distill-cross-shlokbest-n4-sft3e-5-frozen-pre-grpo-kl3e-4-lr1e-6-seed1 \
        --progdistill-exp balanced-progdistill-sftlr1e-6-grpo-lr1e-6-kl3e-4-seed1
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_DATASET = "balanced"            # n=3,4 in-distribution
OOD_DATASETS = ["balanced5", "balanced6"]
OOD_LABEL = {"balanced5": "n=5 (OOD)", "balanced6": "n=6 (far OOD)"}


def _step_from_path(path):
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))
    if "/sft_final/" in path:
        return 1600
    m = re.search(r"round_(\d+)", path)
    return int(m.group(1)) if m else None


def load_mean1_by_step(result_dir, model, exp, dataset):
    """{step: mean@1} read straight from the .scores caches for one (exp, dataset)."""
    pattern = os.path.join(result_dir, model, exp, "global_step_*",
                           f"{dataset}_temp*.json.scores")
    out = {}
    for sp in glob.glob(pattern):
        step = _step_from_path(sp)
        if step is None:
            continue
        try:
            with open(sp) as f:
                rec = json.load(f)            # [step, mean@1, mean@32]
            out[step] = float(rec[1])
        except Exception:
            continue
    return out


def trajectory(id_by_step, ood_by_step):
    """Steps present in both ID and OOD, sorted; returns (steps, xs, ys)."""
    steps = sorted(set(id_by_step) & set(ood_by_step))
    xs = [id_by_step[s] for s in steps]
    ys = [ood_by_step[s] for s in steps]
    return steps, xs, ys


def plot_one(ood_dataset, arms, out_path, caveat):
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    colors = {"distill→GRPO": "#d62728", "progdistill→GRPO": "#1f77b4"}
    for name, (steps, xs, ys) in arms.items():
        if not steps:
            continue
        c = colors.get(name, None)
        ax.plot(xs, ys, "-", color=c, alpha=0.5, zorder=1)
        ax.scatter(xs, ys, s=42, color=c, label=name, zorder=2)
        # mark start (open) and end (star)
        ax.scatter([xs[0]], [ys[0]], s=120, facecolors="none",
                   edgecolors=c, linewidths=1.8, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=180, marker="*", color=c,
                   edgecolors="k", linewidths=0.5, zorder=4)
        for s, x, y in zip(steps, xs, ys):
            ax.annotate(str(s), (x, y), fontsize=6.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"In-dist accuracy  ({ID_DATASET}, n=3,4)  mean@1")
    ax.set_ylabel(f"OOD accuracy  ({OOD_LABEL[ood_dataset]})  mean@1")
    ax.set_title(f"Pareto: distill→GRPO vs progdistill→GRPO  —  {OOD_LABEL[ood_dataset]}\n"
                 "○ = first ckpt (150)   ★ = last ckpt (1600)", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.005, caveat, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def dominance_summary(ood_dataset, distill, progdistill):
    """Print per-step dominance + overall Pareto-frontier verdict."""
    ds, dx, dy = distill
    ps, px, py = progdistill
    d_by = {s: (x, y) for s, x, y in zip(ds, dx, dy)}
    p_by = {s: (x, y) for s, x, y in zip(ps, px, py)}
    common = sorted(set(d_by) & set(p_by))

    print(f"\n=== {OOD_LABEL[ood_dataset]} : per-step (ID mean@1, OOD mean@1) ===")
    print(f"  {'step':>5}  {'distill (ID/OOD)':>22}  {'progdistill (ID/OOD)':>24}  verdict")
    pd_dominates = 0
    for s in common:
        dxi, dyi = d_by[s]
        pxi, pyi = p_by[s]
        # progdistill dominates distill at this step?
        if pxi >= dxi and pyi >= dyi and (pxi > dxi or pyi > dyi):
            v = "progdistill dominates"
            pd_dominates += 1
        elif dxi >= pxi and dyi >= pyi and (dxi > pxi or dyi > pyi):
            v = "distill dominates"
        else:
            v = "neither (trade-off)"
        print(f"  {s:>5}  {dxi:7.3f} / {dyi:7.3f}        "
              f"{pxi:7.3f} / {pyi:7.3f}         {v}")
    print(f"  -> progdistill strictly dominates distill at {pd_dominates}/{len(common)} matched steps")

    # Overall: does any progdistill point dominate ALL distill points (i.e.
    # sit on/above the distill Pareto frontier)?
    def dominates_frontier(point, frontier_pts):
        x, y = point
        return all(x >= fx and y >= fy for fx, fy in frontier_pts)
    distill_pts = list(zip(dx, dy))
    best = [(s, px[i], py[i]) for i, s in enumerate(ps)
            if dominates_frontier((px[i], py[i]), distill_pts)]
    if best:
        print("  -> progdistill checkpoints that dominate the ENTIRE distill set "
              f"(best-vs-best): {[b[0] for b in best]}")
    else:
        print("  -> NO single progdistill checkpoint dominates every distill checkpoint "
              "(no global best-vs-best dominance)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/n/netscratch/kdbrantley_lab/Lab/mwalden/"
                                             "rl-skill-comp-results/results")
    ap.add_argument("--model", default="Qwen2.5-0.5B")
    ap.add_argument("--distill-exp", required=True)
    ap.add_argument("--progdistill-exp", required=True)
    ap.add_argument("--figures-dir", default=None,
                    help="default: figures/<model>/pareto")
    ap.add_argument("--tag", default="",
                    help="suffix appended to output filenames, e.g. sftlr1e-6")
    args = ap.parse_args()

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", args.model, "pareto")

    caveat = (f"distill-exp={args.distill_exp}  |  progdistill-exp={args.progdistill_exp}")

    exps = {"distill→GRPO": args.distill_exp,
            "progdistill→GRPO": args.progdistill_exp}

    # ID curve per arm
    id_curves = {name: load_mean1_by_step(args.result_dir, args.model, exp, ID_DATASET)
                 for name, exp in exps.items()}
    for name, c in id_curves.items():
        print(f"[{name}] {ID_DATASET}: {len(c)} checkpoints with scores")

    for ood in OOD_DATASETS:
        arms = {}
        for name, exp in exps.items():
            ood_curve = load_mean1_by_step(args.result_dir, args.model, exp, ood)
            arms[name] = trajectory(id_curves[name], ood_curve)
            print(f"[{name}] {ood}: {len(ood_curve)} ckpts -> {len(arms[name][0])} matched with ID")
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = os.path.join(figures_dir, f"pareto_{ood}{suffix}.png")
        plot_one(ood, arms, out_path, caveat)
        dominance_summary(ood, arms["distill→GRPO"], arms["progdistill→GRPO"])


if __name__ == "__main__":
    main()
