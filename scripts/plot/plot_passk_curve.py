#!/usr/bin/env python3
"""pass@k vs k for post-SFT (pre-GRPO) checkpoints — one curve per model.

Reads an eval JSON produced with a large --n (e.g. 512 samples/prompt) and
reports pass@k for k = 32, 64, ... up to N.

Estimator. With N samples per prompt of which c are correct, the probability
that a random size-k subset contains at least one correct sample is

    pass@k = 1 - C(N-c, k) / C(N, k)

(Chen et al. 2021, "Evaluating Large Language Models Trained on Code"). This is
the unbiased estimate over all C(N,k) subsets. Taking "the first k samples"
instead would be a single noisy draw — at k=32 out of 512 that throws away 15/16
of the data and adds variance for nothing. Computed in log-space to keep
C(512, 256) from overflowing.

    python3 scripts/plot/plot_passk_curve.py \
        --run "1B 4001B=OLMo-2-0425-1B-pt4001B" \
        --run "7B 126B=OLMo-2-1124-7B-pt126B" \
        --exp balanced-simple-gtsft200-lr3e-5-seed1 \
        --dataset balanced-simple --tag sft-students

Writes figures/olmo2-ladder/<tag>/passk_curve_<tag>.png and a CSV of the same numbers.
"""
import argparse
import csv
import glob
import json
import os
from math import lgamma, exp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
from grader_utils import compute_score  # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def pass_at_k(n_total, n_correct, k):
    """Unbiased pass@k for one prompt: 1 - C(n-c, k)/C(n, k)."""
    if k > n_total:
        return None
    if n_correct <= 0:
        return 0.0
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - exp(_log_comb(n_total - n_correct, k) - _log_comb(n_total, k))


def correct_counts(path):
    """[(n_samples, n_correct)] per prompt, using the same scorer as every other
    metric in the repo (score == 1.0 is exactly correct)."""
    with open(path) as f:
        recs = json.load(f)
    out = []
    for r in recs:
        outs = r["outputs"]
        gt = r["target"]
        # Same call convention as scripts/compute_passk.py: data_source=None,
        # numbers carried in extra_info under the key "numbers" (the JSON field
        # itself is "nums"), verbose off.
        extra = {"numbers": r["nums"]}
        c = sum(1 for o in outs
                if compute_score(None, o, gt, extra, verbose=False) == 1.0)
        out.append((len(outs), c))
    return out


def find_json(result_dir, model, exp, subdir, dataset, n):
    pat = os.path.join(result_dir, model, exp, subdir, f"{dataset}_temp*_n{n}_max*.json")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=MODEL")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--subdir", default="sft_final")
    ap.add_argument("--dataset", action="append", default=None, metavar="DATASET=LABEL",
                    help="repeatable; one sub-plot per dataset. Default: the three "
                         "countdown splits (n=3,4 in-dist, n=5 OOD, n=6 OOD).")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--n", type=int, default=512, help="samples per prompt in the eval JSON")
    ap.add_argument("--ks", default="32,64,128,256,512")
    ap.add_argument("--set", dest="set_dir", default=None, metavar="FOLDER",
                    help="sub-directory of figures/olmo2-ladder/ for this comparison "
                         "set (default: --tag). Folder names are descriptive and need "
                         "not match the tag.")
    ap.add_argument("--figures-dir", default=None,
                    help="default: figures/olmo2-ladder/<--set or --tag>/ — one "
                         "sub-directory per comparison set.")
    ap.add_argument("--tag", default="sft-students")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    datasets = ([tuple(d.split("=", 1)) for d in args.dataset] if args.dataset
                else [("balanced-simple", "n=3,4 (in-dist)"),
                      ("balanced5-simple", "n=5 (OOD)"),
                      ("balanced6-simple", "n=6 (OOD)")])

    # panels[ds] = list of per-model series
    panels = {}
    for ds, _ in datasets:
        series = []
        for ci, spec in enumerate(args.run):
            label, model = spec.split("=", 1)
            p = find_json(args.result_dir, model, args.exp, args.subdir, ds, args.n)
            if not p:
                print(f"  {ds:18} SKIP {label}: no n={args.n} eval JSON")
                continue
            counts = correct_counts(p)
            n_prompts = len(counts)
            n_avail = min(c[0] for c in counts)
            curve = []
            for k in ks:
                if k > n_avail:
                    curve.append((k, None))
                    continue
                vals = [pass_at_k(n, c, k) for n, c in counts]
                curve.append((k, sum(vals) / len(vals)))
            series.append({"label": label, "model": model, "curve": curve,
                           "n_prompts": n_prompts, "n_avail": n_avail,
                           "color": COLORS[ci % len(COLORS)],
                           "marker": MARKERS[ci % len(MARKERS)]})
            shown = "  ".join(f"pass@{k} {v:.4f}" for k, v in curve if v is not None)
            print(f"  {ds:18} {label:12} prompts={n_prompts} N={n_avail}  {shown}")
        panels[ds] = series

    if not any(panels.values()):
        raise SystemExit("ERROR: no runs had an eval JSON — nothing to plot.")

    if args.figures_dir is None:
        args.figures_dir = os.path.join("figures", "olmo2-ladder",
                                        args.set_dir or args.tag or "untagged")
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs("analysis_data", exist_ok=True)
    csv_path = os.path.join("analysis_data", f"passk_curve_{args.tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "model", "dataset", "n_prompts", "k", "pass_at_k"])
        for ds, _ in datasets:
            for s in panels[ds]:
                for k, v in s["curve"]:
                    if v is not None:
                        w.writerow([s["label"], s["model"], ds, s["n_prompts"], k, f"{v:.6f}"])
    print(f"  wrote {csv_path}")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.3 * len(datasets), 5.0),
                             squeeze=False)
    for ax, (ds, ds_label) in zip(axes[0], datasets):
        series = panels[ds]
        # Endpoint labels are staggered by rank: where models converge (in-dist
        # pass@k saturates near the same value) an unstaggered label stack
        # overlaps itself and clips the axis.
        ends = sorted(range(len(series)),
                      key=lambda i: [v for _, v in series[i]["curve"] if v is not None][-1]
                      if any(v is not None for _, v in series[i]["curve"]) else 0)
        rank = {si: r for r, si in enumerate(ends)}
        for si, s in enumerate(series):
            xs = [k for k, v in s["curve"] if v is not None]
            ys = [v for k, v in s["curve"] if v is not None]
            if not xs:
                continue
            ax.plot(xs, ys, marker=s["marker"], ms=7, lw=2, color=s["color"], label=s["label"])
            # Direct-label the endpoint so identity never rests on colour alone.
            dy = -14 + 11 * rank[si]
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(7, dy), fontsize=8, color=s["color"],
                        annotation_clip=False)
        # Room on the right for endpoint labels, and on top so the highest
        # label is not clipped by the axis frame.
        ax.margins(x=0.16, y=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k   (samples drawn per prompt)")
        ax.set_ylabel("pass@k   (>=1 of k samples exactly correct)")
        n_p = series[0]["n_prompts"] if series else 0
        ax.set_title(f"{ds_label}   ({n_p} prompts)" if n_p else f"{ds_label}   (no data)",
                     fontsize=11)
        ax.grid(alpha=0.3)
        if series:
            ax.legend(fontsize=9, loc="best")
        missing = [spec.split("=", 1)[0] for spec in args.run
                   if spec.split("=", 1)[0] not in {s["label"] for s in series}]
        if missing:
            ax.text(0.5, -0.15, "no n=%d eval: %s" % (args.n, ", ".join(missing)),
                    transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#b00")

    N = next((s["n_avail"] for ss in panels.values() for s in ss), args.n)
    title = args.title or ("Post-SFT (pre-GRPO) coverage vs sample budget  ·  "
                           f"temp 0.6, unbiased pass@k from {N} samples/prompt")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = os.path.join(args.figures_dir, f"passk_curve_{args.tag}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
