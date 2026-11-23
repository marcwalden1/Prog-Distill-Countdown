"""Pass@k coverage comparison across models and sampling temperatures.

Reads eval JSONs (32 samples/prompt) under results/<model>/<run>/<step>/,
scores them with grader_utils.compute_score (correct == score 1.0), and
computes the unbiased pass@k estimator (Chen et al. 2021):

    pass@k = mean over prompts of  1 - C(n-c, k) / C(n, k)

where n is the number of samples per prompt and c the number of correct ones.
Per-prompt correct counts are cached in a `.passk` file next to each JSON
(same mtime-freshness convention as plot_results.py's `.scores`).

Usage:
    python3 scripts/plot/plot_passk.py \
        --run base=base/global_step_0 \
        --run grpo=balanced-grpo-seed1/global_step_1600 \
        --run grpo-warmup=balanced-grpo-warmup0.03-beta2_0.95-seed1/global_step_1600
"""

import argparse
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grader_utils import compute_score

MAX_PROMPTS = None  # None = full prompt set, same as plot_results.py. A head-slice
                    # biases n=5/n=6 (file is ordered by pattern index). Set an int
                    # only for a deliberately fast, approximate plot.

DATASET_LABELS = {
    "balanced": "n=3,4 (in-dist)",
    "balanced5": "n=5 (OOD)",
    "balanced6": "n=6 (OOD)",
}

# Fixed categorical assignment (validated palette); markers double as
# secondary encoding so identity never rides on color alone.
SERIES_STYLE = [
    {"color": "#2a78d6", "marker": "o"},
    {"color": "#1baf7a", "marker": "s"},
    {"color": "#eda100", "marker": "^"},
    {"color": "#4a3aa7", "marker": "D"},
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def series_styles(n_series):
    """Categorical palette for few series; sequential viridis for many
    (e.g. one series per checkpoint step, where order is meaningful)."""
    if n_series <= len(SERIES_STYLE):
        return SERIES_STYLE[:n_series]
    cmap = plt.get_cmap("viridis")
    denom = max(n_series - 1, 1)
    return [
        {"color": cmap(0.9 * i / denom), "marker": MARKERS[i % len(MARKERS)]}
        for i in range(n_series)
    ]


def _count_correct(args):
    outputs, target, nums = args
    extra = {"numbers": nums}
    return sum(
        compute_score(None, out, target, extra, verbose=False) == 1.0
        for out in outputs
    )


def correct_counts(path):
    """Per-prompt correct counts (and n samples/prompt) for one eval JSON."""
    cache_path = path + ".passk"
    try:
        if os.path.getmtime(cache_path) >= os.path.getmtime(path):
            with open(cache_path) as f:
                cached = json.load(f)
            return cached["n"], cached["correct_counts"]
    except Exception:
        pass

    with open(path) as f:
        data = json.load(f)
    data = [item for item in data if item.get("outputs")]
    if MAX_PROMPTS is not None:
        data = data[:MAX_PROMPTS]
    if not data:
        return None, None

    n = min(len(item["outputs"]) for item in data)
    items = [(item["outputs"][:n], item["target"], item["nums"]) for item in data]

    from multiprocessing import Pool

    with Pool(8) as pool:
        counts = pool.map(_count_correct, items)

    with open(cache_path, "w") as f:
        json.dump({"n": n, "correct_counts": counts}, f)
    return n, counts


def pass_at_k(n, counts, k):
    """Unbiased pass@k averaged over prompts."""
    k = min(k, n)
    return sum(1.0 - comb(n - c, k) / comb(n, k) for c in counts) / len(counts)


def _darken(color, frac=0.35):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - frac), g * (1 - frac), b * (1 - frac))


def _annotate_all_points(ax, ks, plotted):
    """Label every point of every series.

    plotted: list of (ys, color). Every label sits just above its point; when
    points in a column are too close, labels stack upward preserving the
    vertical order of the lines (so the k-th label from the bottom always
    belongs to the k-th line from the bottom). Labels are tinted with their
    series color to reinforce the association.
    """
    y0, y1 = ax.get_ylim()
    fig = ax.figure
    axis_h_pt = ax.get_position().height * fig.get_size_inches()[1] * 72
    h = 9.5 * (y1 - y0) / axis_h_pt  # label height in data units
    pad = 0.45 * h
    for xi, x in enumerate(ks):
        col = sorted(((ys[xi], color) for ys, color in plotted), key=lambda t: t[0])
        prev_top = y0
        for y, color in col:
            lo = max(y + pad, prev_top)
            prev_top = lo + h
            ax.annotate(
                f"{y:.2f}", (x, lo + h / 2), ha="center", va="center",
                fontsize=7, color=_darken(color), clip_on=False,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/Qwen2.5-0.5B")
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=SUBDIR",
        help="series to plot, as <label>=<subdir under result root>",
    )
    parser.add_argument("--temps", type=float, nargs="+", default=[0.6, 1.0])
    parser.add_argument(
        "--datasets", nargs="+", default=["balanced", "balanced5", "balanced6"]
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--n", type=int, default=32, help="samples per prompt in filenames")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default="figures/Qwen2.5-0.5B/passk/passk_grid.png")
    parser.add_argument(
        "--by-step-out", default=None,
        help="optional second figure: pass@k vs. training step (one line per k); "
        "requires every --run label to be an integer step",
    )
    parser.add_argument(
        "--title-tag", default=None,
        help="model tag for the figure title (default: basename of --result-root)",
    )
    parser.add_argument(
        "--point-labels", choices=["last", "all", "none"], default="last",
        help="annotate pass@k values at the last point, every point, or not at all",
    )
    parser.add_argument("--no-title", action="store_true", help="omit the figure title")
    parser.add_argument(
        "--legend-loc", choices=["top", "bottom"], default="top",
        help="place the legend above or below the grid",
    )
    parser.add_argument(
        "--style-slots", default=None,
        help="comma-separated SERIES_STYLE indices, one per --run (e.g. 0,1,3), "
        "so a series keeps its color when other series are dropped",
    )
    args = parser.parse_args()

    runs = [r.split("=", 1) for r in args.run]
    if args.style_slots:
        run_styles = [SERIES_STYLE[int(i)] for i in args.style_slots.split(",")]
        assert len(run_styles) == len(runs), "--style-slots needs one index per --run"
    else:
        run_styles = series_styles(len(runs))

    # results[(temp, dataset, label)] = {k: pass@k}
    results = {}
    for temp in args.temps:
        for ds in args.datasets:
            for label, subdir in runs:
                path = os.path.join(
                    args.result_root, subdir,
                    f"{ds}_temp{temp:.1f}_n{args.n}_max{args.max_tokens}.json",
                )
                if not os.path.exists(path):
                    print(f"[missing] {path}")
                    continue
                print(f"[scoring] {path}", flush=True)
                n, counts = correct_counts(path)
                if not counts:
                    continue
                results[(temp, ds, label)] = {
                    "n_prompts": len(counts),
                    **{k: pass_at_k(n, counts, k) for k in args.ks},
                }

    if not results:
        print("No result files found — nothing to plot.")
        return

    # Stdout table
    header = f"{'temp':>5} {'dataset':>10} {'model':>28} {'#prompts':>8} " + " ".join(
        f"{'pass@' + str(k):>9}" for k in args.ks
    )
    print("\n" + header)
    print("-" * len(header))
    for (temp, ds, label), vals in sorted(results.items()):
        row = f"{temp:>5} {ds:>10} {label:>28} {vals['n_prompts']:>8} " + " ".join(
            f"{vals[k]:>9.4f}" for k in args.ks
        )
        print(row)

    # temperature (rows) x dataset (cols) grid of pass@k curves
    nrows, ncols = len(args.temps), len(args.datasets)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False, sharey="row"
    )
    for i, temp in enumerate(args.temps):
        for j, ds in enumerate(args.datasets):
            ax = axes[i][j]
            has_data = False
            plotted = []
            for (label, _), style in zip(runs, run_styles):
                vals = results.get((temp, ds, label))
                if vals is None:
                    continue
                has_data = True
                ys = [vals[k] for k in args.ks]
                ax.plot(
                    args.ks, ys, label=label, linewidth=2,
                    marker=style["marker"], markersize=6, color=style["color"],
                )
                plotted.append((ys, style["color"]))
                if args.point_labels == "last" and len(runs) <= len(SERIES_STYLE):
                    ax.annotate(
                        f"{ys[-1]:.2f}", (args.ks[-1], ys[-1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color="#555555",
                    )
            ax.set_xscale("log", base=2)
            ax.set_xticks(args.ks)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_ylim(bottom=0)
            if args.point_labels == "all" and plotted:
                _annotate_all_points(ax, args.ks, plotted)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i == nrows - 1:
                ax.set_xlabel("k")
            if j == 0:
                ax.set_ylabel(f"temp {temp}\npass@k")
            if i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
            if not has_data:
                ax.text(
                    0.5, 0.5, "eval pending", transform=ax.transAxes,
                    ha="center", va="center", color="#888888",
                )

    handles, labels = None, None
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if len(l) > (len(labels) if labels else 0):
                handles, labels = h, l
    if handles:
        if args.legend_loc == "bottom":
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
        else:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
                       frameon=False, bbox_to_anchor=(0.5, 1.02))
    model_tag = args.title_tag or os.path.basename(os.path.normpath(args.result_root))
    if not args.no_title:
        fig.suptitle(
            f"{model_tag} coverage: unbiased pass@k ({args.n} samples/prompt)",
            y=1.08, fontsize=12,
        )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {args.out}")

    if args.by_step_out:
        plot_by_step(args, runs, results, model_tag)


def plot_by_step(args, runs, results, model_tag):
    """pass@k vs. training step: one line per k, all temp x dataset panels in a row."""
    try:
        steps = {label: int(label) for label, _ in runs}
    except ValueError:
        print("[by-step] skipped: --run labels are not all integer steps")
        return

    combos = [(temp, ds) for temp in args.temps for ds in args.datasets]
    fig, axes = plt.subplots(
        1, len(combos), figsize=(4.2 * len(combos), 3.4), squeeze=False, sharey=True
    )
    k_styles = series_styles(len(args.ks))
    for ax, (temp, ds) in zip(axes[0], combos):
        has_data = False
        for k, style in zip(args.ks, k_styles):
            pts = sorted(
                (steps[label], results[(temp, ds, label)][k])
                for label, _ in runs
                if (temp, ds, label) in results
            )
            if not pts:
                continue
            has_data = True
            xs, ys = zip(*pts)
            ax.plot(
                xs, ys, label=f"pass@{k}", linewidth=2,
                marker=style["marker"], markersize=5, color=style["color"],
            )
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("training step")
        title = f"temp {temp}"
        if len(args.datasets) > 1:
            title = f"{DATASET_LABELS.get(ds, ds)}, {title}"
        ax.set_title(title, fontsize=11)
        if not has_data:
            ax.text(
                0.5, 0.5, "eval pending", transform=ax.transAxes,
                ha="center", va="center", color="#888888",
            )
    axes[0][0].set_ylabel("pass@k")

    handles, labels = None, None
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        if len(l) > (len(labels) if labels else 0):
            handles, labels = h, l
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
                   frameon=True, bbox_to_anchor=(0.5, 1.14))
    if not args.no_title:
        fig.suptitle(
            f"{model_tag} coverage vs. training step ({args.n} samples/prompt)",
            y=1.24, fontsize=12,
        )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.by_step_out), exist_ok=True)
    fig.savefig(args.by_step_out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.by_step_out}")


if __name__ == "__main__":
    main()
