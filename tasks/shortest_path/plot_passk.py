"""pass@1 / pass@128 vs GRPO step, one column per d (pass@1 solid, pass@128 dashed).

    python3 tasks/shortest_path/plot_passk.py \
        --run "0.6B lr3e-6" analysis_data/spath_grpo_0.6B_lr3e-6_passk.csv Qwen3-0.6B-Base \
        --out figures-spath/Qwen3/passk_0.6B_lr3e-6_kl3e-3.png

--run LABEL CSV BASE_MODEL (repeatable): CSV is score_passk.py output whose model
column ends in stepN. Step 0 is read from the base-model table (--base-csv) for
BASE_MODEL and drawn as a hollow marker.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(csv, base_csv, base_model):
    df = pd.read_csv(csv)
    df = df[df.paths == "all"].copy()
    df["step"] = df.model.str.extract(r"step(\d+)$")[0].astype(int)
    b = pd.read_csv(base_csv)
    b = b[(b.paths == "all") & (b.model == base_model)].copy()
    b["step"] = 0
    return pd.concat([b, df]).sort_values(["distance", "step"])


def label_final(ax, groups, met, gap=0.075):
    """Final values of `met` right of the last point, nudged apart so they don't overlap;
    a short colored tick ties each label to its run."""
    pts = sorted(((g.iloc[-1].step, g.iloc[-1][met], c) for g, c in groups if len(g)), key=lambda t: t[1])
    ys = [y for _, y, _ in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - 1.04 if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
    for (x, y, c), ly in zip(pts, ys):
        xt = x * 1.07
        ax.plot([x * 1.02, xt], [y, ly], color=c, lw=1, zorder=2)
        ax.annotate(f"{y:.2f}", (xt, ly), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", required=True, metavar=("LABEL", "CSV", "BASE_MODEL"))
    ap.add_argument("--base-csv", default="analysis_data/spath_v1_base_passk.csv")
    ap.add_argument("--metrics", nargs="+", default=["pass@1", "pass@128"])
    ap.add_argument("--train-ds", default="2,3,4", help="d values seen in training (panel titles)")
    ap.add_argument("--label-metric", default="pass@1", help="metric whose final values are labelled (multi-run)")
    ap.add_argument("--title", default=None, help="suffix after the default title (multi-run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [(lab, load(csv, args.base_csv, bm)) for lab, csv, bm in args.run]
    ds = sorted(set().union(*[set(r.distance) for _, r in runs]))
    train = {int(x) for x in args.train_ds.split(",")}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK})
    styles = ["-", "--", ":", "-."]  # one line style per metric (pass@1 solid, pass@128 dashed)
    fig, axes = plt.subplots(1, len(ds), figsize=(2.4 * len(ds), 3.0), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for j, d in enumerate(ds):
        ax = axes[j]
        for k, (lab, r) in enumerate(runs):
            g = r[r.distance == d]
            c = SERIES[k]
            for i, met in enumerate(args.metrics):
                ls = styles[i]
                ax.plot(g.step, g[met], color=c, lw=2, ls=ls, zorder=3)
                trained = g[g.step > 0]
                ax.plot(trained.step, trained[met], "o", ms=5, color=c, mec=c, mew=1.5, zorder=4)
                base = g[g.step == 0]
                ax.plot(base.step, base[met], "o", ms=5, mfc="white", mec=c, mew=1.5, zorder=4)
                if len(runs) == 1:
                    # last metric (largest k) labelled above its point, others below
                    above = i == len(args.metrics) - 1
                    last = g.iloc[-1]
                    ax.annotate(f"{last[met]:.2f}", (last.step, last[met]), xytext=(0, 6 if above else -13),
                                textcoords="offset points", ha="center", fontsize=8, color=INK)
                    b0 = base[met].iloc[0]
                    ax.annotate(f"{b0:.2f}", (0, b0), xytext=(-7, -3),
                                textcoords="offset points", ha="right", fontsize=8, color=INK2)
        if len(runs) > 1:
            label_final(ax, [(r[r.distance == d], c) for (_, r), c in zip(runs, SERIES)], args.label_metric)
        ax.set_ylim(0, 1.08)
        if len(runs) > 1:
            smax = max(r.step.max() for _, r in runs)
            ax.set_xlim(-0.05 * smax, 1.32 * smax)
            ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= smax])
        if len(runs) == 1:  # room for the step-0 value labels left of the base markers
            ax.set_xlim(-0.28 * max(r.step.max() for _, r in runs), 1.06 * max(r.step.max() for _, r in runs))
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"d={d}" + (" (train)" if d in train else " (OOD)"), fontsize=10)
        ax.set_xlabel("GRPO step")
    axes[0].set_ylabel("pass@k")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK2, lw=2, ls=styles[i], label=met) for i, met in enumerate(args.metrics)]
    if len(runs) > 1:
        handles += [Line2D([], [], color=SERIES[k], lw=2, label=lab) for k, (lab, _) in enumerate(runs)]
    if len(runs) == 1:
        axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    else:  # one row under the panels: inside any panel it covers a curve
        fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=len(handles))
    title = "Shortest path: pass@k vs GRPO step (step 0 = base, hollow)"
    if len(runs) == 1:
        title += f" | {runs[0][0]}"
    elif args.title:
        title += f" | {args.title}"
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07 if len(runs) > 1 else 0, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
