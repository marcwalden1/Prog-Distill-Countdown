"""Plot teacher logit falloff from scripts/sanity_teacher_logits.py output.

  python3 scripts/plot/plot_teacher_logit_sanity.py --in_dir <out_dir> --out <png>

Panels: (a) prob of the r-th ranked token vs r (median + 10-90% / 1-99% bands);
(b) CDF over positions of tail mass outside top-k for k in {5, 20, 64};
(c) tail mass beyond top-20 vs response position (median, p90).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(os.path.join(a.in_dir, "sanity.npz"))
    tp, pos = d["top_probs"], d["pos"]
    cum = np.cumsum(tp, -1)
    r = np.arange(1, tp.shape[1] + 1)

    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), facecolor="#fcfcfb")
    for x in ax:
        x.set_facecolor("#fcfcfb"); x.grid(color=GRID, lw=0.8)

    fl = np.clip(tp, 1e-12, None)
    q = np.percentile(fl, [1, 10, 50, 90, 99], axis=0)
    ax[0].fill_between(r, q[0], q[4], color=BLUE, alpha=0.12, lw=0, label="1–99%")
    ax[0].fill_between(r, q[1], q[3], color=BLUE, alpha=0.28, lw=0, label="10–90%")
    ax[0].plot(r, q[2], color=BLUE, lw=2, label="median")
    for k in (20, 64):
        ax[0].axvline(k, color=INK2, lw=1, ls="--"); ax[0].text(k * 1.07, 3e-11, f"k={k}", color=INK2)
    ax[0].set(xscale="log", yscale="log", xlabel="rank r", ylabel="p(rank-r token)",
              title="(a) Probability by rank, per position", ylim=(1e-11, 1.5))
    ax[0].legend(frameon=False, loc="lower left")

    for k, c in ((5, AQUA), (20, BLUE), (64, ORANGE)):
        tail = np.sort(np.clip(1 - cum[:, k - 1], 1e-9, None))
        ax[1].plot(tail, np.arange(1, len(tail) + 1) / len(tail), color=c, lw=2, label=f"top-{k}")
    ax[1].set(xscale="log", xlabel="tail mass outside top-k", ylabel="fraction of positions ≤ x",
              title="(b) CDF of tail mass", xlim=(1e-9, 1))
    ax[1].legend(frameon=False, loc="lower right")

    tail20 = 1 - cum[:, 19]
    bins = np.arange(0, pos.max() + 33, 32)
    idx = np.digitize(pos, bins) - 1
    xs, med, p90, mean = [], [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 20: continue
        xs.append(bins[b] + 16); med.append(np.median(tail20[m]))
        p90.append(np.percentile(tail20[m], 90)); mean.append(tail20[m].mean())
    ax[2].plot(xs, mean, color=BLUE, lw=2, marker="o", ms=4, label="mean")
    ax[2].plot(xs, p90, color=ORANGE, lw=2, marker="o", ms=4, label="p90")
    ax[2].plot(xs, med, color=AQUA, lw=2, marker="o", ms=4, label="median")
    ax[2].set(yscale="log", xlabel="response position (tokens, 32-token bins)",
              ylabel="tail mass outside top-20", title="(c) Top-20 tail mass vs position")
    ax[2].legend(frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=fig.get_facecolor())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
