"""Cross-arm comparison: does SFT move P(true) vs P(target), and does GRPO keep it?

One panel per SFT format, three models each -- the split is both the readable
one (3 categorical slots validate all-pairs; 6 on one axis would not) and the
scientific one: `answer`-format SFT never demonstrates a single `= value`, so it
is the control that isolates whether the `search` format's correct evaluations
of wrong expressions are what move the metric, as opposed to SFT in general.

Arms use different save grids (1B every 100, 7B every 200), so the GRPO steps go
on a numeric axis; `base` and `SFT` are placed at fixed negative offsets before a
divider, since they are not RL steps.

    python3 scripts/plot/plot_arith_ptrue_by_arm.py --out figures/diagnosis/arith_ptrue_by_arm.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = {"1B-pt4001B": "#2a78d6", "7B-pt135B": "#eb6834", "7B-pt3896B": "#1baf7a"}
MK = {"1B-pt4001B": "o", "7B-pt135B": "s", "7B-pt3896B": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d6d5d0"
X_BASE, X_SFT, X_DIV = -420, -210, -105          # pre-RL slots, then the rule

ARMS = {
    "search": [("1B-pt4001B", "analysis_data/arith_ntp_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv"),
               ("7B-pt135B", "analysis_data/arith_ntp_olmo2_7B_pt135B_search_sftlr1e-5_grpo.csv"),
               ("7B-pt3896B", "analysis_data/arith_ntp_olmo2_7B_pt3896B_search_sftlr3e-5_grpo.csv")],
    "answer": [("1B-pt4001B", "analysis_data/arith_ntp_olmo2_1B_pt4001B_answer_sftlr3e-5_grpo.csv"),
               ("7B-pt135B", "analysis_data/arith_ntp_olmo2_7B_pt135B_answer_sftlr3e-5_grpo.csv"),
               ("7B-pt3896B", "analysis_data/arith_ntp_olmo2_7B_pt3896B_answer_sftlr1e-5_grpo.csv")],
}
TITLES = {
    "search": "search GT-SFT  —  the trace demonstrates correct evaluation of wrong expressions",
    "answer": "answer GT-SFT (control)  —  the trace never contains a single `= value`",
}


def xy(df, col):
    xs, ys = [], []
    for _, r in df.sort_values("order").iterrows():
        x = X_BASE if r["stage"] == "base" else X_SFT if str(r["stage"]).startswith("sft") \
            else float(r["step"])
        xs.append(x)
        ys.append(float(r[col]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="p_true_norm_macro")
    ap.add_argument("--out", default="figures/diagnosis/arith_ptrue_by_arm.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, fmt in zip(axes, ("search", "answer")):
        ax.set_facecolor("#fcfcfb")
        for name, path in ARMS[fmt]:
            if not os.path.exists(path):
                print(f"  missing {path}")
                continue
            xs, ys = xy(pd.read_csv(path), args.col)
            ax.plot(xs, ys, color=C[name], lw=2, marker=MK[name], ms=5.5,
                    mec="white", mew=0.8, label=name, zorder=3)
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=7.5, color=C[name],
                        fontweight="bold", zorder=4)
        ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=2)
        ax.axvspan(X_BASE - 110, X_DIV, color=GRID, alpha=0.30, lw=0, zorder=0)
        ax.axvline(X_DIV, color=INK2, lw=1, ls=":", zorder=1)
        ax.set_xticks([X_BASE, X_SFT] + list(range(0, 1601, 400)))
        ax.set_xticklabels(["base", "SFT"] + [str(s) for s in range(0, 1601, 400)],
                           fontsize=8, color=INK2)
        ax.set_xlim(X_BASE - 130, 1810)
        ax.set_ylim(0, 1)
        ax.set_title(TITLES[fmt], color=INK, fontsize=9, loc="left", pad=6)
        ax.set_xlabel("GRPO step  (first two ticks are not RL steps)", color=INK2, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    axes[0].set_ylabel("P(true) / (P(true) + P(target))", color=INK)
    axes[0].annotate("0.5 = indifferent", (1790, 0.5), fontsize=7, color=INK2,
                     va="bottom", ha="right")
    fig.suptitle("Does the true value beat the target at an identical prefix? "
                 "(teacher-forced, entropy-invariant)", color=INK, fontsize=11, x=0.005, ha="left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
