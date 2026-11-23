"""All arithmetic-fidelity arms on one figure (replaces the per-arm plots).

Layout: one metric per row, one column per SFT format, one line per model.
Colour encodes the model (3 categorical slots, which validate all-pairs under
the dataviz palette; six arms on a single axis would not), and the
search-vs-answer split is the scientific one -- `answer` GT-SFT never
demonstrates a single `= value`, so it is the control for whether the `search`
format's correct evaluations of wrong expressions are what move the metric.

y is shared across each row so the two formats are directly comparable; arms use
different save grids (1B every 100, 7B every 200), so GRPO steps go on a numeric
axis with `base` and `SFT` at fixed negative offsets before a divider.

    python3 scripts/plot/plot_arith_grid.py --which fidelity
    python3 scripts/plot/plot_arith_grid.py --which drift
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = {"1B-pt4001B": "#2a78d6", "7B-pt135B": "#eb6834", "7B-pt3896B": "#1baf7a"}
MK = {"1B-pt4001B": "o", "7B-pt135B": "s", "7B-pt3896B": "^"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
X_BASE, X_SFT, X_DIV = -420, -210, -105

# arm -> (teacher-forced csv, behavioral csv or None)
A = "analysis_data"
ARMS = {
    "search": [
        ("1B-pt4001B",  f"{A}/arith_ntp_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv",
                        f"{A}/arith_fidelity_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv"),
        ("7B-pt135B",   f"{A}/arith_ntp_olmo2_7B_pt135B_search_sftlr1e-5_grpo.csv",
                        f"{A}/arith_fidelity_olmo2_7B_pt135B_search_sftlr1e-5_grpo.csv"),
        ("7B-pt3896B",  f"{A}/arith_ntp_olmo2_7B_pt3896B_search_sftlr3e-5_grpo.csv",
                        f"{A}/arith_fidelity_olmo2_7B_pt3896B_search_sftlr3e-5_grpo.csv"),
    ],
    "answer": [
        ("1B-pt4001B",  f"{A}/arith_ntp_olmo2_1B_pt4001B_answer_sftlr3e-5_grpo.csv", None),
        ("7B-pt135B",   f"{A}/arith_ntp_olmo2_7B_pt135B_answer_sftlr3e-5_grpo.csv", None),
        ("7B-pt3896B",  f"{A}/arith_ntp_olmo2_7B_pt3896B_answer_sftlr1e-5_grpo.csv", None),
    ],
}
COL_TITLE = {
    "search": "search GT-SFT  —  trace demonstrates correct evaluation of wrong expressions",
    "answer": "answer GT-SFT (control)  —  trace never contains a single `= value`",
}

# (column, source, row label, unit hint)
METRICS = {
    "fidelity": [
        ("ce_reject_value_macro", "tf",  "CE: true value of\na rejected attempt", "nats"),
        ("ce_verdict_macro",      "tf",  "CE: too high /\ntoo low verdict", "nats"),
        ("ce_final_value_macro",  "tf",  "CE: final value\n(= target)", "nats"),
        ("p_true_norm_macro",     "tf",  "P(true) /\n(P(true)+P(target))", "unit"),
        ("rej_miscompute",        "beh", "sampled: arithmetic\nerror rate", "unit"),
        ("verdict_error",         "beh", "sampled: verdict\nerror rate", "unit"),
    ],
    "drift": [
        ("entropy_macro", "tf", "predictive\nentropy", "nats"),
        ("top1_macro",    "tf", "top-1\nprobability", "unit"),
        ("kl_base_macro", "tf", "KL(ckpt || base)", "nats"),
    ],
}
SUPTITLE = {
    "fidelity": "Arithmetic fidelity across base -> GT-SFT -> GRPO, by SFT format",
    "drift": "Distribution drift at the probe positions, by SFT format",
}


def xy(path, col):
    """(x, y) for one metric of one arm; base/SFT map to fixed pre-RL offsets."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if col not in df.columns:
        return None
    xs, ys = [], []
    for _, r in df.sort_values("order").iterrows():
        st = str(r["stage"])
        if pd.isna(r["step"]):
            if st.startswith("base"):
                if st not in ("base", "base-8shot"):
                    continue                    # skip the 2-shot behavioral row
                x = X_BASE
            elif st.startswith("sft"):
                x = X_SFT
            else:
                continue
        else:
            x = float(r["step"])
        if pd.notna(r[col]):
            xs.append(x)
            ys.append(float(r[col]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("fidelity", "drift"), required=True)
    ap.add_argument("--out-dir", default="figures/diagnosis")
    args = ap.parse_args()

    rows = METRICS[args.which]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11.6, 2.05 * len(rows) + 0.9),
                             sharex=True, sharey="row", constrained_layout=True)
    fig.patch.set_facecolor(SURF)

    for ri, (col, src, label, unit) in enumerate(rows):
        for ci, fmt in enumerate(("search", "answer")):
            ax = axes[ri][ci]
            ax.set_facecolor(SURF)
            drew = False
            for name, tf_path, beh_path in ARMS[fmt]:
                got = xy(tf_path if src == "tf" else beh_path, col)
                if not got or not got[0]:
                    continue
                ax.plot(*got, color=C[name], lw=2, marker=MK[name], ms=4.5,
                        mec="white", mew=0.7, label=name, zorder=3)
                drew = True
            if not drew:
                ax.text(0.5, 0.5, "no behavioral probe\n(rollouts are a bare <answer> tag)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color=INK2, style="italic")
            if unit == "unit":
                ax.set_ylim(-0.03, 1.03)
            if col == "p_true_norm_macro":
                ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=2)
            ax.axvspan(X_BASE - 110, X_DIV, color=GRID, alpha=0.30, lw=0, zorder=0)
            ax.axvline(X_DIV, color=INK2, lw=1, ls=":", zorder=1)
            ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_xlim(X_BASE - 130, 1700)
            if ri == 0:
                ax.set_title(COL_TITLE[fmt], color=INK, fontsize=9, loc="left", pad=6)
                # one legend for the whole grid: colour means the same model in
                # every panel, so repeating it per column is noise.
                if drew and ci == 0:
                    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2,
                              loc="upper center", ncol=3, columnspacing=1.0,
                              handlelength=1.4, bbox_to_anchor=(0.5, 1.02))
            if ci == 0:
                ax.set_ylabel(label, color=INK, fontsize=8.5)
            if ri == len(rows) - 1:
                ax.set_xticks([X_BASE, X_SFT] + list(range(0, 1601, 400)))
                ax.set_xticklabels(["base", "SFT"] + [str(s) for s in range(0, 1601, 400)],
                                   fontsize=8, color=INK2)
                ax.set_xlabel("GRPO step  (first two ticks are not RL steps)",
                              color=INK2, fontsize=9)

    fig.suptitle(SUPTITLE[args.which], color=INK, fontsize=11.5, x=0.005, ha="left")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"arith_{args.which}_all.png")
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
