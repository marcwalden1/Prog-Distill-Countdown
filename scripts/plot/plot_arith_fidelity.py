"""Arithmetic fidelity across base -> SFT -> GRPO, for `search`-format GT-SFT runs.

Two figures from the two probes:

  arith_fidelity.png    teacher-forced CE at the value/verdict spans
                        (scripts/diagnose_arith_ntp.py), the entropy-invariant
                        P(true)/(P(true)+P(target)) at an identical prefix, and
                        the behavioral miscompute / verdict-error rates
                        (scripts/probe_arith_fidelity.py)
  arith_drift.png       drift controls at the same positions: predictive
                        entropy, top-1 prob, KL(ckpt || base)

The x axis is the RL step, with the base model and the SFT checkpoint as the
first two ticks: they are not RL steps, so the ticks are equally spaced and a
rule separates the pre-RL block from the GRPO curve. Never a second y-axis --
each panel carries one unit (nats, or a fraction in [0, 1]).

Usage:
    python3 scripts/plot/plot_arith_fidelity.py \
        --behavioral analysis_data/arith_fidelity_<tag>.csv \
        --teacher-forced analysis_data/arith_ntp_<tag>.csv \
        --out-dir figures/<MODEL>/<EXP_NAME>/diagnosis \
        --title "OLMo-2-1B-pt4001B | search GT-SFT -> GRPO"
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# dataviz reference palette, categorical slots 1-3 (light mode). Validated
# all-pairs: worst CVD dE 9.2, worst normal-vision dE 24.0. Slot 3 (aqua) is
# below 3:1 on a light surface, so the relief rule applies -> every series is
# also direct-labeled at its right end and carries a distinct marker.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d6d5d0"


def stage_axis(*frames):
    """Shared, equally spaced x for the pre-RL stages then the GRPO steps."""
    pre, steps = [], set()
    for df in frames:
        if df is None:
            continue
        for _, r in df.sort_values("order").iterrows():
            if pd.isna(r["step"]):
                if r["stage"] not in pre:
                    pre.append(r["stage"])
            else:
                steps.add(int(r["step"]))
    labels = pre + [str(s) for s in sorted(steps)]
    return {lab: i for i, lab in enumerate(labels)}, labels, len(pre)


def key_of(row):
    return row["stage"] if pd.isna(row["step"]) else str(int(row["step"]))


def series(df, col, pos):
    """(x, y) for one metric column, ordered along the shared axis."""
    xs, ys = [], []
    for _, r in df.iterrows():
        k = key_of(r)
        if k in pos and pd.notna(r.get(col)):
            xs.append(pos[k])
            ys.append(float(r[col]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def style(ax, labels, n_pre, ylabel, title=None):
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=INK2)
    # The pre-RL block is not on the step axis; mark it rather than pretend.
    ax.axvspan(-0.5, n_pre - 0.5, color=GRID, alpha=0.30, lw=0, zorder=0)
    ax.axvline(n_pre - 0.5, color=INK2, lw=1, ls=":", zorder=1)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    # right margin leaves room for the direct end-labels (relief rule)
    ax.set_xlim(-0.7, len(labels) + 1.4)


def plot_line(ax, xs, ys, color, marker, label, end_label=None):
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5.5,
            mec="white", mew=0.8, label=label, zorder=3)
    if end_label and xs:
        ax.annotate(end_label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), va="center", fontsize=7.5, color=color,
                    fontweight="bold", zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioral",
                    help="omit for answer-format arms: their rollouts are a bare "
                         "<answer> tag, so there are no arithmetic claims to parse")
    ap.add_argument("--teacher-forced")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True,
                    help="filename suffix, e.g. OLMo-2-0425-1B-pt4001B_search-sftlr3e-5")
    ap.add_argument("--title", default="")
    ap.add_argument("--base-stage", default="base-8shot",
                    help="which behavioral base row to place on the axis")
    args = ap.parse_args()

    beh = None
    if args.behavioral:
        beh = pd.read_csv(args.behavioral)
        beh = beh[beh["step"].notna() | (beh["stage"] == args.base_stage) |
                  beh["stage"].str.startswith("sft")].copy()
    tf = None
    if args.teacher_forced and os.path.exists(args.teacher_forced):
        tf = pd.read_csv(args.teacher_forced)
    elif args.teacher_forced:
        print(f"note: no teacher-forced CSV at {args.teacher_forced!r} — "
              "plotting behavioral panel only")

    # Align the pre-RL labels of the two probes onto one axis.
    if tf is not None and beh is not None:
        beh["stage"] = beh["stage"].replace({args.base_stage: "base"})
        beh.loc[beh["stage"].str.startswith("sft"), "stage"] = "SFT"
    if tf is not None:
        tf["stage"] = tf["stage"].replace({"base": "base", "sft": "SFT"})
    pos, labels, n_pre = stage_axis(tf, beh)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- figure 1: the two things asked for --------------------
    n_panels = (2 if tf is not None else 0) + (1 if beh is not None else 0)
    if not n_panels:
        raise SystemExit("nothing to plot: pass --behavioral and/or --teacher-forced")
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.2, 3.0 * n_panels),
                             sharex=True, constrained_layout=True)
    axes = [axes] if n_panels == 1 else list(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    if tf is not None:
        ax = axes[0]
        for col, c, mk, lab, end in (
            ("ce_reject_value_macro", C1, "o", "true value of a rejected attempt", "arith"),
            ("ce_verdict_macro", C3, "^", "too high / too low verdict", "verdict"),
            ("ce_final_value_macro", C2, "s", "final value (= target)", "format"),
        ):
            plot_line(ax, *series(tf, col, pos), color=c, marker=mk, label=lab, end_label=end)
        style(ax, labels, n_pre, "teacher-forced CE (nats)",
              "Teacher-forced loss on the forced trace, by span")
        # mid-band is the free space here (spans sit high and low), and every
        # series is direct-labeled anyway -- the box just names them.
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="center left", ncol=1)
        ax.margins(y=0.14)

        ax = axes[1]
        plot_line(ax, *series(tf, "p_true_norm_macro", pos), color=C1, marker="o",
                  label=None, end_label="P(true)")
        ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=2)
        ax.annotate("0.5 = indifferent", (len(labels) - 1.2, 0.5), fontsize=7,
                    color=INK2, va="bottom", ha="right")
        ax.set_ylim(0, 1)
        style(ax, labels, n_pre, "P(true) / (P(true)+P(target))",
              "True value vs the target at an identical prefix (entropy-invariant)")

    if beh is not None:
        ax = axes[-1]
        for col, c, mk, lab, end in (
            ("rej_miscompute", C1, "o", "arithmetic error on rejected attempts", "miscompute"),
            ("verdict_error", C2, "s", "verdict inconsistent with own claimed value", "verdict err"),
        ):
            plot_line(ax, *series(beh, col, pos), color=c, marker=mk, label=lab, end_label=end)
        ax.set_ylim(-0.03, 1.03)
        style(ax, labels, n_pre, "rate", "Sampled rollouts (temp 0.6): error rates")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="center left")
    axes[-1].set_xlabel("GRPO step  (first two ticks are not RL steps)",
                        color=INK2, fontsize=9)
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=11, x=0.01, ha="left")

    p1 = os.path.join(args.out_dir, f"arith_fidelity_{args.tag}.png")
    fig.savefig(p1, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", p1)

    # ---------------- figure 2: drift controls ------------------------------
    if tf is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (col, ylab, ttl) in zip(axes, (
        ("entropy_macro", "nats", "Predictive entropy"),
        ("top1_macro", "probability", "Top-1 probability"),
        ("kl_base_macro", "nats", "KL(ckpt || base)"),
    )):
        ax.set_facecolor("#fcfcfb")
        plot_line(ax, *series(tf, col, pos), color=C1, marker="o", label=None)
        style(ax, labels, n_pre, ylab, ttl)
        ax.set_xlabel("GRPO step", color=INK2, fontsize=9)
    if args.title:
        fig.suptitle(args.title + "  —  distribution drift at the probe positions",
                     color=INK, fontsize=11, x=0.01, ha="left")
    p2 = os.path.join(args.out_dir, f"arith_drift_{args.tag}.png")
    fig.savefig(p2, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
