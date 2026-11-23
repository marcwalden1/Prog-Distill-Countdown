#!/usr/bin/env python3
"""
Qwen3 base GRPO, n=3,4: mean response length (tokens) by outcome, across
training, at generation caps 1024 and 2048.

    rows    = cap 1024, cap 2048
    columns = correct | correct & finished | incorrect

"Correct & finished" excludes responses that hit the cap (>= cap - 2 tokens):
these models open with a guessed <answer>, so a response whose first guess was
right and that then rambled to the cap is graded correct (the grader takes the
last <answer>, which is that one) -- it would inflate "correct" length.

Reads the per-response cache written by plot_qwen3_eval_group_signal.py
(analysis_data/.eval_group_cache/); run that first. CPU only, seconds.

Usage:
    python3 scripts/plot/plot_qwen3_correct_lengths.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_qwen3_size_comparison as base
import plot_qwen3_trunc_pass1_vs_pass32 as tr
import plot_qwen3_eval_group_signal as eg

REPO = tr.REPO
OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_n34_length_by_outcome.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "length_by_outcome_n34.png")


def collect(runs):
    rows = []
    for model, exp, label, *_ in runs:
        steps = tr.STEPS_2EP if "-ep2-" in exp else tr.STEPS_1EP
        for cap in tr.CAPS:
            for step in steps:
                p = tr.result_path(model, exp, step, cap)
                if not os.path.exists(p):
                    continue
                res = eg.per_response(p, pool=None)  # cache hit; pool unused
                c, cf, w, a = [], [], [], []
                for scores, lens in res:
                    for s, L in zip(scores, lens):
                        a.append(L)
                        if s == 1.0:
                            c.append(L)
                            if L < cap - 2:
                                cf.append(L)
                        else:
                            w.append(L)
                mean = lambda x: sum(x) / len(x) if x else float("nan")
                med = lambda x: sorted(x)[len(x) // 2] if x else float("nan")
                rows.append(dict(model=model, exp_name=exp, label=label, cap=cap, step=step,
                                 n_correct=len(c), n_correct_fin=len(cf), n_wrong=len(w),
                                 len_correct=mean(c), len_correct_fin=mean(cf),
                                 len_wrong=mean(w), len_all=mean(a),
                                 med_correct_fin=med(cf),
                                 frac_correct_truncated=1 - len(cf) / len(c) if c else float("nan")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def main():
    runs = base.PRESETS["all"][0]
    df = collect(runs)
    cols = [("len_correct", "correct"), ("len_correct_fin", "correct & finished (not truncated)"),
            ("len_wrong", "incorrect")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), sharex=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for r, cap in enumerate(tr.CAPS):
        for c, (col, title) in enumerate(cols):
            ax = axes[r][c]
            for model, exp, label, color, marker, ls in runs:
                d = df[(df.model == model) & (df.exp_name == exp) & (df.cap == cap)].sort_values("step")
                if d.empty:
                    continue
                ax.plot(d.step, d[col], color=color, linestyle=ls, linewidth=2, marker=marker,
                        markersize=5.5, markeredgecolor=base.SURFACE, markeredgewidth=1,
                        label=label, zorder=3)
                drawn.add(label)
            ax.axvline(tr.EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
            if c == 2:
                ax.axhline(cap, color="#cfcec9", linewidth=1, zorder=1)
            base.style_axes(ax, title if r == 0 else "", "training step" if r == 1 else "",
                            f"cap {cap}\nmean length (tokens)" if c == 0 else "")
    # correct columns: zoomed, one range shared by all four panels so caps and
    # the finished filter compare directly; incorrect column: 0..cap
    lo = min(axes[r][c].get_ylim()[0] for r in range(2) for c in range(2))
    hi = max(axes[r][c].get_ylim()[1] for r in range(2) for c in range(2))
    for r, cap in enumerate(tr.CAPS):
        for c in range(2):
            axes[r][c].set_ylim(lo, hi)
        axes[r][2].set_ylim(0, cap * 1.04)
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO, n=3,4: response length by outcome", fontsize=12.5,
                 color=base.INK, y=0.99)
    fig.text(0.5, 0.06, "eval samples (997 prompts x 32, temp 0.6), checkpoints every 200 steps"
             "   ·   tokens re-tokenized from text   ·   solid gray line = cap (incorrect column)"
             "   ·   correct = reward 1.0", ha="center", fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 0.965])
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
