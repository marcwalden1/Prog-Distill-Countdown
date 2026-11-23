#!/usr/bin/env python3
"""
Qwen3 base GRPO: fraction of rollout groups that carry no GRPO signal, per
training step, from the per-step rollout dumps `<exp dir>/<step>.jsonl`
(256 prompts x rollout.n=4 consecutive lines, each with the training reward
`score` in {0, 0.1, 1.0}).

Per group (one prompt's 4 rollouts):
  no signal     all 4 rewards identical -> every advantage is 0 (GRPO here is
                mean-centred, no std normalisation), so the group adds no gradient
  all correct   all 4 rewards == 1.0
  all incorrect all 4 rewards < 1.0. NB a group mixing 0.1 (format ok, wrong)
                and 0.0 (no valid answer) still has a small format-only
                advantage, so "no signal" = all-correct + the all-incorrect
                groups whose rewards are also identical; the CSV keeps that
                split (`frac_all_wrong_same`).
  truncated     share of the rollouts inside all-incorrect groups that hit the
                1024-token training cap (re-tokenized length >= cap - 2; the
                dump stores text, not token counts). Responses < 1000 chars are
                skipped as provably shorter (every token is >= 1 char).

Curves are 50-step window means (single steps are 256 groups, ~3 pt noise).

Usage:
    python3 scripts/plot/plot_qwen3_grpo_group_signal.py
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_qwen3_size_comparison as base  # styling + run list

CKPT = "/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown/checkpoints"
GROUP = 4
CAP = 1024
TOKENIZER = "/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/Qwen3-0.6B-Base"
WINDOW = 50
EPOCH_STEPS = 1633
OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_grpo_group_signal.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "grpo_group_signal.png")


_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOKENIZER)


def step_stats(path):
    recs = [json.loads(l) for l in open(path)]
    n = len(recs) // GROUP
    out = dict(step=int(os.path.basename(path).split(".")[0]), n_groups=n,
               no_signal=0, all_correct=0, all_wrong=0, all_wrong_same=0, misaligned=0,
               wrong_trunc=0)
    long_wrong = []
    for g in range(n):
        grp = recs[g * GROUP:(g + 1) * GROUP]
        if any(r["input"] != grp[0]["input"] for r in grp):
            out["misaligned"] += 1
        s = [r["score"] for r in grp]
        same = max(s) - min(s) < 1e-9
        out["no_signal"] += same
        out["all_correct"] += all(x == 1.0 for x in s)
        wrong = all(x < 1.0 for x in s)
        out["all_wrong"] += wrong
        out["all_wrong_same"] += wrong and same
        if wrong:
            long_wrong += [r["output"] for r in grp if len(r["output"]) >= CAP - 24]
    if long_wrong:
        lens = [len(x) for x in _tok(long_wrong, add_special_tokens=False).input_ids]
        out["wrong_trunc"] = sum(L >= CAP - 2 for L in lens)
    return out


def collect(runs):
    rows = []
    with Pool(16, initializer=_init) as pool:
        for model, exp, label, *_ in runs:
            files = glob.glob(os.path.join(CKPT, model, exp, "*.jsonl"))
            stats = pool.map(step_stats, files, chunksize=16)
            for s in stats:
                rows.append({"model": model, "exp_name": exp, "label": label, **s})
            bad = sum(s["misaligned"] for s in stats)
            print(f"  {label:22s} {len(files)} steps, misaligned groups: {bad}", flush=True)
    df = pd.DataFrame(rows).sort_values(["model", "exp_name", "step"])
    for k in ("no_signal", "all_correct", "all_wrong", "all_wrong_same"):
        df[f"frac_{k}"] = df[k] / df.n_groups
    df["frac_wrong_trunc"] = df.wrong_trunc / (GROUP * df.all_wrong)
    df.to_csv(OUT_CSV, index=False)
    return df


def main():
    runs = base.PRESETS["all"][0]
    df = pd.read_csv(OUT_CSV) if "--cached" in sys.argv and os.path.exists(OUT_CSV) else collect(runs)

    panels = [("frac_no_signal", "no GRPO signal  (all 4 rewards equal)"),
              ("frac_all_correct", "all 4 correct"),
              ("frac_all_wrong", "all 4 incorrect"),
              ("frac_wrong_trunc", "truncated, within all-incorrect groups")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for ax, (col, title) in zip(axes, panels):
        for model, exp, label, color, marker, ls in runs:
            d = df[(df.model == model) & (df.exp_name == exp)].sort_values("step")
            if d.empty:
                continue
            g = d.assign(bin=(d.step - 1) // WINDOW).groupby("bin")
            if col == "frac_wrong_trunc":  # pooled ratio: steps differ in #wrong groups
                b = g.agg(step=("step", "mean"), t=("wrong_trunc", "sum"), w=("all_wrong", "sum"))
                b["v"] = b.t / (GROUP * b.w)
            else:
                b = g.agg(step=("step", "mean"), v=(col, "mean"))
            ax.plot(b.step, b.v, color=color, linestyle=ls, linewidth=2, label=label, zorder=3)
            ax.scatter(b.step.iloc[-1], b.v.iloc[-1], s=60, marker=marker, color=color,
                       edgecolor=base.INK, linewidth=0.8, zorder=4)
            drawn.add(label)
        ax.axvline(EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        base.style_axes(ax, title, "training step",
                        "fraction of groups" if ax is axes[0] else
                        "fraction of rollouts" if col == "frac_wrong_trunc" else "")
    axes[0].set_ylim(0, 1)
    axes[0].text(EPOCH_STEPS, 0.98, " 1 epoch", fontsize=7.5, color=base.INK_MUTED, va="top")
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO (n=3,4, rollout.n=4): groups without GRPO signal "
                 "during training", fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.085, f"training rollouts (temp 1.0), 256 groups/step, {WINDOW}-step means"
             "   ·   correct = reward 1.0   ·   truncated = >= 1022 of the 1024-token training cap"
             "   ·   marker = last step", ha="center",
             fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
