#!/usr/bin/env python3
"""
Qwen3 base GRPO, n=3,4 (balanced-simple): pass@1 vs pass@32 at two generation
caps, over all responses vs over non-truncated responses only.

    rows    = generation cap (1024, 2048)
    columns = all responses | non-truncated responses only

One trajectory per run (every size x lr in {1e-6, 3e-6}, plus 0.6B lr=1e-6 for
2 epochs), styled as the "all" preset of plot_qwen3_size_comparison.py.

Metrics, per (run, step, cap), over the full n=3,4 test set (997 prompts x 32):
  all responses     pass@1  = mean over prompts of c/n
                    pass@32 = fraction of prompts with >=1 correct of 32
  non-truncated     the same, with each prompt's responses restricted to those
                    that did not hit the cap: pass@1 = mean of c_nt/n_nt, pass@32 =
                    fraction with >=1 correct non-truncated response. Prompts
                    with no non-truncated response are dropped (their count is
                    printed and in the CSV). With n_nt < 32 the second number is
                    "any correct among the finished ones", not a strict pass@32.
"Truncated" = response re-tokenizes to >= cap - 2 tokens (vLLM's exact count is
not stored; re-tokenizing the decoded text can differ by a token or so).
"Correct" = compute_score == 1.0.

Per-file results are cached in analysis_data/.trunc_passk_cache/ (path+mtime+size).

Usage:
    python3 scripts/plot/plot_qwen3_trunc_pass1_vs_pass32.py
"""
import contextlib
import hashlib
import io
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_qwen3_size_comparison as base  # styling + run list
from grader_utils import compute_score

TOKENIZER = "/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/Qwen3-0.6B-Base"
DATASET = "balanced-simple"
CAPS = [1024, 2048]
STEPS_1EP = list(range(200, 1601, 200))
STEPS_2EP = list(range(200, 3201, 200))
CACHE = os.path.join(REPO, "analysis_data", ".trunc_passk_cache")
OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_n34_trunc_passk.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "pass1_vs_pass32_n34_trunc.png")
OUT_FIG_STEP = os.path.join(REPO, "figures", "Qwen3-base-grpo", "pass_vs_step_n34_trunc.png")
EPOCH_STEPS = 1633  # 1 epoch of balanced-simple at batch 256

_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOKENIZER)


def _score_prompt(args):
    p, cap = args
    outs = p["outputs"]
    lens = [len(x) for x in _tok(outs, add_special_tokens=False).input_ids]
    rows = []
    for o, L in zip(outs, lens):
        with contextlib.redirect_stdout(io.StringIO()):
            s = compute_score("countdown", o, p["target"], {"numbers": p["nums"]})
        rows.append((s == 1.0, L >= cap - 2))
    return rows


def result_path(model, exp, step, cap):
    d = os.path.join(REPO, "results", model, exp, f"global_step_{step}")
    if cap != 1024:
        d = os.path.join(d, f"max{cap}")
    return os.path.join(d, f"{DATASET}_temp0.6_n32_max{cap}.json")


def score_file(path, cap, pool):
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime}|{st.st_size}|v1".encode()).hexdigest()
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    data = json.load(open(path))
    per_prompt = pool.map(_score_prompt, [(p, cap) for p in data], chunksize=8)
    n_resp = sum(len(r) for r in per_prompt)
    rec = {
        "n_prompts": len(per_prompt),
        "trunc_rate": sum(t for r in per_prompt for _, t in r) / n_resp,
        "pass_at_1": sum(sum(c for c, _ in r) / len(r) for r in per_prompt) / len(per_prompt),
        "pass_at_32": sum(any(c for c, _ in r) for r in per_prompt) / len(per_prompt),
    }
    nt = [[c for c, t in r if not t] for r in per_prompt]
    nt = [r for r in nt if r]
    rec["nt_n_prompts"] = len(nt)
    rec["nt_pass_at_1"] = sum(sum(r) / len(r) for r in nt) / len(nt) if nt else float("nan")
    rec["nt_pass_at_32"] = sum(any(r) for r in nt) / len(nt) if nt else float("nan")
    os.makedirs(CACHE, exist_ok=True)
    json.dump(rec, open(cp, "w"))
    return rec


def collect(runs):
    rows = []
    with Pool(16, initializer=_init) as pool:
        for model, exp, label, *_ in runs:
            steps = STEPS_2EP if "-ep2-" in exp else STEPS_1EP
            for cap in CAPS:
                for step in steps:
                    p = result_path(model, exp, step, cap)
                    if not os.path.exists(p):
                        print(f"  missing: {p}")
                        continue
                    rec = score_file(p, cap, pool)
                    print(f"  {label:22s} cap={cap} step={step:5d} trunc={rec['trunc_rate']:.3f} "
                          f"p1={rec['pass_at_1']:.3f} p32={rec['pass_at_32']:.3f} "
                          f"nt_p1={rec['nt_pass_at_1']:.3f} nt_p32={rec['nt_pass_at_32']:.3f} "
                          f"(nt prompts {rec['nt_n_prompts']}/{rec['n_prompts']})", flush=True)
                    rows.append({"model": model, "exp_name": exp, "label": label,
                                 "cap": cap, "dataset": str(cap), "step": step, **rec})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def fig_vs_step(df, runs):
    """Accuracy vs training step: columns = pass@1/@32 x cap, rows = all /
    non-truncated responses. y is shared within each metric (all 4 panels), so
    the cap and the truncation filter both read as vertical offsets."""
    cols = [("pass_at_1", "pass@1", 1024), ("pass_at_1", "pass@1", 2048),
            ("pass_at_32", "pass@32", 1024), ("pass_at_32", "pass@32", 2048)]
    rows = [("", "all responses"), ("nt_", "non-truncated responses only")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), sharex=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for r, (pre, rname) in enumerate(rows):
        for c, (metric, mname, cap) in enumerate(cols):
            ax = axes[r][c]
            col = pre + metric
            for model, exp, label, color, marker, ls in runs:
                d = df[(df.exp_name == exp) & (df.model == model) & (df.cap == cap)].sort_values("step")
                if d.empty:
                    continue
                ax.plot(d.step, d[col], color=color, linestyle=ls, linewidth=2,
                        marker=marker, markersize=5.5, markeredgecolor=base.SURFACE,
                        markeredgewidth=1, label=label, zorder=3)
                drawn.add(label)
            ax.axvline(EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
            if r == 0 and c == 0:
                ax.text(EPOCH_STEPS, 0.02, " 1 epoch", transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color=base.INK_MUTED, va="bottom")
            sub = df[df.cap == cap]
            note = (f"truncated: {sub.trunc_rate.min():.0%}-{sub.trunc_rate.max():.0%}" if r == 0 else
                    f"prompts kept: {(sub.nt_n_prompts / sub.n_prompts).min():.0%}-"
                    f"{(sub.nt_n_prompts / sub.n_prompts).max():.0%}")
            ylabel = ""
            if c == 0:
                ylabel = f"{rname}\naccuracy"
            ytxt = mname if r == 0 or metric == "pass_at_1" else "any correct among finished"
            base.style_axes(ax, f"{ytxt}  ·  cap {cap}", "training step" if r == 1 else "",
                            ylabel, note)
    # shared y per metric: columns 0-1 (pass@1) and 2-3 (pass@32), both rows
    for cs in ((0, 1), (2, 3)):
        lo = min(axes[r][c].get_ylim()[0] for r in range(2) for c in cs)
        hi = max(axes[r][c].get_ylim()[1] for r in range(2) for c in cs)
        for r in range(2):
            for c in cs:
                axes[r][c].set_ylim(lo, min(hi, 1.005))
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO, n=3,4 (balanced-simple): accuracy vs training step, "
                 "by generation cap and truncation", fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.055, "checkpoints every 200 steps   ·   rows share y within pass@1 and within "
             "pass@32   ·   non-truncated: prompts with no finished response dropped",
             ha="center", fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.07, 1, 0.965])
    fig.savefig(OUT_FIG_STEP, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG_STEP}")


def main():
    runs = base.PRESETS["all"][0]
    df = collect(runs)
    fig_vs_step(df, runs)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.6))
    fig.patch.set_facecolor(base.SURFACE)
    cols = [(("pass_at_1", "pass_at_32"), "all responses"),
            (("nt_pass_at_1", "nt_pass_at_32"), "non-truncated responses only")]
    for r, cap in enumerate(CAPS):
        for c, (mcols, cname) in enumerate(cols):
            ax = axes[r][c]
            drawn = base.draw_curves(ax, df, runs, str(cap), mcols,
                                     annotate=False, label_last=False)
            if drawn:
                base.diagonal(ax)
            sub = df[df.cap == cap]
            note = (f"responses truncated: {sub.trunc_rate.min():.0%}–{sub.trunc_rate.max():.0%}"
                    if c == 0 else
                    f"prompts with >=1 finished response: "
                    f"{(sub.nt_n_prompts / sub.n_prompts).min():.0%}–"
                    f"{(sub.nt_n_prompts / sub.n_prompts).max():.0%}")
            base.style_axes(ax, f"cap {cap} tokens  ·  {cname}", "pass@1",
                            "pass@32" if c == 0 else "any correct among finished",
                            note)
    # same axes within each column so the cap rows compare directly
    for c in range(2):
        x0 = min(axes[r][c].get_xlim()[0] for r in range(2))
        x1 = max(axes[r][c].get_xlim()[1] for r in range(2))
        y0 = min(axes[r][c].get_ylim()[0] for r in range(2))
        y1 = max(axes[r][c].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r][c].set_xlim(x0, x1)
            axes[r][c].set_ylim(y0, y1)

    drawn_labels = {l for a in axes.flat for l in a.get_legend_handles_labels()[1]}
    handles, labels = base.run_legend(runs, drawn_labels)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO, n=3,4 (balanced-simple): pass@1 vs pass@32 "
                 "by generation cap", fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.075, "checkpoints every 200 steps   ·   open ring = first (step 200)"
             "   ·   ink-edged marker = last   ·   dashed = y=x", ha="center",
             fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.095, 1, 0.965])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
