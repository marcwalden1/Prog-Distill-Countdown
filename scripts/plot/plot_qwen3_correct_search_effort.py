#!/usr/bin/env python3
"""
Qwen3 base GRPO, n=3,4: search effort vs verbosity in CORRECT, finished responses.

Splits response length into "how many things were tried" and "how many tokens
each try costs", to separate search strategy from writing style:

  candidates / response   chains (scripts/probe_arith_chains.py) whose first
                          expression uses exactly the puzzle's numbers -- one
                          full candidate solution evaluated
  `=` claims / response   every arithmetic step written out
  tokens / candidate      length / max(candidates, 1)  (verbosity per attempt)
  tokens / claim          length / max(claims, 1)

Only responses with reward 1.0 that did not hit the cap. Lengths and scores
come from the per-response cache of plot_qwen3_eval_group_signal.py (same
order as the eval JSON); text from the eval JSON.

Also writes a prompt-matched table at the last 1-epoch step (prompts every
1-epoch run solves), since larger models also solve harder prompts.

Usage (CPU; sbatch, a few minutes on 32 cores):
    sbatch -p serial_requeue --account=bingbin_lab -c 32 --mem=96G -t 2:00:00 \
        -o logs/search_effort-%j.out \
        --wrap "~/.conda/envs/verl/bin/python3 scripts/plot/plot_qwen3_correct_search_effort.py"
"""
import json
import os
import re
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_qwen3_size_comparison as base
import plot_qwen3_trunc_pass1_vs_pass32 as tr
import plot_qwen3_eval_group_signal as eg
from probe_arith_chains import chains

OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_n34_correct_search_effort.csv")
OUT_MATCHED = os.path.join(REPO, "analysis_data", "qwen3_n34_correct_search_effort_matched.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "length_by_outcome_search_effort_n34.png")
NUM = re.compile(r"\d+(?:\.\d+)?")


def effort(text, nums):
    want = sorted(int(n) for n in nums)
    cand = claims = 0
    for ch in chains(text):
        claims += len(ch) - 1
        got = NUM.findall(ch[0][0])
        if all("." not in g for g in got) and sorted(int(g) for g in got) == want:
            cand += 1
    return cand, claims


def file_task(args):
    model, exp, label, cap, step, path = args
    res = eg.per_response(path, pool=None)
    data = json.load(open(path))
    per_prompt = []  # (prompt idx, [(tokens, cand, claims), ...]) for correct finished
    for i, (p, (scores, lens)) in enumerate(zip(data, res)):
        rs = []
        for o, s, L in zip(p["outputs"], scores, lens):
            if s == 1.0 and L < cap - 2:
                c, k = effort(o, p["nums"])
                rs.append((L, c, k))
        per_prompt.append(rs)
    return (model, exp, label, cap, step), per_prompt


def summarize(rs):
    n = len(rs)
    if not n:
        return {}
    return dict(n_resp=n,
                tokens=sum(r[0] for r in rs) / n,
                candidates=sum(r[1] for r in rs) / n,
                claims=sum(r[2] for r in rs) / n,
                tok_per_cand=sum(r[0] / max(r[1], 1) for r in rs) / n,
                tok_per_claim=sum(r[0] / max(r[2], 1) for r in rs) / n,
                frac_zero_cand=sum(r[1] == 0 for r in rs) / n)


def main():
    runs = base.PRESETS["all"][0]
    ts = []
    for model, exp, label, *_ in runs:
        steps = tr.STEPS_2EP if "-ep2-" in exp else tr.STEPS_1EP
        for cap in tr.CAPS:
            for step in steps:
                p = tr.result_path(model, exp, step, cap)
                if os.path.exists(p):
                    ts.append((model, exp, label, cap, step, p))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        out = pool.map(file_task, ts, chunksize=1)

    rows, by_key = [], {}
    for (model, exp, label, cap, step), pp in out:
        by_key[(label, cap, step)] = pp
        rows.append(dict(model=model, exp_name=exp, label=label, cap=cap, step=step,
                         **summarize([r for rs in pp for r in rs])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # prompt-matched at step 1600 (1-epoch runs)
    mrows = []
    labels = [l for _, e, l, *_ in runs if "-ep2-" not in e]
    for cap in tr.CAPS:
        common = [i for i in range(997) if all(by_key[(l, cap, 1600)][i] for l in labels)]
        for l in labels:
            pp = by_key[(l, cap, 1600)]
            # mean over prompts of per-prompt means: each prompt weighs the same
            per = [summarize(pp[i]) for i in common]
            mrows.append(dict(label=l, cap=cap, n_prompts=len(common),
                              **{k: sum(p[k] for p in per) / len(per)
                                 for k in ("tokens", "candidates", "claims",
                                           "tok_per_cand", "tok_per_claim")}))
    m = pd.DataFrame(mrows)
    m.to_csv(OUT_MATCHED, index=False)
    print(m.round(2).to_string())

    cols = [("tokens", "tokens / response"), ("candidates", "candidates tried / response"),
            ("claims", "`=` claims / response"), ("tok_per_cand", "tokens / candidate")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.4), sharex=True)
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
            base.style_axes(ax, title if r == 0 else "", "training step" if r == 1 else "",
                            f"cap {cap}\nper correct response" if c == 0 else "")
    for c in range(4):
        hi = max(axes[r][c].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r][c].set_ylim(0, hi)
    handles, labels_ = base.run_legend(runs, drawn)
    fig.legend(handles, labels_, loc="lower center", ncol=len(labels_), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO, n=3,4: search effort vs verbosity in correct, finished responses",
                 fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.06, "candidate = an evaluated expression using exactly the puzzle's numbers"
             "   ·   eval samples (997 prompts x 32, temp 0.6), every 200 steps"
             "   ·   not prompt-matched (see *_matched.csv)", ha="center", fontsize=8.5,
             color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 0.965])
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}\nwrote {OUT_MATCHED}")


if __name__ == "__main__":
    main()
