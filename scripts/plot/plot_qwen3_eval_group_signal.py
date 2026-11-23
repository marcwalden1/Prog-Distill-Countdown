#!/usr/bin/env python3
"""
GRPO-group signal at generation caps 1024 vs 2048, estimated from EVAL samples.

The training-rollout version (plot_qwen3_grpo_group_signal.py) exists only at
the 1024 training cap. To see what a 2048 cap would do, this uses the n=3,4 eval
files instead: per test prompt, the 32 samples (temp 0.6) are split into 8
pseudo-groups of 4 (consecutive samples, i.e. rollout.n=4), at both caps, for
checkpoints every 200 steps. Both rows come from eval data so they differ only
in the cap. Expect higher "no signal" than in training: temp 0.6 < 1.0 makes the
4 samples agree more often, and these are test prompts.

    rows    = cap 1024, cap 2048
    columns = no signal (4 equal rewards) | all 4 correct | all 4 incorrect |
              truncated share of rollouts inside all-incorrect groups

Reward = compute_score (1.0 / 0.1 / 0.0), i.e. the training reward. Truncated =
re-tokenized length >= cap - 2. Per-response scores and lengths are cached in
analysis_data/.eval_group_cache/ (one file per eval JSON), so a preempted job
resumes and later analyses can skip re-scoring.

Usage (CPU; ~1 h on 32 cores):
    sbatch -p serial_requeue --account=bingbin_lab -c 32 --mem=64G -t 4:00:00 \
        -o logs/eval_group_signal-%j.out \
        --wrap "~/.conda/envs/verl/bin/python3 scripts/plot/plot_qwen3_eval_group_signal.py"
    python3 scripts/plot/plot_qwen3_eval_group_signal.py --cached   # replot from CSV
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

import plot_qwen3_size_comparison as base
import plot_qwen3_trunc_pass1_vs_pass32 as tr  # result_path, steps, caps, tokenizer

GROUP = 4
CACHE = os.path.join(REPO, "analysis_data", ".eval_group_cache")
OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_eval_group_signal.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "grpo_group_signal_eval_caps.png")
EPOCH_STEPS = 1633

_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(tr.TOKENIZER)


def _score_prompt(p):
    outs = p["outputs"]
    lens = [len(x) for x in _tok(outs, add_special_tokens=False).input_ids]
    scores = []
    for o in outs:
        with contextlib.redirect_stdout(io.StringIO()):
            scores.append(compute_score("countdown", o, p["target"], {"numbers": p["nums"]}))
    return scores, lens


from grader_utils import compute_score  # noqa: E402  (after sys.path setup)


def per_response(path, pool):
    """[(scores[32], lens[32]) per prompt], cached by path+mtime+size."""
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime}|{st.st_size}".encode()).hexdigest()
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    data = json.load(open(path))
    res = pool.map(_score_prompt, data, chunksize=8)
    os.makedirs(CACHE, exist_ok=True)
    with open(cp + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(cp + ".tmp", cp)
    return res


def group_stats(res, cap):
    n = no_sig = all_c = all_w = w_roll = w_trunc = 0
    for scores, lens in res:
        for g in range(0, len(scores) - GROUP + 1, GROUP):
            s, L = scores[g:g + GROUP], lens[g:g + GROUP]
            n += 1
            no_sig += max(s) - min(s) < 1e-9
            all_c += all(x == 1.0 for x in s)
            if all(x < 1.0 for x in s):
                all_w += 1
                w_roll += GROUP
                w_trunc += sum(l >= cap - 2 for l in L)
    return dict(n_groups=n, no_signal=no_sig, all_correct=all_c, all_wrong=all_w,
                wrong_trunc=w_trunc, frac_no_signal=no_sig / n, frac_all_correct=all_c / n,
                frac_all_wrong=all_w / n,
                frac_wrong_trunc=w_trunc / w_roll if w_roll else float("nan"))


def collect(runs):
    rows = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16)), initializer=_init) as pool:
        for model, exp, label, *_ in runs:
            steps = tr.STEPS_2EP if "-ep2-" in exp else tr.STEPS_1EP
            for cap in tr.CAPS:
                for step in steps:
                    p = tr.result_path(model, exp, step, cap)
                    if not os.path.exists(p):
                        print(f"  missing: {p}", flush=True)
                        continue
                    r = group_stats(per_response(p, pool), cap)
                    print(f"  {label:22s} cap={cap} step={step:5d} nosig={r['frac_no_signal']:.3f} "
                          f"allC={r['frac_all_correct']:.3f} allW={r['frac_all_wrong']:.3f} "
                          f"truncW={r['frac_wrong_trunc']:.3f}", flush=True)
                    rows.append({"model": model, "exp_name": exp, "label": label,
                                 "cap": cap, "step": step, **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def main():
    runs = base.PRESETS["all"][0]
    df = pd.read_csv(OUT_CSV) if "--cached" in sys.argv and os.path.exists(OUT_CSV) else collect(runs)

    panels = [("frac_no_signal", "no GRPO signal  (all 4 rewards equal)", "fraction of groups"),
              ("frac_all_correct", "all 4 correct", "fraction of groups"),
              ("frac_all_wrong", "all 4 incorrect", "fraction of groups"),
              ("frac_wrong_trunc", "truncated, within all-incorrect groups", "fraction of rollouts")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for r, cap in enumerate(tr.CAPS):
        for c, (col, title, ylab) in enumerate(panels):
            ax = axes[r][c]
            for model, exp, label, color, marker, ls in runs:
                d = df[(df.model == model) & (df.exp_name == exp) & (df.cap == cap)].sort_values("step")
                if d.empty:
                    continue
                ax.plot(d.step, d[col], color=color, linestyle=ls, linewidth=2, marker=marker,
                        markersize=5.5, markeredgecolor=base.SURFACE, markeredgewidth=1,
                        label=label, zorder=3)
                drawn.add(label)
            ax.axvline(EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
            base.style_axes(ax, f"{title}" if r == 0 else "", "training step" if r == 1 else "",
                            f"cap {cap}\n{ylab}" if c == 0 else (ylab if c == 3 else ""))
    axes[0][0].set_ylim(0, 1)
    axes[0][0].text(EPOCH_STEPS, 0.98, " 1 epoch", fontsize=7.5, color=base.INK_MUTED, va="top")
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO, n=3,4: GRPO-group signal at generation cap 1024 vs 2048 "
                 "(estimated from eval samples)", fontsize=12.5, color=base.INK, y=0.99)
    fig.text(0.5, 0.05, "eval samples (test prompts, temp 0.6): 32 per prompt split into 8 groups "
             "of 4   ·   checkpoints every 200 steps   ·   reward = training reward (1 / 0.1 / 0)"
             "   ·   truncated = >= cap - 2 tokens", ha="center", fontsize=8.5,
             color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.065, 1, 0.965])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
