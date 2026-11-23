#!/usr/bin/env python3
"""
Arithmetic fidelity of Qwen3 base GRPO generations across training, from the
stored eval JSONs (cap 1024, temp 0.6), via scripts/probe_arith_chains.py.

Per (run, step, dataset), over all prompts for n=3,4 and a seeded 1000-prompt
subset for n=5/n=6, the first 8 of 32 samples each:

  link error rate    wrong `a = b` claims / all claims
  chain error rate   chains whose last value != true value of their first expr
  false-hit rate     chains whose expr does NOT equal the target but whose
                     claimed value is the target, / such chains
  links / response   how much arithmetic is shown at all (drift control: a rate
                     over fewer, simpler claims is not the same skill)

Step 0 = zero-shot base model (results/baselines/<M>/, n=3,4 only).

Usage (CPU; sbatch on serial_requeue, ~30 min on 32 cores):
    sbatch -p serial_requeue --account=bingbin_lab -c 32 --mem=96G -t 3:00:00 \
        -o logs/arith_chains-%j.out \
        --wrap "~/.conda/envs/verl/bin/python3 scripts/plot/plot_qwen3_arith_chains.py"
    python3 scripts/plot/plot_qwen3_arith_chains.py --cached     # replot from CSV
"""
import hashlib
import json
import os
import random
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
from probe_arith_chains import probe_text

DATASETS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
N_SAMPLES = 8
N_PROMPTS_OOD = 1000
CACHE = os.path.join(REPO, "analysis_data", ".arith_chain_cache")
OUT_CSV = os.path.join(REPO, "analysis_data", "qwen3_arith_chains.csv")
OUT_FIG = os.path.join(REPO, "figures", "Qwen3-base-grpo", "arith_fidelity.png")
EPOCH_STEPS = 1633
KEYS = ["links", "link_err", "chains", "chain_err", "false_hit", "wrong_val_chains",
        "missed_hit", "target_val_chains"]


def score_file(task):
    path, ds = task
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime}|{st.st_size}|{N_SAMPLES}|{N_PROMPTS_OOD}|v3"
                       .encode()).hexdigest()
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    data = json.load(open(path))
    if ds != "balanced-simple" and len(data) > N_PROMPTS_OOD:
        data = [data[i] for i in sorted(random.Random(0).sample(range(len(data)), N_PROMPTS_OOD))]
    tot = dict.fromkeys(KEYS, 0)
    tot.update(n_responses=0, resp_with_err=0)
    for p in data:
        for o in p["outputs"][:N_SAMPLES]:
            r = probe_text(o, p["target"])
            for k in KEYS:
                tot[k] += r[k]
            tot["n_responses"] += 1
            tot["resp_with_err"] += r["link_err"] > 0
    os.makedirs(CACHE, exist_ok=True)
    with open(cp + ".tmp", "w") as f:
        json.dump(tot, f)
    os.replace(cp + ".tmp", cp)
    return tot


def tasks(runs):
    out = []
    for model, exp, label, *_ in runs:
        for ds, _ in DATASETS:
            if "-ep2-" not in exp or True:
                bl = os.path.join(REPO, "results", "baselines", model, f"{ds}_temp0.6_n32_max1024.json")
                if os.path.exists(bl):
                    out.append((model, exp, label, ds, 0, bl))
            root = os.path.join(REPO, "results", model, exp)
            for d in sorted(os.listdir(root)):
                if not d.startswith("global_step_"):
                    continue
                p = os.path.join(root, d, f"{ds}_temp0.6_n32_max1024.json")
                if os.path.exists(p):
                    out.append((model, exp, label, ds, int(d.split("_")[-1]), p))
    return out


def collect(runs):
    ts = tasks(runs)
    print(f"{len(ts)} files", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        res = pool.map(score_file, [(t[5], t[3]) for t in ts], chunksize=1)
    rows = [dict(model=m, exp_name=e, label=l, dataset=ds, step=s, **r)
            for (m, e, l, ds, s, _), r in zip(ts, res)]
    df = pd.DataFrame(rows).sort_values(["model", "exp_name", "dataset", "step"])
    df["link_err_rate"] = df.link_err / df.links
    df["chain_err_rate"] = df.chain_err / df.chains
    df["false_hit_rate"] = df.false_hit / df.wrong_val_chains
    df["links_per_resp"] = df.links / df.n_responses
    df["resp_err_rate"] = df.resp_with_err / df.n_responses
    df.to_csv(OUT_CSV, index=False)
    return df


def main():
    runs = base.PRESETS["all"][0]
    df = pd.read_csv(OUT_CSV) if "--cached" in sys.argv and os.path.exists(OUT_CSV) else collect(runs)

    cols = [("link_err_rate", "wrong `=` claims", "fraction of claims"),
            ("chain_err_rate", "chains ending at a wrong value", "fraction of chains"),
            ("false_hit_rate", "false hit: claims target, isn't", "fraction of non-target chains"),
            ("links_per_resp", "`=` claims per response", "claims")]
    fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(19, 12.5), sharex=True)
    fig.patch.set_facecolor(base.SURFACE)
    drawn = set()
    for r, (ds, dname) in enumerate(DATASETS):
        for c, (col, title, ylab) in enumerate(cols):
            ax = axes[r][c]
            for model, exp, label, color, marker, ls in runs:
                d = df[(df.model == model) & (df.exp_name == exp) & (df.dataset == ds)].sort_values("step")
                if d.empty:
                    continue
                ax.plot(d.step, d[col], color=color, linestyle=ls, linewidth=1.8, marker=marker,
                        markersize=4, markeredgecolor=base.SURFACE, markeredgewidth=0.8,
                        label=label, zorder=3)
                drawn.add(label)
            ax.axvline(EPOCH_STEPS, color="#cfcec9", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
            base.style_axes(ax, title if r == 0 else "", "training step" if r == len(DATASETS) - 1 else "",
                            f"{dname}\n{ylab}" if c == 0 else ylab)
            if c < 3:
                ax.set_ylim(bottom=0)
    # rates share y within a column so datasets compare directly
    for c in range(len(cols)):
        hi = max(axes[r][c].get_ylim()[1] for r in range(len(DATASETS)))
        for r in range(len(DATASETS)):
            axes[r][c].set_ylim(0, hi)
    handles, labels = base.run_legend(runs, drawn)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, labelcolor=base.INK_MUTED, bbox_to_anchor=(0.5, 0.005),
               handlelength=3)
    fig.suptitle("Qwen3 base GRPO: arithmetic fidelity of `expr = ... = value` chains in generations",
                 fontsize=12.5, color=base.INK, y=0.995)
    fig.text(0.5, 0.04, f"eval samples, cap 1024, temp 0.6, first {N_SAMPLES} of 32 per prompt "
             f"(n=5/6: {N_PROMPTS_OOD} seeded prompts)   ·   step 0 = zero-shot base (n=3,4 only)"
             "   ·   cut-off last line of truncated responses dropped", ha="center",
             fontsize=8.5, color=base.INK_MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.975])
    fig.savefig(OUT_FIG, dpi=200, facecolor=base.SURFACE)
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
