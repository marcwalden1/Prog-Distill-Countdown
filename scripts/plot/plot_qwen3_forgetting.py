"""Per-problem consistency of Qwen3-Base GRPO (size sweep): which problems got
significantly better / worse between an intermediate checkpoint and the final one.

For each problem we have #correct out of 32 samples at every checkpoint
(analysis_data/.qwen3_size_cache, written by qwen3_size_curve_data.py; row order is
identical across checkpoints). A problem counts as "worse at the end" if a one-sided
Fisher exact test of k_A/32 vs k_final/32 gives p < ALPHA (and "better" symmetrically).
Unlike "sample #1 right then wrong", this doesn't count sampling noise as forgetting:
under no real change, < ALPHA of tested problems are flagged.

Also reported (CSV only): the majority-threshold view (>=16/32 at step A, <16/32 at the
end) with its exact no-change expectation (pool the 64 samples, hypergeometric split).

Run with the verl env (needs numpy/scipy/matplotlib):
    ~/.conda/envs/verl/bin/python3 scripts/plot/plot_qwen3_forgetting.py
"""
import csv, json, os
import numpy as np
from scipy.stats import fisher_exact, hypergeom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "analysis_data/.qwen3_size_cache"
RUNS = {  # same selection as research_logs/2026-11-05_qwen3-size-overlap-and-best-ckpt.md
    "0.6B": "Qwen3-0.6B-Base__balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
    "1.7B": "Qwen3-1.7B-Base__balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
    "4B":   "Qwen3-4B-Base__balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
}
DS = {"n=3,4 (in-dist)": "balanced-simple", "n=5 (OOD)": "balanced5-simple", "n=6 (OOD)": "balanced6-simple"}
FROM_STEPS = [400, 800, 1200]
FINAL = 1600
ALPHA = 0.01
COLORS = {"0.6B": "#2a78d6", "1.7B": "#eb6834", "4B": "#1baf7a"}  # categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT_PNG = "figures/Qwen3-base-grpo/consistency_vs_step1600.png"
OUT_CSV = "analysis_data/qwen3_consistency_vs_step1600.csv"


def load(run, step, ds):
    return np.array(json.load(open(f"{CACHE}/{run}__{step}__{ds}.json")))


def sig_counts(a, b):
    worse = better = 0
    for x, y in zip(a, b):
        if x > y and fisher_exact([[x, 32 - x], [y, 32 - y]], alternative="greater")[1] < ALPHA:
            worse += 1
        elif y > x and fisher_exact([[y, 32 - y], [x, 32 - x]], alternative="greater")[1] < ALPHA:
            better += 1
    return worse, better


def null_lost_majority(a, b):
    """E[#(k_A>=16 and k_B<16)] if the 64 samples were exchangeable."""
    tot, ks = 0.0, np.arange(33)
    for T in np.unique(a + b):
        pk = hypergeom(64, T, 32).pmf(ks)
        tot += np.sum(a + b == T) * pk[(ks >= 16) & (T - ks < 16) & (T - ks >= 0)].sum()
    return tot


rows = []
for m, run in RUNS.items():
    for dlabel, ds in DS.items():
        b = load(run, FINAL, ds)
        for s in FROM_STEPS:
            a = load(run, s, ds)
            w, g = sig_counts(a, b)
            rows.append(dict(model=m, dataset=ds, from_step=s, to_step=FINAL, n_problems=len(a),
                             pass1_from=round(a.mean() / 32, 4), pass1_to=round(b.mean() / 32, 4),
                             sig_worse=w, sig_better=g,
                             sig_worse_pct=round(100 * w / len(a), 2), sig_better_pct=round(100 * g / len(a), 2),
                             maj_from=int((a >= 16).sum()), maj_lost=int(((a >= 16) & (b < 16)).sum()),
                             maj_lost_null=round(null_lost_majority(a, b), 1),
                             maj_gained=int(((a < 16) & (b >= 16)).sum()),
                             maj_to_zero=int(((a >= 16) & (b == 0)).sum())))
            print({k: rows[-1][k] for k in ("model", "dataset", "from_step", "sig_worse", "sig_better",
                                            "maj_lost", "maj_lost_null", "maj_gained")}, flush=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)

# ---------------------------------------------------------------- plot
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
fig.patch.set_facecolor("#fcfcfb")
models = list(RUNS)
bw, gap = 0.26, 0.02
for ax, (dlabel, ds) in zip(axes, DS.items()):
    ax.set_facecolor("#fcfcfb")
    sub = [r for r in rows if r["dataset"] == ds]
    ymax = max(r["sig_better_pct"] for r in sub) * 1.22
    ymin = max(r["sig_worse_pct"] for r in sub) * 1.9
    for i, s in enumerate(FROM_STEPS):
        for j, m in enumerate(models):
            r = next(r for r in sub if r["model"] == m and r["from_step"] == s)
            x = i + (j - 1) * (bw + gap)
            up, dn = r["sig_better_pct"], r["sig_worse_pct"]
            ax.bar(x, up, bw, color=COLORS[m], alpha=0.35, linewidth=0)
            ax.bar(x, -dn, bw, color=COLORS[m], linewidth=0)
            ax.text(x, up + ymax * 0.02, f"{up:.0f}" if up >= 10 else f"{up:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.text(x, -dn - ymax * 0.02, f"{dn:.1f}", ha="center", va="top", fontsize=7.5,
                    color=INK, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(-ymin, ymax)
    ax.set_xticks(range(len(FROM_STEPS)))
    ax.set_xticklabels([f"step {s}" for s in FROM_STEPS])
    ax.set_xlabel(f"compared against final step {FINAL}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n = sub[0]["n_problems"]
    ax.set_title(f"{dlabel}   ({n} problems)", fontsize=11, color=INK, loc="left")
    ax.text(-0.5, ymax * 0.97, "▲ better at end", fontsize=8.5, color=INK2, va="top")
    ax.text(-0.5, -ymin * 0.97, "▼ worse at end (forgotten)", fontsize=8.5, color=INK, va="bottom")
axes[0].set_ylabel("% of test problems")

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in models]
fig.legend(handles, [f"Qwen3-{m}-Base" for m in models], loc="upper left", ncol=3, frameon=False,
           bbox_to_anchor=(0.005, 0.90), fontsize=9.5)
fig.suptitle("GRPO consistency: problems that got significantly worse vs better by the final checkpoint",
             x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
fig.text(0.01, 0.925, f"Per problem: 32 samples at each checkpoint, one-sided Fisher exact test, p < {ALPHA}. "
         f"Sampling noise alone would flag < {ALPHA:.0%} of problems. Labels are % of problems.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.84))
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=180, facecolor=fig.get_facecolor())
print("wrote", OUT_PNG, OUT_CSV)
