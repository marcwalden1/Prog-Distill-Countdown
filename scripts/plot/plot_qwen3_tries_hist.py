"""Histograms of tries per response and tokens per try, Qwen3-Base GRPO (1 epoch, rollout n=4).

A "try" is a candidate as defined in plot_qwen3_correct_search_effort.py: an
arithmetic chain whose first expression uses exactly the puzzle's numbers (one
full candidate solution written out and evaluated). tokens/try = response
length / tries, over responses with >=1 try.

Only non-truncated responses (length < cap) are plotted; the truncation rate
is in the legend. Lengths come from the .lenhist.npy caches written by
plot_qwen3_length_hist.py (run that first); per-response try counts are
cached next to each eval JSON as .tries.npy (all responses, same order).

Usage (CPU; a few minutes on 32 cores):
    sbatch -p serial_requeue -c 32 --mem=96G -t 1:00:00 -o logs/qwen3_tries-%j.out \
        --wrap "~/.conda/envs/verl/bin/python3 scripts/plot/plot_qwen3_tries_hist.py"
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from plot_qwen3_correct_search_effort import effort  # noqa: E402

RUNS = [  # (model, exp, label)
    ("Qwen3-0.6B-Base", "balanced-simple-grpo-lr3e-6-kl3e-3-seed1", "0.6B, lr=3e-6"),
    ("Qwen3-1.7B-Base", "balanced-simple-grpo-lr1e-6-kl3e-3-seed1", "1.7B, lr=1e-6"),
    ("Qwen3-4B-Base",   "balanced-simple-grpo-lr1e-6-kl3e-3-seed1", "4B, lr=1e-6"),
]
DATASETS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
COLORS = ["#2a78d6", "#e3871c", "#2f9e6b"]
MAX = 1024
TRIES_CLIP = 40     # last tries bin collects >= this
TPT_CLIP = 500      # last tokens/try bin collects >= this


def _tries(args):
    text, nums = args
    return effort(text, nums)[0]


ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=1600)
ap.add_argument("--results-dir", default="results")
ap.add_argument("--out-dir", default="figures/Qwen3-base-grpo/lengths")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

data = {}  # (run_idx, ds) -> (lengths, tries)
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
    for i, (model, exp, _) in enumerate(RUNS):
        for ds, _ in DATASETS:
            p = f"{args.results_dir}/{model}/{exp}/global_step_{args.step}/{ds}_temp0.6_n32_max{MAX}.json"
            L = np.load(p + ".lenhist.npy")
            cache = p + ".tries.npy"
            if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(p):
                T = np.load(cache)
            else:
                items = [(o, it["nums"]) for it in json.load(open(p)) for o in it["outputs"]]
                T = np.array(pool.map(_tries, items, chunksize=500)); del items
                np.save(cache, T)
            assert len(T) == len(L), (p, len(T), len(L))
            data[(i, ds)] = (L, T)
            ok = L < MAX
            t = T[ok]
            print(f"{model:16s} {ds:17s} non-trunc={ok.sum():6d}  tries mean={t.mean():5.2f} "
                  f"median={np.median(t):3.0f} zero={np.mean(t == 0)*100:5.1f}%  "
                  f"tok/try median={np.median(L[ok][t > 0] / t[t > 0]):6.1f}  "
                  f"| truncated tries mean={T[~ok].mean():5.2f}", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
tbins = np.arange(0, TRIES_CLIP + 2) - 0.5
pbins = np.arange(0, TPT_CLIP + 11, 10)
for c, (ds, dl) in enumerate(DATASETS):
    for i, (_, _, lab) in enumerate(RUNS):
        L, T = data[(i, ds)]
        ok = L < MAX
        trunc = (~ok).mean() * 100
        t = T[ok]
        w = np.ones(len(t)) / len(t) * 100
        x = np.minimum(t, TRIES_CLIP)
        ax = axes[0][c]
        ax.hist(x, bins=tbins, weights=w, histtype="stepfilled", alpha=0.25, color=COLORS[i])
        ax.hist(x, bins=tbins, weights=w, histtype="step", lw=1.8, color=COLORS[i],
                label=f"{lab}: trunc {trunc:.1f}%, mean {t.mean():.1f}, 0 tries {np.mean(t == 0)*100:.0f}%")
        pos = t > 0
        tpt = np.minimum(L[ok][pos] / t[pos], TPT_CLIP)
        w = np.ones(len(tpt)) / len(tpt) * 100
        ax = axes[1][c]
        ax.hist(tpt, bins=pbins, weights=w, histtype="stepfilled", alpha=0.25, color=COLORS[i])
        ax.hist(tpt, bins=pbins, weights=w, histtype="step", lw=1.8, color=COLORS[i],
                label=f"{lab}: trunc {trunc:.1f}%, median {np.median(L[ok][pos] / t[pos]):.0f}")
    axes[0][c].set_title(f"{dl}  ({ds})")
    axes[0][c].set_xlabel(f"Tries per response (last bin = {TRIES_CLIP}+)")
    axes[1][c].set_xlabel(f"Tokens per try (responses with >=1 try; last bin = {TPT_CLIP}+)")
    axes[0][c].set_xlim(-0.5, TRIES_CLIP + 0.5)
    axes[1][c].set_xlim(0, TPT_CLIP + 10)
    for r in range(2):
        axes[r][c].grid(alpha=0.3)
        axes[r][c].legend(fontsize=8.5, frameon=False)
axes[0][0].set_ylabel("% of non-truncated responses")
axes[1][0].set_ylabel("% of non-truncated responses with >=1 try")
fig.suptitle(f"Qwen3-Base GRPO (1 epoch, rollout n=4, kl=3e-3), step {args.step}: tries per response and "
             f"tokens per try, non-truncated responses (cap {MAX}), eval temp 0.6, 32 samples/prompt\n"
             "try = a written-out evaluation of an expression using exactly the puzzle's numbers",
             fontsize=11.5)
fig.tight_layout()
out = f"{args.out_dir}/tries_hist_nontrunc_step{args.step}.png"
fig.savefig(out, dpi=150); print("saved", out)
