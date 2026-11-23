"""Histogram of per-sample response lengths (tokens) for Qwen3-Base GRPO runs.

Reads eval JSONs (n=32 samples/prompt, temp 0.6, max 1024) at a given step and
tokenizes every output with the model's own tokenizer (same rule as
plot_results.py's .lengths cache, add_special_tokens=False).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

RUNS = [  # (model, exp, label)
    ("Qwen3-0.6B-Base", "balanced-simple-grpo-lr3e-6-kl3e-3-seed1", "0.6B, lr=3e-6"),
    ("Qwen3-1.7B-Base", "balanced-simple-grpo-lr1e-6-kl3e-3-seed1", "1.7B, lr=1e-6"),
    ("Qwen3-4B-Base",   "balanced-simple-grpo-lr1e-6-kl3e-3-seed1", "4B, lr=1e-6"),
]
DATASETS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
COLORS = ["#2a78d6", "#e3871c", "#2f9e6b"]
MAX = 1024

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=1600)
ap.add_argument("--model-dir", default="/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models")
ap.add_argument("--results-dir", default="results")
ap.add_argument("--out-dir", default="figures/Qwen3-base-grpo/lengths")
ap.add_argument("--include-truncated", action="store_true",
                help="histogram ALL responses (truncated ones pile up in the last bin) -> length_hist_step{step}.png")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

lengths = {}  # (run_idx, ds) -> np.array
for i, (model, exp, _) in enumerate(RUNS):
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, model))
    for ds, _ in DATASETS:
        p = f"{args.results_dir}/{model}/{exp}/global_step_{args.step}/{ds}_temp0.6_n32_max{MAX}.json"
        cache = p + f".lenhist.npy"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(p):
            arr = np.load(cache)
        else:
            outs = [o for item in json.load(open(p)) for o in item["outputs"]]
            arr = []
            for k in range(0, len(outs), 4096):  # chunked: one big call OOMs a login node
                arr += [len(x) for x in tok(outs[k:k + 4096], add_special_tokens=False)["input_ids"]]
            del outs
            arr = np.array(arr); np.save(cache, arr)
        lengths[(i, ds)] = arr
        print(f"{model:16s} {ds:17s} N={len(arr):6d} mean={arr.mean():6.1f} "
              f"median={np.median(arr):6.0f} p90={np.percentile(arr,90):6.0f} "
              f">={MAX}: {(arr>=MAX).mean()*100:5.1f}%", flush=True)

bins = np.arange(0, MAX + 33, 32) if args.include_truncated else np.arange(0, MAX + 1, 32)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
for ax, (ds, dl) in zip(axes, DATASETS):
    for i, (_, _, lab) in enumerate(RUNS):
        full = lengths[(i, ds)]
        trunc = (full >= MAX).mean() * 100
        a = full if args.include_truncated else full[full < MAX]  # default: non-truncated only, normalized within it
        w = np.ones_like(a, dtype=float) / len(a) * 100
        ax.hist(a, bins=bins, weights=w, histtype="stepfilled", alpha=0.25, color=COLORS[i])
        ax.hist(a, bins=bins, weights=w, histtype="step", lw=1.8, color=COLORS[i],
                label=f"{lab}: truncated {trunc:.1f}%, mean {a.mean():.0f} (N={len(a)})")
    ax.set_title(f"{dl}  ({ds})")
    ax.set_xlabel("Response length (tokens)" + ("" if args.include_truncated else ", non-truncated only"))
    ax.set_xlim(0, MAX + 32 if args.include_truncated else MAX)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
axes[0].set_ylabel("% of samples" if args.include_truncated else "% of non-truncated samples")
fig.suptitle(f"Qwen3-Base GRPO (1 epoch, rollout n=4, kl=3e-3) — {'' if args.include_truncated else 'non-truncated '}response length at step {args.step}, "
             f"cap {MAX}, eval temp 0.6, 32 samples/prompt", fontsize=12)
fig.tight_layout()
out = f"{args.out_dir}/length_hist{'' if args.include_truncated else '_nontrunc'}_step{args.step}.png"
fig.savefig(out, dpi=150); print("saved", out)
