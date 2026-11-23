"""3-way Venn of per-problem correctness: Qwen3 0.6B / 1.7B / 4B GRPO (kl3e-3, 1 epoch, n=4).
Reads analysis_data/qwen3_venn_step{STEP}.json (built by qwen3_venn_data.py)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

STEP = int(os.environ.get("STEP", 1600))
d = json.load(open(f"analysis_data/qwen3_venn_step{STEP}.json"))
MODELS = [("0.6B", "0.6B (lr 3e-6)", "#2a78d6"),
          ("1.7B", "1.7B (lr 1e-6)", "#eb6834"),
          ("4B",   "4B (lr 1e-6)",   "#1baf7a")]
DS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
DEFS = [("sample0", "sample #1 correct\n(mean@1)", lambda c: c[:, 0] == 1),
        ("maj",     "≥16/32 samples correct\n(majority)", lambda c: c.mean(1) >= 0.5),
        ("pass32",  "any of 32 correct\n(pass@32)", lambda c: c.max(1) == 1)]
INK, MUTED = "#1f1f1e", "#6b6a63"
centers = [(-0.55, 0.32), (0.55, 0.32), (0.0, -0.62)]
R = 1.0
# region label positions keyed by membership bits (0.6B,1.7B,4B)
POS = {(1,0,0): (-1.05, 0.62), (0,1,0): (1.05, 0.62), (0,0,1): (0, -1.25),
       (1,1,0): (0, 0.78), (1,0,1): (-0.72, -0.42), (0,1,1): (0.72, -0.42), (1,1,1): (0, 0.0)}

fig, axes = plt.subplots(len(DEFS), len(DS), figsize=(15, 15.5))
summary = {}
for i, (dkey, dlabel, fn) in enumerate(DEFS):
    for j, (ds, dsl) in enumerate(DS):
        ax = axes[i, j]
        S = [fn(np.array(d[f"{m}|{ds}"]["correct"])) for m, _, _ in MODELS]
        N = len(S[0])
        for (cx, cy), (_, name, col) in zip(centers, MODELS):
            ax.add_patch(Circle((cx, cy), R, facecolor=col, alpha=0.18, edgecolor=col, lw=2))
        counts = {}
        for bits, (x, y) in POS.items():
            mask = np.ones(N, bool)
            for b, s in zip(bits, S): mask &= (s if b else ~s)
            counts[bits] = int(mask.sum())
            ax.text(x, y, f"{mask.sum()}\n{100*mask.mean():.1f}%", ha="center", va="center",
                    fontsize=10.5, color=INK, linespacing=1.1)
        none = int((~S[0] & ~S[1] & ~S[2]).sum()); counts[(0,0,0)] = none
        summary[f"{dkey}|{ds}"] = {"N": N, **{"".join(map(str, k)): v for k, v in counts.items()}}
        # set name labels outside circles
        for (cx, cy), (_, name, col), s, (dx, dy) in zip(centers, MODELS, S, [(-1.15, 1.48), (1.15, 1.48), (0, -1.8)]):
            ax.text(dx, dy, f"{name}\n{s.sum()} correct ({100*s.mean():.1f}%)", ha="center", va="center",
                    fontsize=10.5, color=INK, fontweight="bold")
            ax.plot([], [])
        ax.text(1.7, -1.55, f"none correct\n{none} ({100*none/N:.1f}%)", ha="right", va="center",
                fontsize=10, color=MUTED)
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-2.0, 1.75); ax.set_aspect("equal"); ax.axis("off")
        if i == 0: ax.set_title(f"{dsl}  (N={N})", fontsize=14, color=INK, pad=8)
        if j == 0: ax.text(-1.95, -0.1, dlabel, rotation=90, ha="center", va="center", fontsize=12.5, color=INK)
fig.suptitle(f"Qwen3-Base GRPO (0.6B lr 3e-6, 1.7B/4B lr 1e-6; kl=3e-3, 1 epoch, rollout n=4) — per-problem correctness at step {STEP}\n"
             "eval: temp 0.6, 32 samples, max 1024 tokens; region labels = #problems / % of test set",
             fontsize=14, color=INK)
plt.tight_layout(rect=[0.02, 0, 1, 0.95])
os.makedirs("figures/Qwen3-base-grpo/venn", exist_ok=True)
out = f"figures/Qwen3-base-grpo/venn/venn_qwen3_0.6B-1.7B-4B_step{STEP}.png"
plt.savefig(out, dpi=130, facecolor="white"); print(out)
json.dump(summary, open(f"analysis_data/qwen3_venn_counts_step{STEP}.json", "w"), indent=1)
