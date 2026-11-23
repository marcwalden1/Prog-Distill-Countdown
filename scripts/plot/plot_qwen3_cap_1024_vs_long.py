"""Qwen3-Base GRPO size sweep, step 1600: pass@1 vs pass@32, default 1024 cap vs a longer cap.

Two panels: n=3/4 at 1024 -> 2048 and n=5 at 1024 -> 4096 (the long caps that
exist on disk at step 1600; n=3/4 has no 4096 eval, and n=6 has a long-cap
eval only for 0.6B, so it is left out). One arrow per model, from its
1024-cap point to its long-cap point. A model whose long-cap eval is not on
disk yet is drawn as the 1024 point alone.

Both caps are scored from the eval JSONs with
grader_utils.compute_score == 1.0; pass@32 = any of the 32 samples correct.
Run with the verl env python (needs matplotlib) from the repo root.
"""
import json
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from grader_utils import compute_score  # noqa: E402

RUNS = {
    "0.6B": "Qwen3-0.6B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
    "1.7B": "Qwen3-1.7B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
    "4B": "Qwen3-4B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
}
PANELS = [("balanced-simple", "n=3/4 (ID)", 2048),
          ("balanced5-simple", "n=5 (OOD)", 4096)]
STEP = 1600
COLORS = {"0.6B": "#2a78d6", "1.7B": "#eb6834", "4B": "#1baf7a"}
OUT = "figures/Qwen3-base-grpo/cap_1024_vs_long_step1600.png"


def eval_path(run, ds, cap):
    base = f"results/{run}/global_step_{STEP}"
    sub = "" if cap == 1024 else f"max{cap}/"
    return f"{base}/{sub}{ds}_temp0.6_n32_max{cap}.json"


def _score(item):
    ex = {"numbers": item["nums"]}
    return [compute_score("x", o, item["target"], ex) == 1.0 for o in item["outputs"]]


def passk(path, pool):
    """(pass@1, pass@32) for one eval JSON, or None if it is not on disk."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # compute_score prints per-sample debug lines; silence them.
    with open(os.devnull, "w") as dn:
        old, sys.stdout = sys.stdout, dn
        try:
            res = pool.map(_score, data, chunksize=50)
        finally:
            sys.stdout = old
    p1 = sum(sum(r) / len(r) for r in res) / len(res)
    p32 = sum(any(r) for r in res) / len(res)
    return p1, p32


def main():
    with Pool(16, initializer=lambda: setattr(sys, "stdout", open(os.devnull, "w"))) as pool:
        pts = {(lab, ds): {c: passk(eval_path(run, ds, c), pool) for c in (1024, cap)}
               for ds, _, cap in PANELS for lab, run in RUNS.items()}
    for k, d in pts.items():
        print(k, {c: (None if v is None else tuple(round(x, 3) for x in v)) for c, v in d.items()})

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.5, 4.6))
    for ax, (ds, title, cap) in zip(axes, PANELS):
        for lab in RUNS:
            c = COLORS[lab]
            x0, y0 = pts[(lab, ds)][1024]
            ax.scatter(x0, y0, s=60, facecolor="white", edgecolor=c, linewidth=2, zorder=3)
            if pts[(lab, ds)][cap] is not None:
                x1, y1 = pts[(lab, ds)][cap]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=2, shrinkA=5, shrinkB=5),
                            zorder=2)
                ax.scatter(x1, y1, s=60, color=c, edgecolor="white", linewidth=1.5, zorder=4)
                ax.annotate(lab, (x1, y1), xytext=(7, -3), textcoords="offset points", fontsize=9.5)
            else:
                ax.annotate(f"{lab}  ({cap} pending)", (x0, y0), xytext=(7, -3),
                            textcoords="offset points", fontsize=9, color="#52514e")
        ax.set_xlabel("pass@1")
        ax.set_title(f"{title}: cap 1024 \u2192 {cap}", fontsize=11)
        ax.grid(color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.margins(x=0.18, y=0.1)
    axes[0].set_ylabel("pass@32")
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[l], label=l, markersize=7)
               for l in RUNS]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="white",
                       markeredgecolor="#52514e", markeredgewidth=2, label="1024 cap", markersize=7),
                Line2D([], [], marker="o", ls="", color="#52514e", label="longer cap", markersize=7)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Qwen3-Base GRPO, step {STEP}: longer response cap", y=1.07, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
