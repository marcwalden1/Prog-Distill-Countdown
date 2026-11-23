#!/usr/bin/env python3
"""
Qwen3-4B-Base -> Qwen3-0.6B-Base NTP distillation: loss vs SFT step, LR sweep.

No val loss was logged during these runs (the only val pass is at the last step,
and it OOMed), so this plots the TRAIN loss parsed from the SLURM logs. The runs
are exactly one epoch, so every batch is unseen when its loss is computed: the
pre-update train loss is an unbiased held-out estimate. Each logged value is ONE
batch (256 rollouts) every SFT_LOG_FREQ=50 steps, so it is noisy -- the left
panel shows the raw points faintly and a rolling mean on top.

All three runs share data.seed, hence the identical batch sequence, so the
per-batch noise is common to them. The right panel uses that: the paired
difference loss(lr) - loss(ref lr) on the SAME batches cancels most of it.

    python3 scripts/plot/plot_distill_ntp_loss.py
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_qwen3_size_comparison import (  # noqa: E402
    INK, INK_MUTED, REPO, SURFACE, plt, style_axes)

import numpy as np  # noqa: E402

# (lr, color, linestyle) -- validated blue/orange/purple trio
LRS = [("3e-6", "#2a78d6", "-"), ("1e-5", "#eb6834", "--"), ("3e-5", "#4a3aa7", ":")]
STEP_LOSS = re.compile(r"\bstep:(\d+) - train/loss:([0-9.]+)")


def read_log(lr):
    paths = sorted(glob.glob(os.path.join(REPO, "logs", f"sft-distill-ntp-lr{lr}-*.out")))
    if not paths:
        sys.exit(f"no log for lr={lr}")
    pts = {}
    for p in paths:  # later logs (requeues) overwrite earlier steps
        with open(p, errors="replace") as f:
            for m in STEP_LOSS.finditer(f.read()):
                pts[int(m.group(1))] = float(m.group(2))
    steps = np.array(sorted(pts))
    return steps, np.array([pts[s] for s in steps])


def rolling(y, w):
    """Centered rolling mean; shrinks the window at the edges."""
    out = np.empty_like(y)
    h = w // 2
    for i in range(len(y)):
        out[i] = y[max(0, i - h): i + h + 1].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20,
                    help="rolling-mean window in logged points (x50 steps)")
    ap.add_argument("--ref", default="1e-5", help="reference lr for the paired difference")
    ap.add_argument("--out", default=os.path.join(
        REPO, "figures", "Qwen3-distill-grpo", "distill_ntp_loss_vs_step.png"))
    args = ap.parse_args()

    data = {lr: read_log(lr) for lr, _, _ in LRS}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.9))
    fig.patch.set_facecolor(SURFACE)
    ws = args.window * 50

    for lr, color, ls in LRS:
        s, y = data[lr]
        a0.scatter(s, y, s=6, color=color, alpha=0.18, linewidths=0, zorder=2)
        a0.plot(s, rolling(y, args.window), ls, color=color, linewidth=2, zorder=3,
                label=f"lr={lr}")
        print(f"lr={lr}: {len(s)} points, steps {s[0]}-{s[-1]}, "
              f"first-{ws} mean {y[s <= ws].mean():.4f}, last-{ws} mean "
              f"{y[s > s[-1] - ws].mean():.4f}")
    style_axes(a0, f"train loss (one epoch, so held-out); line = {ws}-step rolling mean",
               "SFT step", "per-token CE (nats)")
    lo = min(np.percentile(y, 1) for _, y in data.values())
    hi = max(np.percentile(y, 99) for _, y in data.values())
    a0.set_ylim(lo - 0.02, hi + 0.02)

    rs, ry = data[args.ref]
    ref = dict(zip(rs, ry))
    a1.axhline(0, color="#c9c8c3", linewidth=1, zorder=1)
    for lr, color, ls in LRS:
        if lr == args.ref:
            continue
        s, y = data[lr]
        keep = np.array([st in ref for st in s])
        s, d = s[keep], y[keep] - np.array([ref[st] for st in s[keep]])
        a1.scatter(s, d, s=6, color=color, alpha=0.18, linewidths=0, zorder=2)
        a1.plot(s, rolling(d, args.window), ls, color=color, linewidth=2, zorder=3,
                label=f"lr={lr}")
        print(f"lr={lr} - lr={args.ref}: last-{ws} mean diff {d[s > s[-1] - ws].mean():+.4f}")
    style_axes(a1, f"paired difference vs lr={args.ref} on the same batches",
               "SFT step", f"loss(lr) − loss(lr={args.ref})  (below 0 = better)")

    h, l = a0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Qwen3-4B → 0.6B NTP distillation (bs256, cosine, wu 3%, 1 epoch = 16,333 steps)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
