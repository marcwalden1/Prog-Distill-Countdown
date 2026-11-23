#!/usr/bin/env python3
"""
Qwen3-4B -> 0.6B distillation: logit-KD (constant LR, 4000 steps) vs NTP (cosine,
full epoch), per LR, on the metric they share -- the per-token CE of the teacher's
SAMPLED token (NTP: train/loss; KD: train/kd_ce). Teacher-entropy floor 0.944
(research_logs/2026-11-06_qwen3-4b-teacher-logit-sanity.md).

KD and NTP use the same parquet order, data.seed, world size and batch size, so
step s sees the SAME batch in every run; the right panel exploits that with a
paired KD - NTP difference (per-batch noise cancels). Pearson r of the raw
per-batch series is printed as a check of that claim. Through step 4000 the NTP
cosine LR is still >= 94% of peak, so the schedules nearly match (warmup differs:
KD 120 steps, NTP 490).

    python3 scripts/plot/plot_distill_kd_vs_ntp_loss.py
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_qwen3_size_comparison import INK, INK_MUTED, REPO, SURFACE, plt, style_axes  # noqa: E402
from plot_distill_ntp_loss import rolling  # noqa: E402

import numpy as np  # noqa: E402

LRS = [("3e-6", "#2a78d6"), ("1e-5", "#eb6834"), ("3e-5", "#4a3aa7")]
FLOOR = 0.944
KD_JOBS = {"3e-6": "59497236", "1e-5": "59497237", "3e-5": "59497238"}


def read(pattern, key):
    rx = re.compile(r"\bstep:(\d+) - .*?" + re.escape(key) + r":([0-9.]+)")
    pts = {}
    for p in sorted(glob.glob(os.path.join(REPO, "logs", pattern))):
        with open(p, errors="replace") as f:
            for line in f.read().replace("\r", "\n").split("\n"):
                for m in rx.finditer(line):
                    pts[int(m.group(1))] = float(m.group(2))
    s = np.array(sorted(pts))
    return s, np.array([pts[k] for k in s])


def main():
    xmax = 4000
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 5.0))
    fig.patch.set_facecolor(SURFACE)
    for lr, color in LRS:
        ks, ky = read(f"sft-distill-kd-lr{lr}-{KD_JOBS[lr]}-*.out", "train/kd_ce")
        ns, ny = read(f"sft-distill-ntp-lr{lr}-*.out", "train/loss")
        _, kl = read(f"sft-distill-kd-lr{lr}-{KD_JOBS[lr]}-*.out", "train/loss")
        m = ns <= xmax
        a0.plot(ns[m], rolling(ny[m], 10), "--", color=color, linewidth=1.4, alpha=0.6)
        a0.plot(ks, rolling(ky, 50), "-", color=color, linewidth=2.2, label=f"lr={lr}")
        a2.plot(ks, rolling(kl, 50), "-", color=color, linewidth=2.2, label=f"lr={lr}")
        # paired on common steps (multiples of 50)
        nmap = dict(zip(ns, ny))
        common = np.array([s for s in ks if s in nmap])
        kmap = dict(zip(ks, ky))
        kv = np.array([kmap[s] for s in common])
        nv = np.array([nmap[s] for s in common])
        d = kv - nv
        a1.scatter(common, d, s=6, color=color, alpha=0.2, linewidths=0)
        a1.plot(common, rolling(d, 10), "-", color=color, linewidth=2.2, label=f"lr={lr}")
        r = np.corrcoef(kv, nv)[0, 1]
        last = common > xmax - 1000
        print(f"lr={lr}: KD pts {len(ks)} (to {ks[-1]}), paired {len(common)}, "
              f"r(KD,NTP per-batch)={r:.3f}; last-1000: KD ce {kv[last].mean():.4f}  "
              f"NTP {nv[last].mean():.4f}  diff {d[last].mean():+.4f}  "
              f"KD gap to floor {kv[last].mean() - FLOOR:.4f}; KD loss {kl[ks > xmax - 1000].mean():.4f}")
    for ax in (a0,):
        ax.axhline(FLOOR, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.text(60, FLOOR + 0.004, "teacher entropy floor 0.944", fontsize=8, color=INK_MUTED)
    a1.axhline(0, color="#c9c8c3", linewidth=1)
    style_axes(a0, "sampled-token CE  (solid = KD, dashed = NTP)", "SFT step",
               "nats / token  (500-step rolling mean)")
    a0.set_ylim(FLOOR - 0.02, 1.16)
    style_axes(a1, "paired KD − NTP on the same batches", "SFT step",
               "Δ sampled-token CE  (below 0 = KD better)")
    style_axes(a2, "KD objective (top-20 bucketed KL)", "SFT step", "train/loss")
    h, l = a0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Qwen3-4B → 0.6B: logit-KD (constant LR) vs NTP (cosine, first 4000 of 16,333 steps), bs256",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = os.path.join(REPO, "figures", "Qwen3-distill-grpo", "distill_kd_vs_ntp_loss.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
