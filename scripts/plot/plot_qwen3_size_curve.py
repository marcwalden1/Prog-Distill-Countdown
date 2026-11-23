"""Accuracy vs model size for the Qwen3-Base GRPO sweep (0.6B lr3e-6, 1.7B/4B lr1e-6; kl3e-3, 1 ep, n=4).
2 rows (pass@1 = mean over 32 samples, pass@32) x 3 cols (n=3,4 / n=5 / n=6).
Solid = final checkpoint (step 1600); dashed = best checkpoint for that model on that exact panel's
metric (oracle: chosen on the test set itself), labelled with its step.
Reads analysis_data/qwen3_size_passk.csv (from qwen3_size_curve_data.py).
--ep2 adds the 0.6B lr1e-6 2-epoch run at x=0.6B (orange star = final ckpt @3200, orange dot = best
of its 32 ckpts) and writes size_curve_pass1_pass32_2epoch.png."""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("analysis_data/qwen3_size_passk.csv")))
MODELS = [("0.6B", 0.6), ("1.7B", 1.7), ("4B", 4.0)]
DS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
METRICS = [("pass@1", "pass@1 (mean of 32 samples)"), ("pass@32", "pass@32")]
FINAL = 1600
INK, MUTED, GRID, COL = "#1f1f1e", "#6b6a63", "#e3e2dc", "#2a78d6"
EP2 = "--ep2" in sys.argv
COL2, EP2_FINAL, EP2_X = "#eb6834", 3200, 0.6 / 1.22   # left of the 1-ep 0.6B points (free space)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
for i, (met, mlabel) in enumerate(METRICS):
    for j, (ds, dsl) in enumerate(DS):
        ax = axes[i, j]; xs, fin, best, bstep = [], [], [], []
        for m, x in MODELS:
            r = [q for q in rows if q["model"] == m and q["dataset"] == ds]
            f = next(q for q in r if int(q["step"]) == FINAL)
            b = max(r, key=lambda q: float(q[met]))
            xs.append(x); fin.append(100 * float(f[met])); best.append(100 * float(b[met])); bstep.append(int(b["step"]))
        ax.plot(xs, fin, "-", color=COL, lw=2, marker="o", ms=8, mec="white", mew=2, label=("0.6B–4B 1 epoch: " if EP2 else "") + f"final ckpt (step {FINAL})", zorder=3)
        ax.plot(xs, best, "--", color=COL, lw=2, marker="o", ms=8, mfc="white", mec=COL, mew=2, label=("0.6B–4B 1 epoch: " if EP2 else "") + "best ckpt on this metric", zorder=4)
        top = max(best)
        for x, yb, yf, s in zip(xs, best, fin, bstep):
            ax.annotate(f"{yb:.1f}" + ("" if s == FINAL else f"\n@{s}"), (x, yb), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.0)
            if s != FINAL:
                ax.annotate(f"{yf:.1f}", (x, yf), xytext=(0, -9), textcoords="offset points",
                            ha="center", va="top", fontsize=8.5, color=MUTED)
        if EP2:
            r2 = [q for q in rows if q["model"] == "0.6B-ep2" and q["dataset"] == ds]
            f2 = 100 * float(next(q for q in r2 if int(q["step"]) == EP2_FINAL)[met])
            b2q = max(r2, key=lambda q: float(q[met])); b2, s2 = 100 * float(b2q[met]), int(b2q["step"])
            ax.plot([EP2_X], [f2], ls="none", marker="*", ms=15, color=COL2, mec="white", mew=1.2, zorder=6,
                    label=f"0.6B 2 epochs: final ckpt (step {EP2_FINAL})")
            ax.plot([EP2_X], [b2], ls="none", marker="o", ms=8, color=COL2, mec="white", mew=1.5, zorder=6,
                    label="0.6B 2 epochs: best ckpt on this metric")
            # labels left of the markers: best above its dot, final below its star (never collide since b2 >= f2)
            if s2 != EP2_FINAL:
                ax.annotate(f"{b2:.1f}\n@{s2}", (EP2_X, b2), xytext=(-7, 1), textcoords="offset points",
                            ha="right", va="bottom", fontsize=8.5, color=COL2, linespacing=1.0)
            ax.annotate(f"{f2:.1f}\n@{EP2_FINAL}", (EP2_X, f2), xytext=(-8, -1), textcoords="offset points",
                        ha="right", va="top", fontsize=8.5, color=COL2, linespacing=1.0)
            top = max(top, b2)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([m for m, _ in MODELS]); ax.minorticks_off()
        ax.set_xlim(0.24 if EP2 else 0.45, 5.3); ax.set_ylim(0, min(100, top * 1.25 + 2))
        ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color("#bdbcb4")
        ax.tick_params(colors=MUTED, labelsize=9.5)
        if i == 0: ax.set_title(dsl, fontsize=13, color=INK)
        if i == 1: ax.set_xlabel("model size (Qwen3-Base)", fontsize=10, color=INK)
        if j == 0: ax.set_ylabel(f"{mlabel}\naccuracy (%)", fontsize=11, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10.5, handlelength=5, bbox_to_anchor=(0.5, 0.91 if EP2 else 0.935))
fig.suptitle("Qwen3-Base GRPO (kl=3e-3, 1 epoch, rollout n=4; 0.6B lr 3e-6, 1.7B/4B lr 1e-6)\n" +
             ("blue: 1 epoch; orange: 0.6B lr 1e-6, 2 epochs (best of 32 ckpts, 100–3200)\n" if EP2 else "") +
             "eval temp 0.6, 32 samples, max 1024 tokens; dashed = oracle best of 16 ckpts (100–1600), '@step' when ≠ 1600",
             fontsize=11.5, color=INK)
plt.tight_layout(rect=[0, 0, 1, 0.86 if EP2 else 0.9])
out = "figures/Qwen3-base-grpo/size_curve_pass1_pass32" + ("_2epoch" if EP2 else "") + ".png"
plt.savefig(out, dpi=140, facecolor="white"); print(out)
