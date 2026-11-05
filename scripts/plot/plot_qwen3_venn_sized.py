"""Size-faithful alternatives to plot_qwen3_venn.py, same data
(analysis_data/qwen3_venn_step{STEP}.json from qwen3_venn_data.py):
  euler_*.png  area-proportional Euler diagrams, one shared scale per column (test set), so sizes
               are comparable down a column. Circle areas and pairwise
               overlaps are exact; the 3-way region can't always be exact (its error is printed).
  upset_*.png  UpSet plots: one bar per exclusive region, so every count is directly readable."""
import json, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

STEP = int(os.environ.get("STEP", 1600))
d = json.load(open(f"analysis_data/qwen3_venn_step{STEP}.json"))
MODELS = [("0.6B", "0.6B (lr 1e-6)", "#2a78d6"),
          ("1.7B", "1.7B (lr 3e-6)", "#eb6834"),
          ("4B",   "4B (lr 3e-6)",   "#1baf7a")]
DS = [("balanced-simple", "n=3,4"), ("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")]
DEFS = [("sample #1 correct\n(mean@1)", lambda c: c[:, 0] == 1),
        ("≥16/32 samples correct\n(majority)", lambda c: c.mean(1) >= 0.5),
        ("any of 32 correct\n(pass@32)", lambda c: c.max(1) == 1)]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
OUT = "figures/Qwen3-base-grpo/venn"; os.makedirs(OUT, exist_ok=True)
REGIONS = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1), (0,0,0)]

def sets(ds, fn):
    return [fn(np.array(d[f"{m}|{ds}"]["correct"])) for m, _, _ in MODELS]

def region_counts(S):
    out = {}
    for bits in REGIONS:
        m = np.ones(len(S[0]), bool)
        for b, s in zip(bits, S): m &= (s if b else ~s)
        out[bits] = int(m.sum())
    return out

# ---------------- Euler geometry ----------------
def lens(r1, r2, dd):
    if dd >= r1 + r2: return 0.0
    if dd <= abs(r1 - r2): return math.pi * min(r1, r2) ** 2
    a = r1*r1*math.acos((dd*dd + r1*r1 - r2*r2) / (2*dd*r1))
    b = r2*r2*math.acos((dd*dd + r2*r2 - r1*r1) / (2*dd*r2))
    c = 0.5*math.sqrt((-dd+r1+r2)*(dd+r1-r2)*(dd-r1+r2)*(dd+r1+r2))
    return a + b - c

def dist_for(r1, r2, area):
    lo, hi = abs(r1 - r2), r1 + r2
    if area <= 0: return hi * 1.02          # disjoint: small gap
    if area >= math.pi * min(r1, r2) ** 2 - 1e-9: return lo * 0.98   # nested
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lens(r1, r2, mid) > area else (lo, mid)
    return (lo + hi) / 2

def layout(S):
    sizes = [s.sum() for s in S]
    r = [math.sqrt(max(n, 1e-9) / math.pi) for n in sizes]
    dab = dist_for(r[0], r[1], (S[0] & S[1]).sum())
    dac = dist_for(r[0], r[2], (S[0] & S[2]).sum())
    dbc = dist_for(r[1], r[2], (S[1] & S[2]).sum())
    # place A at origin, B on x-axis, C below by law of cosines (clamped if triangle infeasible)
    cosv = (dab**2 + dac**2 - dbc**2) / (2 * dab * dac) if dab * dac > 0 else 0
    th = math.acos(max(-1, min(1, cosv)))
    return np.array([[0, 0], [dab, 0], [dac*math.cos(th), -dac*math.sin(th)]]), np.array(r)

def mc_regions(C, R, n=400_000, seed=0):
    """Monte-Carlo area of each exclusive region, plus a representative point for labelling."""
    rng = np.random.default_rng(seed)
    lo = (C - R[:, None]).min(0); hi = (C + R[:, None]).max(0)
    P = rng.uniform(lo, hi, size=(n, 2)); cell = np.prod(hi - lo) / n
    inside = [((P - c) ** 2).sum(1) <= rr * rr for c, rr in zip(C, R)]
    out = {}
    for bits in REGIONS[:-1]:
        m = np.ones(n, bool)
        for b, s in zip(bits, inside): m &= (s if b else ~s)
        if m.sum() == 0: out[bits] = (0.0, None); continue
        pts = P[m]
        # pick the region point farthest from every circle boundary (most room for a label)
        margin = np.min([np.abs(np.sqrt(((pts - c) ** 2).sum(1)) - rr) for c, rr in zip(C, R)], axis=0)
        out[bits] = (m.sum() * cell, pts[np.argmax(margin)], margin.max())
    return out

def euler():
    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    L = {(i, j): layout(sets(ds, fn)) for i, (_, fn) in enumerate(DEFS) for j, (ds, _) in enumerate(DS)}
    # shared scale per column: half-width big enough for the largest layout (+ room for text)
    half = {j: 1.25 * max(max(((C + R[:, None]).max(0) - (C - R[:, None]).min(0)) / 2) for (i, jj), (C, R) in L.items() if jj == j)
            for j in range(3)}
    for i, (dlabel, fn) in enumerate(DEFS):
        for j, (ds, dsl) in enumerate(DS):
            ax = axes[i, j]; S = sets(ds, fn); N = len(S[0]); cnt = region_counts(S)
            C, R = L[(i, j)]
            side = 2 * half[j]; ctr = ((C - R[:, None]).min(0) + (C + R[:, None]).max(0)) / 2
            ctr = ctr + np.array([0, -0.06 * side])   # leave the top-left for the totals text
            for c, rr, (_, _, col) in zip(C, R, MODELS):
                ax.add_patch(Circle(c, rr, facecolor=col, alpha=0.30, edgecolor="white", lw=2))
                ax.add_patch(Circle(c, rr, facecolor="none", edgecolor=col, lw=1.5))
            mc = mc_regions(C, R)
            err = abs(mc[(1,1,1)][0] - cnt[(1,1,1)]) / N * 100
            minroom = 0.03 * side              # only label regions with room for text
            for bits, v in mc.items():
                if v[1] is None or cnt[bits] == 0: continue
                if v[2] >= minroom:
                    ax.text(*v[1], f"{cnt[bits]}", ha="center", va="center", fontsize=9.5, color=INK)
            ax.text(ctr[0] + side/2 - 0.02*side, ctr[1] - side/2 + 0.02*side,
                    f"none correct: {cnt[(0,0,0)]} ({100*cnt[(0,0,0)]/N:.1f}%)",
                    ha="right", va="bottom", fontsize=9.5, color=MUTED)
            ax.text(ctr[0] - side/2 + 0.02*side, ctr[1] + side/2 - 0.02*side,
                    "\n".join(f"{nm.split()[0]}: {s.sum()} ({100*s.mean():.1f}%)" for (_, nm, _), s in zip(MODELS, S)),
                    ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.3)
            if err > 1.0:
                ax.text(ctr[0] - side/2 + 0.02*side, ctr[1] - side/2 + 0.02*side,
                        f"3-way overlap area off by {err:.1f}% of N", ha="left", va="bottom", fontsize=8, color=MUTED)
            pad = side * 0.5
            ax.set_xlim(ctr[0] - pad, ctr[0] + pad); ax.set_ylim(ctr[1] - pad, ctr[1] + pad)
            ax.set_aspect("equal"); ax.axis("off")
            if i == 0: ax.set_title(f"{dsl}  (N={N})", fontsize=14, color=INK)
            if j == 0: ax.text(-0.04, 0.5, dlabel, transform=ax.transAxes, rotation=90,
                               ha="center", va="center", fontsize=12.5, color=INK)
    handles = [Circle((0, 0), 1, facecolor=col, alpha=0.4, edgecolor=col) for _, _, col in MODELS]
    fig.legend(handles, [nm for _, nm, _ in MODELS], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=12)
    fig.suptitle(f"Qwen3-Base GRPO (kl=3e-3, 1 epoch, rollout n=4), step {STEP}: area-proportional Euler diagrams\n"
                 "circle area = #problems solved (same scale within each column); numbers = exclusive region counts "
                 "(small regions unlabeled, see UpSet)", fontsize=13, color=INK)
    plt.tight_layout(rect=[0.02, 0, 1, 0.93])
    out = f"{OUT}/euler_qwen3_0.6B-1.7B-4B_step{STEP}.png"; plt.savefig(out, dpi=130, facecolor="white"); print(out)

# ---------------- UpSet ----------------
def upset():
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(6, 3, height_ratios=[3, 1.1] * 3, hspace=0.12, wspace=0.18)
    for i, (dlabel, fn) in enumerate(DEFS):
        for j, (ds, dsl) in enumerate(DS):
            S = sets(ds, fn); N = len(S[0]); cnt = region_counts(S)
            order = REGIONS          # fixed order across panels so columns line up
            ax = fig.add_subplot(gs[2*i, j]); axm = fig.add_subplot(gs[2*i+1, j], sharex=ax)
            x = np.arange(len(order)); vals = [100 * cnt[b] / N for b in order]
            ax.bar(x, vals, width=0.7, color=["#8a8980"] * 7 + ["#d9d8d2"], edgecolor="white", lw=2)
            for xi, b, v in zip(x, order, vals):
                ax.text(xi, v + 1.0, f"{cnt[b]}", ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="x", bottom=False, labelbottom=False); ax.tick_params(axis="y", colors=MUTED, labelsize=9)
            if j == 0: ax.set_ylabel(f"{dlabel}\n% of test set", fontsize=11, color=INK)
            if i == 0: ax.set_title(f"{dsl}  (N={N})", fontsize=14, color=INK)
            # dot matrix
            for k, (_, nm, col) in enumerate(MODELS):
                yk = 2 - k
                for xi, b in zip(x, order):
                    on = b[k] == 1
                    axm.scatter(xi, yk, s=70, color=col if on else "#e6e5df", zorder=3,
                                edgecolors="white", linewidths=1.5)
            for xi, b in zip(x, order):
                ys = [2 - k for k in range(3) if b[k]]
                if len(ys) > 1: axm.plot([xi, xi], [min(ys), max(ys)], color="#8a8980", lw=2, zorder=2)
            axm.text(x[-1], -0.9, "none", ha="center", va="center", fontsize=8.5, color=MUTED)
            axm.set_ylim(-1.3, 2.6); axm.set_yticks([2, 1, 0])
            axm.set_yticklabels([f"{nm.split()[0]} ({s.sum()})" for (_, nm, _), s in zip(MODELS, S)], fontsize=9, color=INK)
            axm.tick_params(axis="both", length=0); axm.tick_params(axis="x", labelbottom=False)
            for s in axm.spines.values(): s.set_visible(False)
    fig.suptitle(f"Qwen3-Base GRPO (kl=3e-3, 1 epoch, rollout n=4), step {STEP}: UpSet view of per-problem correctness\n"
                 "each bar = problems solved by exactly the highlighted models (count above bar); "
                 "0.6B lr 1e-6, 1.7B/4B lr 3e-6", fontsize=13, color=INK, y=0.97)
    out = f"{OUT}/upset_qwen3_0.6B-1.7B-4B_step{STEP}.png"
    plt.savefig(out, dpi=130, facecolor="white", bbox_inches="tight"); print(out)

euler(); upset()
