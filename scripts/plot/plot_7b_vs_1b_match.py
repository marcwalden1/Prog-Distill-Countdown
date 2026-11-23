#!/usr/bin/env python3
"""Find the 7B pretraining checkpoint whose post-SFT n=3,4 performance best matches a
reference model, and plot every candidate in the pass@1-vs-pass@32 plane.

For each 7B arm the "best LR" is the one maximising pass@1 on the ID set (n=3,4);
that arm's (pass@1, pass@32) is its candidate point. Candidates are ranked against
the reference by:

  * joint    — Euclidean distance in the (pass@1, pass@32) plane. Both metrics are
               in [0,1] and this is what "matches on both" means, so it is the
               headline ranking.
  * pass@1   — |delta pass@1| alone.
  * pass@32  — |delta pass@32| alone.

These can disagree, so all three are printed; the plot shows the raw points and
lets you judge.

Reads only `.passk` caches (scripts/compute_passk.py). No mean@32 — see
research_logs/2026-08-25_olmo2-pretrain-ladder-grpo.md for why it is misleading here.
"""
import argparse
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def tokens_of(tag):
    m = re.search(r"pt(\d+)B", tag)
    return int(m.group(1)) if m else None


def passk(result_dir, model, exp, ds):
    for p in sorted(glob.glob(os.path.join(result_dir, model, exp, "sft_final",
                                           f"{ds}_temp*.json.passk"))):
        try:
            with open(p) as f:
                r = json.load(f)
            if isinstance(r, list) and len(r) >= 3:
                return r[1], r[2]
        except Exception:
            pass
    return None


def best_lr_point(result_dir, model, lrs, tmpl, steps, seed, ds):
    """(lr, pass@1, pass@32) maximising ID pass@1, or None if nothing scored."""
    best = None
    for lr in lrs:
        v = passk(result_dir, model, tmpl.format(steps=steps, lr=lr, seed=seed), ds)
        if v and (best is None or v[0] > best[1]):
            best = (lr, v[0], v[1])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-prefix", default="OLMo-2-1124-7B")
    ap.add_argument("--arms", required=True, help="comma-separated, e.g. pt42B,pt63B,...")
    ap.add_argument("--lrs", default="1e-5,3e-5,1e-4,3e-4")
    ap.add_argument("--ref-model", default="OLMo-2-0425-1B-pt4001B")
    ap.add_argument("--ref-lrs", default="1e-5,3e-5,1e-4")
    ap.add_argument("--ref-label", default="OLMo-2-1B, 4001B tokens (target)")
    ap.add_argument("--dataset", default="balanced-simple")
    ap.add_argument("--exp-template", default="balanced-simple-gtsft{steps}-lr{lr}-seed{seed}")
    ap.add_argument("--steps", default="200")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--figures-dir", default="figures/gtsft-lr-sweep")
    ap.add_argument("--best-only", action="store_true",
                    help="plot only each arm's best-LR point (clean companion view)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lrs = args.lrs.split(",")
    arms = args.arms.split(",")

    ref = best_lr_point(args.result_dir, args.ref_model, args.ref_lrs.split(","),
                        args.exp_template, args.steps, args.seed, args.dataset)
    if ref is None:
        raise SystemExit(f"ERROR: no .passk for reference {args.ref_model}")
    print(f"Reference {args.ref_model}: best lr{ref[0]}  pass@1 {ref[1]:.4f}  pass@32 {ref[2]:.4f}")
    ref_all = []
    for lr in args.ref_lrs.split(","):
        v = passk(args.result_dir, args.ref_model,
                  args.exp_template.format(steps=args.steps, lr=lr, seed=args.seed),
                  args.dataset)
        if v:
            ref_all.append({"lr": lr, "p1": v[0], "p32": v[1]})

    cands, missing, allpts = [], [], []
    for arm in arms:
        model = f"{args.model_prefix}-{arm}"
        for lr in lrs:
            v = passk(args.result_dir, model,
                      args.exp_template.format(steps=args.steps, lr=lr, seed=args.seed),
                      args.dataset)
            if v:
                allpts.append({"arm": arm, "tokens": tokens_of(arm), "lr": lr,
                               "p1": v[0], "p32": v[1]})
        b = best_lr_point(args.result_dir, model, lrs, args.exp_template,
                          args.steps, args.seed, args.dataset)
        if b is None:
            missing.append(arm)
            continue
        cands.append({"arm": arm, "tokens": tokens_of(arm), "lr": b[0],
                      "p1": b[1], "p32": b[2],
                      "d1": b[1] - ref[1], "d32": b[2] - ref[2],
                      "dist": math.hypot(b[1] - ref[1], b[2] - ref[2])})
    if not cands:
        raise SystemExit("ERROR: no 7B candidates scored yet")
    cands.sort(key=lambda c: c["tokens"] or 0)

    w = max(len(c["arm"]) for c in cands) + 1
    print(f"\nCandidates ({args.dataset}, best LR by pass@1):")
    hdr = f"  {'arm':<{w}}{'lr':<7}{'pass@1':>9}{'pass@32':>10}{'d pass@1':>11}{'d pass@32':>11}{'joint dist':>12}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for c in cands:
        print(f"  {c['arm']:<{w}}{c['lr']:<7}{c['p1']:>9.4f}{c['p32']:>10.4f}"
              f"{c['d1']:>+11.4f}{c['d32']:>+11.4f}{c['dist']:>12.4f}")

    print("\nBest match:")
    for key, name in (("dist", "joint (pass@1 & pass@32)"),
                      ("d1", "pass@1 only"), ("d32", "pass@32 only")):
        b = min(cands, key=lambda c: abs(c[key]))
        print(f"  {name:<26} {b['arm']} (lr{b['lr']})  "
              f"{b['p1']:.4f}/{b['p32']:.4f}  d={b[key]:+.4f}")
    if missing:
        print(f"\nNOT YET SCORED (excluded): {', '.join(missing)}")

    # ---- plot -------------------------------------------------------------
    # Every (arm, LR) point is drawn: colour = pretraining tokens, marker = SFT LR.
    # The arm's best-LR point (the one the match uses) is drawn large with a black
    # edge; the other LRs are small and faded, joined per arm by a faint line so the
    # LR response of each checkpoint is visible.
    LR_MARKER = {"1e-5": "v", "3e-5": "o", "1e-4": "s", "3e-4": "D"}
    lr_order = {lr: k for k, lr in enumerate(lrs)}
    toks = [c["tokens"] for c in allpts if c["tokens"]]
    norm = matplotlib.colors.LogNorm(vmin=min(toks), vmax=max(toks))
    cmap = matplotlib.cm.viridis

    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    best_key = {(c["arm"], c["lr"]) for c in cands}
    if args.best_only:
        allpts = [p for p in allpts if (p["arm"], p["lr"]) in best_key]
        ref_all = [p for p in ref_all if p["lr"] == ref[0]]

    # per-arm LR trajectory
    for arm in arms:
        pts = sorted([p for p in allpts if p["arm"] == arm],
                     key=lambda p: lr_order.get(p["lr"], 99))
        if len(pts) > 1:
            ax.plot([p["p1"] for p in pts], [p["p32"] for p in pts], "-",
                    color=cmap(norm(pts[0]["tokens"])), lw=0.6, alpha=0.22, zorder=1)

    for p in allpts:
        is_best = (p["arm"], p["lr"]) in best_key
        ax.plot(p["p1"], p["p32"], ls="none",
                marker=LR_MARKER.get(p["lr"], "o"),
                ms=13 if is_best else 7,
                color=cmap(norm(p["tokens"])),
                mec="black" if is_best else "0.5",
                mew=1.0 if is_best else 0.4,
                alpha=1.0 if is_best else 0.65,
                zorder=4 if is_best else 2)
    # label only the best-LR point of each arm, to keep the plot readable
    for c in cands:
        ax.annotate(f"{c['tokens']}B", (c["p1"], c["p32"]), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, fontweight="bold")

    # reference model: full LR sweep in red, best-LR as the star
    if ref_all:
        rs = sorted(ref_all, key=lambda p: lr_order.get(p["lr"], 99))
        ax.plot([p["p1"] for p in rs], [p["p32"] for p in rs], "-", color="#d62728",
                lw=1.0, alpha=0.5, zorder=3)
        for p in rs:
            if p["lr"] == ref[0]:
                continue
            ax.plot(p["p1"], p["p32"], ls="none", marker=LR_MARKER.get(p["lr"], "o"),
                    ms=8, color="#d62728", mec="0.3", mew=0.5, alpha=0.6, zorder=3)
    ax.plot(ref[1], ref[2], marker="*", ms=26, color="#d62728", mec="black", mew=0.9,
            ls="none", zorder=6,
            label=f"{args.ref_label}, best lr{ref[0]}\n{ref[1]:.4f}/{ref[2]:.4f}")
    best = min(cands, key=lambda c: c["dist"])
    ax.plot(best["p1"], best["p32"], marker="o", ms=22, mfc="none", mec="#d62728",
            mew=2.0, ls="none", zorder=5,
            label=f"closest 7B: {best['tokens']}B (lr{best['lr']})\n"
                  f"{best['p1']:.4f}/{best['p32']:.4f}  dist {best['dist']:.4f}")

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax); cb.set_label("7B pretraining tokens (B, log)")

    shape_handles = [Line2D([], [], ls="none", marker=LR_MARKER.get(lr, "o"), ms=9,
                            mfc="0.6", mec="black", mew=0.5, label=f"SFT lr={lr}")
                     for lr in lrs]
    shape_handles.append(Line2D([], [], ls="none", marker="o", ms=13, mfc="0.6",
                                mec="black", mew=1.0,
                                label="large = arm's best LR (used for matching)"))
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + shape_handles, fontsize=8, loc="lower right")

    ax.set_xlabel("pass@1   (first sample exactly correct)")
    ax.set_ylabel("pass@32  (>=1 of 32 samples exactly correct)")
    ax.set_title(("Post-GT-SFT on n=3,4, best LR per arm:\n" if args.best_only else
                  "Post-GT-SFT on n=3,4, every LR in the sweep:\n") +
                 "OLMo-2-7B pretraining ladder vs the 1B 4001B-token model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(args.figures_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.figures_dir, f"pass1_vs_pass32_7b_vs_1b{suffix}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}  ({len(allpts)} 7B points + {len(ref_all)} reference points)")


if __name__ == "__main__":
    main()
