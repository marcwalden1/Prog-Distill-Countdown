#!/usr/bin/env python3
"""
2D early-stopping Pareto plots sourced from W&B (project prog_distill).

ID accuracy (n=3,4 mean@1) on x, OOD accuracy (n=5 or n=6 mean@1) on y, plotting
each GRPO-on-top run's trajectory across its evaluated checkpoints (steps
150..1600). One figure per OOD dataset.

Why W&B and not local .scores: only two GRPO-on-top runs have eval caches on
disk; the canonical sweep's eval data lives in the `*-eval` W&B runs
(schema eval/n3_4/mean_at_1, eval/n5/mean_at_1, eval/n6/mean_at_1).

Arms are defined in ARMS below as (label, wandb_run_id, color, is_distill).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

PROJECT = "prog_distill"
OOD = {"n5": "eval/n5/mean_at_1", "n6": "eval/n6/mean_at_1"}
OOD_TITLE = {"n5": "n=5 (OOD)", "n6": "n=6 (far OOD)"}
ID_KEY = "eval/n3_4/mean_at_1"

# label -> (wandb eval-run id, color, is_distill)
ARMS = {
    "distill→GRPO  sftlr3e-5 (kl3e-4)":     ("fd2ee841", "#d62728", True),
    "progdistill→GRPO sftlr3e-5 (kl3e-3)":  ("f257314d", "#1f77b4", False),
    "progdistill→GRPO sftlr1e-5 (kl3e-3)":  ("74531d33", "#2ca02c", False),
    "progdistill→GRPO sftlr3e-6 (kl3e-3)":  ("c5aa45bf", "#9467bd", False),
}

CAVEAT = ("Matched by student sftlr where possible; each arm uses its own standard KL "
          "(distill=3e-4, progdistill=3e-3). distill→GRPO only had one stable sftlr (3e-5).")


def load_arm(api, run_id):
    """{step: (ID, n5, n6)} for rows that carry the ID eval metric."""
    r = api.run(f"{PROJECT}/{run_id}")
    keys = [ID_KEY] + list(OOD.values()) + ["_step"]
    out = {}
    for row in r.history(samples=4000, pandas=False, keys=keys):
        if ID_KEY not in row or not isinstance(row[ID_KEY], (int, float)):
            continue
        step = int(row["_step"])
        out[step] = (row[ID_KEY], row.get(OOD["n5"]), row.get(OOD["n6"]))
    return out


def plot(ood, arms_xy, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for label, (steps, xs, ys, color, is_distill) in arms_xy.items():
        if not steps:
            continue
        ls = "-" if is_distill else "--"
        ax.plot(xs, ys, ls, color=color, alpha=0.45, zorder=1)
        ax.scatter(xs, ys, s=40, color=color, label=label, zorder=2)
        ax.scatter([xs[0]], [ys[0]], s=120, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=3)        # start
        ax.scatter([xs[-1]], [ys[-1]], s=190, marker="*", color=color,
                   edgecolors="k", linewidths=0.5, zorder=4)          # end
        for s, x, y in zip(steps, xs, ys):
            if s in (steps[0], steps[-1]) or s % 300 == 0:
                ax.annotate(str(s), (x, y), fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("In-dist accuracy  (n=3,4)  mean@1")
    ax.set_ylabel(f"OOD accuracy  ({OOD_TITLE[ood]})  mean@1")
    ax.set_title(f"Early-stopping Pareto — {OOD_TITLE[ood]}\n"
                 "○ = step 150   ★ = step 1600", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def dominance(ood, distill_xy, prog_arms):
    """For each progdistill arm: does any checkpoint dominate the WHOLE distill trajectory?"""
    d_pts = list(zip(distill_xy[1], distill_xy[2]))   # (x,y) over distill steps
    print(f"\n=== {OOD_TITLE[ood]} : does a progdistill checkpoint dominate the entire distill curve? ===")
    for label, (steps, xs, ys, *_), in prog_arms.items():
        wins = [s for s, x, y in zip(steps, xs, ys)
                if all(x >= dx and y >= dy for dx, dy in d_pts)
                and any(x > dx or y > dy for dx, dy in d_pts)]
        best_ood = max(zip(steps, xs, ys), key=lambda t: t[2]) if steps else None
        msg = f"dominating ckpts: {wins}" if wins else "none dominate full distill curve"
        bo = f"(best OOD: step {best_ood[0]} @ ID={best_ood[1]:.3f}, OOD={best_ood[2]:.3f})" if best_ood else ""
        print(f"  {label:42s} {msg}  {bo}")


def main():
    api = wandb.Api()
    figdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures", "Qwen2.5-0.5B", "pareto_wandb")
    raw = {label: (load_arm(api, rid), color, is_d)
           for label, (rid, color, is_d) in ARMS.items()}
    for label, (data, *_), in raw.items():
        print(f"[{label}] {len(data)} eval checkpoints, steps {sorted(data)[:3]}...{sorted(data)[-2:]}")

    for ood, ood_key in OOD.items():
        idx = 1 if ood == "n5" else 2
        arms_xy, distill_xy, prog_arms = {}, None, {}
        for label, (data, color, is_d) in raw.items():
            steps = sorted(s for s in data if data[s][idx] is not None)
            xs = [data[s][0] for s in steps]
            ys = [data[s][idx] for s in steps]
            arms_xy[label] = (steps, xs, ys, color, is_d)
            if is_d:
                distill_xy = (steps, xs, ys)
            else:
                prog_arms[label] = (steps, xs, ys, color, is_d)
        plot(ood, arms_xy, os.path.join(figdir, f"pareto_{ood}.png"))
        if distill_xy:
            dominance(ood, distill_xy, prog_arms)


if __name__ == "__main__":
    main()
