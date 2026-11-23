"""Plot the GT-NTP / distribution diagnosis CSV from scripts/diagnose_gt_ntp.py.

Produces two figures under figures/<model>/<exp>/diagnosis/:
  gt_ntp_metrics.png — 2x3 small multiples over training step:
      ce_answer (macro + p90), ce_full, entropy, top1, kl_base, stop_prob
  gt_ntp_vs_eval.png — ce_answer alongside held-out eval mean@1/mean@32
      (read from results/<model>/<exp>/global_step_*/<dataset>_*.scores)

Each panel: one line per dataset (ID balanced = blue, OOD balanced5 = orange).
Step 0 = base model.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# validated 2-slot categorical palette (dataviz slots 1-2, light mode)
SERIES = {
    "balanced": {"color": "#2a78d6", "label": "ID (n=3,4)"},
    "balanced5": {"color": "#eb6834", "label": "OOD (n=5)"},
}
TEXT, MUTED = "#0b0b0b", "#52514e"

PANELS = [
    ("ce_answer_macro", "GT-answer NTP loss (CE, nats/token)", "ce_answer"),
    ("ce_full_macro", "Full-continuation NTP loss", "ce_full"),
    ("entropy_macro", "Predictive entropy (nats)", "entropy"),
    ("top1_macro", "Top-1 probability", "top1"),
    ("kl_base_macro", "KL(ckpt || base)", "kl_base"),
    ("stop_prob_macro", "p(stop | ...</answer>)", "stop_prob"),
]


def style_ax(ax, title):
    ax.set_title(title, fontsize=10, color=TEXT)
    ax.grid(True, color="#e8e8e6", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def load_eval_scores(results_dir, dataset):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "global_step_*", f"{dataset}_*.scores")):
        with open(f) as fh:
            step, m1, m32 = json.load(fh)
        rows.append({"step": step, "mean@1": m1, "mean@32": m32})
    return pd.DataFrame(rows).sort_values("step") if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--results-dir", required=True,
                    help="results/<MODEL>/<EXP_NAME> (for the eval cross-reference)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("step")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Figure 1: 2x3 metric small multiples ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (col, title, _) in zip(axes.flat, PANELS):
        for name, spec in SERIES.items():
            d = df[df["dataset"] == name]
            ax.plot(d["step"], d[col], color=spec["color"], linewidth=2,
                    marker="o", markersize=4, label=spec["label"], zorder=3)
            if col == "ce_answer_macro":
                ax.plot(d["step"], d["ce_answer_p90"], color=spec["color"],
                        linewidth=1.2, linestyle=":", zorder=2)
        style_ax(ax, title)
        if col == "ce_answer_macro":
            ax.text(0.98, 0.02, "dotted = p90 across prompts", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("training step", fontsize=9, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("GT-answer NTP + predictive-distribution diagnosis (step 0 = base model)",
                 fontsize=11, color=TEXT, y=1.04)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "gt_ntp_metrics.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    print(out1)

    # --- Figure 2: ce_answer vs held-out eval accuracy ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for name, spec in SERIES.items():
        d = df[df["dataset"] == name]
        axes[0].plot(d["step"], d["ce_answer_macro"], color=spec["color"], linewidth=2,
                     marker="o", markersize=4, label=spec["label"], zorder=3)
        ev = load_eval_scores(args.results_dir, name)
        if not ev.empty:
            axes[1].plot(ev["step"], ev["mean@1"], color=spec["color"], linewidth=2,
                         marker="o", markersize=4, zorder=3)
            axes[2].plot(ev["step"], ev["mean@32"], color=spec["color"], linewidth=2,
                         marker="o", markersize=4, zorder=3)
    for ax, title in zip(axes, ["GT-answer NTP loss", "eval mean@1 (temp 0.6)",
                                "eval mean@32 (temp 0.6)"]):
        style_ax(ax, title)
        ax.set_xlabel("training step", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "gt_ntp_vs_eval.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    print(out2)


if __name__ == "__main__":
    main()
