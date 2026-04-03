"""
plot_results.py

Produces 4 plots for a given model/experiment run:
  1. val_reward_vs_step.png  — val reward mean@4 vs. training step (from log)
  2. eval_n34.png            — mean@1 + mean@32 vs. checkpoint step (n=3,4)
  3. eval_n5.png             — mean@1 + mean@32 vs. checkpoint step (n=5)
  4. eval_n6.png             — mean@1 + mean@32 vs. checkpoint step (n=6)

Plots 2-4 are skipped gracefully if the eval results don't exist.

Usage:
    python3 plot_results.py --model_name Qwen2.5-1.5B --exp_name balanced-grpo-seed1
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gather_experiment import find_training_log, parse_step_metrics
from grader_utils import compute_score

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STYLE = {"font.size": 11, "figure.dpi": 150}
COLOR_VAL   = "#2563eb"   # blue  — val reward
COLOR_MEAN1 = "#16a34a"   # green — mean@1
COLOR_MEAN32 = "#dc2626"  # red   — mean@32
LINEWIDTH  = 1.5
MARKERSIZE = 3
FIGSIZE    = (7, 4)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   required=True)
    parser.add_argument("--exp_name",     required=True)
    parser.add_argument("--result_dir",   default="results")
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--figures_dir",  default="figures")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_reward_curve(log_dir, model_name, exp_name):
    """Return sorted list of (step, val_reward) from the training log."""
    log_path, _ = find_training_log(log_dir, model_name, exp_name)
    if log_path is None:
        print(f"  WARNING: No training log found for {model_name}/{exp_name}")
        return []
    with open(log_path) as f:
        content = f.read()
    step_metrics = parse_step_metrics(content)
    return [
        (s, d["val_reward"])
        for s, d in sorted(step_metrics.items())
        if "val_reward" in d
    ]


def compute_mean_metrics(result_dir, model_name, exp_name, eval_dataset):
    """
    For each checkpoint step, compute mean@1 and mean@32.

    mean@1  — score outputs[0] as binary (1.0 correct else 0.0), average across prompts
    mean@32 — average raw compute_score() across all 32 outputs per prompt, then across prompts

    Returns sorted list of (step, mean1, mean32), or [] if no results found.
    """
    pattern = os.path.join(
        result_dir, model_name, exp_name,
        "global_step_*",
        f"{eval_dataset}_temp*.json",
    )
    files = glob.glob(pattern)
    if not files:
        return []

    devnull = open(os.devnull, "w")
    results = []

    for path in files:
        m = re.search(r"global_step_(\d+)", path)
        if not m:
            continue
        step = int(m.group(1))

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        sum1 = sum32 = 0.0
        n_prompts = 0

        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            for item in data:
                outputs = item.get("outputs", [])
                if not outputs:
                    continue
                target = item["target"]
                extra  = {"numbers": item["nums"]}

                # mean@1: binary score of first output
                s1 = compute_score(None, outputs[0], target, extra)
                sum1 += float(s1 == 1.0)

                # mean@32: average raw score across all outputs
                scores = [compute_score(None, out, target, extra) for out in outputs]
                sum32 += sum(scores) / len(scores)

                n_prompts += 1
        finally:
            sys.stdout = old_stdout

        if n_prompts > 0:
            results.append((step, sum1 / n_prompts, sum32 / n_prompts))

    devnull.close()
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_val_reward(val_curve, model_name, exp_name, figures_dir):
    if not val_curve:
        print("  Skipping val_reward plot (no data)")
        return
    xs, ys = zip(*val_curve)
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, linewidth=LINEWIDTH, color=COLOR_VAL)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Val reward mean@4")
    ax.set_title(f"Val reward — {model_name} / {exp_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_dir, "val_reward_vs_step.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_eval(metrics, n_label, out_path, title_prefix):
    """Plot mean@1 and mean@32 on the same axes vs. checkpoint step."""
    xs      = [r[0] for r in metrics]
    mean1s  = [r[1] for r in metrics]
    mean32s = [r[2] for r in metrics]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, mean1s,  linewidth=LINEWIDTH, color=COLOR_MEAN1,
            marker="o", markersize=MARKERSIZE, label="mean@1")
    ax.plot(xs, mean32s, linewidth=LINEWIDTH, color=COLOR_MEAN32,
            marker="o", markersize=MARKERSIZE, label="mean@32")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Eval [{n_label}] — {title_prefix}")
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_name = args.model_name
    exp_name   = args.exp_name

    figures_dir = os.path.join(args.figures_dir, model_name, exp_name)
    os.makedirs(figures_dir, exist_ok=True)

    title_prefix = f"{model_name} / {exp_name}"
    print(f"\nPlotting: {title_prefix}")
    print("=" * 60)

    # --- Plot 1: val reward vs. training step ---
    print("\n[1] Val reward vs. training step...")
    val_curve = load_val_reward_curve(args.log_dir, model_name, exp_name)
    plot_val_reward(val_curve, model_name, exp_name, figures_dir)

    # --- Plots 2-4: mean@1 + mean@32 per eval dataset ---
    dataset_configs = [
        ("balanced",  "n=3,4", "eval_n34.png"),
        ("balanced5", "n=5",   "eval_n5.png"),
        ("balanced6", "n=6",   "eval_n6.png"),
    ]

    for dataset, n_label, fname in dataset_configs:
        print(f"\n[eval] Dataset: {dataset} ({n_label})")
        metrics = compute_mean_metrics(args.result_dir, model_name, exp_name, dataset)
        if not metrics:
            print(f"  Skipping: no eval results found for {dataset}")
            continue
        print(f"  Checkpoints found: {len(metrics)}")
        out = os.path.join(figures_dir, fname)
        plot_eval(metrics, n_label, out, title_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
