"""Plot train-time actor entropy and response length from wandb histories.

Downloads `actor/entropy` and `response_length/mean` for the Qwen2.5-0.5B runs
listed in results/wandb_Qwen0.5B.md (entity/project progressive_distill/
prog_distill), caches the full per-step histories as CSVs under
results/wandb_Qwen0.5B/, subsamples to 20-step intervals, and writes one plot
per field to figures/entropy_length/.

Unlike plot_entropy_length.py (val-set sample entropy from eval JSONs), these
are TRAIN-time metrics: actor/entropy is the token-level policy entropy
(nats/token) over rollout tokens, response_length/mean the mean rollout length.

Usage: python3 scripts/plot/plot_wandb_entropy_length.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_entropy_length import COLORS  # user-maintained label -> color map

ENTITY_PROJECT = "progressive_distill/prog_distill"
RUNS = [  # (label matching COLORS, wandb run id) — from results/wandb_Qwen0.5B.md
    ("Qwen2.5-0.5B GRPO", "508ua0kv"),
    ("Qwen2.5-0.5B KD + GRPO", "3c6e19ww"),
    ("Qwen2.5-0.5B PD + GRPO", "w3g85krj"),
    ("Qwen2.5-0.5B PD mixture + GRPO", "f42n1d95"),
]
MARKERS = {
    "Qwen2.5-0.5B GRPO": "s",
    "Qwen2.5-0.5B KD + GRPO": "^",
    "Qwen2.5-0.5B PD + GRPO": "v",
    "Qwen2.5-0.5B PD mixture + GRPO": "D",
}

FIELDS = ["actor/entropy", "response_length/mean"]
STEP_INTERVAL = 20
CACHE_DIR = "results/wandb_Qwen0.5B"
FIGURES_DIR = "figures/entropy_length"


def fetch_history(run_id):
    cache = os.path.join(CACHE_DIR, f"{run_id}.csv")
    if os.path.isfile(cache):
        return pd.read_csv(cache)
    import wandb

    run = wandb.Api().run(f"{ENTITY_PROJECT}/{run_id}")
    df = run.history(keys=FIELDS, samples=100000, pandas=True)
    df = df[["_step"] + FIELDS].sort_values("_step")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def plot_field(dfs_by_label, field, ylabel, out_path, logy=False, legend_bg=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in dfs_by_label.items():
        sub = df[df["_step"] % STEP_INTERVAL == 0]
        ax.plot(sub["_step"], sub[field], color=COLORS[label],
                marker=MARKERS[label], markersize=3, linewidth=1.5, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if legend_bg:
        ax.legend(fontsize=9, frameon=True, framealpha=0.8,
                  facecolor="white", edgecolor="none")
    else:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    dfs = {}
    for label, rid in RUNS:
        dfs[label] = fetch_history(rid)
        print(f"{label} ({rid}): {len(dfs[label])} steps, "
              f"{dfs[label]['_step'].min()}..{dfs[label]['_step'].max()}")

    plot_field(dfs, "actor/entropy", "Policy entropy (nats/token)",
               os.path.join(FIGURES_DIR, "wandb_actor_entropy_vs_step.png"))
    plot_field(dfs, "actor/entropy", "Policy entropy (nats/token, log scale)",
               os.path.join(FIGURES_DIR, "wandb_actor_entropy_vs_step_logy.png"),
               logy=True)
    plot_field(dfs, "response_length/mean", "Mean rollout length (tokens)",
               os.path.join(FIGURES_DIR, "wandb_response_length_vs_step.png"),
               legend_bg=True)


if __name__ == "__main__":
    main()
