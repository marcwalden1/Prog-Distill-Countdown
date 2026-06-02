#!/usr/bin/env python3
"""Upload eval scores/lengths to wandb as an offline run.

Reads cached .scores ([step, mean@1, mean@32]) and .lengths ([step, mean_len])
from results/<model>/<exp>/global_step_*/<dataset>_temp*_n*_max*.json.{scores,lengths}
and logs per-step scalars to a new offline wandb run.

Usage:
    python scripts/upload_evals_to_wandb.py \
        --results-dir results/gemma-3-270m/<exp> \
        --model gemma-3-270m \
        --exp <exp>

After it prints, sync with: `wandb sync <offline-run-dir>`.
"""
import argparse
import glob
import json
import os
import re
import sys

DATASETS = ["balanced", "balanced5", "balanced6"]
SUFFIX = "_temp0.6_n32_max1024.json"


def find_steps(results_dir):
    steps = []
    for p in os.listdir(results_dir):
        m = re.match(r"global_step_(\d+)$", p)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def read_pair(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"  WARN: failed to read {path}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--project", default="prog_distill")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "progressive_distill"))
    ap.add_argument("--tags", nargs="*", default=["evals", "grpo"])
    ap.add_argument("--wandb-dir", default=None,
                    help="WANDB_DIR for the offline run. Defaults to /tmp/wandb_evals_<exp>_<pid>. "
                         "Must be on a local filesystem; wandb-core's IPC port file bursts the "
                         "30s startup timeout on NFS (which is where $HOME lives on FAS-RC).")
    ap.add_argument("--final-dir", default=None,
                    help="If set, the offline run dir is moved here after finish() so the artifacts "
                         "survive after /tmp gets reaped. Defaults to ./wandb_evals_<exp>/.")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"results-dir does not exist: {args.results_dir}")

    wandb_dir = args.wandb_dir or f"/tmp/wandb_evals_{args.exp}_{os.getpid()}"
    final_dir = args.final_dir or f"./wandb_evals_{args.exp}"
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    # WANDB_MODE defaults to online; override with `WANDB_MODE=offline ...` to
    # write locally and sync later.
    os.environ.setdefault("WANDB_MODE", "online")

    import wandb

    steps = find_steps(args.results_dir)
    if not steps:
        sys.exit(f"no global_step_* dirs in {args.results_dir}")
    print(f"Found {len(steps)} checkpoints: {steps}")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"{args.model}-{args.exp}-evals",
        tags=args.tags,
        config={
            "model": args.model,
            "exp": args.exp,
            "results_dir": os.path.abspath(args.results_dir),
            "datasets": DATASETS,
            "eval_temp": 0.6,
            "eval_n": 32,
            "eval_max_length": 1024,
        },
    )

    summary_rows = []
    for step in steps:
        row = {"step": step}
        for ds in DATASETS:
            base = f"{args.results_dir}/global_step_{step}/{ds}{SUFFIX}"
            sc = read_pair(base + ".scores")
            ln = read_pair(base + ".lengths")
            if sc is not None:
                # sc = [step, mean@1, mean@32]
                row[f"eval/{ds}/mean@1"] = sc[1]
                row[f"eval/{ds}/mean@32"] = sc[2]
            if ln is not None:
                # ln = [step, mean_length]
                row[f"eval/{ds}/length_mean"] = ln[1]
        wandb.log(row, step=step)
        summary_rows.append(row)

    # Pretty-print a compact summary table to stdout.
    print("\nLogged rows:")
    for r in summary_rows:
        print(" ", r)

    run.finish()

    # Move the offline run out of /tmp so it survives node reap. wandb writes
    # the actual run dir at $WANDB_DIR/wandb/offline-run-*; move that into
    # final_dir/wandb/ for a stable path the user can sync from later.
    import shutil
    src = os.path.join(wandb_dir, "wandb")
    dst = os.path.join(final_dir, "wandb")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    moved = []
    if os.path.isdir(src):
        for entry in os.listdir(src):
            if entry.startswith("offline-run-"):
                shutil.move(os.path.join(src, entry), os.path.join(dst, entry))
                moved.append(entry)
    print(f"\nOffline wandb run(s) moved to: {dst}")
    for m in moved:
        print(f"  {m}")
    print(f"\nSync with:\n  wandb sync {dst}/offline-run-*")


if __name__ == "__main__":
    main()
