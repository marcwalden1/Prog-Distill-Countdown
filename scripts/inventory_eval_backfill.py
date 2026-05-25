"""Inventory checkpoints and completed evals for backfill planning."""

import argparse
import csv
import glob
import os
import re


DEFAULT_CHECKPOINT_ROOT = (
    "/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/"
    "rl-checkpoints/checkpoints"
)
DEFAULT_STEPS = [150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600]
DEFAULT_DATASETS = ["balanced", "balanced5", "balanced6"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen2.5-0.5B")
    parser.add_argument("--run-glob", default="balanced-distill-*grpo-lr1e-6-kl3e-4-seed1*")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--output", default="analysis_data/qwen05b_seed1_distill_backfill_inventory.csv")
    parser.add_argument("--include-obsolete", action="store_true")
    return parser.parse_args()


def infer_sft_lr(exp_name):
    match = re.search(r"sftlr([0-9]+e-[0-9]+|[0-9.]+)", exp_name)
    return match.group(1) if match else ""


def infer_seed(exp_name):
    match = re.search(r"seed(\d+)", exp_name)
    return match.group(1) if match else ""


def eval_file(result_run_dir, step, dataset, suffix):
    pattern = os.path.join(
        result_run_dir,
        f"global_step_{step}",
        f"{dataset}_temp*_n*_max*.json{suffix}",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


def checkpoint_has_files(path):
    if not path or not os.path.isdir(path):
        return False
    pattern = os.path.join(path, "**", "*")
    return any(os.path.isfile(candidate) for candidate in glob.iglob(pattern, recursive=True))


def main():
    args = parse_args()
    checkpoint_model_dir = os.path.join(args.checkpoint_root, args.model_name)
    checkpoint_runs = {
        os.path.basename(path): path
        for path in glob.glob(os.path.join(checkpoint_model_dir, args.run_glob))
        if os.path.isdir(path)
    }
    result_model_dir = os.path.join(args.result_dir, args.model_name)
    result_runs = {
        os.path.basename(path): path
        for path in glob.glob(os.path.join(result_model_dir, args.run_glob))
        if os.path.isdir(path)
    }

    run_names = sorted(set(checkpoint_runs) | set(result_runs))
    if not args.include_obsolete:
        run_names = [name for name in run_names if "obsolete" not in name]
    rows = []
    for run_name in run_names:
        checkpoint_run_dir = checkpoint_runs.get(run_name, "")
        result_run_dir = result_runs.get(run_name, "")
        for step in DEFAULT_STEPS:
            checkpoint_path = (
                os.path.join(checkpoint_run_dir, f"global_step_{step}")
                if checkpoint_run_dir
                else ""
            )
            checkpoint_dir_exists = bool(checkpoint_path and os.path.isdir(checkpoint_path))
            checkpoint_materialized = checkpoint_has_files(checkpoint_path)
            for dataset in DEFAULT_DATASETS:
                json_path = eval_file(result_run_dir, step, dataset, "") if result_run_dir else ""
                score_path = eval_file(result_run_dir, step, dataset, ".scores") if result_run_dir else ""
                rows.append({
                    "model_name": args.model_name,
                    "exp_name": run_name,
                    "sft_lr": infer_sft_lr(run_name),
                    "seed": infer_seed(run_name),
                    "checkpoint_step": step,
                    "checkpoint_dir_exists": str(checkpoint_dir_exists).lower(),
                    "checkpoint_materialized": str(checkpoint_materialized).lower(),
                    "eval_dataset": dataset,
                    "eval_json_exists": str(bool(json_path)).lower(),
                    "score_cache_exists": str(bool(score_path)).lower(),
                    "checkpoint_path": checkpoint_path,
                    "eval_json_path": json_path,
                    "score_cache_path": score_path,
                })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        "model_name",
        "exp_name",
        "sft_lr",
        "seed",
        "checkpoint_step",
        "checkpoint_dir_exists",
        "checkpoint_materialized",
        "eval_dataset",
        "eval_json_exists",
        "score_cache_exists",
        "checkpoint_path",
        "eval_json_path",
        "score_cache_path",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
