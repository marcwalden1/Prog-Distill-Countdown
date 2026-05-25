"""Export cached eval scores for ID-vs-OOD Pareto plots.

The output is a git-friendly CSV with one row per
(experiment, checkpoint, eval dataset). It is intentionally derived from
`.scores` caches written by plot_results.py, so regenerating it is cheap.
"""

import argparse
import csv
import glob
import json
import os
import re


DATASET_META = {
    "balanced": ("id", "n=3/4"),
    "balanced5": ("ood", "n=5"),
    "balanced6": ("ood", "n=6"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen2.5-0.5B")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--output", default="analysis_data/qwen05b_grpo_on_top_pareto_eval.csv")
    parser.add_argument("--grpo-lr", default="1e-6")
    parser.add_argument("--grpo-kl", default="3e-4")
    return parser.parse_args()


def infer_method(exp_name):
    if "progdistill" in exp_name:
        return "progdistill"
    if "distill" in exp_name:
        return "distill"
    return None


def infer_field(pattern, text):
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def parse_exp_name(exp_name):
    return {
        "method": infer_method(exp_name),
        "sft_lr": infer_field(r"sftlr([0-9]+e-[0-9]+|[0-9.]+)", exp_name),
        "grpo_lr": infer_field(r"grpo-lr([0-9]+e-[0-9]+|[0-9.]+)", exp_name),
        "grpo_kl": infer_field(r"kl([0-9]+e-[0-9]+|[0-9.]+)", exp_name),
        "seed": infer_field(r"seed(\d+)", exp_name),
    }


def parse_score_path(path, result_dir, model_name):
    rel = os.path.relpath(path, os.path.join(result_dir, model_name))
    parts = rel.split(os.sep)
    if len(parts) < 3:
        return None

    exp_name, checkpoint, filename = parts[0], parts[1], parts[2]
    dataset = filename.split("_temp", 1)[0]
    if dataset not in DATASET_META:
        return None

    if checkpoint.startswith("global_step_"):
        stage = "post_grpo"
        checkpoint_step = checkpoint.removeprefix("global_step_")
    elif checkpoint.startswith("round_"):
        stage = "pre_grpo"
        checkpoint_step = checkpoint.removeprefix("round_")
    elif checkpoint == "sft_final":
        stage = "pre_grpo"
        checkpoint_step = "1600"
    else:
        return None

    temp = infer_field(r"_temp([^_]+)", filename)
    samples = infer_field(r"_n(\d+)", filename)
    max_tokens = infer_field(r"_max(\d+)", filename)

    return exp_name, checkpoint, checkpoint_step, dataset, stage, temp, samples, max_tokens


def main():
    args = parse_args()
    pattern = os.path.join(args.result_dir, args.model_name, "*", "*", "*.json.scores")
    rows = []

    for score_file in sorted(glob.glob(pattern)):
        parsed = parse_score_path(score_file, args.result_dir, args.model_name)
        if parsed is None:
            continue

        exp_name, checkpoint, checkpoint_step, dataset, stage, temp, samples, max_tokens = parsed
        if "obsolete" in exp_name:
            continue
        exp = parse_exp_name(exp_name)
        if exp["method"] not in {"distill", "progdistill"}:
            continue
        if exp["grpo_lr"] != args.grpo_lr or exp["grpo_kl"] != args.grpo_kl:
            continue

        with open(score_file) as f:
            step, mean_at_1, mean_at_32 = json.load(f)

        split, n_label = DATASET_META[dataset]
        rows.append({
            "model_name": args.model_name,
            "method": exp["method"],
            "stage": stage,
            "sft_lr": exp["sft_lr"],
            "grpo_lr": exp["grpo_lr"],
            "grpo_kl": exp["grpo_kl"],
            "seed": exp["seed"],
            "exp_name": exp_name,
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint_step or step,
            "eval_dataset": dataset,
            "split": split,
            "n_label": n_label,
            "temperature": temp,
            "samples": samples,
            "max_tokens": max_tokens,
            "mean_at_1": f"{mean_at_1:.6f}",
            "mean_at_32": f"{mean_at_32:.6f}",
            "source_json": score_file.removesuffix(".scores"),
            "score_file": score_file,
        })

    rows.sort(key=lambda r: (
        r["model_name"],
        r["method"],
        r["sft_lr"],
        int(r["seed"] or 0),
        r["exp_name"],
        int(r["checkpoint_step"] or 0),
        r["eval_dataset"],
    ))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        "model_name",
        "method",
        "stage",
        "sft_lr",
        "grpo_lr",
        "grpo_kl",
        "seed",
        "exp_name",
        "checkpoint",
        "checkpoint_step",
        "eval_dataset",
        "split",
        "n_label",
        "temperature",
        "samples",
        "max_tokens",
        "mean_at_1",
        "mean_at_32",
        "source_json",
        "score_file",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
