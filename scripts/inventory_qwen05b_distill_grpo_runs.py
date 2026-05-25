"""Inventory Qwen 0.5B distill+GRPO runs used for ID/OOD plotting."""

import csv
import glob
import os


MODEL_NAME = "Qwen2.5-0.5B"
CHECKPOINT_ROOT = (
    "/n/netscratch/kempner_bingbin_lab/Lab/sdholakia/archive/"
    "rl-checkpoints/checkpoints/Qwen2.5-0.5B"
)
EVAL_ROOT = "results/Qwen2.5-0.5B"
STEPS = [150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600]
DATASETS = ["balanced", "balanced5", "balanced6"]
RUNS = [
    {
        "exp_name": "balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1",
        "sft_lr": "1e-5",
        "seed": "1",
        "role": "canonical_seed1_missing_artifacts",
        "notes": (
            "W&B confirms finished run 1hvzfql4 on 2026-04-29, ending at "
            "global_step=300 after disk quota error at step 301. Earlier "
            "crashed attempt olq6oijq reached global_step=1088. No current "
            "checkpoint/eval artifacts found in lab or netscratch roots."
        ),
    },
    {
        "exp_name": "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1",
        "sft_lr": "3e-5",
        "seed": "1",
        "role": "canonical_seed1",
        "notes": "Clean run; archived seed1-v2 renamed to this canonical name.",
    },
    {
        "exp_name": "balanced-distill-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed1",
        "sft_lr": "3e-6",
        "seed": "1",
        "role": "canonical_seed1",
        "notes": "Seed1 run for lower SFT LR.",
    },
    {
        "exp_name": "balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed21",
        "sft_lr": "1e-5",
        "seed": "21",
        "role": "comparison_seed21",
        "notes": "Complete seed21 comparison run.",
    },
    {
        "exp_name": "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed21",
        "sft_lr": "3e-5",
        "seed": "21",
        "role": "comparison_seed21",
        "notes": "Complete seed21 comparison run.",
    },
    {
        "exp_name": "balanced-distill-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed21",
        "sft_lr": "3e-6",
        "seed": "21",
        "role": "comparison_seed21",
        "notes": "Complete seed21 comparison run.",
    },
]


def find_eval_file(eval_run_dir, step, dataset, suffix):
    pattern = os.path.join(
        eval_run_dir,
        f"global_step_{step}",
        f"{dataset}_temp*_n32_max1024.json{suffix}",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


def checkpoint_state(checkpoint_dir):
    merged = (
        os.path.isfile(os.path.join(checkpoint_dir, "config.json"))
        and bool(glob.glob(os.path.join(checkpoint_dir, "model*.safetensors")))
    )
    fsdp = (
        os.path.isfile(os.path.join(checkpoint_dir, "actor", "huggingface", "config.json"))
        and bool(glob.glob(os.path.join(checkpoint_dir, "actor", "model_world_size_*.pt")))
    )
    if merged:
        return "merged_hf"
    if fsdp:
        return "fsdp_actor"
    if os.path.isdir(checkpoint_dir):
        return "dir_only"
    return "missing"


def compress_steps(steps):
    return " ".join(str(step) for step in steps) if steps else ""


def main():
    os.makedirs("analysis_data", exist_ok=True)
    detail_rows = []
    registry_rows = []

    for run in RUNS:
        exp_name = run["exp_name"]
        checkpoint_run_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
        eval_run_dir = os.path.join(EVAL_ROOT, exp_name)
        checkpoint_steps = []
        missing_eval = []

        for step in STEPS:
            checkpoint_dir = os.path.join(checkpoint_run_dir, f"global_step_{step}")
            state = checkpoint_state(checkpoint_dir)
            if state in {"merged_hf", "fsdp_actor"}:
                checkpoint_steps.append(step)

            for dataset in DATASETS:
                eval_json = find_eval_file(eval_run_dir, step, dataset, "")
                score_cache = find_eval_file(eval_run_dir, step, dataset, ".scores")
                if not score_cache:
                    missing_eval.append(f"{dataset}@{step}")
                detail_rows.append({
                    "model_name": MODEL_NAME,
                    "exp_name": exp_name,
                    "sft_lr": run["sft_lr"],
                    "seed": run["seed"],
                    "role": run["role"],
                    "checkpoint_step": step,
                    "eval_dataset": dataset,
                    "checkpoint_state": state,
                    "eval_json_exists": str(bool(eval_json)).lower(),
                    "score_cache_exists": str(bool(score_cache)).lower(),
                    "checkpoint_path": checkpoint_dir,
                    "eval_json_path": eval_json,
                    "score_cache_path": score_cache,
                })

        registry_rows.append({
            "model_name": MODEL_NAME,
            "exp_name": exp_name,
            "sft_lr": run["sft_lr"],
            "seed": run["seed"],
            "role": run["role"],
            "checkpoint_root": checkpoint_run_dir,
            "eval_root": eval_run_dir,
            "checkpoint_steps_available": compress_steps(checkpoint_steps),
            "eval_score_complete": str(not missing_eval).lower(),
            "missing_eval_scores": " ".join(missing_eval),
            "notes": run["notes"],
        })

    with open("analysis_data/qwen05b_distill_grpo_run_registry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)

    with open("analysis_data/qwen05b_distill_grpo_eval_inventory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Wrote {len(registry_rows)} run rows")
    print(f"Wrote {len(detail_rows)} detailed inventory rows")


if __name__ == "__main__":
    main()
