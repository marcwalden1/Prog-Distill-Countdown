# CLAUDE.md — RL-skill-comp

## What this repo is

Research code for the paper **"How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown"** (arXiv:2512.01775, NeurIPS 2025 workshops). It trains LLMs with GRPO (or PPO) on the **Countdown task**: given a target number and N input numbers, generate an arithmetic equation using all N numbers exactly once that equals the target, using +, −, ×, ÷.

---

## Cluster environment

- **Cluster:** Harvard Kempner (SLURM)
- **Conda env:** verl (Miniforge3/26.1.0-fasrc01)
- **Python:** 3.10
- **GPUs:** nvidia_h100_80gb_hbm3 (96 GB) or A100 (80 GB) depending on user partition
  - 0.5B / 1.5B / 3B: 4 GPUs per job
  - 7B: 8 GPUs per job

> Each user has their own partition, account, and storage paths. See your personal ~/.claude/CLAUDE.md.

---

## Key paths

| What | Path |
|---|---|
| Training data | data/balanced/train.parquet (282 MB) |
| Test data (n=3,4) | data/balanced/test.parquet (~997 rows, deduplicated) |
| Test data (n=5) | data/balanced5/test.parquet |
| Test data (n=6) | data/balanced6/test.parquet |
| Raw puzzles | data/countdown_{size}_pattern_{idx}.json |
| SLURM logs | logs/%x-%A-%a.out |
| WandB (offline) | verl/wandb/ |
| Models | USER-SPECIFIC — see personal CLAUDE.md |
| Checkpoints | USER-SPECIFIC — see personal CLAUDE.md |

**Available models:**
- Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B
- Llama-3.2-3B (requires tokenizer_config.json override)

---

## Training: scripts/train_grpo.sh

### Launching

    export MODEL_NAME=Qwen2.5-1.5B   # required
    sbatch scripts/train_grpo.sh

    # Override array for multiple seeds:
    sbatch --array=1-3 scripts/train_grpo.sh

    # 7B needs 8 GPUs:
    sbatch --gres=gpu:nvidia_h100_80gb_hbm3:8 --array=1-3 -t 30:00:00 scripts/train_grpo.sh

### Environment variables (all optional overrides)

| Var | Default | Meaning |
|---|---|---|
| MODEL_NAME | Qwen2.5-1.5B | Model to train |
| PROJECT_DIR | $PWD | Repo root |
| CHECKPOINT_DIR | USER-SPECIFIC | Checkpoint base dir — set in personal CLAUDE.md |
| DATA_SOURCE | balanced | Data directory under data/ |
| EXP_NAME | {data_source}-grpo-seed{SLURM_ARRAY_TASK_ID} | Experiment name |
| MAX_LENGTH | 1024 | Max response length |
| CHECKPOINT_PATH | {checkpoint_dir}/checkpoints/{model_name} | Output dir |

### Key hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | GRPO (algorithm.adv_estimator=grpo) |
| Learning rate | 1e-6 |
| Train batch size | 256 |
| Rollout samples per prompt | 4 (rollout.n=4) |
| Validation samples per prompt | 4 (val_kwargs.n=4) |
| Val sampling | do_sample=True, temperature=1.0 |
| KL loss | use_kl_loss=True, coef=0.001, type=low_var_kl |
| KL in reward | False |
| Advantage normalization | False (norm_adv_by_std_in_grpo=False) |
| Entropy coeff | 0 |
| Loss aggregation | token-mean |
| Save/test freq | every 50 steps |
| Total epochs | 1 |
| Gradient checkpointing | True |
| Strategy | FSDP2 |
| Generation engine | vLLM, gpu_memory_utilization=0.6 |
| Prompt max length | 1024 |
| Response max length | 1024 |

### WandB

- Mode: offline (WANDB_MODE=offline)
- Project: countdown
- Run name: {MODEL_NAME}-{exp_name} (e.g. Qwen2.5-1.5B-balanced-grpo-seed1)
- Sync later with: wandb sync verl/wandb/offline-run-*/

### Checkpoints

Saved to {output_dir}/{exp_name}/global_step_{N}/ every 50 steps.

---

## Training: scripts/train_ppo.sh

Same structure as GRPO but:
- algorithm.adv_estimator=gae (standard PPO with critic)
- Has a separate critic model (critic.model.path, critic.optim.lr=1e-5)
- Default model: Qwen2.5-3B
- Smaller mini batch (64 vs 256)

---

## Data pipeline

### Puzzle sizes and pattern counts

| Size | Patterns | Samples/pattern | Total |
|---|---|---|---|
| n=3 | 18 | 4000 | 72k |
| n=4 | 96 | 4000 | 384k |
| n=5 | 558 | 10 | 5580 |
| n=6 | 4328 | 1 | 4328 |

### Regenerating data

    for i in {0..17};   do python generate_puzzles.py --puzzle_size 3 --pattern_index ${i} --num_data 4000; done
    for i in {0..95};   do python generate_puzzles.py --puzzle_size 4 --pattern_index ${i} --num_data 4000; done
    for i in {0..557};  do python generate_puzzles.py --puzzle_size 5 --pattern_index ${i} --num_data 10;   done
    for i in {0..4327}; do python generate_puzzles.py --puzzle_size 6 --pattern_index ${i} --num_data 1;    done
    python preprocess_balanced.py

### Train/test split (preprocess_balanced.py)

- n=3,4: first 10 samples/pattern -> test, rest -> train
- Test set deduplicated by (sorted(nums), target) -> ~997 unique puzzles
- n=5,6: test-only sets (balanced5/, balanced6/)

### Parquet schema

    prompt: [{"role": "user", "content": "<prompt text>"}]
    ability: "math"
    reward_model: {"style": "rule", "ground_truth": <target int>}
    extra_info: {
      "numbers": [list of ints],
      "puzzle_size": int,
      "canonical_pattern_index": int,
      "canonical_pattern": str  (e.g. "((A-B)*C)+D")
    }

### Prompt format (template_type='base')

    A conversation between User and Assistant...
    User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
    Assistant: Let me solve this step by step.
    <think>

---

## Reward function: grader_utils.py

compute_score(data_source, solution_str, ground_truth, extra_info, format_score=0.1, score=1.0)

- Extracts last <answer>...</answer> match from model output
- Validates all numbers used exactly once
- Evaluates equation safely (restricted eval)
- Returns: 1.0 (correct), 0.1 (valid format, wrong answer), 0.0 (no answer / invalid)

---

## Running the pipeline: scripts/run_pipeline.sh

**Always use this to launch a new experiment.** Submits train → eval → plot_results as a single SLURM dependency chain — no manual follow-up needed.

```bash
# Standard run
MODEL_NAME=Qwen2.5-1.5B EXP_NAME=balanced-grpo-seed1 bash scripts/run_pipeline.sh

# With KL coef override
MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-grpo-kl1e-4-seed1 KL_COEF=0.0001 bash scripts/run_pipeline.sh

# 7B (needs 8 GPUs)
MODEL_NAME=Qwen2.5-7B EXP_NAME=balanced-grpo-seed1 \
  TRAIN_SBATCH_ARGS="--gres=gpu:nvidia_h100_80gb_hbm3:8 -t 30:00:00" \
  bash scripts/run_pipeline.sh
```

This submits:
1. `train_grpo.sh` — trains the model
2. `eval.sh` × 3 datasets (`balanced`, `balanced5`, `balanced6`) — each as a 10-task array job, all pending on train
3. `plot_results.sh` — pending on all 3 eval jobs completing

**Scripts that are NEVER run:** `analyze.sh`, `analyze.py`, `gather_experiment.py`. Tables are produced manually on request.

---

## Evaluation: scripts/eval.sh + eval.py

- Default array: `3,6,9,12,15,18,21,24,28,32` → evaluates checkpoints at steps 150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600
- Merges FSDP shards first (`verl.model_merger merge`)
- Runs `eval.py` with vLLM, n=32 samples/prompt, temperature=0.6
- Output: `results/{model}/{exp}/global_step_{N}/{dataset}_temp{T}_n{N}_max{M}.json`

---

## Plotting: scripts/plot_results.sh + plot_results.py

- CPU-only job on `serial_requeue`, 8 CPUs, 64GB RAM, 1h time limit
- Produces **5 plots** per run in `figures/{model}/{exp}/`:
  - `val_reward_vs_step.png` — val reward mean@4 vs. training step (from log)
  - `eval_n34.png` — mean@1 + mean@32 vs. checkpoint (n=3,4)
  - `eval_n5.png` — same for n=5
  - `eval_n6.png` — same for n=6
  - `response_length.png` — mean response length (tokens) vs. checkpoint, all 3 datasets
- Uses `.scores` and `.lengths` cache files to avoid recomputing unchanged results
- Scoring uses `Pool(8)` for parallelism; capped at 1000 prompts per file for n=5,6

---

## verl framework

Pinned fork of volcengine/verl at commit 083da9ab130efa2dc284eeb821a3edd6ce570fe3.
Entry point: python3 -m verl.trainer.main_ppo (used for both GRPO and PPO).

Key deps: torch==2.6.0, transformers==4.51.1, vllm==0.8.5.post1, flash-attn==2.7.4.post1, flashinfer-python==0.2.2

---

## Common operations

    # Check job status (replace with your username)
    squeue -u $USER

    # Cancel a job
    scancel <jobid>

    # Submit 0.5B job
    MODEL_NAME=Qwen2.5-0.5B sbatch scripts/train_grpo.sh

    # Submit with multiple seeds
    MODEL_NAME=Qwen2.5-1.5B sbatch --array=1-3 -t 10:00:00 scripts/train_grpo.sh

    # Tail a training log
    tail -f logs/train_grpo.sh-<jobid>-1.out

    # Sync WandB after training
    wandb sync verl/wandb/offline-run-*/

---

## Known issues / history

- Test set deduplication fix (2025-03): Original test set had duplicate prompts (same puzzle presented multiple times). Fixed in preprocess_balanced.py by deduplicating on (sorted(nums), target). Current test set: ~997 unique puzzles.
- val_kwargs.n: Was previously 7; corrected to 4 to match rollout.n=4.
- Buggy runs: Any checkpoints from before the deduplication fix should be discarded.