#!/bin/bash
# scripts/run_pipeline.sh
# Submits the full train → eval → plot_results pipeline as a SLURM dependency chain.
#
# Usage:
#   MODEL_NAME=Qwen2.5-1.5B EXP_NAME=my-exp bash scripts/run_pipeline.sh
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=my-exp KL_COEF=0.0001 bash scripts/run_pipeline.sh
#
# For 7B (needs 8 GPUs):
#   MODEL_NAME=Qwen2.5-7B EXP_NAME=my-exp \
#     TRAIN_SBATCH_ARGS="--gres=gpu:nvidia_h100_80gb_hbm3:8 -t 30:00:00" \
#     bash scripts/run_pipeline.sh
#
# Optional env var overrides (passed through to sub-scripts):
#   KL_COEF       — KL loss coefficient (default: 0.001)
#   DATA_SOURCE   — data directory under data/ (default: balanced)
#   EXP_NAME      — experiment name (default: balanced-grpo-seed1)
#   EXTRA_ARGS    — extra args forwarded to eval.py

set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-Qwen2.5-1.5B}
export EXP_NAME=${EXP_NAME:-balanced-grpo-seed1}

TRAIN_SBATCH_ARGS=${TRAIN_SBATCH_ARGS:-""}

echo "===== Pipeline: $MODEL_NAME / $EXP_NAME ====="

# 1. Train
TRAIN_JID=$(sbatch --parsable $TRAIN_SBATCH_ARGS scripts/train_grpo.sh)
echo "Train:  job $TRAIN_JID"

# 2. Eval — 3 datasets in parallel, all depend on train finishing
EVAL_JID1=$(EVAL_DATASET=balanced  sbatch --parsable --dependency=afterok:${TRAIN_JID} scripts/eval.sh)
EVAL_JID2=$(EVAL_DATASET=balanced5 sbatch --parsable --dependency=afterok:${TRAIN_JID} scripts/eval.sh)
EVAL_JID3=$(EVAL_DATASET=balanced6 sbatch --parsable --dependency=afterok:${TRAIN_JID} scripts/eval.sh)
echo "Eval:   job $EVAL_JID1 (n=3,4)  $EVAL_JID2 (n=5)  $EVAL_JID3 (n=6)"

# 3. Plot — depends on ALL 3 eval array jobs completing successfully
PLOT_JID=$(sbatch --parsable \
  --dependency=afterok:${EVAL_JID1}:${EVAL_JID2}:${EVAL_JID3} \
  scripts/plot_results.sh)
echo "Plot:   job $PLOT_JID"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
