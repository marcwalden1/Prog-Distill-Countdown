#!/bin/bash
# Resubmit gemma student-RL chains after the verl fsdp_vllm.py fix.
# Each chain: train_grpo (2 H100s) -> eval×3 (1 H100 each, array) -> plot.
# Run from the repo root.
set -euo pipefail

cd /n/home06/sdholakia/RL-skill-comp

ACCT=kempner_bingbin_lab
PART=kempner_h100
GEMMA_EXTRA="actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4"
SFT_BASE=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints/sft-checkpoints/gemma-3-270m

submit_chain() {
    local exp="$1" mp="$2" kl="$3" distmode="$4"
    echo
    echo "=== Submitting chain: $exp"
    echo "    MODEL_PATH=$mp"
    echo "    KL_COEF=$kl  DISTILL_MODE=$distmode"

    local TRAIN_JID
    TRAIN_JID=$(MODEL_NAME=gemma-3-270m \
        MODEL_PATH="$mp" \
        EXP_NAME="$exp" \
        KL_COEF="$kl" \
        EXTRA_ARGS="$GEMMA_EXTRA" \
        DISTILL_MODE="$distmode" \
        sbatch --parsable --account=$ACCT --partition=$PART \
        --gres=gpu:nvidia_h100_80gb_hbm3:2 scripts/train_grpo.sh)
    echo "    TRAIN_JID=$TRAIN_JID"

    local EVAL_JIDS=""
    for dataset in balanced balanced5 balanced6; do
        local EJID
        EJID=$(MODEL_NAME=gemma-3-270m \
            EXP_NAME="$exp" \
            EVAL_DATASET="$dataset" \
            sbatch --parsable --account=$ACCT --dependency=afterok:$TRAIN_JID \
            scripts/eval.sh)
        EVAL_JIDS="${EVAL_JIDS}${EVAL_JIDS:+:}$EJID"
        echo "    EVAL[$dataset]=$EJID"
    done

    local PJID
    PJID=$(MODEL_NAME=gemma-3-270m \
        EXP_NAME="$exp" \
        sbatch --parsable --account=$ACCT --partition=kempner_requeue \
        --gres=gpu:nvidia_h100_80gb_hbm3:1 \
        --dependency=afterok:$EVAL_JIDS scripts/plot_results.sh)
    echo "    PLOT_JID=$PJID"
}

# 5 student-RL chains + 2 sweep — same configs as the broken ones we cancelled
# (1) distill-sftlr1e-5 + kl3e-4
submit_chain \
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1" \
    "$SFT_BASE/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-5-seed1" \
    "3e-4" "distill"

# (2) distill-sftlr3e-5 + kl3e-4
submit_chain \
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1" \
    "$SFT_BASE/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-5-seed1" \
    "3e-4" "distill"

# (3) distill-sftlr3e-6 + kl3e-4 (was 10369546, already cancelled before; resubmit)
submit_chain \
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed1" \
    "$SFT_BASE/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-seed1" \
    "3e-4" "distill"

# (4) progdistill-sftlr3e-6 + kl3e-3
submit_chain \
    "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr3e-6-grpo-lr1e-6-kl3e-3-seed1" \
    "$SFT_BASE/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr3e-6-seed1/round_1600" \
    "3e-3" "progdistill"

# (5) progdistill-sftlr3e-5 + kl3e-3
submit_chain \
    "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr3e-5-grpo-lr1e-6-kl3e-3-seed1" \
    "$SFT_BASE/gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr3e-5-seed1/round_1600" \
    "3e-3" "progdistill"

# (6) sftlr3e-6 distill + kl1e-3 (sweep)
submit_chain \
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-grpo-lr1e-6-kl1e-3-seed1" \
    "$SFT_BASE/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-seed1" \
    "1e-3" "distill"

# (7) sftlr3e-6 distill + kl3e-3 (sweep)
submit_chain \
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-grpo-lr1e-6-kl3e-3-seed1" \
    "$SFT_BASE/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr3e-6-seed1" \
    "3e-3" "distill"

echo
echo "All 7 chains submitted."
