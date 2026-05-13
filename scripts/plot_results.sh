#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --account=kdbrantley_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A.out
#SBATCH -t 01:00:00

if [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster: see scripts/train_grpo.sh for notes.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${MIT_CONDA_ENV:-base}"
else
    module load Miniforge3/26.1.0-fasrc01
    source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
    conda activate verl
fi

export WANDB_MODE="online"
export WANDB_ENTITY="progressive_distill"
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
elif [ "$USER" = "sdholakia" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
elif [ "${CLUSTER:-}" = "mit" ]; then
    _model_dir=${HOME}/models
else
    echo "ERROR: Unknown user $USER. Set MODEL_DIR explicitly." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}/results}
log_dir=${LOG_DIR:-${project_dir}/logs}
figures_dir=${FIGURES_DIR:-${project_dir}/figures}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
condition=${CONDITION:-}

condition_arg=""
if [ -n "$condition" ]; then
    condition_arg="--condition ${condition}"
fi

echo "W&B mode:   ${WANDB_MODE}"
echo "W&B entity: ${WANDB_ENTITY}"

python3 ${project_dir}/plot_results.py \
    --model_name     ${model_name} \
    --exp_name       ${exp_name} \
    --result_dir     ${result_dir} \
    --log_dir        ${log_dir} \
    --figures_dir    ${figures_dir} \
    --model_base_dir ${MODEL_DIR:-$_model_dir} \
    ${condition_arg}
