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
# `conda activate` sets CONDA_PREFIX but on this cluster can leave PATH pointing
# at the base interpreter (personal miniconda from ~/.bashrc conflicts with the
# module Miniforge), so python3 loses numpy/torch. Force the env's bin onto PATH.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-bingbin}"
# Put wandb-core's IPC port-file on local /tmp — NFS-backed paths bust the
# 30s service-startup timeout (ServicePollForTokenError). Matches train_grpo.sh.
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "${WANDB_DIR}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
elif [ "$USER" = "sdholakia" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
elif [ "$USER" = "bingbin" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models
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

# Eval-set dir suffix, derived from DATA_SOURCE (see run_pipeline.sh):
# DATA_SOURCE=balanced-simple -> '-simple', unset/balanced -> ''.
dataset_suffix=${DATA_SOURCE:-balanced}
dataset_suffix=${dataset_suffix#balanced}

# Eval generation cap: eval.py files non-1024 caps under <step>/max<M>/, so the
# plotter must know it. It is defined by --max_tokens in EVAL_EXTRA_ARGS (inherited
# from run_pipeline.sh's environment); absent = eval.py's default 1024.
eval_max_tokens=$(echo " ${EVAL_EXTRA_ARGS:-} " | grep -oE -- "--max_tokens[= ]+[0-9]+" | grep -oE "[0-9]+$")
eval_max_tokens=${eval_max_tokens:-1024}

python3 ${project_dir}/plot_results.py \
    --model_name     ${model_name} \
    --exp_name       ${exp_name} \
    --result_dir     ${result_dir} \
    --log_dir        ${log_dir} \
    --figures_dir    ${figures_dir} \
    --model_base_dir ${MODEL_DIR:-$_model_dir} \
    --dataset_suffix="${dataset_suffix}" \
    --max_tokens=${eval_max_tokens} \
    ${OOD_SUBSAMPLE:+--ood_suffix=-r${OOD_SUBSAMPLE}} \
    ${condition_arg}
