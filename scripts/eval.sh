#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1  # explicit H100: vLLM 0.8.5 crashes on MIG slices ('MIG-<uuid>' device IDs)
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=kempner_requeue
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 01:30:00
#SBATCH --array 3,6,9,12,15,18,21,24,28,32

if [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster: see scripts/train_grpo.sh for notes.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${MIT_CONDA_ENV:-base}"
else
    module load Miniforge3/26.1.0-fasrc01
    source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
    conda activate verl
fi

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# User-specific defaults. Account is set here (not via #SBATCH) so it tracks
# $USER without requiring a per-user copy of this script; sbatch parses
# directives at submission time, so the static --account directive was removed
# and we verify $SLURM_JOB_ACCOUNT matches below.
if [ "$USER" = "mwalden" ]; then
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
    # Allow either the Kempner account (default for kempner_* partitions) or
    # the lab account (needed for gpu_requeue, which DenyAccounts kempner_*).
    _account=kempner_kdbrantley_lab
    _account_alt=kdbrantley_lab
elif [ "$USER" = "sdholakia" ]; then
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
    _account=kempner_bingbin_lab
elif [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster (CLUSTER=mit). CHECKPOINT_DIR override expected; account
    # left empty so the strict match check below is skipped (MIT account is
    # passed via the sbatch CLI / SBATCH_ACCOUNT, not pinned here).
    _checkpoint_dir=${HOME}/rl-checkpoints
    _account=""
else
    echo "ERROR: Unknown user $USER. Set CHECKPOINT_DIR explicitly." >&2
    exit 1
fi
if [ -n "$_account" ] && [ -n "${SLURM_JOB_ACCOUNT:-}" ] \
        && [ "$SLURM_JOB_ACCOUNT" != "$_account" ] \
        && [ "$SLURM_JOB_ACCOUNT" != "${_account_alt:-}" ]; then
    echo "ERROR: job running on account=$SLURM_JOB_ACCOUNT, expected $_account${_account_alt:+ or $_account_alt} for user $USER." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}
result_dir=${RESULT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
eval_dataset=${EVAL_DATASET:-balanced}
extra_args=${EVAL_EXTRA_ARGS:-""}

if [ -n "${EVAL_CHECKPOINT_PATH:-}" ]; then
    # SFT-only mode: eval a specific checkpoint (no array-index derivation).
    checkpoint_path=${EVAL_CHECKPOINT_PATH}
    result_path=${EVAL_RESULT_PATH:?EVAL_RESULT_PATH must be set when EVAL_CHECKPOINT_PATH is set}
else
    checkpoint_path=${checkpoint_dir}/checkpoints/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))
    result_path=${result_dir}/results/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))
fi

if [ -f "${checkpoint_path}/model.safetensors" ]; then
    echo "Merged checkpoint already present at ${checkpoint_path}"
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path}/actor/ --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/actor/
    rm -rf ${checkpoint_path}/critic/
    rm -rf ${checkpoint_path}/data.pt
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path} --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/*.pt
fi

python3 eval.py --checkpoint_path ${checkpoint_path} --result_path ${result_path} --eval_dataset ${eval_dataset} ${extra_args}

chmod -R 770 ${checkpoint_dir}/checkpoints/${model_name}/${exp_name}
chmod -R 770 ${result_dir}/results/${model_name}/${exp_name}
