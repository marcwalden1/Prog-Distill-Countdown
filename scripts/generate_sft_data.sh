#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 04:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
elif [ "$USER" = "sdholakia" ]; then
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
else
    echo "ERROR: Unknown user $USER. Set CHECKPOINT_DIR explicitly." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-distill-grpo-seed1}
teacher_exp_name=${TEACHER_EXP_NAME:?TEACHER_EXP_NAME must be set}
teacher_model_name=${TEACHER_MODEL_NAME:-${model_name}}
teacher_step=${TEACHER_STEP:-final}
data_source=${DATA_SOURCE:-balanced}
n_responses=${N_RESPONSES:-4}
max_length=${MAX_LENGTH:-1024}
filter_correct_only=${FILTER_CORRECT_ONLY:-false}
truncate=${TRUNCATE:-false}
# Default namespace suffixes with -n${n_responses} (and -truncate when
# truncate=true) so the pipeline and generate_sft_data.sh resolve to the
# same directory. Must be computed *after* n_responses / truncate.
_ns_default="teacher-${teacher_model_name}-${teacher_exp_name}-n${n_responses}"
if [ "${truncate}" = "true" ] || [ "${truncate}" = "1" ]; then
    _ns_default="${_ns_default}-truncate"
fi
sft_data_namespace=${SFT_DATA_NAMESPACE:-${_ns_default}}

# Resolve teacher checkpoint path
if [ "$teacher_step" = "final" ]; then
    # Find the highest global_step directory
    teacher_base=${checkpoint_dir}/checkpoints/${teacher_model_name}/${teacher_exp_name}
    teacher_step_num=$(ls -d ${teacher_base}/global_step_* 2>/dev/null | \
        sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$teacher_step_num" ]; then
        echo "ERROR: No checkpoints found under ${teacher_base}" >&2
        exit 1
    fi
    teacher_ckpt_path=${teacher_base}/global_step_${teacher_step_num}
else
    teacher_ckpt_path=${checkpoint_dir}/checkpoints/${teacher_model_name}/${teacher_exp_name}/global_step_${teacher_step}
fi

extra_args=""
if [ "${filter_correct_only}" = "true" ] || [ "${filter_correct_only}" = "1" ]; then
    extra_args="${extra_args} --filter_correct_only"
fi
if [ "${truncate}" = "true" ] || [ "${truncate}" = "1" ]; then
    extra_args="${extra_args} --truncate"
fi

output_path=${checkpoint_dir}/sft-data/${model_name}/${sft_data_namespace}/step_${teacher_step}.parquet
metadata_path=${output_path}.metadata
data_path=${project_dir}/data/${data_source}/train.parquet

echo "============================================================"
echo "GENERATE SFT DATA"
echo "Date:              $(date)"
echo "SLURM Job ID:      ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Student model:     ${model_name}"
echo "Exp name:          ${exp_name}"
echo "Data namespace:    ${sft_data_namespace}"
echo "Teacher model:     ${teacher_model_name}"
echo "Teacher exp:       ${teacher_exp_name}"
echo "Teacher step:      ${teacher_step}"
echo "Teacher ckpt:      ${teacher_ckpt_path}"
echo "Data source:       ${data_source}"
echo "N responses:       ${n_responses}"
echo "Max length:        ${max_length}"
echo "Filter correct:    ${filter_correct_only}"
echo "Truncate:          ${truncate}"
echo "Output path:       ${output_path}"
echo "============================================================"

# Merge teacher checkpoint if needed
if [ -f "${teacher_ckpt_path}/model.safetensors" ]; then
    echo "Teacher checkpoint already merged at ${teacher_ckpt_path}"
elif python3 -m verl.model_merger merge --backend fsdp \
        --local_dir ${teacher_ckpt_path}/actor/ \
        --target_dir ${teacher_ckpt_path}; then
    echo "Merged teacher checkpoint (actor/ subdir)"
elif python3 -m verl.model_merger merge --backend fsdp \
        --local_dir ${teacher_ckpt_path} \
        --target_dir ${teacher_ckpt_path}; then
    echo "Merged teacher checkpoint (flat)"
else
    echo "ERROR: Could not find or merge teacher checkpoint at ${teacher_ckpt_path}" >&2
    exit 1
fi

python3 ${project_dir}/scripts/generate_sft_data.py \
    --teacher_checkpoint_path ${teacher_ckpt_path} \
    --output_path ${output_path} \
    --data_path ${data_path} \
    --n_responses ${n_responses} \
    --max_length ${max_length} \
    ${extra_args}

mkdir -p "$(dirname "${metadata_path}")"
{
    echo "teacher_model_name=${teacher_model_name}"
    echo "teacher_exp_name=${teacher_exp_name}"
    echo "teacher_step=${teacher_step}"
    echo "data_source=${data_source}"
    echo "n_responses=${n_responses}"
    echo "filter_correct_only=${filter_correct_only}"
    echo "truncate=${truncate}"
    echo "max_length=${max_length}"
} > "${metadata_path}"

chmod -R 770 ${checkpoint_dir}/sft-data/${model_name}/${sft_data_namespace}
