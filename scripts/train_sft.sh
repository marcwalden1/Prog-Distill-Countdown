#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 06:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}
export WANDB_MODE="online"
export WANDB_ENTITY="progressive_distill"

# Tag this wandb run by pipeline mode so progdistill/distill SFT runs are filterable.
# DISTILL_MODE is set by run_pipeline.sh; falls back to "sft" when SFT is run standalone.
_sft_tags="sft"
if [ -n "${DISTILL_MODE:-}" ]; then
    _sft_tags="${DISTILL_MODE},sft"
fi
export WANDB_TAGS="${WANDB_TAGS:-${_sft_tags}}"

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
elif [ "$USER" = "sdholakia" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
else
    echo "ERROR: Unknown user $USER. Set CHECKPOINT_DIR explicitly." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-distill-grpo-seed1}

# SFT-specific parameters
sft_data_path=${SFT_DATA_PATH:?SFT_DATA_PATH must be set}
sft_base_model_path=${SFT_BASE_MODEL_PATH:-${MODEL_DIR:-$_model_dir}/${model_name}}
sft_output_dir=${SFT_OUTPUT_DIR:?SFT_OUTPUT_DIR must be set}
sft_train_steps=${SFT_TRAIN_STEPS:-1600}
sft_lr=${SFT_LR:-1e-5}
sft_seed=${SFT_SEED:-1}
sft_experiment_label=${SFT_EXPERIMENT_LABEL:-sft}

N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"

# Use SLURM job ID to derive a unique master port, avoiding conflicts between
# concurrent jobs that might land on the same node.
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))

echo "============================================================"
echo "SFT TRAINING CONFIG"
echo "Date:              $(date)"
echo "SLURM Job ID:      ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "GPUs:              ${N_GPUS}"
echo "Master port:       ${MASTER_PORT}"
echo "Model:             ${model_name}"
echo "Exp name:          ${exp_name}"
echo "SFT data:          ${sft_data_path}"
echo "Base model:        ${sft_base_model_path}"
echo "Output dir:        ${sft_output_dir}"
echo "Train steps:       ${sft_train_steps}"
echo "Learning rate:     ${sft_lr}"
echo "Seed:              ${sft_seed}"
echo "============================================================"

cd ${project_dir}/verl

set -o pipefail
python3 -m torch.distributed.run \
    --nproc_per_node=${N_GPUS} \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${sft_data_path}" \
    data.val_files="${sft_data_path}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.truncation=right \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    +data.seed=${sft_seed} \
    model.partial_pretrain=${sft_base_model_path} \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    optim.lr=${sft_lr} \
    trainer.total_training_steps=${sft_train_steps} \
    trainer.total_epochs=9999 \
    trainer.project_name=prog_distill \
    "trainer.experiment_name=${model_name}-${exp_name}-${sft_experiment_label}" \
    trainer.default_local_dir=${sft_output_dir} \
    "trainer.logger=['console','wandb']" \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    2>&1 | python3 -u ${project_dir}/scripts/timestamp_filter.py
SFT_EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail

if [ $SFT_EXIT_CODE -ne 0 ]; then
    echo "ERROR: SFT training failed with exit code $SFT_EXIT_CODE" >&2
    exit $SFT_EXIT_CODE
fi

# -----------------------------------------------------------------------
# Merge FSDP shards to HF format so the checkpoint can be used as:
#   - model.partial_pretrain for the next progdistill round
#   - MODEL_PATH for GRPO training
#
# The SFT trainer saves FSDP shards at global_step_N/ (same format as
# GRPO actor checkpoints). model_merger reads the shards + the
# huggingface/ config subdir, and writes a complete HF model
# (model.safetensors + config.json + tokenizer) to sft_output_dir/.
# -----------------------------------------------------------------------
step_dir=${sft_output_dir}/global_step_${sft_train_steps}
echo "Merging FSDP checkpoint from ${step_dir} to ${sft_output_dir}"
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir ${step_dir} \
    --target_dir ${sft_output_dir}
MERGE_EXIT_CODE=$?

if [ $MERGE_EXIT_CODE -ne 0 ]; then
    echo "ERROR: model_merger failed with exit code $MERGE_EXIT_CODE" >&2
    exit $MERGE_EXIT_CODE
fi

# Clean up FSDP shards to save disk space (HF model is now in sft_output_dir/)
rm -rf ${step_dir}
echo "Cleaned up FSDP shards: ${step_dir}"

chmod -R 770 ${sft_output_dir}
