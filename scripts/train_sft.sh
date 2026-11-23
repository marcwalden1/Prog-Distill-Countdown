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
# `conda activate` sets CONDA_PREFIX but on this cluster can leave PATH pointing
# at the base interpreter (personal miniconda from ~/.bashrc conflicts with the
# module Miniforge), so python3 loses numpy/torch. Force the env's bin onto PATH.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"

# Node-local, per-job triton kernel cache. The shared ~/.triton/cache default is
# poisoned by `cuda_utils` extensions built by other CPython versions, which raise
# "SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats" on the
# first kernel launch. See the same block in scripts/train_grpo.sh.
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-$$}"
mkdir -p "${TRITON_CACHE_DIR}"

# vLLM's torch.compile cache defaults to ~/.cache/vllm (NFS home) and is keyed
# only by the model graph hash, so two concurrent jobs compiling the SAME model
# write into the SAME directory. One job's rename into the cache invalidates the
# other's open handle and it dies at engine init with
#   OSError: [Errno 116] Stale file handle
# in triton CompiledKernel.__init__ -> BackendCompilerFailed. Observed 2026-09-22
# on job 47811579 (third of three Qwen3-0.6B GRPO jobs launched together; the two
# 0.6B jobs shared graph hash 23afc4795d). This is also the likely root cause of
# the historical "JSONDecodeError: Extra data" in the vLLM compile cache that was
# worked around with --enforce_eager. Keep it node-local and per-job, like
# TRITON_CACHE_DIR above; the cost is one ~1-2 min recompile per job.
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${VLLM_CACHE_ROOT}"

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-bingbin}"
# Put wandb-core's IPC port-file on local /tmp — NFS-backed paths bust the
# 30s service-startup timeout (ServicePollForTokenError). Matches train_grpo.sh.
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "${WANDB_DIR}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"

# Tag this wandb run by pipeline mode so progdistill/distill SFT runs are filterable.
# DISTILL_MODE is set by run_pipeline.sh; falls back to "sft" when SFT is run standalone.
# Also tag by student model size (e.g. "270m", "0.5B", "1.5B") — derived from the trailing
# segment of MODEL_NAME after the last hyphen. MODEL_NAME is the student being SFT'd.
_sft_tags="sft"
if [ -n "${DISTILL_MODE:-}" ]; then
    _sft_tags="${DISTILL_MODE},sft"
fi
_size_tag="${MODEL_NAME##*-}"
# From-scratch 40M models are named Qwen2.5-40M-<shape>; keep the size in the tag
# (e.g. "40M-wide", "40M-deep") rather than just the shape suffix.
if [[ "${MODEL_NAME}" == *-40M-* ]]; then
    _size_tag="40M-${MODEL_NAME##*-}"
fi
_sft_tags="${_sft_tags},${_size_tag}"
# Tag by model family ("Qwen2.5", "Qwen3", "OLMo-2", "gemma-3") so runs are
# filterable by family in the wandb UI. The size tag above deliberately drops the
# family, so otherwise it only survives as a prefix of the run name (not filterable
# as a tag). Rule: keep the leading hyphen-separated segments of MODEL_NAME up to
# the first size-like ("0.5B", "270m", "40M") or 4-digit release-date ("0425")
# segment -- Qwen3-0.6B-Base -> Qwen3, OLMo-2-1124-7B-pt135B -> OLMo-2.
_model_family() {
    local _out="" _seg
    local IFS='-'
    for _seg in $1; do
        [[ "${_seg}" =~ ^[0-9.]+[BbMm]$ || "${_seg}" =~ ^[0-9]{4}$ ]] && break
        _out="${_out:+${_out}-}${_seg}"
    done
    # No size/date segment anywhere (hand-named model dir): emit nothing rather
    # than a junk tag equal to the whole name.
    [ "${_out}" = "$1" ] && _out=""
    printf '%s' "${_out}"
}
_family_tag="$(_model_family "${MODEL_NAME}")"   # appended to WANDB_TAGS below
# Tag by prompt variant (e.g. "balanced", "balanced-simple") when the launcher
# exports DATA_SOURCE -- run_pipeline.sh does. SFT itself is driven by
# SFT_DATA_PATH, whose parquet path does not always encode the variant
# (teacher-distill rounds live under sft-data/<MODEL>/teacher-.../), so a
# standalone run with no DATA_SOURCE is left untagged rather than guessed.
if [ -n "${DATA_SOURCE:-}" ]; then
    _sft_tags="${_sft_tags},${DATA_SOURCE}"
fi
export WANDB_TAGS="${WANDB_TAGS:-${_sft_tags}}"
# Always carry the family tag, even when the caller preset WANDB_TAGS (the
# run_olmo2_*.sh launchers do), so filtering by family works for every run.
if [ -n "${_family_tag}" ] && [[ ",${WANDB_TAGS}," != *",${_family_tag},"* ]]; then
    export WANDB_TAGS="${WANDB_TAGS},${_family_tag}"
fi

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
elif [ "$USER" = "sdholakia" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
elif [ "$USER" = "bingbin" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models
    _checkpoint_dir=/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown
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
sft_log_freq=${SFT_LOG_FREQ:-50}
sft_warmup_ratio=${SFT_WARMUP_RATIO:-0.1}
# Padded sequence length. verl's SFTDataset pads every sample to this, so it sets
# the lm_head logits tensor size (micro_batch x max_length x vocab) and therefore
# peak memory — not just a truncation ceiling. Keep it just above the longest
# prompt+response in the data: the GT answer-only sets are ~120 tokens, where the
# historical 2048 wasted ~94% of every batch and OOMed 7B in the final validation
# pass (3.06 GiB logits alloc). Default stays 2048 so teacher-rollout SFT data,
# whose responses run to the 1024-token cap, is unaffected.
sft_max_length=${SFT_MAX_LENGTH:-2048}
# val_files defaults to the full train mixture (below), which can be ~400k
# rows; cap the final validation pass so it takes minutes, not hours.
# Set SFT_VAL_MAX_BATCHES=-1 to validate on everything.
sft_val_max_batches=${SFT_VAL_MAX_BATCHES:-500}
# Opt-in overrides (defaults = historical progdistill settings, so existing
# launchers are unaffected). Used by the Qwen3 4B->0.6B offline-distill runs
# (research_logs/2026-11-06_qwen3-4b-teacher-logit-sanity.md):
#   SFT_TRAIN_BATCH_SIZE / SFT_MICRO_BATCH_SIZE_PER_GPU  global / per-GPU micro batch
#   SFT_LR_SCHEDULER   cosine | wsd | constant (constant = local verl patch 7)
#   SFT_SAVE_FREQ      -1 = only the final step; >0 also saves every N steps
#   SFT_RESUME_MODE    disable | auto (auto resumes from the latest saved step)
#   SFT_EXTRA_ARGS     extra Hydra overrides, e.g. data.custom_cls.* for token-id datasets
sft_train_batch_size=${SFT_TRAIN_BATCH_SIZE:-64}
sft_micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE_PER_GPU:-4}
sft_lr_scheduler=${SFT_LR_SCHEDULER:-cosine}
# cosine floor as a fraction of peak LR (local verl patch 9); unset = stock cosine to 0
sft_min_lr_ratio=${SFT_MIN_LR_RATIO:-}
sft_save_freq=${SFT_SAVE_FREQ:--1}
sft_resume_mode=${SFT_RESUME_MODE:-disable}

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
echo "WandB tags:        ${WANDB_TAGS}"
echo "Base model:        ${sft_base_model_path}"
echo "Output dir:        ${sft_output_dir}"
echo "Train steps:       ${sft_train_steps}"
echo "Learning rate:     ${sft_lr}"
echo "Seed:              ${sft_seed}"
echo "Log freq:          ${sft_log_freq}"
echo "Warmup ratio:      ${sft_warmup_ratio}"
echo "Max length:        ${sft_max_length} (pads every sample; drives logits memory)"
echo "Batch (global/micro-per-gpu): ${sft_train_batch_size} / ${sft_micro_batch_size_per_gpu}"
echo "LR scheduler:      ${sft_lr_scheduler}"
echo "Save freq / resume: ${sft_save_freq} / ${sft_resume_mode}"
echo "Extra args:        ${SFT_EXTRA_ARGS:-}"
echo "============================================================"

cd ${project_dir}/verl

set -o pipefail
bash ${project_dir}/scripts/with_gpu_watchdog.sh python3 -m torch.distributed.run \
    --nproc_per_node=${N_GPUS} \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${sft_data_path}" \
    data.val_files="${sft_data_path}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=${sft_max_length} \
    data.truncation=right \
    data.train_batch_size=${sft_train_batch_size} \
    data.micro_batch_size_per_gpu=${sft_micro_batch_size_per_gpu} \
    +data.seed=${sft_seed} \
    model.partial_pretrain=${sft_base_model_path} \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    optim.lr=${sft_lr} \
    optim.warmup_steps_ratio=${sft_warmup_ratio} \
    optim.lr_scheduler=${sft_lr_scheduler} \
    ${sft_min_lr_ratio:++optim.min_lr_ratio=${sft_min_lr_ratio}} \
    trainer.total_training_steps=${sft_train_steps} \
    trainer.logging_freq=${sft_log_freq} \
    +trainer.max_val_batches=${sft_val_max_batches} \
    trainer.total_epochs=9999 \
    trainer.project_name=prog_distill \
    "trainer.experiment_name=${model_name}-${exp_name}-${sft_experiment_label}" \
    trainer.default_local_dir=${sft_output_dir} \
    "trainer.logger=['console','wandb']" \
    trainer.save_freq=${sft_save_freq} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.resume_mode=${sft_resume_mode} \
    ${SFT_EXTRA_ARGS:-} \
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

# Clean up FSDP shards to save disk space (HF model is now in sft_output_dir/).
# SFT_KEEP_LAST_SHARDS=true keeps them (model + fp32 optimizer + dataloader
# state, ~10 GB for 0.6B) so a longer run with the same SFT_OUTPUT_DIR and
# SFT_RESUME_MODE=auto continues from here instead of warm-starting from the
# merged weights with a fresh optimizer. Pair with a constant LR schedule.
if [ "${SFT_KEEP_LAST_SHARDS:-false}" = true ]; then
    echo "SFT_KEEP_LAST_SHARDS=true: keeping ${step_dir} for resume"
else
    rm -rf ${step_dir}
    echo "Cleaned up FSDP shards: ${step_dir}"
fi

chmod -R 770 ${sft_output_dir}
