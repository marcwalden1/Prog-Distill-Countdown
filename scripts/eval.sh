#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1  # explicit H100: vLLM 0.8.5 crashes on MIG slices ('MIG-<uuid>' device IDs)
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=kempner_requeue
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 01:30:00
# holygpu8a17602: two evals died there with 'CUDA error: unspecified launch failure'
# (2026-10-11, tasks 52292735_4 / 52292779_4). Add further bad nodes comma-separated.
#SBATCH --exclude=holygpu8a17602
# Checkpoint step = EVAL_STEP_MULT (default 100) x array index, so 1-16 covers
# every saved checkpoint of a default 1600-step run at save_freq=100.
# Longer runs need a wider array -- do NOT rely on this default covering them:
#   TOTAL_EPOCHS=2 (~3266 steps) -> EVAL_ARRAY=1-32
# Override the array at submission with EVAL_ARRAY (run_pipeline.sh) or
# SBATCH_ARRAY_INX / sbatch --array. Steps with no checkpoint on disk exit 0.
#SBATCH --array 1-16

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

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# eval.py loads the test parquet via HF datasets; keep its Arrow cache off
# NFS home (quota) — matches train_grpo.sh.
export HF_DATASETS_CACHE="/tmp/hf_datasets_${SLURM_JOB_ID:-$$}"
mkdir -p "${HF_DATASETS_CACHE}"

# Triton's kernel cache defaults to ~/.triton/cache, which is shared across every
# Python interpreter that ever ran triton from this home dir. A `cuda_utils` C
# extension built by a different CPython imports fine but raises
#   SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
# the moment a kernel is launched (triton compiler.py `_init_handles` ->
# `load_binary`). That killed every OLMo-2 GRPO run at step 1 in flash-attn's
# fused cross-entropy (verl logprobs_from_logits), and is the same underlying
# breakage behind the gemma eval failures in CLAUDE.md. Keep the cache node-local
# and per-job so it is always built by the interpreter that uses it.
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
elif [ "$USER" = "bingbin" ]; then
    _checkpoint_dir=/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown
    _account=kempner_bingbin_lab
elif [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster (CLUSTER=mit). CHECKPOINT_DIR override expected; account
    # left empty so the strict match check below is skipped (MIT account is
    # passed via the sbatch CLI / SBATCH_ACCOUNT, not pinned here).
    _checkpoint_dir=${HOME}/rl-checkpoints
    _account=""
elif [ -n "${CHECKPOINT_DIR:-}" ]; then
    # Generic fallback: export CHECKPOINT_DIR; account left empty so the strict
    # SLURM_JOB_ACCOUNT match below is skipped (account comes from the sbatch CLI
    # / run_pipeline.sh, as with the MIT branch).
    _checkpoint_dir=${CHECKPOINT_DIR}
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
    # Step = EVAL_STEP_MULT x array index. EVAL_STEP_MULT must match the training
    # save_freq (train_grpo.sh / train_ppo.sh SAVE_FREQ, default 100). Runs saved
    # at the legacy 50-step cadence are re-evaluated with the old settings:
    #   EVAL_STEP_MULT=50 EVAL_ARRAY=3,6,9,12,15,18,21,24,28,32
    eval_step_mult=${EVAL_STEP_MULT:-100}
    eval_step=$((eval_step_mult * SLURM_ARRAY_TASK_ID))
    checkpoint_path=${checkpoint_dir}/checkpoints/${model_name}/${exp_name}/global_step_${eval_step}
    result_path=${result_dir}/results/${model_name}/${exp_name}/global_step_${eval_step}

    # A wide array is deliberately allowed to overshoot the end of training (run
    # lengths vary with TOTAL_EPOCHS / dynamic batching). Absent steps are not an
    # error: exit 0 so the afterok plot dependency still fires. Never create the
    # result dir before this check -- an empty results/.../global_step_N/ reads as
    # "eval attempted" to the cleanup and inventory scripts.
    if [ ! -d "${checkpoint_path}" ]; then
        echo "No checkpoint at ${checkpoint_path} (step ${eval_step} beyond end of training, or not saved) -- nothing to evaluate."
        exit 0
    fi
fi

# A merged HF checkpoint is either a single model.safetensors (<=~5 GB, e.g. 0.5B/1B)
# or a sharded set with model.safetensors.index.json (7B). Checking only the former
# made every 7B eval run two doomed verl.model_merger attempts before falling
# through to eval.py, and breaks the "merged == already evaluated" heuristic.
# --- keep the FINAL checkpoint resumable -------------------------------------
# Merging deletes actor/ (FSDP shards + fp32 optimizer state), critic/ and
# data.pt. That is what forces a later continuation to be a weights-only warm
# start: Adam moments restart from zero, the LR schedule re-warms up, the
# dataloader restarts at row 0, and -- because the warm-start weights are passed
# as actor_rollout_ref.model.path -- the KL term re-anchors on them instead of
# the base model. Keeping those files for the run's LAST checkpoint lets verl
# resume properly instead (resume_mode=auto reads
# latest_checkpointed_iteration.txt, which training already wrote).
#
# Cost: one step's optimizer state stays on disk (~10 GB for 0.6B, ~82 GB for a
# 7B run). Set KEEP_LAST_OPTIMIZER=false for the old delete-everything behaviour.
#
# NB two consequences:
#   - The "merged (flat model.safetensors, no actor/) == already evaluated"
#     heuristic no longer holds for the final step; test for the eval JSON.
#   - Resubmitting the SAME EXP_NAME now genuinely auto-resumes from the last
#     step rather than half-resuming off a stripped dir. Move or delete the exp
#     dir first if you want a clean restart.
keep_last_optimizer=${KEEP_LAST_OPTIMIZER:-true}
preserve_optimizer=false
if [ "${keep_last_optimizer,,}" = "true" ] && [ -z "${EVAL_CHECKPOINT_PATH:-}" ]; then
    latest_file="$(dirname "${checkpoint_path}")/latest_checkpointed_iteration.txt"
    if [ -f "${latest_file}" ]; then
        latest_step="$(tr -dc '0-9' < "${latest_file}")"
        if [ -n "${latest_step}" ] && [ "${latest_step}" = "${eval_step}" ]; then
            preserve_optimizer=true
            echo "Step ${eval_step} is this run's final checkpoint (latest_checkpointed_iteration.txt) -- keeping actor/, critic/, data.pt so it stays resumable."
        fi
    fi
fi

if [ -f "${checkpoint_path}/model.safetensors" ] || [ -f "${checkpoint_path}/model.safetensors.index.json" ]; then
    echo "Merged checkpoint already present at ${checkpoint_path}"
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path}/actor/ --target_dir ${checkpoint_path}; then
    if [ "${preserve_optimizer}" = "true" ]; then
        echo "Preserving ${checkpoint_path}/{actor,critic,data.pt} (final checkpoint)."
    else
        rm -rf ${checkpoint_path}/actor/
        rm -rf ${checkpoint_path}/critic/
        rm -rf ${checkpoint_path}/data.pt
    fi
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path} --target_dir ${checkpoint_path}; then
    if [ "${preserve_optimizer}" = "true" ]; then
        echo "Preserving ${checkpoint_path}/*.pt (final checkpoint)."
    else
        rm -rf ${checkpoint_path}/*.pt
    fi
fi

bash ${project_dir}/scripts/with_gpu_watchdog.sh python3 eval.py --checkpoint_path ${checkpoint_path} --result_path ${result_path} --eval_dataset ${eval_dataset} ${extra_args}
eval_rc=$?

# Exit with eval.py's status: the chmods used to be the last command, so a crashed
# eval (e.g. a CUDA fault) showed as COMPLETED, and an SFT-only eval (no
# checkpoints/<...> dir to chmod) showed as FAILED with a valid JSON.
chmod -R 770 ${checkpoint_dir}/checkpoints/${model_name}/${exp_name} 2>/dev/null
chmod -R 770 ${result_dir}/results/${model_name}/${exp_name} 2>/dev/null
exit ${eval_rc}
