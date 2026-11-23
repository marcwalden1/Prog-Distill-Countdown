#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=kempner_h100
#SBATCH --output=logs/%x-%A.out
#SBATCH -t 01:30:00
# No #SBATCH --account: it is set per-$USER below the same way scripts/eval.sh
# does it (sbatch parses directives at submission, so a hard-coded account would
# be wrong for every user but one). Your SLURM default account is used; pass
# --account=... at submission to override.
#
# Zero-shot (historical):
#   MODEL_NAME=OLMo-2-0425-1B-pt4001B EVAL_DATASET=balanced-simple \
#     sbatch scripts/eval_base.sh
# Few-shot, answer-only vs. search-trace exemplars:
#   MODEL_NAME=... EVAL_DATASET=balanced-simple N_SHOT=4 SHOT_FORMAT=answer \
#     sbatch scripts/eval_base.sh
#   MODEL_NAME=... EVAL_DATASET=balanced-simple N_SHOT=4 SHOT_FORMAT=search \
#     sbatch scripts/eval_base.sh

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

# `conda activate` sets CONDA_PREFIX but on this cluster can leave PATH pointing
# at the base interpreter (a personal miniconda from ~/.bashrc shadows the module
# Miniforge), so `python3 eval.py` dies with ModuleNotFoundError: transformers.
# Same two lines as scripts/eval.sh.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}
# Node-local triton cache (see scripts/train_grpo.sh for why).
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
# Node-local HF datasets cache: load_dataset wedges in uninterruptible I/O
# acquiring its filelock under this cluster's NFS home (same hang that forced
# scripts/make_prompt_variant_data.py off datasets; see research_logs/2026-09-10).
export HF_DATASETS_CACHE="/tmp/hf_datasets_${SLURM_JOB_ID:-$$}"
mkdir -p "${HF_DATASETS_CACHE}"

# User-specific model roots + expected account (mirrors scripts/eval.sh).
if [ "$USER" = "mwalden" ]; then
    _model_root=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _account=kempner_kdbrantley_lab
    _account_alt=kdbrantley_lab
elif [ "$USER" = "sdholakia" ]; then
    _model_root=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _account=kempner_bingbin_lab
elif [ "$USER" = "bingbin" ]; then
    _model_root=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models
    _account=kempner_bingbin_lab
elif [ -n "${MODEL_DIR:-}${MODEL_ROOT:-}" ]; then
    _model_root=${MODEL_ROOT:-${MODEL_DIR}}
    _account=""
else
    echo "ERROR: Unknown user $USER. Set MODEL_ROOT (or MODEL_DIR) explicitly." >&2
    exit 1
fi
if [ -n "$_account" ] && [ -n "${SLURM_JOB_ACCOUNT:-}" ] \
        && [ "$SLURM_JOB_ACCOUNT" != "$_account" ] \
        && [ "$SLURM_JOB_ACCOUNT" != "${_account_alt:-}" ]; then
    echo "ERROR: job running on account=$SLURM_JOB_ACCOUNT, expected $_account${_account_alt:+ or $_account_alt} for user $USER." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}}
model_root=${MODEL_ROOT:-${MODEL_DIR:-$_model_root}}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
eval_dataset=${EVAL_DATASET:-balanced}
temperature=${TEMPERATURE:-0.6}
num_samples=${NUM_SAMPLES:-32}
max_tokens=${MAX_TOKENS:-1024}
extra_args=${EXTRA_ARGS:-""}

# Few-shot (see sft_formats.py). N_SHOT=0 keeps the historical zero-shot eval.
# Results land in results/baselines/<model>/<K>shot-<format>/ so they never mix
# with the zero-shot JSONs plot_results.py globs.
n_shot=${N_SHOT:-0}
shot_args=""
if [ "${n_shot}" -gt 0 ]; then
    shot_args="--n_shot ${n_shot} --shot_format ${SHOT_FORMAT:-answer} \
--shot_seed ${SHOT_SEED:-0} --shot_attempts ${SHOT_ATTEMPTS:-1-3} \
--shot_strategy ${SHOT_STRATEGY:-mixed} --prompt_format ${PROMPT_FORMAT:-raw}"
    [ -n "${SHOT_SOURCE:-}" ] && shot_args="${shot_args} --shot_source ${SHOT_SOURCE}"
elif [ -n "${PROMPT_FORMAT:-}" ]; then
    shot_args="--prompt_format ${PROMPT_FORMAT}"
fi

checkpoint_path=${model_root}/${model_name}
base_result_dir=${result_dir}/results/baselines/${model_name}

bash ${project_dir}/scripts/with_gpu_watchdog.sh python3 eval.py \
    --checkpoint_path ${checkpoint_path} \
    --result_path ${base_result_dir} \
    --eval_dataset ${eval_dataset} \
    --temperature ${temperature} \
    --n ${num_samples} \
    --max_tokens ${max_tokens} \
    ${shot_args} \
    ${extra_args}

chmod -R 770 ${base_result_dir}
