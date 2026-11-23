#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G 
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 24:00:00
#SBATCH --array 1-1

conda activate verl
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
# HF datasets Arrow cache off NFS home (quota) — matches train_grpo.sh.
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
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-60}"

project_dir=${PROJECT_DIR:-$PWD}
cd ${project_dir}/verl
checkpoint_dir=${CHECKPOINT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-3B}
data_source=${DATA_SOURCE:-balanced}
exp_name=${EXP_NAME:-${data_source}-ppo-seed${SLURM_ARRAY_TASK_ID}}

max_length=${MAX_LENGTH:-1024}
train_path=../data/${data_source}/train.parquet
test_path=../data/${data_source}/test.parquet

# Checkpoint cadence — see train_grpo.sh: save_freq must match eval.sh's
# EVAL_STEP_MULT (default 100) or the checkpoints are never evaluated.
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}

output_dir=${CHECKPOINT_PATH:-${checkpoint_dir}/checkpoints/${model_name}}

N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"

set -o pipefail
bash ${project_dir}/scripts/with_gpu_watchdog.sh python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    +data.seed=${SLURM_ARRAY_TASK_ID} \
    actor_rollout_ref.model.path=${project_dir}/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    critic.optim.lr=1e-5 \
    critic.model.path=${project_dir}/models/${model_name} \
    critic.ppo_micro_batch_size_per_gpu=8 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${output_dir}/${exp_name} \
    trainer.rollout_data_dir=${output_dir}/${exp_name} \
    trainer.project_name=countdown \
    trainer.experiment_name=${model_name}-${exp_name} \
    trainer.balance_batch=False \
    custom_reward_function.path=../grader_utils.py \
    +custom_reward_function.reward_kwargs.trailing_penalty=${TRAILING_PENALTY:-0} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    2>&1 | python3 -u ${project_dir}/scripts/timestamp_filter.py
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail

chmod -R 770 ${output_dir}/${exp_name}

exit $TRAIN_EXIT_CODE