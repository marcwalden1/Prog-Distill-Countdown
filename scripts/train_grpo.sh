#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2 # for larger models, override with TRAIN_SBATCH_ARGS
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 24:00:00
#SBATCH --array 1-1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab

if [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster: no FAS-RC module system. Conda must already be on PATH
    # from the submitter's shell (~/.bashrc). MIT_CONDA_ENV overrides the
    # default env name ("base"). Opt-in via CLUSTER=mit; default unset =
    # Harvard.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${MIT_CONDA_ENV:-base}"
else
    module load Miniforge3/26.1.0-fasrc01
    source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
    conda activate verl
fi
# `conda activate` sets CONDA_PREFIX but on this cluster can leave PATH pointing
# at the base interpreter (personal miniconda from ~/.bashrc conflicts with the
# module Miniforge), so python3 loses numpy/torch and the job dies at import.
# Force the activated env's bin onto PATH.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-bingbin}"
# Put wandb-core's IPC port-file on local /tmp. The default lives under CWD
# (which is the NFS-backed verl/ dir), and on Kempner compute nodes NFS
# write-then-read latency reliably busts wandb-core's 30s service-startup
# timeout — manifests as ServicePollForTokenError → job dies before step 1.
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "${WANDB_DIR}"
# WANDB__SERVICE_WAIT is set later (line ~100) to 300s via the same pattern.

# verl's RLHFDataset loads parquets via HF datasets, which materializes an
# Arrow cache. The default cache dir is NFS home (~/.cache/huggingface/datasets),
# which fills the home quota and then kills the job at dataset load with
# "OSError: Not enough disk space". Keep it node-local like WANDB_DIR.
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

# All ranks on the node share the per-job TRITON_CACHE_DIR above, and inductor's
# Triton bundler races on it: two ranks compiling the same kernel each build a
# temp dir and both call os.replace() into the same final name, so the loser dies
# with FileExistsError in TritonBundler.read_and_emit -> the whole job fails at
# actor compile (~7 min in, before any checkpoint). Hit 3 of 4 OLMo-2-7B GRPO
# jobs launched together on 2026-09-02 (43822416/43822426/43822433, identical
# kernel hash). Disabling the bundler skips that code path; it only affects
# caching of compiled-kernel artifacts, so training is numerically unchanged.
export TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE=0

# Tag this wandb run so the post-distill GRPO leg is filterable alongside its SFT rounds.
# DISTILL_MODE is set by run_pipeline.sh when this is the final GRPO stage of a (prog)distill chain.
# Also tag by student model size (e.g. "270m", "0.5B", "1.5B") — derived from the trailing
# segment of MODEL_NAME after the last hyphen. MODEL_NAME is the student in both standalone
# GRPO and the post-distill GRPO leg, so this tag always reflects the model being trained.
_grpo_tags="grpo"
if [ -n "${DISTILL_MODE:-}" ]; then
    _grpo_tags="${DISTILL_MODE},grpo"
fi
# Qwen3 base models are named Qwen3-<size>-Base; drop the -Base suffix so the
# tag is the size ("0.6B") rather than the literal "Base".
_size_tag="${MODEL_NAME%-Base}"
_size_tag="${_size_tag##*-}"
# From-scratch 40M models are named Qwen2.5-40M-<shape>; keep the size in the tag
# (e.g. "40M-wide", "40M-deep") rather than just the shape suffix.
if [[ "${MODEL_NAME}" == *-40M-* ]]; then
    _size_tag="40M-${MODEL_NAME##*-}"
fi
_grpo_tags="${_grpo_tags},${_size_tag}"
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
# (WANDB_TAGS is exported below, once data_source is known.)
export RAY_DISABLE_DASHBOARD=1
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

echo "USER: $USER"

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
elif [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster (CLUSTER=mit). Defaults live under $HOME; override with
    # MODEL_DIR / CHECKPOINT_DIR env vars (or MODEL_PATH for one-off paths).
    _model_dir=${HOME}/models
    _checkpoint_dir=${HOME}/rl-checkpoints
elif [ -n "${MODEL_DIR:-}" ] && [ -n "${CHECKPOINT_DIR:-}" ]; then
    # Generic fallback: any other user/cluster works by exporting MODEL_DIR +
    # CHECKPOINT_DIR (and passing --account=... to sbatch, as run_pipeline.sh
    # does). No per-user branch needed.
    _model_dir=${MODEL_DIR}
    _checkpoint_dir=${CHECKPOINT_DIR}
else
    echo "ERROR: Unknown user $USER. Set MODEL_DIR and CHECKPOINT_DIR explicitly (and pass --account=... to sbatch)." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
cd ${project_dir}/verl
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}

# Checkpoint cadence. save_freq MUST stay in sync with eval.sh's EVAL_STEP_MULT
# (also 100) — eval derives the step to evaluate as EVAL_STEP_MULT x array index,
# so a checkpoint saved off that grid is never evaluated. test_freq is the
# validation cadence and is independent of what lands on disk.
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
data_source=${DATA_SOURCE:-balanced}
# Scaffold-free prompt variants (data/*-simple): train on the literal parquet
# string — blank the tokenizer's chat template so Qwen's default
# "You are a helpful assistant." system wrapper is NOT prepended (see
# research_logs/2026-08-10_base-simple-crash-quota-chat-template.md).
# Override with DISABLE_CHAT_TEMPLATE=True/False.
if [[ "${data_source}" == *-simple* ]]; then
    _dct_default=True
else
    _dct_default=False
fi
disable_chat_template=${DISABLE_CHAT_TEMPLATE:-${_dct_default}}

# Tag by prompt variant so wandb runs are filterable by prompt format
# (e.g. "balanced", "balanced-simple", "balanced-simple-gsm"). config.data.train_files
# carries the same information, but only as a path — a tag is filterable in the UI
# without adding a config column. Exported here rather than in the tag block above
# because data_source is not resolved until this point.
_grpo_tags="${_grpo_tags},${data_source}"
export WANDB_TAGS="${WANDB_TAGS:-${_grpo_tags}}"
# Always carry the family tag, even when the caller preset WANDB_TAGS (the
# run_olmo2_*.sh launchers do), so filtering by family works for every run.
if [ -n "${_family_tag}" ] && [[ ",${WANDB_TAGS}," != *",${_family_tag},"* ]]; then
    export WANDB_TAGS="${WANDB_TAGS},${_family_tag}"
fi
exp_name=${EXP_NAME:-${data_source}-grpo-seed${SLURM_ARRAY_TASK_ID}}

max_length=${MAX_LENGTH:-1024}
kl_loss_coef=${KL_COEF:-0.001}
lr=${LR:-1e-6}
total_steps=${TOTAL_STEPS:-1635}
# verl's fit() loops `for epoch in range(trainer.total_epochs)` and only stops
# early once global_steps reaches total_training_steps. data/balanced/train.parquet
# is 418,136 rows = 1633 steps at train_batch_size=256, so with total_epochs=1 a
# run is data-limited to 1633 steps no matter how large TOTAL_STEPS is. To train
# multiple passes, raise BOTH: e.g. TOTAL_EPOCHS=2 TOTAL_STEPS=3266.
total_epochs=${TOTAL_EPOCHS:-1}
# Wandb/console cadence for per-step train metrics (val/save are unaffected and
# stay at test_freq/save_freq). Requires the local verl ray_trainer.py
# `trainer.log_freq` patch.
grpo_log_freq=${GRPO_LOG_FREQ:-50}
# Reward deducted per 1000 chars of junk after the final </answer> (0 = off,
# the historical reward). Plumbed into grader_utils.compute_score via
# custom_reward_function.reward_kwargs; eval-side scoring is unaffected.
trailing_penalty=${TRAILING_PENALTY:-0}
rollout_n=${ROLLOUT_N:-4}
val_kwargs_n=${VAL_KWARGS_N:-${rollout_n}}
# Entropy bonus. 0 = stock (no bonus). Non-zero flips verl's calculate_entropy=True,
# which computes entropy over the full vocab in the actor forward — extra time and
# memory (set +actor_rollout_ref.actor.entropy_from_logits_with_chunking=True via
# EXTRA_ARGS if that OOMs on a large-vocab model).
entropy_coeff=${ENTROPY_COEFF:-0}
train_path=../data/${data_source}/train.parquet
test_path=../data/${data_source}/test.parquet

output_dir=${CHECKPOINT_PATH:-${checkpoint_dir}/checkpoints/${model_name}}

ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}
# vLLM rollout memory. Both were hard-coded; 7B on 4x80GB needs them lower, because
# the FSDP2 actor (~98 GB of bf16 params + fp32 master + Adam, so ~25 GB/GPU on 4
# GPUs) and the resident vLLM weights share each device. tensor_model_parallel_size
# stays 1 by default = one independent vLLM replica per GPU (data-parallel rollout);
# raise it to shard the weights instead if a model does not fit.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_tp_size=${ROLLOUT_TP_SIZE:-1}
if [[ "${model_name,,}" == *gemma* ]]; then
    ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
    rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
    ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
fi

# Dynamic (token-budgeted) micro-batching. Off by default = stock behavior, where
# micro-batches hold a fixed *sequence* count and so vary wildly in token count.
# With USE_DYNAMIC_BSZ=True, verl packs each micro-batch to ~ppo_max_token_len_per_gpu
# tokens instead, which both raises utilization and makes micro-batches near-equal
# in tokens — the latter matters because loss_agg_mode=token-mean normalizes *within*
# a micro-batch and then averages with weight 1/G (verl dp_actor.py), so unequal
# micro-batch token counts perturb the effective per-token weighting.
# NB: switching this on changes that weighting relative to fixed-micro-batch runs;
# it is a speed *and* numerics change, not a pure speed knob.
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
dynamic_bsz_args=""
if [[ "${use_dynamic_bsz,,}" == "true" ]]; then
    # ref/rollout log-prob passes inherit these via oc.select in verl's ref.yaml /
    # rollout.yaml, so setting the actor keys is enough.
    dynamic_bsz_args="actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
fi

# Give W&B's local service more time to start on busy cluster filesystems.
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}

N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"

echo "============================================================"
echo "EXPERIMENT CONFIG"
echo "Date:             $(date)"
echo "SLURM Job ID:     ${SLURM_JOB_ID}"
echo "SLURM Array ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:             $(hostname)"
echo "GPUs:             ${N_GPUS}"
echo "Model:            ${model_name}"
echo "Exp name:         ${exp_name}"
echo "Data source:      ${data_source}"
echo "WandB tags:       ${WANDB_TAGS}"
echo "Chat template:    disabled=${disable_chat_template}"
echo "Max length:       ${max_length}"
echo "Learning rate:    ${lr}"
echo "KL coef:          ${kl_loss_coef}"
echo "Rollout n:        ${rollout_n}"
echo "Val kwargs n:     ${val_kwargs_n}"
echo "Total steps:      ${total_steps}"
echo "Total epochs:     ${total_epochs}"
echo "Save freq:        ${save_freq} (eval.sh EVAL_STEP_MULT must match)"
echo "Test freq:        ${test_freq}"
echo "Entropy coeff:    ${entropy_coeff}"
echo "Dynamic bsz:      ${use_dynamic_bsz} (max_token_len_per_gpu=${ppo_max_token_len_per_gpu})"
echo "PPO microbatch:   ${ppo_micro_batch_size_per_gpu}"
echo "Rollout logprob:  ${rollout_log_prob_micro_batch_size_per_gpu}"
echo "Rollout vLLM:     gpu_mem_util=${rollout_gpu_mem_util} tp=${rollout_tp_size}"
echo "Ref logprob:      ${ref_log_prob_micro_batch_size_per_gpu}"
echo "Checkpoint path:  ${output_dir}/${exp_name}"
echo "Project dir:      ${project_dir}"
echo "============================================================"

ulimit -n 65536
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
# Bump Ray's raylet startup wait (default 10s). One of the 2026-05-18 resubmits
# died because the in-process raylet didn't register with GCS inside 10s on a
# busy node. Ray's own error message literally suggests raising this; 60s gives
# slow-starting nodes plenty of headroom and is no-op when startup is fast.
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-60}"

if [ "${CLUSTER:-}" = "mit" ]; then
    # MIT mit_preemptable: multiple SLURM jobs can land on the same compute
    # node and race for default Ray ports. Cleanup is safe here because each
    # job runs its own external head and port-collides only with stale runs
    # from prior preemptions on the same node — not with concurrent peers.
    ray stop 2>/dev/null || true
    rm -rf /tmp/ray/ 2>/dev/null || true

    # Run an external `ray start --head` with per-job port isolation.
    # Surgical fixes from prior debugging:
    #   - Don't pass `--node-ip-address=127.0.0.1` (cosmetic; Ray rewrites
    #     it to the LAN IP via resolve_ip_for_localhost anyway).
    #   - Set RAY_ADDRESS to the same LAN IP Ray's get_node_ip_address()
    #     produces, so the driver bootstraps without an address-rewrite step.
    RAY_PORT=$((20000 + (SLURM_JOB_ID % 300) * 100))
    NODE_IP=$(python3 -c 'import ray._private.services as s; print(s.get_node_ip_address())')
    export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
    ray start --head \
        --include-dashboard=false \
        --num-gpus=${N_GPUS} \
        --temp-dir=/tmp/ray_${SLURM_JOB_ID} \
        --port=${RAY_PORT} \
        --node-manager-port=$((RAY_PORT + 1)) \
        --object-manager-port=$((RAY_PORT + 2)) \
        --dashboard-agent-listen-port=$((RAY_PORT + 3)) \
        --dashboard-agent-grpc-port=$((RAY_PORT + 4)) \
        --runtime-env-agent-port=$((RAY_PORT + 5)) \
        --metrics-export-port=$((RAY_PORT + 6)) \
        --min-worker-port=$((RAY_PORT + 10)) \
        --max-worker-port=$((RAY_PORT + 99))
else
    # Harvard kempner_h100: 4-GPU nodes can host two concurrent 2-GPU jobs.
    # Do NOT run `ray stop` or `rm -rf /tmp/ray/` — both are process- and
    # filesystem-scoped to the whole user on the host, and would kill a
    # peer job's Ray instance. Skip the external head entirely and let
    # verl's `ray.init()` (main_ppo.py:58) start an in-process Ray inside
    # this python process. RAY_TMPDIR isolates the per-job session dir;
    # the Ray instance dies naturally when python exits.
    echo "Single-tenant or shared cluster: letting verl's ray.init() start in-process Ray (no global ray stop)."
fi

set -o pipefail
bash ${project_dir}/scripts/with_gpu_watchdog.sh python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    +data.disable_chat_template=${disable_chat_template} \
    +data.seed=${SLURM_ARRAY_TASK_ID} \
    actor_rollout_ref.model.path=${MODEL_PATH:-${MODEL_DIR:-$_model_dir}/${model_name}} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    ${dynamic_bsz_args} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    +trainer.log_freq=${grpo_log_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_steps} \
    trainer.default_local_dir=${output_dir}/${exp_name} \
    trainer.rollout_data_dir=${output_dir}/${exp_name} \
    trainer.project_name=prog_distill \
    trainer.experiment_name=${model_name}-${exp_name} \
    trainer.balance_batch=False \
    custom_reward_function.path=../grader_utils.py \
    +custom_reward_function.reward_kwargs.trailing_penalty=${trailing_penalty} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ${EXTRA_ARGS:-} \
    2>&1 | python3 -u ${project_dir}/scripts/timestamp_filter.py
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail

chmod -R 770 ${output_dir}/${exp_name}

if [ "${CLUSTER:-}" = "mit" ]; then
    # Only safe to global-stop Ray on MIT's external-head path; on Harvard
    # this would kill a peer job's in-process Ray on the same node.
    ray stop
fi

exit $TRAIN_EXIT_CODE
