#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4 # for 7B, use 8 GPUs
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

export WANDB_MODE="online"
export WANDB_ENTITY="progressive_distill"

# Tag this wandb run so the post-distill GRPO leg is filterable alongside its SFT rounds.
# DISTILL_MODE is set by run_pipeline.sh when this is the final GRPO stage of a (prog)distill chain.
# Also tag by student model size (e.g. "270m", "0.5B", "1.5B") — derived from the trailing
# segment of MODEL_NAME after the last hyphen. MODEL_NAME is the student in both standalone
# GRPO and the post-distill GRPO leg, so this tag always reflects the model being trained.
_grpo_tags="grpo"
if [ -n "${DISTILL_MODE:-}" ]; then
    _grpo_tags="${DISTILL_MODE},grpo"
fi
_size_tag="${MODEL_NAME##*-}"
_grpo_tags="${_grpo_tags},${_size_tag}"
export WANDB_TAGS="${WANDB_TAGS:-${_grpo_tags}}"
export RAY_DISABLE_DASHBOARD=1
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

# User-specific default paths
if [ "$USER" = "mwalden" ]; then
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
elif [ "$USER" = "sdholakia" ]; then
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
elif [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster (CLUSTER=mit). Defaults live under $HOME; override with
    # MODEL_DIR / CHECKPOINT_DIR env vars (or MODEL_PATH for one-off paths).
    _model_dir=${HOME}/models
    _checkpoint_dir=${HOME}/rl-checkpoints
else
    echo "ERROR: Unknown user $USER. Set MODEL_DIR and CHECKPOINT_DIR explicitly." >&2
    exit 1
fi

project_dir=${PROJECT_DIR:-$PWD}
cd ${project_dir}/verl
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
data_source=${DATA_SOURCE:-balanced}
exp_name=${EXP_NAME:-${data_source}-grpo-seed${SLURM_ARRAY_TASK_ID}}

max_length=${MAX_LENGTH:-1024}
kl_loss_coef=${KL_COEF:-0.001}
lr=${LR:-1e-6}
total_steps=${TOTAL_STEPS:-1635}
rollout_n=${ROLLOUT_N:-4}
val_kwargs_n=${VAL_KWARGS_N:-${rollout_n}}
train_path=../data/${data_source}/train.parquet
test_path=../data/${data_source}/test.parquet

output_dir=${CHECKPOINT_PATH:-${checkpoint_dir}/checkpoints/${model_name}}

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
echo "Max length:       ${max_length}"
echo "Learning rate:    ${lr}"
echo "KL coef:          ${kl_loss_coef}"
echo "Rollout n:        ${rollout_n}"
echo "Val kwargs n:     ${val_kwargs_n}"
echo "Total steps:      ${total_steps}"
echo "Checkpoint path:  ${output_dir}/${exp_name}"
echo "Project dir:      ${project_dir}"
echo "============================================================"

ulimit -n 65536
ray stop 2>/dev/null || true
rm -rf /tmp/ray/ 2>/dev/null || true
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}

# Per-job Ray port range. Ray's --temp-dir isolates the filesystem session
# pointer, but the GCS server still binds the default port 6379. On shared
# partitions (MIT mit_preemptable), two SLURM jobs can land on the same
# compute node and race for that port — the loser sees "Session name ...
# does not match persisted value" from _write_cluster_info_to_kv, then
# python's ray.init() falls back to a local instance that can't talk to
# the stale raylet socket. Deriving the port from SLURM_JOB_ID avoids
# the collision entirely. Setting RAY_ADDRESS pins ray.init() to our head.
# Per-job port plan. Cover EVERY component Ray pre-allocates — the
# default dashboard_agent_http=52365, runtime_env_agent=48457, etc.
# are fixed and will collide with our worker range otherwise, and
# also collide cross-job when two SLURM jobs land on the same node.
# Worker range kept tight (90 ports) to fit inside the per-job slot.
RAY_PORT=$((20000 + (SLURM_JOB_ID % 300) * 100))
export RAY_ADDRESS=127.0.0.1:${RAY_PORT}
ray start --head \
    --include-dashboard=false \
    --num-gpus=${N_GPUS} \
    --temp-dir=/tmp/ray_${SLURM_JOB_ID} \
    --node-ip-address=127.0.0.1 \
    --port=${RAY_PORT} \
    --node-manager-port=$((RAY_PORT + 1)) \
    --object-manager-port=$((RAY_PORT + 2)) \
    --dashboard-agent-listen-port=$((RAY_PORT + 3)) \
    --dashboard-agent-grpc-port=$((RAY_PORT + 4)) \
    --runtime-env-agent-port=$((RAY_PORT + 5)) \
    --metrics-export-port=$((RAY_PORT + 6)) \
    --min-worker-port=$((RAY_PORT + 10)) \
    --max-worker-port=$((RAY_PORT + 99))

set -o pipefail
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    +data.seed=${SLURM_ARRAY_TASK_ID} \
    actor_rollout_ref.model.path=${MODEL_PATH:-${MODEL_DIR:-$_model_dir}/${model_name}} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_steps} \
    trainer.default_local_dir=${output_dir}/${exp_name} \
    trainer.rollout_data_dir=${output_dir}/${exp_name} \
    trainer.project_name=prog_distill \
    trainer.experiment_name=${model_name}-${exp_name} \
    trainer.balance_batch=False \
    custom_reward_function.path=../grader_utils.py \
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

ray stop

exit $TRAIN_EXIT_CODE