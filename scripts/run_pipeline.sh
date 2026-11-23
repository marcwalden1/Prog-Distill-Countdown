#!/bin/bash
# scripts/run_pipeline.sh
# Submits the full train → eval → plot_results pipeline as a SLURM dependency chain.
#
# Usage:
#   MODEL_NAME=Qwen2.5-1.5B EXP_NAME=my-exp bash scripts/run_pipeline.sh
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=my-exp KL_COEF=0.0001 bash scripts/run_pipeline.sh
#
# Distillation only — SFT the student on the teacher's responses, no GRPO:
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-distill-seed1 \
#     bash scripts/run_pipeline.sh --distill balanced-grpo-kl1e-3-lr1e-6-seed1
#
# Distillation + GRPO (SFT then continue with GRPO, eval, plot):
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-distill-grpo-seed1 \
#     bash scripts/run_pipeline.sh --distill balanced-grpo-kl1e-3-lr1e-6-seed1 --GRPO
#
# Cross-size distillation (1.5B teacher → 0.5B student, SFT only):
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-distill-cross-seed1 \
#     TEACHER_MODEL_NAME=Qwen2.5-1.5B \
#     bash scripts/run_pipeline.sh --distill balanced-grpo-kl3e-3-lr3e-6-seed1
#
# Progressive distillation ((N,T) curriculum per arxiv:2410.05464, SFT only):
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-progdistill-seed1 \
#     bash scripts/run_pipeline.sh --progdistill balanced-grpo-kl1e-3-lr1e-6-seed1
#
# Progressive-distillation mixture — same per-checkpoint data as --progdistill,
# but pooled into one dataset for a single SFT pass (not sequential rounds):
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-progdistill-mixture-seed1 \
#     bash scripts/run_pipeline.sh --progdistill_mixture balanced-grpo-kl1e-3-lr1e-6-seed1
#
# Offline on-policy distillation (arXiv:2604.13010) on top of an existing SFT
# student — samples rollouts once from the student, precomputes per-token
# teacher logprobs, trains with the OPD loss, then evals all checkpoints:
#   MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-opd-from-distill-lr2e-6-seed1 \
#     TEACHER_MODEL_NAME=Qwen2.5-1.5B TEACHER_STEP=1600 \
#     OPD_STUDENT_PATH=/path/to/merged/sft/student \
#     bash scripts/run_pipeline.sh --opd <teacher-EXP_NAME>
# Teacher consistency: pass the SAME teacher checkpoint that generated the
# student's SFT data (for progdistill students, the final-round teacher).
#
# --GRPO can be combined with --distill, --progdistill, or --progdistill_mixture
# to run RL after distillation. Without --GRPO those modes stop at SFT.
# --opd does not support --GRPO.
#
# For 7B (needs 8 GPUs):
#   MODEL_NAME=Qwen2.5-7B EXP_NAME=my-exp \
#     TRAIN_SBATCH_ARGS="--gres=gpu:nvidia_h100_80gb_hbm3:8 -t 30:00:00" \
#     bash scripts/run_pipeline.sh
#
# Optional env var overrides:
#   KL_COEF                    — KL loss coefficient (default: 0.001)
#   TRAILING_PENALTY           — train-time reward deduction per 1000 chars of junk after the
#                                final </answer> (default: 0 = off; eval scoring unaffected)
#   DATA_SOURCE                — data directory under data/ (default: balanced)
#   EXTRA_ARGS                 — Hydra-style overrides forwarded to train_grpo.sh / train_ppo.sh
#   EVAL_EXTRA_ARGS            — extra argparse args forwarded to eval.py
#   TRAIN_SBATCH_ARGS          — extra sbatch args for train_grpo.sh
#   EVAL_SBATCH_ARGS           — extra sbatch args for eval.sh (e.g. --gres override on a non-FAS-RC cluster)
#   SFT_SBATCH_ARGS            — extra sbatch args for train_sft.sh / generate_sft_data.sh
#   SAVE_FREQ                  — checkpoint save cadence in steps (default: 100; must match EVAL_STEP_MULT)
#   TEST_FREQ                  — validation cadence in steps (default: 50; independent of SAVE_FREQ)
#   EVAL_ARRAY                 — sbatch array for the eval sweep; step = EVAL_STEP_MULT x index
#                                (default: eval.sh's 1-16 = steps 100..1600. TOTAL_EPOCHS=2 needs 1-32)
#   EVAL_STEP_MULT             — step-per-array-index for eval.sh (default: 100; use 50 for legacy runs)
#   TEACHER_STEPS              — space-separated teacher checkpoint steps for --progdistill / mixture
#                                (default: "200 400 600 800 1000 1200 1400 1600"; must exist on the teacher)
#   PROGDISTILL_STEPS_PER_ROUND — SFT steps per teacher checkpoint round (default: 160)
#   SFT_TRAIN_STEPS            — total SFT steps for --distill mode (default: 1600, matches progdistill budget)
#   SFT_LR                     — SFT learning rate (default: 1e-5)
#   N_RESPONSES                — teacher responses per prompt for data gen (default: 4)
#   TEACHER_MODEL_NAME         — teacher arch/size for cross-size distill (default: MODEL_NAME = student)
#   TEACHER_CHECKPOINT_DIR     — read teacher checkpoints from another tree (e.g. another user's,
#                                read-only); outputs still go to CHECKPOINT_DIR (default: CHECKPOINT_DIR)
#   FILTER_CORRECT_ONLY        — keep only score==1.0 teacher responses (default: false; keeps all)
#   SFT_DATA_NAMESPACE         — optional override for where generated SFT parquet data is stored
#                                and reused; default is derived from the teacher checkpoint lineage
#   SFT_DATA_SOURCE_EXP_NAME   — legacy override to reuse older SFT parquet data stored under a
#                                previous EXP_NAME-based layout
#   --opd-specific:
#   OPD_STUDENT_PATH           — REQUIRED: merged SFT student checkpoint (pi_ref: rollout
#                                source AND training init)
#   TEACHER_STEP               — teacher checkpoint step for logprob scoring (default: 1600;
#                                must be numeric here so the data path is derivable at submit time)
#   OPD_ROLLOUT_TEMP           — student rollout temperature (default: 1.0; paper used 0.8)
#   OPD_MAX_PROMPTS            — prompts to roll out (default: 10000, x N_RESPONSES samples)
#   OPD_TRAIN_STEPS            — OPD steps (default: 1600, matches the SFT budget)
#   OPD_LR                     — OPD learning rate (default: 2e-6, from the paper)
#   OPD_CLIP                   — advantage clip range (default: 10.0)
#   OPD_DATA_NAMESPACE         — override where the OPD parquet is stored/reused

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DISTILL_MODE=""
TEACHER_EXP_NAME=""
RUN_GRPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --distill)
            DISTILL_MODE=distill
            TEACHER_EXP_NAME="${2:?--distill requires a teacher EXP_NAME}"
            shift 2
            ;;
        --progdistill)
            DISTILL_MODE=progdistill
            TEACHER_EXP_NAME="${2:?--progdistill requires a teacher EXP_NAME}"
            shift 2
            ;;
        --progdistill_mixture|--progdistill-mixture)
            DISTILL_MODE=progdistill_mixture
            TEACHER_EXP_NAME="${2:?--progdistill_mixture requires a teacher EXP_NAME}"
            shift 2
            ;;
        --opd)
            DISTILL_MODE=opd
            TEACHER_EXP_NAME="${2:?--opd requires a teacher EXP_NAME}"
            shift 2
            ;;
        --GRPO|--grpo)
            RUN_GRPO=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Plain RL mode always runs GRPO; distill/progdistill need --GRPO to enable it.
if [ -z "$DISTILL_MODE" ]; then
    RUN_GRPO=1
fi

# Export so sbatch'd child scripts (train_grpo.sh, train_sft.sh) can derive WANDB_TAGS
export DISTILL_MODE
export RUN_GRPO

export MODEL_NAME=${MODEL_NAME:-Qwen2.5-1.5B}
export EXP_NAME=${EXP_NAME:-balanced-grpo-seed1}
export DATA_SOURCE=${DATA_SOURCE:-balanced}
# Eval-set names follow DATA_SOURCE: prompt-variant data written by
# scripts/make_prompt_variant_data.py lives in data/balanced<suffix>{,5,6}, so
# DATA_SOURCE=balanced<suffix> evals on balanced5<suffix>/balanced6<suffix>
# (plain 'balanced' keeps the historical balanced5/balanced6).
# The derivation assumes DATA_SOURCE=balanced<suffix>; it does not hold when the
# size set also varies (DATA_SOURCE=balanced234-simple would give the nonexistent
# balanced5234-simple), so all three are overridable by env.
_ds_suffix=${DATA_SOURCE#balanced}
EVAL_DS_N34=${EVAL_DS_N34:-${DATA_SOURCE}}
EVAL_DS_N5=${EVAL_DS_N5:-balanced5${_ds_suffix}}
EVAL_DS_N6=${EVAL_DS_N6:-balanced6${_ds_suffix}}
# OOD_SUBSAMPLE=<K>: eval (and therefore score/plot) n=5 and n=6 on a seeded
# uniform-random K-prompt subset instead of the full 4620/3519 prompts, ~4x cheaper
# at K=1000. The subsets are separate dataset dirs data/<ds>-r<K>/ built by
# scripts/make_eval_subsample.py, so their result files never mix with full-set
# ones. n=3,4 (997 prompts) is always evaluated in full. Exported so the plot job
# (plot_results.sh) looks for the same dataset names.
export OOD_SUBSAMPLE=${OOD_SUBSAMPLE:-}
if [ -n "${OOD_SUBSAMPLE}" ]; then
    EVAL_DS_N5="${EVAL_DS_N5}-r${OOD_SUBSAMPLE}"
    EVAL_DS_N6="${EVAL_DS_N6}-r${OOD_SUBSAMPLE}"
    for _ds in "${EVAL_DS_N5}" "${EVAL_DS_N6}"; do
        if [ ! -f "data/${_ds}/test.parquet" ]; then
            echo "ERROR: data/${_ds}/test.parquet missing. Create it with:" >&2
            echo "  python3 scripts/make_eval_subsample.py --source ${_ds%-r${OOD_SUBSAMPLE}} --n ${OOD_SUBSAMPLE}" >&2
            exit 1
        fi
    done
fi
SFT_DATA_NAMESPACE=${SFT_DATA_NAMESPACE:-""}
SFT_DATA_SOURCE_EXP_NAME=${SFT_DATA_SOURCE_EXP_NAME:-""}

TRAIN_SBATCH_ARGS=${TRAIN_SBATCH_ARGS:-""}
SFT_SBATCH_ARGS=${SFT_SBATCH_ARGS:-""}
EVAL_SBATCH_ARGS=${EVAL_SBATCH_ARGS:-""}
PROGDISTILL_STEPS_PER_ROUND=${PROGDISTILL_STEPS_PER_ROUND:-160}

# Checkpoint/eval cadence must agree: training saves every SAVE_FREQ steps and
# eval.sh evaluates step = EVAL_STEP_MULT x array index. If they diverge, most
# checkpoints are silently never evaluated (and then look like dead weight to
# any disk cleanup). Both default to 100; warn loudly rather than fail, since a
# deliberate mismatch is occasionally useful (e.g. save dense, eval sparse).
if [ "${SAVE_FREQ:-100}" != "${EVAL_STEP_MULT:-100}" ]; then
    echo "WARNING: SAVE_FREQ=${SAVE_FREQ:-100} != EVAL_STEP_MULT=${EVAL_STEP_MULT:-100}." >&2
    echo "         Only checkpoints on the EVAL_STEP_MULT grid will be evaluated." >&2
fi

# User-specific checkpoint dir (mirrors logic in train_grpo.sh / eval.sh)
if [ "$USER" = "mwalden" ]; then
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _account=kempner_kdbrantley_lab
    _train_partition=kempner_h100
    _eval_partition=kempner_requeue
    _eval_account=kempner_kdbrantley_lab
    _plot_extra=""
    _plot_partition=serial_requeue
    _plot_account=kdbrantley_lab
elif [ "$USER" = "sdholakia" ]; then
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _account=kempner_bingbin_lab
    _train_partition=kempner_h100
    _eval_partition=kempner_requeue
    _eval_account=kempner_bingbin_lab
    _plot_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    _plot_partition=kempner_requeue
    _plot_account=kempner_bingbin_lab
elif [ "$USER" = "bingbin" ]; then
    _checkpoint_dir=/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models
    _account=kempner_bingbin_lab
    _train_partition=kempner_h100
    _eval_partition=kempner_requeue
    _eval_account=kempner_bingbin_lab
    _plot_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    _plot_partition=kempner_requeue
    _plot_account=kempner_bingbin_lab
elif [ "${CLUSTER:-}" = "mit" ]; then
    # MIT cluster (CLUSTER=mit). mit_normal_gpu has a 6h wall-clock cap which
    # is too short for gemma GRPO (~22h on Harvard 4xH100), so train+plot
    # default to mit_preemptable. Eval is short (~30min/task) and fits in
    # mit_normal_gpu. Override the account / partitions via env vars if your
    # MIT account differs from "mit_general" or you prefer different queues.
    _checkpoint_dir=${HOME}/rl-checkpoints
    _model_dir=${HOME}/models
    _account=${MIT_ACCOUNT:-mit_general}
    _train_partition=${MIT_TRAIN_PARTITION:-mit_preemptable}
    _eval_partition=${MIT_EVAL_PARTITION:-mit_normal_gpu}
    _plot_extra=""
    _plot_partition=${MIT_PLOT_PARTITION:-mit_preemptable}
    _plot_account=${MIT_PLOT_ACCOUNT:-${_account}}
elif [ -n "${CHECKPOINT_DIR:-}" ] && [ -n "${MODEL_DIR:-}" ] && [ -n "${ACCOUNT:-}" ]; then
    # Generic fallback for any other user/cluster on SLURM: export CHECKPOINT_DIR,
    # MODEL_DIR, and ACCOUNT. Partitions default to Kempner's kempner_h100;
    # override any of them via TRAIN_PARTITION / EVAL_PARTITION / PLOT_PARTITION /
    # EVAL_ACCOUNT / PLOT_ACCOUNT / PLOT_EXTRA env vars.
    _checkpoint_dir=${CHECKPOINT_DIR}
    _model_dir=${MODEL_DIR}
    _account=${ACCOUNT}
    _train_partition=${TRAIN_PARTITION:-kempner_h100}
    _eval_partition=${EVAL_PARTITION:-kempner_h100}
    _eval_account=${EVAL_ACCOUNT:-${ACCOUNT}}
    _plot_extra=${PLOT_EXTRA:-}
    _plot_partition=${PLOT_PARTITION:-${_train_partition}}
    _plot_account=${PLOT_ACCOUNT:-${ACCOUNT}}
else
    echo "ERROR: Unknown user $USER. Set CHECKPOINT_DIR, MODEL_DIR, and ACCOUNT explicitly." >&2
    exit 1
fi
# Default eval account to train account if not set per-user above.
_eval_account=${_eval_account:-$_account}
checkpoint_dir=${CHECKPOINT_DIR:-$_checkpoint_dir}
model_dir=${MODEL_DIR:-$_model_dir}

mode_label="${DISTILL_MODE:-rl}"
if [ -n "$DISTILL_MODE" ] && [ -n "$RUN_GRPO" ]; then
    mode_label="${DISTILL_MODE}+grpo"
fi
echo "===== Pipeline: $MODEL_NAME / $EXP_NAME  [mode: ${mode_label}] ====="

# ---------------------------------------------------------------------------
# Helper: submit generate_sft_data.sh for a given teacher step
# Returns the job ID via stdout.
# ---------------------------------------------------------------------------
submit_gendata() {
    local teacher_step="$1"
    local dep_arg="${2:-}"  # e.g. "--dependency=afterok:12345" or ""
    TEACHER_EXP_NAME="${TEACHER_EXP_NAME}" \
    TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-}" \
    TEACHER_CHECKPOINT_DIR="${TEACHER_CHECKPOINT_DIR:-}" \
    SFT_DATA_NAMESPACE="${SFT_DATA_NAMESPACE:-}" \
    TEACHER_STEP="${teacher_step}" \
    N_RESPONSES="${N_RESPONSES:-4}" \
    FILTER_CORRECT_ONLY="${FILTER_CORRECT_ONLY:-false}" \
    sbatch --parsable \
        --partition=$_train_partition --account=$_account \
        ${dep_arg} \
        ${SFT_SBATCH_ARGS} \
        scripts/generate_sft_data.sh
}

# ---------------------------------------------------------------------------
# Helper: submit train_sft.sh
# Returns the job ID via stdout.
# ---------------------------------------------------------------------------
submit_sft() {
    local sft_data_path="$1"
    local sft_base_model_path="$2"
    local sft_output_dir="$3"
    local sft_train_steps="$4"
    local sft_experiment_label="$5"
    local dep_arg="${6:-}"
    SFT_DATA_PATH="${sft_data_path}" \
    SFT_BASE_MODEL_PATH="${sft_base_model_path}" \
    SFT_OUTPUT_DIR="${sft_output_dir}" \
    SFT_TRAIN_STEPS="${sft_train_steps}" \
    SFT_LR="${SFT_LR:-1e-5}" \
    SFT_SEED="${SFT_SEED:-1}" \
    SFT_LOG_FREQ="${SFT_LOG_FREQ:-50}" \
    SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.1}" \
    SFT_EXPERIMENT_LABEL="${sft_experiment_label}" \
    sbatch --parsable \
        --partition=$_train_partition --account=$_account \
        ${dep_arg} \
        ${SFT_SBATCH_ARGS} \
        scripts/train_sft.sh
}

sft_data_namespace() {
    if [ -n "${SFT_DATA_NAMESPACE}" ]; then
        echo "${SFT_DATA_NAMESPACE}"
    else
        # Default namespace suffixes with -n${N_RESPONSES} so different rollout
        # counts point at different data dirs; default N_RESPONSES=4. Truncated
        # data gets an additional -truncate suffix.
        local base="teacher-${TEACHER_MODEL_NAME:-${MODEL_NAME}}-${TEACHER_EXP_NAME}-n${N_RESPONSES:-4}"
        if [ "${TRUNCATE:-false}" = "true" ] || [ "${TRUNCATE:-false}" = "1" ]; then
            base="${base}-truncate"
        fi
        echo "${base}"
    fi
}

sft_data_path() {
    local data_namespace="$1"
    local teacher_step="$2"
    echo "${checkpoint_dir}/sft-data/${MODEL_NAME}/${data_namespace}/step_${teacher_step}.parquet"
}

metadata_path() {
    local data_path="$1"
    echo "${data_path}.metadata"
}

metadata_matches() {
    local data_path="$1"
    local teacher_step="$2"
    local meta_path

    meta_path=$(metadata_path "${data_path}")
    if [ ! -f "${meta_path}" ]; then
        return 1
    fi

    local teacher_model_name_expected="${TEACHER_MODEL_NAME:-${MODEL_NAME}}"
    local teacher_exp_name_actual=""
    local teacher_model_name_actual=""
    local teacher_step_actual=""
    local data_source_actual=""
    local n_responses_actual=""
    local filter_correct_only_actual=""
    local truncate_actual=""
    local max_length_actual=""

    while IFS='=' read -r key value; do
        case "${key}" in
            teacher_exp_name) teacher_exp_name_actual="${value}" ;;
            teacher_model_name) teacher_model_name_actual="${value}" ;;
            teacher_step) teacher_step_actual="${value}" ;;
            data_source) data_source_actual="${value}" ;;
            n_responses) n_responses_actual="${value}" ;;
            filter_correct_only) filter_correct_only_actual="${value}" ;;
            truncate) truncate_actual="${value}" ;;
            max_length) max_length_actual="${value}" ;;
        esac
    done < "${meta_path}"

    [ "${teacher_exp_name_actual}" = "${TEACHER_EXP_NAME}" ] || return 1
    [ "${teacher_model_name_actual}" = "${teacher_model_name_expected}" ] || return 1
    [ "${teacher_step_actual}" = "${teacher_step}" ] || return 1
    [ "${data_source_actual}" = "${DATA_SOURCE}" ] || return 1
    [ "${n_responses_actual}" = "${N_RESPONSES:-4}" ] || return 1
    [ "${filter_correct_only_actual}" = "${FILTER_CORRECT_ONLY:-false}" ] || return 1
    [ "${truncate_actual}" = "${TRUNCATE:-false}" ] || return 1
    [ "${max_length_actual}" = "${MAX_LENGTH:-1024}" ] || return 1
}

prepare_sft_data() {
    local teacher_step="$1"
    local current_path
    local source_path=""

    current_path=$(sft_data_path "$(sft_data_namespace)" "${teacher_step}")
    LAST_SFT_DATA_PATH="${current_path}"
    LAST_GENDATA_JID=""
    LAST_GENDATA_ACTION="generate"

    if [ -f "${current_path}" ] && metadata_matches "${current_path}" "${teacher_step}"; then
        LAST_GENDATA_ACTION="reuse-current"
        return
    fi

    if [ -n "${SFT_DATA_SOURCE_EXP_NAME}" ] && [ "${SFT_DATA_SOURCE_EXP_NAME}" != "${EXP_NAME}" ]; then
        source_path=$(sft_data_path "${SFT_DATA_SOURCE_EXP_NAME}" "${teacher_step}")
        if [ -f "${source_path}" ] && metadata_matches "${source_path}" "${teacher_step}"; then
            LAST_SFT_DATA_PATH="${source_path}"
            LAST_GENDATA_ACTION="reuse-source"
            return
        fi
    fi

    LAST_GENDATA_JID=$(submit_gendata "${teacher_step}")
}

# ---------------------------------------------------------------------------
# Mode: plain RL (no distillation)
# ---------------------------------------------------------------------------
if [ -z "$DISTILL_MODE" ]; then
    TRAIN_JID=$(sbatch --parsable --account=$_account --partition=$_train_partition $TRAIN_SBATCH_ARGS scripts/train_grpo.sh)
    echo "Train:  job $TRAIN_JID"
    FINAL_JID="${TRAIN_JID}"
    FINAL_EVAL_CKPT=""
    FINAL_EVAL_SUBDIR=""

# ---------------------------------------------------------------------------
# Mode: distill — SFT on teacher's final checkpoint, then GRPO
# ---------------------------------------------------------------------------
elif [ "$DISTILL_MODE" = "distill" ]; then
    sft_ckpt_dir=${checkpoint_dir}/sft-checkpoints/${MODEL_NAME}/${EXP_NAME}
    base_model_path=${model_dir}/${MODEL_NAME}
    sft_steps=${SFT_TRAIN_STEPS:-1600}

    # 1. Generate SFT data from teacher's final checkpoint
    prepare_sft_data "final"
    sft_data_path="${LAST_SFT_DATA_PATH}"
    GENDATA_JID="${LAST_GENDATA_JID}"
    if [ -n "${GENDATA_JID}" ]; then
        echo "GenData: job $GENDATA_JID (teacher final checkpoint)"
        sft_dep="--dependency=afterok:${GENDATA_JID}"
    else
        echo "GenData: reuse ${sft_data_path}"
        sft_dep=""
    fi

    # 2. SFT on base model
    SFT_JID=$(submit_sft \
        "${sft_data_path}" \
        "${base_model_path}" \
        "${sft_ckpt_dir}" \
        "${sft_steps}" \
        "sft" \
        "${sft_dep}")
    echo "SFT:    job $SFT_JID (${sft_steps} steps)"

    # 3. GRPO starting from SFT checkpoint (only if --GRPO specified)
    TRAIN_JID=""
    FINAL_JID="${SFT_JID}"
    FINAL_EVAL_CKPT="${sft_ckpt_dir}"
    FINAL_EVAL_SUBDIR="sft_final"
    if [ -n "$RUN_GRPO" ]; then
        TRAIN_JID=$(MODEL_PATH="${sft_ckpt_dir}" sbatch --parsable \
            --account=$_account --partition=$_train_partition \
            --dependency=afterok:${SFT_JID} \
            $TRAIN_SBATCH_ARGS \
            scripts/train_grpo.sh)
        echo "Train:  job $TRAIN_JID (GRPO from SFT checkpoint)"
        FINAL_JID="${TRAIN_JID}"
        FINAL_EVAL_CKPT=""
        FINAL_EVAL_SUBDIR=""
    fi

# ---------------------------------------------------------------------------
# Mode: progdistill — (N=32, T=PROGDISTILL_STEPS_PER_ROUND) progressive
#   distillation following arxiv:2410.05464, then GRPO
# ---------------------------------------------------------------------------
elif [ "$DISTILL_MODE" = "progdistill" ]; then
    sft_ckpt_base=${checkpoint_dir}/sft-checkpoints/${MODEL_NAME}/${EXP_NAME}
    base_model_path=${model_dir}/${MODEL_NAME}

    # Teacher checkpoints to distill from, in order. Must be a subset of the
    # steps the teacher actually saved: at save_freq=100 (train_grpo.sh default)
    # that means multiples of 100, so the old 150-spaced list is no longer
    # valid — 8 rounds at every 200 steps is the 100-aligned equivalent.
    # Teachers trained at the legacy save_freq=50 keep their old rounds via
    #   TEACHER_STEPS="150 300 450 600 750 900 1050 1200 1400 1600"
    # NB: total SFT budget = ${#teacher_steps[@]} x PROGDISTILL_STEPS_PER_ROUND,
    # so 8 rounds x 160 = 1280; use PROGDISTILL_STEPS_PER_ROUND=200 for 1600.
    read -r -a teacher_steps <<< "${TEACHER_STEPS:-200 400 600 800 1000 1200 1400 1600}"
    T=${PROGDISTILL_STEPS_PER_ROUND}

    prev_sft_jid=""
    prev_sft_ckpt="${base_model_path}"

    for step in "${teacher_steps[@]}"; do
        sft_out_dir=${sft_ckpt_base}/round_${step}
        prepare_sft_data "${step}"
        sft_data_path="${LAST_SFT_DATA_PATH}"
        GENDATA_JID="${LAST_GENDATA_JID}"

        # SFT depends on: (a) data for this step if it must be generated,
        # (b) previous SFT round
        if [ -z "${GENDATA_JID}" ] && [ -z "${prev_sft_jid}" ]; then
            dep=""
        elif [ -z "${GENDATA_JID}" ]; then
            dep="--dependency=afterok:${prev_sft_jid}"
        elif [ -z "${prev_sft_jid}" ]; then
            dep="--dependency=afterok:${GENDATA_JID}"
        else
            dep="--dependency=afterok:${GENDATA_JID}:${prev_sft_jid}"
        fi

        SFT_JID=$(submit_sft \
            "${sft_data_path}" \
            "${prev_sft_ckpt}" \
            "${sft_out_dir}" \
            "${T}" \
            "sft-round-${step}" \
            "${dep}")

        if [ -n "${GENDATA_JID}" ]; then
            echo "Step ${step}: GenData=$GENDATA_JID  SFT=$SFT_JID  (base: $(basename ${prev_sft_ckpt}))"
        else
            echo "Step ${step}: ReuseData=$(basename "${sft_data_path}")  SFT=$SFT_JID  (base: $(basename ${prev_sft_ckpt}))"
        fi

        prev_sft_jid="${SFT_JID}"
        prev_sft_ckpt="${sft_ckpt_base}/round_${step}"
    done

    final_sft_ckpt="${sft_ckpt_base}/round_1600"

    # GRPO from final progressive SFT checkpoint (only if --GRPO specified)
    TRAIN_JID=""
    FINAL_JID="${prev_sft_jid}"
    FINAL_EVAL_CKPT="${final_sft_ckpt}"
    FINAL_EVAL_SUBDIR="round_1600"
    if [ -n "$RUN_GRPO" ]; then
        TRAIN_JID=$(MODEL_PATH="${final_sft_ckpt}" sbatch --parsable \
            --account=$_account --partition=$_train_partition \
            --dependency=afterok:${prev_sft_jid} \
            $TRAIN_SBATCH_ARGS \
            scripts/train_grpo.sh)
        echo "Train:  job $TRAIN_JID (GRPO from progdistill round_1600 checkpoint)"
        FINAL_JID="${TRAIN_JID}"
        FINAL_EVAL_CKPT=""
        FINAL_EVAL_SUBDIR=""
    fi

# ---------------------------------------------------------------------------
# Mode: progdistill_mixture — like progdistill, generate data from all 10
#   teacher checkpoints, but pool every checkpoint's responses into ONE dataset
#   and run a single SFT pass over the mixture (from the base model) instead of
#   one sequential SFT round per checkpoint. Then optional GRPO.
#
#   Reuses exactly the same per-step SFT parquets as progdistill (same data
#   namespace), so if progdistill data already exists for this teacher it is
#   reused. verl's SFTDataset accepts a list of parquets and concatenates them
#   (sft_dataset.py), so the mixture is passed as a Hydra list literal — no
#   separate merge step.
# ---------------------------------------------------------------------------
elif [ "$DISTILL_MODE" = "progdistill_mixture" ]; then
    sft_ckpt_dir=${checkpoint_dir}/sft-checkpoints/${MODEL_NAME}/${EXP_NAME}
    base_model_path=${model_dir}/${MODEL_NAME}
    sft_steps=${SFT_TRAIN_STEPS:-1600}

    # Same teacher checkpoints as progdistill (see the notes there on save_freq
    # alignment and the TEACHER_STEPS override); here their generated data is
    # concatenated rather than consumed one round at a time.
    read -r -a teacher_steps <<< "${TEACHER_STEPS:-200 400 600 800 1000 1200 1400 1600}"

    # 1. Ensure data exists for every teacher step (generate or reuse), and
    #    collect the parquet paths + the gen-data jobs the SFT must wait on.
    mixture_paths=()
    gendata_jids=()
    for step in "${teacher_steps[@]}"; do
        prepare_sft_data "${step}"
        mixture_paths+=("${LAST_SFT_DATA_PATH}")
        if [ -n "${LAST_GENDATA_JID}" ]; then
            gendata_jids+=("${LAST_GENDATA_JID}")
            echo "Step ${step}: GenData=${LAST_GENDATA_JID}  -> $(basename "${LAST_SFT_DATA_PATH}")"
        else
            echo "Step ${step}: ReuseData=$(basename "${LAST_SFT_DATA_PATH}")"
        fi
    done

    # Hydra list literal of all per-step parquets (paths have no spaces).
    mixture_list="[$(IFS=,; echo "${mixture_paths[*]}")]"

    # 2. Single SFT job depends on every gen-data job that was actually submitted.
    if [ ${#gendata_jids[@]} -gt 0 ]; then
        sft_dep="--dependency=afterok:$(IFS=:; echo "${gendata_jids[*]}")"
        echo "GenData: ${#gendata_jids[@]} job(s) to complete before SFT"
    else
        sft_dep=""
        echo "GenData: all ${#teacher_steps[@]} step datasets reused"
    fi

    # 3. Single SFT pass over the pooled mixture, starting from the base model.
    SFT_JID=$(submit_sft \
        "${mixture_list}" \
        "${base_model_path}" \
        "${sft_ckpt_dir}" \
        "${sft_steps}" \
        "sft-mixture" \
        "${sft_dep}")
    echo "SFT:    job $SFT_JID (${sft_steps} steps, mixture of ${#teacher_steps[@]} checkpoints)"

    # 4. GRPO from the mixture SFT checkpoint (only if --GRPO specified)
    TRAIN_JID=""
    FINAL_JID="${SFT_JID}"
    FINAL_EVAL_CKPT="${sft_ckpt_dir}"
    FINAL_EVAL_SUBDIR="sft_final"
    if [ -n "$RUN_GRPO" ]; then
        TRAIN_JID=$(MODEL_PATH="${sft_ckpt_dir}" sbatch --parsable \
            --account=$_account --partition=$_train_partition \
            --dependency=afterok:${SFT_JID} \
            $TRAIN_SBATCH_ARGS \
            scripts/train_grpo.sh)
        echo "Train:  job $TRAIN_JID (GRPO from progdistill_mixture SFT checkpoint)"
        FINAL_JID="${TRAIN_JID}"
        FINAL_EVAL_CKPT=""
        FINAL_EVAL_SUBDIR=""
    fi

# ---------------------------------------------------------------------------
# Mode: opd — offline on-policy distillation (arXiv:2604.13010) on top of an
#   existing SFT student. One gen-data job (student rollouts + teacher logprob
#   precompute), one training job. Checkpoints land at
#   checkpoints/${MODEL_NAME}/${EXP_NAME}/global_step_{50..1600}, so the
#   standard eval array + plot chain below applies unchanged.
# ---------------------------------------------------------------------------
elif [ "$DISTILL_MODE" = "opd" ]; then
    if [ -n "$RUN_GRPO" ]; then
        echo "ERROR: --GRPO is not supported with --opd." >&2
        exit 1
    fi
    opd_student_path=${OPD_STUDENT_PATH:?OPD_STUDENT_PATH must be set for --opd (merged SFT student checkpoint)}
    opd_teacher_step=${TEACHER_STEP:-1600}
    case "${opd_teacher_step}" in
        ''|*[!0-9]*)
            echo "ERROR: TEACHER_STEP must be numeric for --opd (got '${opd_teacher_step}')." >&2
            exit 1
            ;;
    esac
    opd_n=${N_RESPONSES:-4}
    opd_temp=${OPD_ROLLOUT_TEMP:-1.0}
    opd_output_dir=${checkpoint_dir}/checkpoints/${MODEL_NAME}/${EXP_NAME}

    # Data namespace/path — must mirror the derivation in generate_opd_data.sh.
    student_tag=$(basename "${opd_student_path}")
    case "${student_tag}" in
        round_*|global_step_*)
            student_tag="$(basename "$(dirname "${opd_student_path}")")-${student_tag}"
            ;;
    esac
    _opd_ns_default="student-${student_tag}-teacher-${TEACHER_MODEL_NAME:-${MODEL_NAME}}-${TEACHER_EXP_NAME}-step${opd_teacher_step}-n${opd_n}-temp${opd_temp}-p${OPD_MAX_PROMPTS:-10000}"
    opd_namespace=${OPD_DATA_NAMESPACE:-${_opd_ns_default}}
    opd_data_path=${checkpoint_dir}/opd-data/${MODEL_NAME}/${opd_namespace}/opd.parquet

    if [ -f "${opd_data_path}" ] && [ -f "${opd_data_path}.metadata" ]; then
        echo "GenData: reuse ${opd_data_path}"
        GENDATA_JID=""
        train_dep=""
    else
        GENDATA_JID=$(OPD_STUDENT_PATH="${opd_student_path}" \
            TEACHER_EXP_NAME="${TEACHER_EXP_NAME}" \
            TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-}" \
            TEACHER_CHECKPOINT_DIR="${TEACHER_CHECKPOINT_DIR:-}" \
            TEACHER_STEP="${opd_teacher_step}" \
            OPD_DATA_NAMESPACE="${opd_namespace}" \
            N_RESPONSES="${opd_n}" \
            OPD_ROLLOUT_TEMP="${opd_temp}" \
            OPD_MAX_PROMPTS="${OPD_MAX_PROMPTS:-10000}" \
            sbatch --parsable \
                --partition=$_train_partition --account=$_account \
                ${SFT_SBATCH_ARGS} \
                scripts/generate_opd_data.sh)
        echo "GenData: job $GENDATA_JID (student rollouts + teacher logprob scoring)"
        train_dep="--dependency=afterok:${GENDATA_JID}"
    fi

    TRAIN_JID=$(OPD_DATA_PATH="${opd_data_path}" \
        OPD_BASE_MODEL_PATH="${opd_student_path}" \
        OPD_OUTPUT_DIR="${opd_output_dir}" \
        OPD_TRAIN_STEPS="${OPD_TRAIN_STEPS:-1600}" \
        OPD_LR="${OPD_LR:-2e-6}" \
        OPD_CLIP="${OPD_CLIP:-10.0}" \
        OPD_SEED="${OPD_SEED:-1}" \
        sbatch --parsable \
            --partition=$_train_partition --account=$_account \
            ${train_dep} \
            ${TRAIN_SBATCH_ARGS} \
            scripts/train_opd.sh)
    echo "Train:  job $TRAIN_JID (OPD, ${OPD_TRAIN_STEPS:-1600} steps from $(basename ${opd_student_path}))"
    FINAL_JID="${TRAIN_JID}"
    FINAL_EVAL_CKPT=""
    FINAL_EVAL_SUBDIR=""

else
    echo "ERROR: Unknown DISTILL_MODE=${DISTILL_MODE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Eval + plot — runs off FINAL_JID, which points at the final training stage
# (GRPO for plain-RL / distill+GRPO / progdistill+GRPO; final SFT job for
# distill-only / progdistill-only).
#
# When GRPO ran, eval uses its baked-in array over global_step_* checkpoints.
# When only SFT ran, eval is submitted as a single non-array task per dataset
# with EVAL_CHECKPOINT_PATH / EVAL_RESULT_PATH pointing at the final SFT dir.
# ---------------------------------------------------------------------------
if [ -n "${FINAL_JID:-}" ]; then
    if [ -z "${TRAIN_JID:-}" ]; then
        # SFT-only: eval the single final SFT checkpoint per dataset.
        project_dir=${PROJECT_DIR:-$PWD}
        eval_result_base=${project_dir}/results/${MODEL_NAME}/${EXP_NAME}/${FINAL_EVAL_SUBDIR}
        EVAL_JID1=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N34} \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_eval_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                ${EVAL_SBATCH_ARGS} \
                scripts/eval.sh)
        EVAL_JID2=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N5} \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_eval_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                ${EVAL_SBATCH_ARGS} \
                scripts/eval.sh)
        EVAL_JID3=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N6} \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_eval_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                ${EVAL_SBATCH_ARGS} \
                scripts/eval.sh)
    else
        # Checkpoint-sweep eval. The step evaluated is EVAL_STEP_MULT x array
        # index (eval.sh), so the array must span the whole run: eval.sh's
        # baked-in 1-16 covers a 1600-step run at save_freq=100, and longer runs
        # need EVAL_ARRAY (e.g. TOTAL_EPOCHS=2 -> EVAL_ARRAY=1-32). Passing it
        # here beats the in-script #SBATCH directive.
        _eval_array_arg=${EVAL_ARRAY:+--array=${EVAL_ARRAY}}
        EVAL_JID1=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N34} sbatch --parsable \
            --partition=$_eval_partition --account=$_eval_account \
            --dependency=afterok:${FINAL_JID} \
            ${_eval_array_arg} \
            ${EVAL_SBATCH_ARGS} \
            scripts/eval.sh)
        EVAL_JID2=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N5} sbatch --parsable \
            --partition=$_eval_partition --account=$_eval_account \
            --dependency=afterok:${FINAL_JID} \
            ${_eval_array_arg} \
            ${EVAL_SBATCH_ARGS} \
            scripts/eval.sh)
        EVAL_JID3=$(EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" EVAL_DATASET=${EVAL_DS_N6} sbatch --parsable \
            --partition=$_eval_partition --account=$_eval_account \
            --dependency=afterok:${FINAL_JID} \
            ${_eval_array_arg} \
            ${EVAL_SBATCH_ARGS} \
            scripts/eval.sh)
    fi
    echo "Eval:   job $EVAL_JID1 (n=3,4)  $EVAL_JID2 (n=5)  $EVAL_JID3 (n=6)"

    PLOT_JID=$(sbatch --parsable \
        --partition=$_plot_partition --account=$_plot_account $_plot_extra \
        --dependency=afterok:${EVAL_JID1}:${EVAL_JID2}:${EVAL_JID3} \
        scripts/plot/plot_results.sh)
    echo "Plot:   job $PLOT_JID"
fi

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
