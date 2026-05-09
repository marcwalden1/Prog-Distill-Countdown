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
# --GRPO can be combined with either --distill or --progdistill to run RL
# after distillation. Without --GRPO those modes stop at SFT.
#
# For 7B (needs 8 GPUs):
#   MODEL_NAME=Qwen2.5-7B EXP_NAME=my-exp \
#     TRAIN_SBATCH_ARGS="--gres=gpu:nvidia_h100_80gb_hbm3:8 -t 30:00:00" \
#     bash scripts/run_pipeline.sh
#
# Optional env var overrides:
#   KL_COEF                    — KL loss coefficient (default: 0.001)
#   DATA_SOURCE                — data directory under data/ (default: balanced)
#   EXTRA_ARGS                 — extra args forwarded to eval.py
#   TRAIN_SBATCH_ARGS          — extra sbatch args for train_grpo.sh
#   SFT_SBATCH_ARGS            — extra sbatch args for train_sft.sh / generate_sft_data.sh
#   PROGDISTILL_STEPS_PER_ROUND — SFT steps per teacher checkpoint round (default: 50)
#   SFT_TRAIN_STEPS            — total SFT steps for --distill mode (default: 1600, matches progdistill budget)
#   SFT_LR                     — SFT learning rate (default: 1e-5)
#   N_RESPONSES                — teacher responses per prompt for data gen (default: 4)
#   TEACHER_MODEL_NAME         — teacher arch/size for cross-size distill (default: MODEL_NAME = student)
#   FILTER_CORRECT_ONLY        — keep only score==1.0 teacher responses (default: false; keeps all)
#   SFT_DATA_NAMESPACE         — optional override for where generated SFT parquet data is stored
#                                and reused; default is derived from the teacher checkpoint lineage
#   SFT_DATA_SOURCE_EXP_NAME   — legacy override to reuse older SFT parquet data stored under a
#                                previous EXP_NAME-based layout

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
SFT_DATA_NAMESPACE=${SFT_DATA_NAMESPACE:-""}
SFT_DATA_SOURCE_EXP_NAME=${SFT_DATA_SOURCE_EXP_NAME:-""}

TRAIN_SBATCH_ARGS=${TRAIN_SBATCH_ARGS:-""}
SFT_SBATCH_ARGS=${SFT_SBATCH_ARGS:-""}
PROGDISTILL_STEPS_PER_ROUND=${PROGDISTILL_STEPS_PER_ROUND:-160}

# User-specific checkpoint dir (mirrors logic in train_grpo.sh / eval.sh)
if [ "$USER" = "mwalden" ]; then
    _checkpoint_dir=/n/netscratch/kdbrantley_lab/Lab/mwalden/rl-checkpoints
    _model_dir=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models
    _account=kempner_kdbrantley_lab
    _train_partition=kempner_h100
    _eval_partition=kempner_h100
    _plot_extra=""
    _plot_partition=serial_requeue
    _plot_account=kdbrantley_lab
elif [ "$USER" = "sdholakia" ]; then
    _checkpoint_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints
    _model_dir=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/models
    _account=kempner_bingbin_lab
    _train_partition=kempner_h100
    _eval_partition=kempner_requeue
    _plot_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    _plot_partition=kempner_requeue
    _plot_account=kempner_bingbin_lab
else
    echo "ERROR: Unknown user $USER. Set CHECKPOINT_DIR and MODEL_DIR explicitly." >&2
    exit 1
fi
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

    # Teacher checkpoints: the 10 steps at which eval.sh evaluates
    # (array indices 3,6,9,12,15,18,21,24,28,32 × 50 = these steps)
    teacher_steps=(150 300 450 600 750 900 1050 1200 1400 1600)
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
        EVAL_JID1=$(EVAL_DATASET=balanced \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                scripts/eval.sh)
        EVAL_JID2=$(EVAL_DATASET=balanced5 \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                scripts/eval.sh)
        EVAL_JID3=$(EVAL_DATASET=balanced6 \
            EVAL_CHECKPOINT_PATH="${FINAL_EVAL_CKPT}" \
            EVAL_RESULT_PATH="${eval_result_base}" \
            sbatch --parsable \
                --partition=$_eval_partition --account=$_account \
                --array=1-1 \
                --dependency=afterok:${FINAL_JID} \
                scripts/eval.sh)
    else
        EVAL_JID1=$(EVAL_DATASET=balanced  sbatch --parsable \
            --partition=$_eval_partition --account=$_account \
            --dependency=afterok:${FINAL_JID} \
            scripts/eval.sh)
        EVAL_JID2=$(EVAL_DATASET=balanced5 sbatch --parsable \
            --partition=$_eval_partition --account=$_account \
            --dependency=afterok:${FINAL_JID} \
            scripts/eval.sh)
        EVAL_JID3=$(EVAL_DATASET=balanced6 sbatch --parsable \
            --partition=$_eval_partition --account=$_account \
            --dependency=afterok:${FINAL_JID} \
            scripts/eval.sh)
    fi
    echo "Eval:   job $EVAL_JID1 (n=3,4)  $EVAL_JID2 (n=5)  $EVAL_JID3 (n=6)"

    PLOT_JID=$(sbatch --parsable \
        --partition=$_plot_partition --account=$_plot_account $_plot_extra \
        --dependency=afterok:${EVAL_JID1}:${EVAL_JID2}:${EVAL_JID3} \
        scripts/plot_results.sh)
    echo "Plot:   job $PLOT_JID"
fi

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
