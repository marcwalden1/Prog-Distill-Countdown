#!/bin/bash
# Tests for scripts/run_pipeline.sh argument parsing and SLURM dependency chains.
#
# Uses a fake `sbatch` that logs its arguments and returns sequential job IDs.
# No actual SLURM cluster is required.
#
# Run with:
#   bash tests/test_run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0; FAIL=0

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

ok() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        ok "$desc"
    else
        fail "$desc — expected '$expected', got '$actual'"
    fi
}

assert_ge() {
    local desc="$1" min="$2" actual="$3"
    if [ "$actual" -ge "$min" ]; then
        ok "$desc"
    else
        fail "$desc — expected >=$min, got $actual"
    fi
}

assert_contains() {
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        ok "$desc"
    else
        fail "$desc — '$needle' not found in output"
    fi
}

assert_not_contains() {
    local desc="$1" needle="$2" haystack="$3"
    if ! echo "$haystack" | grep -q "$needle"; then
        ok "$desc"
    else
        fail "$desc — '$needle' unexpectedly found in output"
    fi
}

# Create a temporary workspace for each test
setup_workspace() {
    TEST_TMPDIR=$(mktemp -d)
    SBATCH_LOG="${TEST_TMPDIR}/sbatch_calls.log"
    JOB_COUNTER_FILE="${TEST_TMPDIR}/job_counter"
    echo "1000" > "$JOB_COUNTER_FILE"

    # Fake sbatch: logs all args, returns sequential job ID
    mkdir -p "${TEST_TMPDIR}/bin"
    cat > "${TEST_TMPDIR}/bin/sbatch" <<'SBATCH_EOF'
#!/bin/bash
LOG_FILE="${SBATCH_LOG}"
COUNTER_FILE="${JOB_COUNTER_FILE}"
# Increment counter
count=$(cat "$COUNTER_FILE")
count=$((count + 1))
echo "$count" > "$COUNTER_FILE"
# Log the call
echo "$*" >> "$LOG_FILE"
# Return the job ID (--parsable just prints the numeric ID)
echo "$count"
SBATCH_EOF
    chmod +x "${TEST_TMPDIR}/bin/sbatch"

    export PATH="${TEST_TMPDIR}/bin:${PATH}"
    export SBATCH_LOG JOB_COUNTER_FILE

    # Export cluster env vars so user-detection in run_pipeline.sh doesn't fail
    export USER=mwalden
    export CHECKPOINT_DIR="${TEST_TMPDIR}/checkpoints"
    export MODEL_DIR="${TEST_TMPDIR}/models"
    mkdir -p "${CHECKPOINT_DIR}" "${MODEL_DIR}"
}

teardown_workspace() {
    rm -rf "$TEST_TMPDIR"
}

run_pipeline() {
    # Run run_pipeline.sh from REPO_ROOT with given args
    # Capture stdout/stderr
    cd "$REPO_ROOT"
    MODEL_NAME=Qwen2.5-0.5B \
    EXP_NAME=test-exp \
    DATA_SOURCE=balanced \
    bash scripts/run_pipeline.sh "$@" 2>&1
}

count_sbatch_calls() {
    # Number of lines in sbatch log = number of sbatch invocations
    [ -f "$SBATCH_LOG" ] && wc -l < "$SBATCH_LOG" || echo "0"
}

sbatch_args_at() {
    # Return the Nth sbatch call (1-indexed)
    sed -n "${1}p" "$SBATCH_LOG"
}

# ---------------------------------------------------------------------------
# Test: plain RL mode (no flags)
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: plain RL mode (no distillation) ==="
setup_workspace
OUTPUT=$(run_pipeline)

total=$(count_sbatch_calls)
assert_eq "Total sbatch calls (1 train + 3 eval + 1 plot = 5)" "5" "$total"
assert_contains "Output mentions Train job" "Train:" "$OUTPUT"
assert_contains "Output mentions Eval job" "Eval:" "$OUTPUT"
assert_contains "Output mentions Plot job" "Plot:" "$OUTPUT"
assert_not_contains "No SFT job in RL mode" "SFT" "$OUTPUT"
assert_not_contains "No GenData job in RL mode" "GenData" "$OUTPUT"

# GRPO job should be the first call and contain train_grpo.sh
call1=$(sbatch_args_at 1)
assert_contains "First job is train_grpo.sh" "train_grpo.sh" "$call1"

# Eval jobs should have dependency on train job
for i in 2 3 4; do
    calli=$(sbatch_args_at $i)
    assert_contains "Eval job $i has afterok dependency" "afterok" "$calli"
    assert_contains "Eval job $i targets eval.sh" "eval.sh" "$calli"
done

# Plot job should depend on all 3 eval jobs
call5=$(sbatch_args_at 5)
assert_contains "Plot job has afterok dependency" "afterok" "$call5"
assert_contains "Plot job targets plot_results.sh" "plot_results.sh" "$call5"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --distill mode (no --GRPO) — SFT only, eval against final SFT dir
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --distill mode (no --GRPO) ==="
setup_workspace
OUTPUT=$(run_pipeline --distill teacher-exp-name)

total=$(count_sbatch_calls)
# gendata(1) + sft(1) + eval×3(3) + plot(1) = 6
assert_eq "Total sbatch calls for --distill (6)" "6" "$total"

assert_contains "Output mentions GenData" "GenData" "$OUTPUT"
assert_contains "Output mentions SFT" "SFT" "$OUTPUT"
assert_not_contains "No GRPO job without --GRPO" "train_grpo.sh" "$(cat "$SBATCH_LOG")"

# Job 1: gendata
call1=$(sbatch_args_at 1)
assert_contains "Job 1 is generate_sft_data.sh" "generate_sft_data.sh" "$call1"
assert_not_contains "GenData has no upstream dependency" "afterok" "$call1"

# Job 2: SFT depends on gendata
call2=$(sbatch_args_at 2)
assert_contains "Job 2 is train_sft.sh" "train_sft.sh" "$call2"
assert_contains "SFT depends on gendata job" "afterok" "$call2"

# Jobs 3,4,5: eval.sh (single-task, --array=1-1, depend on SFT)
for i in 3 4 5; do
    calli=$(sbatch_args_at $i)
    assert_contains "Job $i is eval.sh" "eval.sh" "$calli"
    assert_contains "Eval $i depends on SFT" "afterok" "$calli"
    assert_contains "Eval $i uses --array=1-1 (SFT-only)" "array=1-1" "$calli"
done

# Job 6: plot
call6=$(sbatch_args_at 6)
assert_contains "Job 6 is plot_results.sh" "plot_results.sh" "$call6"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --distill --GRPO mode — SFT then GRPO then eval array + plot
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --distill --GRPO mode ==="
setup_workspace
OUTPUT=$(run_pipeline --distill teacher-exp-name --GRPO)

total=$(count_sbatch_calls)
# gendata(1) + sft(1) + grpo(1) + eval×3(3) + plot(1) = 7
assert_eq "Total sbatch calls for --distill --GRPO (7)" "7" "$total"

assert_contains "Output mentions GenData" "GenData" "$OUTPUT"
assert_contains "Output mentions SFT" "SFT" "$OUTPUT"
assert_contains "Output mentions GRPO" "GRPO" "$OUTPUT"

# Job 3: GRPO depends on SFT
call3=$(sbatch_args_at 3)
assert_contains "Job 3 is train_grpo.sh" "train_grpo.sh" "$call3"
assert_contains "GRPO depends on SFT" "afterok" "$call3"

# Jobs 4,5,6: eval.sh (default array, no --array=1-1 override, depend on GRPO)
for i in 4 5 6; do
    calli=$(sbatch_args_at $i)
    assert_contains "Job $i is eval.sh" "eval.sh" "$calli"
    assert_contains "Eval $i depends on GRPO" "afterok" "$calli"
    assert_not_contains "Eval $i does not use --array=1-1 (GRPO array mode)" "array=1-1" "$calli"
done

# Job 7: plot
call7=$(sbatch_args_at 7)
assert_contains "Job 7 is plot_results.sh" "plot_results.sh" "$call7"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --progdistill mode (no --GRPO) — 10-round SFT chain, eval final round
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --progdistill mode (no --GRPO) ==="
setup_workspace
OUTPUT=$(run_pipeline --progdistill teacher-exp-name)

total=$(count_sbatch_calls)
# 10 gendata + 10 sft + 3 eval + 1 plot = 24
assert_eq "Total sbatch calls for --progdistill (24)" "24" "$total"

gendata_count=$(grep -c "generate_sft_data.sh" "$SBATCH_LOG" || true)
assert_eq "10 generate_sft_data.sh calls" "10" "$gendata_count"

sft_count=$(grep -c "train_sft.sh" "$SBATCH_LOG" || true)
assert_eq "10 train_sft.sh calls" "10" "$sft_count"

# SFT jobs must each depend on the prior SFT job (chain) — all have afterok
sft_with_dep=$(grep "train_sft.sh" "$SBATCH_LOG" | grep -c "afterok" || true)
assert_eq "All 10 SFT jobs have dependencies" "10" "$sft_with_dep"

# No GRPO job without --GRPO
grpo_count=$(grep -c "train_grpo.sh" "$SBATCH_LOG" || true)
assert_eq "No train_grpo.sh call" "0" "$grpo_count"

# 3 eval jobs
eval_count=$(grep -c "eval.sh" "$SBATCH_LOG" || true)
assert_eq "Exactly 3 eval.sh calls" "3" "$eval_count"

# All 3 eval jobs use --array=1-1 (SFT-only mode)
eval_array1=$(grep "eval.sh" "$SBATCH_LOG" | grep -c "array=1-1" || true)
assert_eq "All 3 eval jobs use --array=1-1" "3" "$eval_array1"

plot_count=$(grep -c "plot_results.sh" "$SBATCH_LOG" || true)
assert_eq "Exactly 1 plot_results.sh call" "1" "$plot_count"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --progdistill reuses existing SFT data before submitting gen-data jobs
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --progdistill reuses existing SFT data ==="
setup_workspace
mkdir -p "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp"
touch "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_150.parquet"
touch "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_300.parquet"
cat > "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_150.parquet.metadata" <<'EOF'
teacher_model_name=Qwen2.5-0.5B
teacher_exp_name=teacher-exp-name
teacher_step=150
data_source=balanced
n_responses=16
filter_correct_only=false
truncate=false
max_length=1024
EOF
cat > "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_300.parquet.metadata" <<'EOF'
teacher_model_name=Qwen2.5-0.5B
teacher_exp_name=teacher-exp-name
teacher_step=300
data_source=balanced
n_responses=16
filter_correct_only=false
truncate=false
max_length=1024
EOF
OUTPUT=$(SFT_DATA_SOURCE_EXP_NAME=source-exp run_pipeline --progdistill teacher-exp-name)

total=$(count_sbatch_calls)
# 8 gendata + 10 sft + 3 eval + 1 plot = 22
assert_eq "Total sbatch calls for --progdistill with reused data (22)" "22" "$total"

gendata_count=$(grep -c "generate_sft_data.sh" "$SBATCH_LOG" || true)
assert_eq "Only 8 generate_sft_data.sh calls when 2 steps are reused" "8" "$gendata_count"

sft_count=$(grep -c "train_sft.sh" "$SBATCH_LOG" || true)
assert_eq "10 train_sft.sh calls with reused data" "10" "$sft_count"

assert_contains "Output notes reused step_150 parquet" "Step 150: ReuseData=step_150.parquet" "$OUTPUT"
assert_contains "Output notes reused step_300 parquet" "Step 300: ReuseData=step_300.parquet" "$OUTPUT"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --progdistill does not reuse mismatched SFT data metadata
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --progdistill rejects mismatched SFT data metadata ==="
setup_workspace
mkdir -p "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp"
touch "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_150.parquet"
cat > "${CHECKPOINT_DIR}/sft-data/Qwen2.5-0.5B/source-exp/step_150.parquet.metadata" <<'EOF'
teacher_model_name=Qwen2.5-0.5B
teacher_exp_name=wrong-teacher-exp
teacher_step=150
data_source=balanced
n_responses=16
filter_correct_only=false
truncate=false
max_length=1024
EOF
OUTPUT=$(SFT_DATA_SOURCE_EXP_NAME=source-exp run_pipeline --progdistill teacher-exp-name)

total=$(count_sbatch_calls)
# All 10 gendata + 10 sft + 3 eval + 1 plot = 24
assert_eq "Total sbatch calls for --progdistill with mismatched metadata (24)" "24" "$total"

gendata_count=$(grep -c "generate_sft_data.sh" "$SBATCH_LOG" || true)
assert_eq "All 10 generate_sft_data.sh calls happen when metadata mismatches" "10" "$gendata_count"

assert_not_contains "Mismatched step_150 parquet is not reused" "Step 150: ReuseData=step_150.parquet" "$OUTPUT"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: --progdistill --GRPO mode
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: --progdistill --GRPO mode ==="
setup_workspace
OUTPUT=$(run_pipeline --progdistill teacher-exp-name --GRPO)

total=$(count_sbatch_calls)
# 10 gendata + 10 sft + 1 grpo + 3 eval + 1 plot = 25
assert_eq "Total sbatch calls for --progdistill --GRPO (25)" "25" "$total"

gendata_count=$(grep -c "generate_sft_data.sh" "$SBATCH_LOG" || true)
assert_eq "10 generate_sft_data.sh calls" "10" "$gendata_count"

sft_count=$(grep -c "train_sft.sh" "$SBATCH_LOG" || true)
assert_eq "10 train_sft.sh calls" "10" "$sft_count"

grpo_count=$(grep -c "train_grpo.sh" "$SBATCH_LOG" || true)
assert_eq "Exactly 1 train_grpo.sh call" "1" "$grpo_count"

eval_count=$(grep -c "eval.sh" "$SBATCH_LOG" || true)
assert_eq "Exactly 3 eval.sh calls" "3" "$eval_count"

# Eval jobs use the default array, NOT --array=1-1
eval_array1=$(grep "eval.sh" "$SBATCH_LOG" | grep -c "array=1-1" || true)
assert_eq "No eval job uses --array=1-1 in GRPO mode" "0" "$eval_array1"

plot_count=$(grep -c "plot_results.sh" "$SBATCH_LOG" || true)
assert_eq "Exactly 1 plot_results.sh call" "1" "$plot_count"

teardown_workspace

# ---------------------------------------------------------------------------
# Test: unknown argument causes non-zero exit
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: unknown argument causes error ==="
setup_workspace
set +e
run_pipeline --unknown-flag 2>/dev/null
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -ne 0 ]; then
    ok "Unknown arg causes non-zero exit"
else
    fail "Unknown arg should have caused non-zero exit"
fi
teardown_workspace

# ---------------------------------------------------------------------------
# Test: MODEL_PATH override in train_grpo.sh
# ---------------------------------------------------------------------------
echo ""
echo "=== Test: MODEL_PATH override in train_grpo.sh ==="
override_count=$(grep -c 'MODEL_PATH:-' scripts/train_grpo.sh || true)
assert_eq "MODEL_PATH override present in train_grpo.sh" "1" "$override_count"

full_pattern=$(grep 'actor_rollout_ref.model.path' scripts/train_grpo.sh)
assert_contains "MODEL_PATH has fallback to MODEL_DIR/model_name" \
    'MODEL_PATH:-' "$full_pattern"
assert_contains "Nested fallback to MODEL_DIR" \
    'MODEL_DIR:-' "$full_pattern"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"
[ "$FAIL" -eq 0 ]
