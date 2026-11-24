#!/bin/bash
# Submit base-model pass@k evals for the shortest-path task: one scripts/eval_base.sh
# job per (model, dataset). Results land in
#   results/baselines/<model>/max${MAX_TOKENS}/<dataset>_temp0.6_n${NUM_SAMPLES}_max${MAX_TOKENS}.json
# and are scored with tasks/shortest_path/score_passk.py.
#
#   MODELS="Qwen3-0.6B-Base" DATASETS="spath-simple" bash tasks/shortest_path/submit_base_eval.sh   # pilot
#   bash tasks/shortest_path/submit_base_eval.sh                                                     # full
#
# Prints the submitted job ids (space-separated) on the last line, for triage_jobs.py.
set -euo pipefail

MODELS=${MODELS:-"Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base"}
DATASETS=${DATASETS:-"spath-simple spath5-simple spath6-simple"}
export NUM_SAMPLES=${NUM_SAMPLES:-128}
export MAX_TOKENS=${MAX_TOKENS:-512}
export TEMPERATURE=${TEMPERATURE:-0.6}
ACCOUNT=${ACCOUNT:-kempner_bingbin_lab}

jids=()
for m in $MODELS; do
    # Walltime: 4B is ~3x slower per token than 0.6B; spath-simple has 900 prompts vs 300.
    case "$m" in *4B*) t=04:00:00 ;; *) t=02:00:00 ;; esac
    for ds in $DATASETS; do
        jid=$(MODEL_NAME=$m EVAL_DATASET=$ds sbatch --parsable --account=$ACCOUNT -t $t \
              --job-name=spath-eval-$m-$ds scripts/eval_base.sh)
        echo "submitted $jid  $m  $ds  (-t $t)"
        jids+=("$jid")
    done
done
echo "${jids[*]}"
