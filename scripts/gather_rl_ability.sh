#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A.out
#SBATCH -t 01:00:00

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=/n/home06/mwalden/.local/lib/python3.10/site-packages:${PYTHONPATH}

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}}
checkpoint_dir=${CHECKPOINT_DIR:-/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/rl-checkpoints}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
eval_dataset=${EVAL_DATASET:-balanced}
temperature=${TEMPERATURE:-0.6}
num_samples=${NUM_SAMPLES:-32}
max_tokens=${MAX_TOKENS:-1024}

baseline_result_path=${BASELINE_RESULT_PATH:-${result_dir}/results/baselines/${model_name}/${eval_dataset}_temp${temperature}_n${num_samples}_max${max_tokens}.json}
checkpoint_root=${checkpoint_dir}/checkpoints/${model_name}/${exp_name}
result_root=${result_dir}/results/${model_name}/${exp_name}

python3 gather_experiment.py --model_name ${model_name} --exp_name ${exp_name} --result_dir ${result_dir}/results --log_dir ${project_dir}/logs --eval_dataset ${eval_dataset}

python3 summarize_rl_ability.py \
    --model_name ${model_name} \
    --exp_name ${exp_name} \
    --baseline_result_path ${baseline_result_path} \
    --result_root ${result_root} \
    --checkpoint_root ${checkpoint_root} \
    --train_job_id ${TRAIN_JOB_ID} \
    --eval_job_id ${EVAL_JOB_ID} \
    --analyze_job_id ${ANALYZE_JOB_ID}
