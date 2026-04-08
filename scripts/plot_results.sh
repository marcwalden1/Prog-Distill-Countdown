#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --account=kdbrantley_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A.out
#SBATCH -t 01:00:00

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=/n/home06/mwalden/.local/lib/python3.10/site-packages:${PYTHONPATH}

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}/results}
log_dir=${LOG_DIR:-${project_dir}/logs}
figures_dir=${FIGURES_DIR:-${project_dir}/figures}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}

python3 ${project_dir}/plot_results.py \
    --model_name  ${model_name} \
    --exp_name    ${exp_name} \
    --result_dir  ${result_dir} \
    --log_dir     ${log_dir} \
    --figures_dir ${figures_dir}
