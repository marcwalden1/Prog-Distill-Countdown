#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --output=logs/%x-%A.out
#SBATCH -t 01:30:00

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=/n/home06/mwalden/.local/lib/python3.10/site-packages:${PYTHONPATH}

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}}
model_root=${MODEL_ROOT:-/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
eval_dataset=${EVAL_DATASET:-balanced}
temperature=${TEMPERATURE:-0.6}
num_samples=${NUM_SAMPLES:-32}
max_tokens=${MAX_TOKENS:-1024}
extra_args=${EXTRA_ARGS:-""}

checkpoint_path=${model_root}/${model_name}
base_result_dir=${result_dir}/results/baselines/${model_name}

python3 eval.py \
    --checkpoint_path ${checkpoint_path} \
    --result_path ${base_result_dir} \
    --eval_dataset ${eval_dataset} \
    --temperature ${temperature} \
    --n ${num_samples} \
    --max_tokens ${max_tokens} \
    ${extra_args}

chmod -R 770 ${base_result_dir}
