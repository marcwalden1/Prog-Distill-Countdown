#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 01:30:00
#SBATCH --array 1-32

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=/n/home06/mwalden/.local/lib/python3.10/site-packages:${PYTHONPATH}

project_dir=${PROJECT_DIR:-$PWD}
checkpoint_dir=${CHECKPOINT_DIR:-/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/rl-checkpoints}
result_dir=${RESULT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
eval_dataset=${EVAL_DATASET:-balanced}
extra_args=${EXTRA_ARGS:-""}

checkpoint_path=${checkpoint_dir}/checkpoints/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))
result_path=${result_dir}/results/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))

if [ -f "${checkpoint_path}/model.safetensors" ]; then
    echo "Merged checkpoint already present at ${checkpoint_path}"
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path}/actor/ --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/actor/
    rm -rf ${checkpoint_path}/critic/
    rm -rf ${checkpoint_path}/data.pt
elif python3 -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path} --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/*.pt
fi

python3 eval.py --checkpoint_path ${checkpoint_path} --result_path ${result_path} --eval_dataset ${eval_dataset} ${extra_args}

chmod -R 770 ${checkpoint_dir}/checkpoints/${model_name}/${exp_name}
chmod -R 770 ${result_dir}/results/${model_name}/${exp_name}
