#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --output=logs/%x-%j.out
#SBATCH -t 00:15:00

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

cd ${PROJECT_DIR:-/n/home06/mwalden/RL-skill-comp}
python scripts/smoke_gemma.py
