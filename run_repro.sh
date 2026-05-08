#!/bin/bash
#SBATCH --job-name=gemma_repro
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bingbin_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -t 00:20:00
#SBATCH --output=/n/home06/sdholakia/RL-skill-comp/logs/gemma_repro-%j.out

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

cd /n/home06/sdholakia/RL-skill-comp
python3 /n/home06/sdholakia/RL-skill-comp/repro_v2.py
