#!/bin/bash
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH -t 01:00:00
# Set SBATCH_ACCOUNT and SBATCH_PARTITION in your shell env (e.g. ~/.rl_skill_comp_env sourced from ~/.bashrc)

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate verl

[ -f ~/.rl_skill_comp_env ] && source ~/.rl_skill_comp_env
export PYTHONPATH=${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}

project_dir=${PROJECT_DIR:-$PWD}
cd ${project_dir}

echo "============================================================"
echo "DATA GENERATION"
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "Project dir: ${project_dir}"
echo "============================================================"

echo "Step 1: Generating annotated_expressions.json..."
python3 annotate_expressions.py
echo "Done."

echo "Step 2: Generating puzzles..."
echo "  n=3 (18 patterns, 4000 samples each)..."
for i in {0..17};   do python3 generate_puzzles.py --puzzle_size 3 --pattern_index ${i} --num_data 4000; done

echo "  n=4 (96 patterns, 4000 samples each)..."
for i in {0..95};   do python3 generate_puzzles.py --puzzle_size 4 --pattern_index ${i} --num_data 4000; done

echo "  n=5 (558 patterns, 10 samples each)..."
for i in {0..557};  do python3 generate_puzzles.py --puzzle_size 5 --pattern_index ${i} --num_data 10; done

echo "  n=6 (4328 patterns, 1 sample each)..."
for i in {0..4327}; do python3 generate_puzzles.py --puzzle_size 6 --pattern_index ${i} --num_data 1; done
echo "Done."

echo "Step 3: Preprocessing into parquets..."
python3 preprocess_balanced.py
echo "Done."

echo "============================================================"
echo "Data generation complete: $(date)"
echo "============================================================"
