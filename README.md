## How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown

This repository is adapted from the official implementation accompanying the paper "How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown" by Park et al. The codebase is adapted from the authors' original implementation and includes additional experiments, analyses, and modifications developed in this repository.


## Quick Links

- [How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown](#RL-skill-comp)
- [Quick Links](#quick-links)
- [Quickstart](#quickstart)
- [Regenerating data from scratch](#regenerating-data-from-scratch-optional)


## Quickstart

A fresh clone does **not** ship with the patched `verl` checkout, the data, or
the base models. The scripts below set all three up. Run from the repo root on a
SLURM cluster with H100/A100 GPUs. (The [Experiments](#experiments) section below
documents the from-scratch equivalents of these steps.)

```Shell
# 1. Build the `verl` conda env + clone verl at the pinned commit.
#    ~20-40 min: compiles flash-attn. Run once on a login node.
bash scripts/setup_verl.sh
conda activate verl

# 2. Apply the REQUIRED local verl + vLLM patches (idempotent; safe to re-run).
#    Without these, gemma-3-270m GRPO produces token salad and progressive-
#    distillation SFT silently trains at LR~=0. See CLAUDE.md for details.
bash scripts/apply_verl_patches.sh

# 3. Download the Countdown train/test parquet data into ./data.
bash scripts/download_data.sh

# 4. Base models -> your models dir.
export MODEL_DIR=/path/to/your/models
huggingface-cli download Qwen/Qwen2.5-0.5B  --local-dir "$MODEL_DIR/Qwen2.5-0.5B"
#    gemma-3-270m is GATED: accept the license at
#    https://huggingface.co/google/gemma-3-270m and run `huggingface-cli login` first.
huggingface-cli download google/gemma-3-270m --local-dir "$MODEL_DIR/gemma-3-270m"
```

### Running experiments

`scripts/run_pipeline.sh` chains train -> eval -> plot as one SLURM dependency
job. It has built-in path/account defaults for the original authors' accounts;
**any other user** supplies them via environment variables:

```Shell
export MODEL_DIR=/path/to/your/models
export CHECKPOINT_DIR=/path/to/your/checkpoints
export ACCOUNT=your_slurm_account          # e.g. kempner_bingbin_lab
# optional overrides: TRAIN_PARTITION / EVAL_PARTITION / PLOT_PARTITION
#                     (default kempner_h100), EVAL_ACCOUNT / PLOT_ACCOUNT

MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-grpo-seed1 bash scripts/run_pipeline.sh
```

See `CLAUDE.md` for the full set of knobs, the distillation modes, and a
description of each required patch.


## Regenerating data from scratch (optional)

`scripts/download_data.sh` fetches the exact train/test splits used in the paper.
To regenerate the raw Countdown puzzles from scratch instead:

```Shell
for i in {0..17}
do
python generate_puzzles.py --puzzle_size 3 --pattern_index ${i} --num_data 4000
done

for i in {0..95}
do
python generate_puzzles.py --puzzle_size 4 --pattern_index ${i} --num_data 4000
done

for i in {0..557}
do
python generate_puzzles.py --puzzle_size 5 --pattern_index ${i} --num_data 10
done

for i in {0..4327}
do
python generate_puzzles.py --puzzle_size 6 --pattern_index ${i} --num_data 1
done

python preprocess_balanced.py
```


