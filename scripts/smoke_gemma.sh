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
# Node-local triton cache (see scripts/train_grpo.sh for why).
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-$$}"
mkdir -p "${TRITON_CACHE_DIR}"

# vLLM's torch.compile cache defaults to ~/.cache/vllm (NFS home) and is keyed
# only by the model graph hash, so two concurrent jobs compiling the SAME model
# write into the SAME directory. One job's rename into the cache invalidates the
# other's open handle and it dies at engine init with
#   OSError: [Errno 116] Stale file handle
# in triton CompiledKernel.__init__ -> BackendCompilerFailed. Observed 2026-09-22
# on job 47811579 (third of three Qwen3-0.6B GRPO jobs launched together; the two
# 0.6B jobs shared graph hash 23afc4795d). This is also the likely root cause of
# the historical "JSONDecodeError: Extra data" in the vLLM compile cache that was
# worked around with --enforce_eager. Keep it node-local and per-job, like
# TRITON_CACHE_DIR above; the cost is one ~1-2 min recompile per job.
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${VLLM_CACHE_ROOT}"

cd ${PROJECT_DIR:-/n/home06/mwalden/RL-skill-comp}
python scripts/smoke_gemma.py
