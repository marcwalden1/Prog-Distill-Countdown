#!/bin/bash
# scripts/download_gemma_students.sh
#
# Download Shlok's Gemma-3-270m distill + progdistill student checkpoints from
# Hugging Face into a local models directory. After this completes you can pass
# the resulting per-subfolder paths as MODEL_PATH to scripts/train_grpo.sh.
#
# Usage:
#   bash scripts/download_gemma_students.sh                # uses $MODEL_DIR or $HOME/models
#   bash scripts/download_gemma_students.sh /path/to/dir   # explicit target dir
#
# What lands on disk:
#   <target_dir>/gemma-3-270m-distill-sftlr1e-4-seed1/
#   <target_dir>/gemma-3-270m-progdistill-sftlr1e-4-seed1-round_1600/
#       └── model.safetensors + config.json + tokenizer.* + ...
#
# Source repo: https://huggingface.co/shlokdho/qwen2.5-1.5b-countdown-teacher
# (Yes, the repo name says qwen2.5-1.5b — it's a multi-checkpoint repo whose
# subfolders include both the Qwen teacher and the Gemma distill students.)

set -euo pipefail

target_dir="${1:-${MODEL_DIR:-${HOME}/models}}"
mkdir -p "${target_dir}"

REPO="shlokdho/qwen2.5-1.5b-countdown-teacher"
SUBFOLDERS=(
    "gemma-3-270m-distill-sftlr1e-4-seed1"
    "gemma-3-270m-progdistill-sftlr1e-4-seed1-round_1600"
)

echo "Target: ${target_dir}"
echo "Repo:   ${REPO}"
echo

for sub in "${SUBFOLDERS[@]}"; do
    dest="${target_dir}/${sub}"
    if [ -f "${dest}/model.safetensors" ] && [ -f "${dest}/config.json" ]; then
        echo "[skip]     ${sub} (already present at ${dest})"
        continue
    fi
    echo "[download] ${sub}"
    python3 - "$REPO" "$sub" "$target_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, sub, target_dir = sys.argv[1:]
# allow_patterns scopes the download to just this subfolder; the resulting
# files land at <target_dir>/<sub>/... preserving the HF layout.
snapshot_download(
    repo_id=repo,
    allow_patterns=[f"{sub}/*"],
    local_dir=target_dir,
)
PY
done

echo
echo "Done. Available student checkpoints:"
for sub in "${SUBFOLDERS[@]}"; do
    if [ -d "${target_dir}/${sub}" ]; then
        size=$(du -sh "${target_dir}/${sub}" 2>/dev/null | cut -f1)
        echo "  ${target_dir}/${sub}   (${size})"
    fi
done
echo
echo "Pass either path as MODEL_PATH when launching scripts/train_grpo.sh."
