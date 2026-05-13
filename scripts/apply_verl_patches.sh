#!/bin/bash
# scripts/apply_verl_patches.sh
#
# Apply the local verl + vLLM patches required for gemma-3-270m training on a
# fresh checkout. Idempotent: skips patches that are already applied. Run from
# the RL-skill-comp repo root after a fresh `git clone`.
#
# What it does:
#   1. Checks the inner `verl/` checkout exists; pins it to commit 083da9ab
#      (the version we develop against) if currently on a different commit.
#   2. Applies patches/local-fixes-vs-083da9ab.patch via `git apply` —
#      this is the combined diff of:
#        - fsdp_sft_trainer.py (SFT scheduler horizon + seed wiring)
#        - fsdp_utils.py        (FSDP wrap-policy skip-missing-classes)
#        - rl_dataset.py        (chat_template fallback, 2 sites)
#        - sft_dataset.py       (chat_template fallback)
#        - fsdp_workers.py      (gemma3 generation_config.eos/pad None fallback)
#        - fsdp_vllm.py         (gemma3 normalizer-buffer restore after dummy load)
#   3. Edits the installed vLLM `gemma3.py` to mark the normalizer buffer
#      as non-persistent (`persistent=False`). This is the canonical fix
#      for the buffer-clobber bug under load_format=dummy_dtensor.
#
# Usage:
#   bash scripts/apply_verl_patches.sh
#
# Notes:
#   - Run inside the conda env you'll train under (so we patch the right vLLM).
#   - If the verl tree has uncommitted/conflicting edits, `git apply` will
#     refuse — clean the tree first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VERL_PIN="083da9ab130efa2dc284eeb821a3edd6ce570fe3"
VERL_PATCH="${REPO_ROOT}/patches/local-fixes-vs-083da9ab.patch"

# ---------------------------------------------------------------------------
# Step 1: verl
# ---------------------------------------------------------------------------
if [ ! -d "${REPO_ROOT}/verl/.git" ]; then
    echo "ERROR: ${REPO_ROOT}/verl is not a git checkout." >&2
    echo "       Clone with: git clone https://github.com/volcengine/verl.git verl" >&2
    exit 1
fi

pushd "${REPO_ROOT}/verl" >/dev/null

current_commit=$(git rev-parse HEAD)
if ! git merge-base --is-ancestor "${VERL_PIN}" HEAD 2>/dev/null; then
    echo "verl: pin commit ${VERL_PIN:0:8} not reachable from HEAD (currently ${current_commit:0:8})."
    echo "verl: fetching and checking out the pin."
    git fetch --quiet origin "${VERL_PIN}" 2>/dev/null || git fetch --quiet origin
    git checkout --quiet "${VERL_PIN}"
fi

# Idempotency: `git apply --check` returns 0 if the patch can be applied
# cleanly to the current tree, non-zero otherwise (already applied OR conflicts).
if git apply --check "${VERL_PATCH}" 2>/dev/null; then
    echo "verl: applying patches/local-fixes-vs-083da9ab.patch"
    git apply "${VERL_PATCH}"
elif git apply --reverse --check "${VERL_PATCH}" 2>/dev/null; then
    echo "verl: patches already applied (reverse-apply check passed). Skipping."
else
    echo "ERROR: verl tree has conflicting changes; cannot apply the patch cleanly." >&2
    echo "       Inspect with: cd verl && git status && git diff" >&2
    exit 1
fi
popd >/dev/null

# ---------------------------------------------------------------------------
# Step 2: vLLM gemma3.py — normalizer buffer must be non-persistent
# ---------------------------------------------------------------------------
vllm_dir=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)
if [ -z "${vllm_dir}" ]; then
    echo "WARNING: vLLM not importable in this env. Skipping gemma3.py patch." >&2
    echo "         Re-run this script inside your training conda env." >&2
    exit 0
fi

gemma3_file="${vllm_dir}/model_executor/models/gemma3.py"
if [ ! -f "${gemma3_file}" ]; then
    echo "WARNING: ${gemma3_file} not found. Your vLLM may not include gemma3 (skip)." >&2
    exit 0
fi

if grep -q 'register_buffer("normalizer", torch.tensor(normalizer), persistent=False)' "${gemma3_file}"; then
    echo "vLLM: gemma3.py normalizer buffer already non-persistent. Skipping."
else
    # Match the unpatched line and rewrite it in place. Use a literal-string
    # sed to avoid regex escaping; back up to .bak so the user can diff/restore.
    sed -i.bak \
        's|register_buffer("normalizer", torch.tensor(normalizer))|register_buffer("normalizer", torch.tensor(normalizer), persistent=False)|' \
        "${gemma3_file}"
    if grep -q 'register_buffer("normalizer", torch.tensor(normalizer), persistent=False)' "${gemma3_file}"; then
        echo "vLLM: patched ${gemma3_file}"
        echo "      (backup at ${gemma3_file}.bak)"
    else
        echo "ERROR: sed did not produce the expected change in ${gemma3_file}." >&2
        echo "       Check the file manually — line ~372 should have persistent=False." >&2
        mv "${gemma3_file}.bak" "${gemma3_file}"
        exit 1
    fi
fi

echo
echo "All patches applied. Verify with:"
echo "  cd verl && git diff --stat   # expect 6 files modified"
echo "  grep -n 'persistent=False' ${gemma3_file} | head"
