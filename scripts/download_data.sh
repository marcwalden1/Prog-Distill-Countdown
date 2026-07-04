#!/bin/bash
# scripts/download_data.sh
#
# Fetch the Countdown train/test parquet data into ./data. This data is NOT
# tracked in git (the train split alone is ~282 MB), so a fresh clone needs it
# before any training/eval can run.
#
# Pulls the exact splits used in the paper from the HuggingFace dataset
# marcwalden/rl-skill-comp-evals:
#   data/balanced/{train,test}.parquet   (n=3,4 train + held-out test)
#   data/balanced5/test.parquet          (n=5 OOD test)
#   data/balanced6/test.parquet          (n=6 OOD test)
#
# Usage (run from anywhere; writes to the repo root):
#   bash scripts/download_data.sh
#
# Requires the huggingface_hub CLI — it ships with the `verl` conda env
# (scripts/setup_verl.sh). Base MODELS are downloaded separately; see the
# "Quickstart" section of README.md (gemma-3-270m is gated on HuggingFace).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_REPO="marcwalden/rl-skill-comp-evals"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found on PATH." >&2
    echo "       Activate the training env first: conda activate verl" >&2
    echo "       (or set it up with scripts/setup_verl.sh)." >&2
    exit 1
fi

echo "==> Downloading Countdown data from ${DATA_REPO} into ${REPO_ROOT}/data ..."
# Only the parquet splits — the dataset's data/ also holds ~5000 raw
# countdown_*_pattern_*.json puzzle files we don't need; pulling them all
# trips HuggingFace's per-5-min request rate limit (HTTP 429).
huggingface-cli download "${DATA_REPO}" --repo-type dataset \
    --include "data/balanced/*.parquet" \
              "data/balanced5/*.parquet" \
              "data/balanced6/*.parquet" \
    --local-dir "${REPO_ROOT}"

echo "==> Verifying expected splits are present:"
missing=0
for f in data/balanced/train.parquet data/balanced/test.parquet \
         data/balanced5/test.parquet data/balanced6/test.parquet; do
    if [ -f "$f" ]; then
        echo "    OK  $f"
    else
        echo "    MISSING  $f" >&2
        missing=1
    fi
done

if [ "$missing" -ne 0 ]; then
    echo "ERROR: one or more expected parquet files did not download." >&2
    exit 1
fi

echo "Done. Data is ready under ${REPO_ROOT}/data."
