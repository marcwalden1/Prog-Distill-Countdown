#!/usr/bin/env python3
"""
Rebuild the balanced/balanced5/balanced6 parquets with a different prompt
template, reusing the puzzles already baked into data/balanced*/. The raw
countdown_*_pattern_*.json files are not on this cluster, and rebuilding from
the existing parquets guarantees the puzzle sets and train/test splits are
byte-identical to the canonical data — only the prompt string changes.

Output goes to data/balanced<suffix>{,5,6}/ so the canonical dirs are never
touched; point DATA_SOURCE=balanced<suffix> at it (run_pipeline.sh derives the
matching eval-set names).

Usage:
    python3 scripts/make_prompt_variant_data.py \
        --template_type base-simple --suffix -simple
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_balanced import make_prefix

SPLITS = {
    "balanced": ["train", "test"],
    "balanced5": ["test"],
    "balanced6": ["test"],
}


def rewrite_prompt(table, template_type):
    """Replace the `prompt` column, re-rendered from the raw fields the parquet carries.

    Uses pyarrow rather than `datasets.Dataset.from_parquet`, which hangs in
    uninterruptible I/O acquiring its cache filelock under ~/.cache/huggingface on
    this cluster's NFS home. Passing the column's own type to pa.array keeps the
    schema (and the struct field order inside `prompt`) byte-compatible.
    """
    # int()/list-of-int so the f-string in make_prefix renders identically
    # to the original JSON-sourced values (numpy types would print e.g.
    # "[73 19 16]" or "np.int64(76)").
    nums = table.column("extra_info").combine_chunks().field("numbers").to_pylist()
    targets = table.column("reward_model").combine_chunks().field("ground_truth").to_pylist()
    prompts = [
        [{"role": "user", "content": make_prefix({"nums": [int(x) for x in n], "target": int(t)},
                                                 template_type)}]
        for n, t in zip(nums, targets)
    ]
    idx = table.schema.get_field_index("prompt")
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(prompts, type=field.type))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_type", required=True,
                    help="template_type branch in preprocess_balanced.make_prefix")
    ap.add_argument("--suffix", required=True,
                    help="appended to each output dir name, e.g. '-simple' -> "
                         "data/balanced-simple, data/balanced5-simple, ...")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output parquets")
    args = ap.parse_args()
    if not args.suffix:
        ap.error("--suffix must be non-empty: refusing to overwrite the canonical dirs")

    for base, splits in SPLITS.items():
        src_dir = os.path.join(args.data_dir, base)
        out_dir = os.path.join(args.data_dir, f"{base}{args.suffix}")
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            src = os.path.join(src_dir, f"{split}.parquet")
            out = os.path.join(out_dir, f"{split}.parquet")
            if os.path.exists(out) and not args.force:
                print(f"[skip] {out} exists (use --force to overwrite)")
                continue
            table = rewrite_prompt(pq.read_table(src), args.template_type)
            pq.write_table(table, out)
            print(f"[ok] {out}  ({table.num_rows} rows)")
            example = table.column("prompt")[0][0]["content"]
            print(f"     example prompt:\n{example}"[:500])


if __name__ == "__main__":
    main()
