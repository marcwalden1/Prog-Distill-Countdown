#!/usr/bin/env python3
"""Write a seeded uniform-random subsample of an eval test set as its own dataset dir.

    python3 scripts/make_eval_subsample.py --source balanced5-simple --n 1000
    # -> data/balanced5-simple-r1000/test.parquet + subsample.json

Why a separate dataset dir rather than a flag on eval.py: every downstream path
(eval.py's result filename, plot_results.py's `{ds}_temp0.6_n32_*.json` glob, the
.scores caches) is keyed on the dataset name, so a distinct name keeps subsample
results from ever being merged with full-set results.

Why random and not a head slice: balanced{5,6}*/test.parquet are sorted by
canonical_pattern_index, and low-index patterns are easier, so data[:1000] is a
biased subsample (see research_logs/2026-07-13_fix-maxprompts-cap-fullset-regen.md).

Rows are drawn uniformly without replacement and written in their original order.
The source row indices, seed, and source file are recorded in subsample.json so the
subset is reproducible and auditable. Re-running with the same arguments is a no-op;
a conflicting existing subsample (different seed/n/source) is an error.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset dir under data/, e.g. balanced5-simple")
    ap.add_argument("--n", type=int, default=1000, help="rows to sample (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dataset dir name (default <source>-r<n>; the tag is always the "
                         "LAST segment so eval.py's prompt-variant/few-shot name rules still apply)")
    args = ap.parse_args()

    src_path = os.path.join(REPO, "data", args.source, "test.parquet")
    out_name = args.out or f"{args.source}-r{args.n}"
    out_dir = os.path.join(REPO, "data", out_name)
    out_path = os.path.join(out_dir, "test.parquet")
    meta_path = os.path.join(out_dir, "subsample.json")

    df = pd.read_parquet(src_path)
    if args.n >= len(df):
        sys.exit(f"ERROR: --n {args.n} >= {len(df)} rows in {src_path}; nothing to subsample.")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(df), size=args.n, replace=False))
    meta = {
        "source": f"data/{args.source}/test.parquet",
        "source_rows": len(df),
        "source_sha256": sha256(src_path),
        "n": args.n,
        "seed": args.seed,
        "sampler": "numpy.random.default_rng(seed).choice(source_rows, n, replace=False), sorted",
        "source_row_indices": idx.tolist(),
    }

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if old == meta and os.path.exists(out_path):
            print(f"Up to date: {out_path}")
            return
        sys.exit(f"ERROR: {out_dir} already holds a different subsample "
                 f"(seed={old.get('seed')}, n={old.get('n')}, source={old.get('source')}). "
                 f"Results under that dataset name would no longer match; pass --out to pick a new name.")

    os.makedirs(out_dir, exist_ok=True)
    df.iloc[idx].reset_index(drop=True).to_parquet(out_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    sub = df.iloc[idx]
    pat = sub["extra_info"].apply(lambda e: e["canonical_pattern_index"])
    all_pat = df["extra_info"].apply(lambda e: e["canonical_pattern_index"])
    print(f"Wrote {out_path}: {args.n}/{len(df)} rows, seed {args.seed}, "
          f"{pat.nunique()}/{all_pat.nunique()} patterns covered, "
          f"pattern-index mean {pat.mean():.1f} (full set {all_pat.mean():.1f})")


if __name__ == "__main__":
    main()
