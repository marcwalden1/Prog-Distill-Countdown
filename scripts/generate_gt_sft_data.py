#!/usr/bin/env python3
"""Build a ground-truth SFT parquet from the Countdown puzzle data.

Unlike scripts/generate_sft_data.py (which samples a *teacher model*), this needs
no GPU and no model: the canonical solution is already in the data. Each row's
equation is reconstructed by substituting `extra_info.numbers` in order into
`extra_info.canonical_pattern` — the same `build_gt_equation` the GT-NTP
diagnosis uses — and verified against `grader_utils.compute_score`.

Output schema matches generate_sft_data.py, i.e. what verl's SFTDataset expects
via train_sft.sh's `data.prompt_key=prompt data.response_key=response`:

    prompt   : str  — the literal parquet prompt (template_type as-built)
    response : str  — the continuation, e.g. " <answer> (53 - 22) * 3 </answer>"

Note on concatenation: with no chat template, SFTDataset uses the prompt string
verbatim and appends `tokenizer.eos_token` to the response, joining the two with
NO separator. The response therefore carries its own leading whitespace so it
continues the prompt the same way a generation would.

Usage:
    python3 scripts/generate_gt_sft_data.py --data-source balanced-simple \
        --output <path>.parquet [--format answer|think_empty|verify|search]
        [--search-attempts LO-HI] [--distractor-strategy mixed|near|random]
        [--max-rows N]

Response formats live in sft_formats.py (repo root), shared with eval.py's
few-shot exemplars so a k-shot base eval measures the format an SFT run would
install.
"""
import argparse
import contextlib
import io
import os
import random
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader_utils import compute_score  # noqa: E402
from sft_formats import FORMATS, DISTRACTOR_STRATEGIES, build_gt_equation, render_response  # noqa: E402


def _render_row(job):
    """(equation, response) for one row. Top-level so it is picklable by Pool."""
    fmt, prompt, nums, target, pattern, seed, n_attempts, strategy = job
    eq = build_gt_equation(pattern, nums)
    return eq, render_response(fmt, prompt, nums, target, pattern, equation=eq,
                               rng=random.Random(seed), n_attempts=n_attempts,
                               strategy=strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="balanced-simple",
                    help="directory under data/ holding train.parquet")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="output .parquet path")
    ap.add_argument("--format", default="answer", choices=list(FORMATS),
                    help="response format; see sft_formats.py. 'search' builds a "
                         "trace of rejected legal expressions before the answer.")
    ap.add_argument("--search-attempts", default="1-3",
                    help="--format search: number of rejected attempts per row, "
                         "'K' or an inclusive range 'LO-HI' sampled per row.")
    ap.add_argument("--distractor-strategy", default="mixed",
                    choices=list(DISTRACTOR_STRATEGIES),
                    help="--format search: how rejected attempts are picked.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = all rows. Subsamples (seeded) when smaller than the split.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify-n", type=int, default=2000,
                    help="rows to score with compute_score as a correctness check "
                         "(0 = skip, -1 = all). Any failure aborts.")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes used to render responses. --format search costs "
                         "~16 ms/row (hundreds of expression evaluations), so a full "
                         "418k-row build is ~2 h single-threaded; the other formats are "
                         "pure string ops and ignore this. Per-row RNG is seeded by row "
                         "index, so the output does not depend on the worker count.")
    ap.add_argument("--project-dir", default=os.path.join(os.path.dirname(__file__), ".."))
    args = ap.parse_args()

    src = os.path.join(args.project_dir, "data", args.data_source, f"{args.split}.parquet")
    df = pd.read_parquet(src)
    print(f"Read {len(df):,} rows from {src}")

    if args.max_rows and args.max_rows < len(df):
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows (seed {args.seed})")

    lo, _, hi = args.search_attempts.partition("-")
    n_attempts = (int(lo), int(hi)) if hi else int(lo)

    jobs = []
    for i, (_, row) in enumerate(df.iterrows()):
        ei = row["extra_info"]
        jobs.append((args.format, str(row["prompt"][0]["content"]),
                     [int(n) for n in ei["numbers"]],
                     int(row["reward_model"]["ground_truth"]),
                     ei["canonical_pattern"],
                     # Per-row seed: reproducible, independent of --max-rows and
                     # of --workers.
                     args.seed * 1_000_003 + i,
                     n_attempts, args.distractor_strategy))

    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1000:
        with Pool(args.workers) as pool:
            rendered = pool.map(_render_row, jobs, chunksize=256)
    else:
        rendered = [_render_row(j) for j in jobs]
    print(f"Rendered {len(rendered):,} responses in {time.time() - t0:.0f}s "
          f"({args.workers} workers)")

    prompts = [j[1] for j in jobs]
    equations = [r[0] for r in rendered]
    responses = [r[1] for r in rendered]

    # Correctness gate. compute_score is chatty on success, so swallow its stdout.
    if args.verify_n != 0:
        n = len(df) if args.verify_n < 0 else min(args.verify_n, len(df))
        idx = df.sample(n=n, random_state=0).index if n < len(df) else df.index
        bad = []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in idx:
                s = compute_score("countdown", responses[i],
                                  df.loc[i, "reward_model"]["ground_truth"],
                                  extra_info=dict(df.loc[i, "extra_info"]))
                if s != 1.0:
                    bad.append((i, equations[i], s))
        if bad:
            print(f"ERROR: {len(bad)}/{n} GT responses did not score 1.0; first few:",
                  file=sys.stderr)
            for b in bad[:5]:
                print(f"  row={b[0]} eq={b[1]!r} score={b[2]}", file=sys.stderr)
            sys.exit(1)
        print(f"Verified {n:,}/{n:,} GT responses score exactly 1.0")

    out = pd.DataFrame({"prompt": prompts, "response": responses})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out):,} rows -> {args.output}")
    print("\nExample row:")
    print(f"  prompt   (tail): ...{out.iloc[0]['prompt'][-70:]!r}")
    print(f"  response       : {out.iloc[0]['response']!r}")


if __name__ == "__main__":
    main()
