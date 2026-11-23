# `data/balanced234` — the `balanced` train set plus operation-balanced n=2 puzzles

**Date:** 2026-09-10
**Builder:** `preprocess_balanced.py --mode balanced234`
**Use:** `DATA_SOURCE=balanced234`

## What it is

| | rows |
|---|---:|
| n=3 (from `balanced`, verbatim) | 64,112 |
| n=4 (from `balanced`, verbatim) | 354,024 |
| n=2 (new) | 964 |
| **train total** | **419,100** |
| test (copied unchanged from `balanced`) | 997 |

The n=3/4 rows are copied byte-identically out of `data/balanced/train.parquet`
(verified with `pyarrow.Table.equals` on the 418,136-row prefix), and the test
parquet is a byte copy, so any `balanced234` run is comparable to an existing
`balanced` run with the n=2 rows as the only difference.

The parquet is the source rather than a regeneration for two reasons: the raw
`data/countdown_{3,4}_pattern_*.json` files are no longer on disk, and a
round-trip through them could not reproduce the current train set anyway. The
JSONs are **not** faithfully reconstructable from the parquets: the test dedup
dropped 143 of 1140 rows and the leakage filter dropped train rows, and neither
set survives in either parquet. (What *is* recoverable is the puzzle content —
every parquet carries `nums`/`target`/`puzzle_size`/`canonical_pattern{,_index}`
at top level next to the rendered prompt, so a new prompt variant can be
re-rendered from an existing parquet without the JSONs; verified 200/200 against
`balanced` and `balanced-simple`.)

The n=2 rows are appended at the end. Order is irrelevant: verl's
`data.shuffle` defaults to `True`, so `create_rl_sampler` builds a
`RandomSampler` over the whole train set.

Written by pyarrow rather than `datasets.to_parquet`, so the file is 24 MB
where the source (with 964 fewer rows) is 295 MB. That is purely an encoding
difference — larger row groups let the near-identical prompt strings
dictionary-compress far better — and the content is verified identical.
`datasets.load_dataset` is deliberately avoided: it hangs acquiring its cache
filelock under `~/.cache/huggingface/datasets` on this cluster's NFS home.

## Operation balance for n=2, and its hard ceiling

`annotated_expressions.json["2"]` has exactly four canonical patterns, one per
operation — index 0 `A+B`, 1 `A-B`, 2 `A*B`, 3 `A/B`. Since
`generate_puzzles.py` samples one pattern per invocation, equal `--num_data`
across the four indices gives exact operation balance with no code change.

The binding constraint is *how many unique n=2 puzzles exist at all* under the
repo's existing generation rules (`nums` drawn from `randint(1,100)`, target
required to be an integer in `1..99`, and `already_seen` dedup on
`tuple(sorted(nums))`). Enumerated exhaustively over all 99x99 ordered pairs:

| op | valid ordered pairs | unique multisets (= dedup ceiling) |
|---|---:|---:|
| `+` | 4851 | 2450 |
| `-` | 4851 | 4851 |
| `*` | 473 | **241** |
| `/` | 473 | 473 |

`*` is capped at 241 because the product must stay below 100 (only `1xn`,
`2x<=49`, `3x<=33`, ...). `/` is capped at 473 by divisibility within 1..99 —
note the target cap is *not* binding there, since `a/b <= 99` always holds.
Widening the number range would not help `*` at all.

So **241 per operation (964 rows) is the largest strictly balanced n=2 set**
obtainable without changing the generation rules. Reaching 500/op (2k rows)
would require one of: allowing repeated puzzles, an unequal split, or lifting
the target<100 cap for n=2 (which would make n=2 targets a different
distribution from n=3/4 and confound the comparison). 964/241-per-op was chosen.

Generation took 523 / 556 / 24,646 / 6,977 draws for `+` / `-` / `*` / `/` —
the `*` figure reflects near-exhaustion of its 241 available puzzles.

## Caveat: 49 puzzles are op-ambiguous

Balance is exact by *generating pattern*, but 49 of the 964 `(sorted(nums),
target)` keys appear twice, so there are 915 distinct n=2 prompts. This is
inherent to n=2, not a pipeline bug: 48 are `A*B`/`A/B` collisions on `{1,n}`
(`1*n == n/1 == n`), plus one `A-B`/`A/B` case (`{2,4}` -> 2, since `4-2 == 4/2`).
The existing n=3/4 train set likewise contains cross-pattern duplicates — the
pipeline only dedups train against test.

## Verified

- Schema (including struct field order and HF metadata) identical to `balanced`.
- All n=2 rows: `len(nums)==2`, nums and targets in 1..99, and the labelled
  canonical pattern evaluates to the recorded target for all 964.
- `extra_info.index` continues contiguously (418,136..419,099), `split="train"`.
- Zero n=2 collisions with the test set (impossible by construction — test is
  n=3/4 only — but checked).
- `grader_utils.compute_score` needs no change for n=2: correct answer -> 1.0,
  format-valid-but-wrong -> 0.1, no `<answer>` tag -> 0.
