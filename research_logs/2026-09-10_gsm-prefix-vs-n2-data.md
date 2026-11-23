# OLMo-2-1B-pt4001B: `base-simple-gsm` prefix vs. n=2 training data

**Date:** 2026-09-10
**Model:** `OLMo-2-0425-1B-pt4001B` (top of the 1B pretraining ladder)
**Shared hyperparams:** `LR=3e-6`, `KL_COEF=3e-3`, `PPO_MICRO_BATCH_SIZE_PER_GPU=32`,
otherwise `train_grpo.sh` defaults (batch 256, `rollout.n=4`, 1 epoch,
`TOTAL_STEPS=1635`, save/test every 100/50, 2 H100s).

## The two arms

Two one-factor-at-a-time changes off the `balanced-simple` baseline
(`balanced-simple-grpo-lr3e-6-kl3e-3-seed1`, see
`2026-08-25_olmo2-pretrain-ladder-grpo.md`):

| Arm | prefix | train data | `DATA_SOURCE` | steps |
|---|---|---|---|---:|
| new prefix, old data | `base-simple-gsm` | n=3,4 (418,136 rows) | `balanced-simple-gsm` | 1633 (data-limited) |
| old prefix, new data | `base-simple` | n=2,3,4 (419,100 rows) | `balanced234-simple` | 1635 (`TOTAL_STEPS`-limited) |

Both stop at the same 16 evaluated checkpoints (steps 100..1600), so the 2-step
difference in run length is immaterial.

`base-simple-gsm` is `base-simple` plus one line: "Use only grade-school math to
find solutions. Nothing complicated required." The n=2 data is 964 rows, 241 per
operation — see `2026-09-10_balanced234-n2-dataset.md` for why 241 is the ceiling.

## Jobs (submitted 2026-09-10)

| Arm | train | eval n=3,4 / n=5 / n=6 | plot |
|---|---|---|---|
| `balanced-simple-gsm-grpo-lr3e-6-kl3e-3-seed1` | `45868322` | `45868324` / `45868325` / `45868326` | `45868327` |
| `balanced234-simple-grpo-lr3e-6-kl3e-3-seed1` | `45868348` | `45868349` / `45868360` / `45868362` | `45868363` |

`TRAIN_SBATCH_ARGS="-t 30:00:00"` (≈21 h/epoch for this model) and
`EVAL_SBATCH_ARGS="-t 08:00:00"` (this model's long responses TIMEOUT eval.sh's
1:30 default on n=5/n=6) both follow the 2026-08-25 precedent.

## Data built for these runs (the prefix is baked into the parquet)

    # arm 1: re-render the canonical puzzles with the gsm prefix
    python3 scripts/make_prompt_variant_data.py --template_type base-simple-gsm --suffix=-simple-gsm
    #   -> data/balanced{,5,6}-simple-gsm

    # arm 2: extend the base-simple train set with n=2 rows in the same prefix
    python3 preprocess_balanced.py --mode balanced234 --template_type base-simple \
        --src_subdir balanced-simple --out_subdir balanced234-simple
    #   -> data/balanced234-simple

Verified: for arm 1, every non-`prompt` column is byte-identical to
`data/balanced{,5,6}-simple` and 100% of prompts carry the gsm line; for arm 2,
the 418,136-row n=3,4 prefix is byte-identical to `balanced-simple/train.parquet`,
the test parquet is a byte copy, and all 964 n=2 prompts render as `base-simple`
(no gsm line).

## Three code fixes these runs needed

1. **`scripts/make_prompt_variant_data.py` was unrunnable on this cluster.**
   `Dataset.from_parquet` wedges in uninterruptible I/O (`D` state, 0% CPU)
   acquiring its HF-datasets cache filelock under the NFS home — the same hang as
   `load_dataset`. Rewritten to use pyarrow: 13 s for all four parquets, and it
   reproduces the existing `data/balanced*-simple` tables exactly (regression-
   tested with `Table.equals` before switching).
2. **`run_pipeline.sh` mangled the OOD eval-set names.** `_ds_suffix=${DATA_SOURCE#balanced}`
   assumes `DATA_SOURCE=balanced<suffix>`, so `balanced234-simple` derived the
   nonexistent `balanced5234-simple`. `EVAL_DS_N34`/`N5`/`N6` are now env-
   overridable (defaults unchanged); arm 2 passes
   `EVAL_DS_N5=balanced5-simple EVAL_DS_N6=balanced6-simple`.
3. **`preprocess_balanced.py --mode balanced234` gained `--src_subdir`/`--out_subdir`**
   so it can extend a prompt variant rather than only `data/balanced`, plus a
   guard that re-renders a real source row and refuses to run if
   `--template_type` disagrees with the prompts already in the source parquet
   (otherwise the n=2 rows would silently carry a different prefix from the
   n=3,4 rows).

`PPO_MICRO_BATCH_SIZE_PER_GPU` is deliberately *not* in either `EXP_NAME`: with
`ppo_mini_batch_size` fixed it only changes gradient-accumulation granularity,
not the objective.
