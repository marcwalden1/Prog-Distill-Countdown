# base-simple prompt variant: data + pipeline support (2026-08-09)

New `template_type='base-simple'` in `preprocess_balanced.make_prefix` (bingbin):
instruction-style prompt ("Solve the Countdown problem. ... Solution:") instead of
the R1-Zero dialogue framing. Notable differences vs `base`:
- no "User and Assistant" transcript scaffold;
- does NOT pre-open `<think>` — the completion starts after `Solution:`, so the
  model must emit the think/answer tags unprompted (grader only requires
  `<answer>`; `<think>` is optional scratch space).

## Data

The raw `countdown_*_pattern_*.json` files are not on this cluster, so the
variant parquets are rebuilt **from the canonical parquets** (guarantees
identical puzzles and train/test splits; only the prompt string changes):

    HF_DATASETS_CACHE=<scratch> python3 scripts/make_prompt_variant_data.py \
        --template_type base-simple --suffix=-simple

(NEW script. `--suffix` needs `=` because the value starts with `-`. HF cache
must be off NFS-home or datasets dies with "Not enough disk space".)

Output (verified: row counts + puzzles match canonical, prompt renders with
plain int lists):

    data/balanced-simple/{train,test}.parquet   418136 / 997 rows
    data/balanced5-simple/test.parquet          4620 rows
    data/balanced6-simple/test.parquet          3519 rows

## Pipeline changes (all backward-compatible; plain `balanced` behaves as before)

- `scripts/run_pipeline.sh`: eval-set names now derive from `DATA_SOURCE` —
  `DATA_SOURCE=balanced<sfx>` evals on `balanced<sfx>`/`balanced5<sfx>`/`balanced6<sfx>`.
  (Models trained on a prompt variant must also be *evaluated* on it; eval
  prompts come from the test parquets.)
- `plot_results.py`: new `--dataset_suffix`; `scripts/plot_results.sh` derives it
  from `DATA_SOURCE`.

Result/eval filenames carry the dataset name (e.g.
`balanced5-simple_temp0.6_n32_max1024.json`), so full-set exports for ID/OOD
plots work unchanged by passing `--datasets balanced-simple balanced5-simple
balanced6-simple` to `scripts/export_idood_scores.py`.

## Launch (0.5B base GRPO on the new prompt, matched to the recent
## dynbsz-n8-lr3e-6-kl3e-3-ep2 baselines)

    MODEL_NAME=Qwen2.5-0.5B \
    DATA_SOURCE=balanced-simple \
    EXP_NAME=balanced-simple-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1 \
    USE_DYNAMIC_BSZ=True ROLLOUT_N=8 LR=3e-6 KL_COEF=3e-3 \
    TOTAL_EPOCHS=2 TOTAL_STEPS=3266 \
    bash scripts/run_pipeline.sh

Caveat for cross-prompt comparisons: `balanced5-simple`/`balanced6-simple`
contain the same puzzles as `balanced5`/`balanced6`, so accuracy numbers are
comparable across prompt variants — but each model should be evaluated on its
own training prompt.
