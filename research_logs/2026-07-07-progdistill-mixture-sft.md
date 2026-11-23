# Progdistill-mixture SFT: 0.5B student on 1.5B-teacher rollout mixture (2026-07-07)

## Setup

- Mode: `launch.sh` → `run_pipeline.sh --progdistill_mixture` — a **single** SFT
  run over the concatenation of all 10 teacher-step parquets (steps 150..1600
  from teacher `Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1`, n=4 responses),
  instead of 10 sequential per-step rounds.
- Student: base Qwen2.5-0.5B. `SFT_TRAIN_STEPS=1600`, `SFT_WARMUP_RATIO=0.03`,
  batch 64, max_length 2048, cosine LR. Sweep: `SFT_LR ∈ {1e-5, 3e-5}`
  (jobs 29107881 / 29107889, 1×H100 each, ~2h07 train each).
- Eval: `sft_final` checkpoint only, n=32, temp 0.6, max 1024 —
  `results/Qwen2.5-0.5B/balanced-progdistill-mixture-...-seed1/sft_final/`.

## Results (mean@1 / mean@32)

| SFT LR | n=3/4 | n=5 | n=6 |
|---:|---:|---:|---:|
| `1e-5` | 0.681 / 0.705 | 0.269 / 0.348 | 0.064 / 0.153 |
| `3e-5` | **0.744 / 0.765** | **0.332 / 0.380** | 0.055 / **0.144**¹ |

¹ n=6 mean@1 is slightly better for 1e-5 (0.064 vs 0.055); everything else
favors 3e-5.

Context: these mixture students beat the strongest *pure* Qwen sequential
students recorded so far (cf. CLAUDE.md tables; e.g. Gemma-scale numbers and
the Qwen 0.5B pre-GRPO students are well below n=5 mean@32 ≈ 0.35-0.38).
Direct same-student comparison vs sequential progdistill at matched SFT LR
still to be tabulated.

## Ops incident + infra fix (same day)

Both jobs "hung" for ~2.5h after step 1600 with wandb frozen. Cause:
`train_sft.sh` sets `data.val_files = data.train_files` (the full ~398k-row
mixture) and `test_freq=-1`, so verl's `fit()` ran its one-and-only validation
pass on the last step — ~100k silent forward passes — **before**
`save_checkpoint`. Jobs finished with ~1h44 walltime to spare; a slower pass
would have lost the runs.

Fixes applied (verl patch **1c** in CLAUDE.md):
- `fsdp_sft_trainer.py`: last-step `save_checkpoint` moved **before**
  validation; validation loop gains tqdm + optional `trainer.max_val_batches`.
- `train_sft.sh`: passes `+trainer.max_val_batches=${SFT_VAL_MAX_BATCHES:-500}`
  (2000 samples ≈ 1 min; `-1` restores the full pass).

Note: the two runs above therefore report `val/loss` over the *full* mixture;
future runs default to a 2000-sample estimate.

## Follow-ups

- `launch.sh` sweep grid mentions `SFT_TRAIN_STEPS ∈ {1600, 3200}` and
  `SFT_LR=1e-6/3e-6`; only steps=1600 × {1e-5, 3e-5} has been run.
- Compare against sequential progdistill (`balanced-progdistill-teacher-...`)
  at matched sftlr for the mixture-vs-curriculum question.
- Optional: GRPO on top of the 3e-5 mixture student
  (`--progdistill_mixture ... --GRPO` path, if wired).
