# Offline on-policy distillation (Lightning OPD) — implementation (2026-07-09)

Implements offline OPD following **arXiv:2604.13010** ("Lightning OPD"): when
the teacher used for off-policy distillation (SFT on teacher rollouts) and for
the subsequent on-policy distillation is the *same* ("teacher consistency"),
the OPD stage can run on offline data with performance comparable to live OPD.

## Method as implemented

1. **Rollout (once):** sample from the SFT student π_ref — `OPD_MAX_PROMPTS`
   =10k prompts × `N_RESPONSES`=4 at `OPD_ROLLOUT_TEMP`=1.0 (paper: 0.8),
   max 1024 new tokens. Token IDs stored verbatim.
2. **Teacher scoring (once):** teacher (same checkpoint that generated the
   student's SFT data; for progdistill, the final-round step-1600 teacher)
   scores every rollout token via vLLM `prompt_logprobs`.
3. **Training:** student initialized from π_ref, 1600 steps @ batch 64 on the
   frozen set (≈2.5 epochs), lr 2e-6 cosine (paper: constant), per-token loss
   `clamp(log π_T − log π_θ, ±10) · (−log π_θ)` with the advantage
   stop-gradiented — implemented as advantage-weighted CE in verl's SFT
   trainer (local patch 6, gated on `+data.opd=True`).

Design decisions vs. the paper:
- **Budget:** 1600 OPD steps to be compute-matched with the off-policy distill
  stage (user's call). That is ~10× the paper's proportional budget (150 OPD
  steps after 3000 SFT steps, 1 epoch); offline-OPD bias grows as π_θ drifts
  from π_ref, so mid-curve regression is possible — checkpoints every 50 steps
  and the standard 10-point eval curve will show where it saturates.
- **Temp 1.0** rollouts (true reference policy) instead of 0.8; knob
  `OPD_ROLLOUT_TEMP`.
- Rollouts are NOT filtered by correctness (OPD is distribution matching);
  grader scores are stored per row for analysis.

## Code map

| Piece | Where |
|---|---|
| Data gen (2-stage) | `scripts/generate_opd_data.py`, `scripts/generate_opd_data.sh` |
| Dataset (verl `data.custom_cls`, stored token IDs, no retokenization) | `opd_dataset.py` (repo root) |
| Loss (verl local patch 6) | `verl/verl/trainer/fsdp_sft_trainer.py`, in `patches/local-fixes-vs-083da9ab.patch` |
| Trainer wrapper | `scripts/train_opd.sh` (save_freq=50 → GRPO-style `global_step_*` layout, stock eval array works) |
| Pipeline | `scripts/run_pipeline.sh --opd <teacher-EXP_NAME>` (gen → train → eval×3 → plot) |

Validated on the login node: OPDDataset shapes/masks/teacher-logprob alignment
(incl. right-truncation), and the loss branch checked against a hand
computation with gradient-mask verification.

## Planned first experiments (Qwen 0.5B student, 1.5B teacher)

Teacher: `Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1` @ step 1600
(sdholakia's tree → `TEACHER_CHECKPOINT_DIR`). Students (see
`sft-checkpoints/best_models.md`):

- **distill arm** (clean teacher consistency): SFT student
  `balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr3e-5-seed1`
- **progdistill arm** (consistency holds only for the final round's data):
  `balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr3e-5-seed1/round_1600`

Comparisons: OPD curve vs. the SFT student (step 0) and vs. GRPO-on-top runs,
on `balanced` (ID) and `balanced5`/`balanced6` (OOD), mean@1 + mean@32.

## Status

- 2026-07-09: implementation complete, smoke-tested on login node. No jobs
  submitted yet.
