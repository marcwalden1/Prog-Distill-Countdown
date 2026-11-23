# research_logs/

Running notes on experiments and analyses, one dated markdown file per
study/thread. Keep small summary tables inline; link out to large artifacts
(eval JSONs, figures, sample dumps) rather than copying them here.

Index:

- `2026-07-07-passk-coverage.md` — pass@k coverage of Qwen2.5-0.5B
  (base / grpo-seed1 / grpo-warmup) + generation inspection, incl.
  Qwen2.5-1.5B post-GRPO comparison.
- `2026-07-08-generation-evolution-across-grpo.md` — how temp-0.6
  generations change across GRPO checkpoints for the three `results/ckpts.md`
  runs (1.5B teacher, 0.5B base-GRPO, 0.5B progdistill-mixture+GRPO);
  style-transition / perseveration / diversity-collapse findings.
- `2026-07-09-offline-opd.md` — offline on-policy distillation
  (Lightning OPD, arXiv:2604.13010) implementation: method mapping, code map,
  design decisions (1600-step compute-matched budget, temp-1.0 rollouts),
  planned Qwen 0.5B distill/progdistill experiments.
- `2026-07-27-opd-40m-wide-distill.md` — first offline-OPD run: 40M-wide
  distill (sftlr 3e-4) student × Qwen2.5-1.5B step-1600 teacher, OPD LR arms
  2e-6 / 2e-5 sharing one rollout dataset; job IDs + comparison plan.
- `2026-08-25_olmo2-pretrain-ladder-grpo.md` — GRPO (lr=3e-6, kl=3e-3, base_simple
  prompts) on three OLMo-2-0425-1B stage1 pretraining checkpoints (1007B / 2937B /
  4001B tokens) read from the shared dam_lab HF cache; setup + compatibility checks.
- `2026-08-31_olmo2-7b-gtsft-lr-sweep.md` — OLMo-2-1124-7B ladder (101/252/500/1007B
  tokens): 117 GB download + verification, and a 4x4 GT-SFT LR sweep (200 steps);
  7B/1B share a byte-identical tokenizer, and 4 H100s suffice for 7B SFT.
- `2026-09-10_balanced234-n2-dataset.md` — `data/balanced234`: the `balanced`
  train set verbatim plus 964 operation-balanced n=2 rows (241 each of +, -, *, /),
  and the per-operation ceilings on unique n=2 puzzles that cap it at 241.
- `2026-09-10_gsm-prefix-vs-n2-data.md` — OLMo-2-1B-pt4001B, two one-factor arms off
  the `balanced-simple` GRPO baseline: the `base-simple-gsm` prefix on the old n=3,4
  data, vs. the old `base-simple` prefix on n=2,3,4 data. Job IDs + the three code
  fixes the launch needed.
- `2026-09-17_arithmetic-fidelity-base-sft-grpo.md` — are the wrong-arithmetic
  claims in rollouts (`(87+17+60)/9 = 13`) developed by RL or already in the base
  model? Behavioral + teacher-forced probes over 6 OLMo-2 GT-SFT arms (1B-pt4001B,
  7B-pt135B, 7B-pt3896B x {search, answer}). Sampled arithmetic error is flat
  ~0.84-0.98 from base through GRPO, so the errors pre-exist; what SFT installs is
  the "= target" assertion habit (0.16 -> 1.00) and verdict coherence
  (0.64 -> 0.00); GRPO improves expression *selection* (final-line miscompute
  0.79 -> 0.49), not evaluation. Teacher-forced, `search` SFT raises
  P(true)/(P(true)+P(target)) from ~0.12-0.26 to 0.39-0.71 while the `answer`
  control never moves it, and how much survives GRPO tracks pretraining
  (7B-pt3896B keeps 0.70; 7B-pt135B returns to base).
- `2026-11-05_qwen3-size-overlap-and-best-ckpt.md` — Qwen3-Base GRPO 0.6B (lr3e-6) /
  1.7B / 4B (lr1e-6), kl3e-3: Venn / area-proportional Euler / UpSet views of which problems each
  size solves, and accuracy vs size at the final vs oracle-best checkpoint. Accuracy is monotone in
  size everywhere; 4B ⊇ smaller models on n=3,4 but not OOD; final ≈ best checkpoint. Plus the
  0.6B lr1e-6 2-epoch run (doesn't beat 1-ep lr3e-6 at the final ckpt).
- `2026-11-06_qwen3-4b-teacher-logit-sanity.md` — raw Qwen3-4B-Base at T=1, 10 prompts × 10:
  how fast the next-token distribution falls off (top-20 tail mass, sampled token outside top-k),
  and the vLLM-vs-HF logprob gap, ahead of 4B→0.6B logit distillation.
