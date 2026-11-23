# 2026-07-09 — Entropy & generation length on ID val set (4 GRPO runs)

Script: `scripts/plot_entropy_length.py` → `figures/entropy_length/{entropy,length}_vs_step_balanced.png`

Runs (from `results/ckpts.md`; KD+GRPO eval JSONs taken from
`/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results/...` — the
`results_shlok` symlink tree only has 4/10 balanced steps):

1. Qwen2.5-1.5B GRPO — `balanced-grpo-lr3e-6-kl3e-3-seed1`
2. Qwen2.5-0.5B GRPO — `balanced-grpo-seed1`
3. Qwen2.5-0.5B KD + GRPO — `balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1`
4. Qwen2.5-0.5B PD mixture + GRPO — `balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl3e-3-seed1`

## Metric definitions

- **Sample entropy (bits):** per prompt, Shannon entropy of the empirical
  distribution over the 32 sampled final answers (last `<answer>` span,
  whitespace-normalized; no-answer = own bucket), averaged over 997 prompts of
  `balanced` test set (n=3,4), temp 0.6. Max = log2(32) = 5.
  **Not token-level policy entropy** — eval JSONs store no logprobs. Cached as
  `.entropy32` next to each eval JSON.
- **Length:** mean response tokens, same definition/caches (`.lengths`) as
  `plot_results.py`.

## Observations

- All four runs show monotone-ish entropy decay (the usual RL sharpening),
  but levels differ: 1.5B GRPO stays highest (~2.9 → 1.6 bits); the two
  distilled students start much lower (~1.2–1.4) and decay gently to
  ~0.7 (KD) / ~0.96 (PD mixture) — PD mixture keeps consistently more answer
  diversity than KD at every step.
- 0.5B base GRPO is wildly non-monotone in both entropy and length
  (length swings 731 → 21 tokens; entropy 2.8 → 0.4 with spikes) —
  the run partially collapses to very short low-diversity outputs from
  step ~750 on.
- Lengths: KD student generates the longest responses (~700–930 tok),
  PD mixture ~530–690, 1.5B teacher ~460–740 (uptick at step 1600),
  0.5B base GRPO collapses to ~20–90 tok after step 600.

## Addendum: wandb train-time metrics (same day)

Script: `scripts/plot_wandb_entropy_length.py` → `figures/entropy_length/
wandb_{actor_entropy,actor_entropy_logy,response_length}_vs_step.png`.
Runs from `results/wandb_Qwen0.5B.md` (entity/project
`progressive_distill/prog_distill`), full per-step histories cached as CSVs in
`results/wandb_Qwen0.5B/`, plotted at 20-step intervals. These are TRAIN
metrics: `actor/entropy` = token-level policy entropy (nats/token) over rollout
tokens; `response_length/mean` = mean rollout length. Note the wandb GRPO run
(`508ua0kv` = lr3e-6-kl3e-3) is a different base-GRPO run than the eval-JSON
one above (`balanced-grpo-seed1`), and PD + GRPO (`w3g85krj`, seed21,
non-mixture) has no counterpart in the eval-JSON plots.

- Base GRPO: entropy dips 2.6 → 0.5 by step ~160, then *rebounds* to ~3 and
  stays high — with rollout length whipsawing 40–430 tok. Consistent with the
  unstable/degenerate behavior seen in the val-set sample entropy.
- All three distilled students sit 1–2 orders of magnitude lower and are flat:
  PD mixture ~0.11–0.15, PD ~0.055–0.08, KD ~0.04–0.07 nats/token, all slowly
  decaying. PD mixture keeps ~2× the token entropy of KD/PD throughout.
- Train lengths: PD ~800 (flat), KD 770 → ~500 → back to ~600, PD mixture
  ~540 → ~430, GRPO noisy ~200–300.

## Addendum: step-1600 accuracy bars (same day)

Script: `scripts/plot_accuracy_bars.py` → `figures/Qwen2.5-0.5B/
accuracy_bars_{mean1,mean32}.png`. Same four runs as the eval-JSON plots
(`results/ckpts.md`); scores via `plot_results._score_checkpoint` (n=5,6
capped at 1000 prompts). The 1.5B run had no `.scores` caches — computed and
cached now.

| Run (step 1600) | n=3,4 m@1/m@32 | n=5 | n=6 |
|---|---|---|---|
| Qwen2.5-1.5B GRPO | 0.851 / 0.873 | 0.412 / 0.465 | 0.085 / 0.172 |
| Qwen2.5-0.5B GRPO | 0.292 / 0.355 | 0.139 / 0.212 | 0.023 / 0.112 |
| Qwen2.5-0.5B KD + GRPO | 0.944 / 0.945 | 0.407 / 0.464 | 0.034 / 0.131 |
| Qwen2.5-0.5B PD mixture + GRPO | 0.925 / 0.931 | 0.483 / 0.513 | 0.086 / 0.168 |

Both distilled students beat the 1.5B teacher in-distribution; PD mixture
beats KD on both OOD sizes (n=5: 0.483 vs 0.407; n=6: 0.086 vs 0.034 mean@1)
and matches the teacher OOD.
