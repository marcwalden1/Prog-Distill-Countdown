# Qwen3-4B-Base teacher logit falloff (sanity check before 4B→0.6B distillation)

**Date:** 2026-11-06. **Job:** 58205789 (1 H100, ~10 min).
**Code:** `scripts/sanity_teacher_logits.{py,sh}`, `scripts/plot/plot_teacher_logit_sanity.py`.
**Data:** `analysis_data/teacher_logit_sanity/Qwen3-4B-Base-T1.0-max512-p10-n10/` (summary.json, sanity.npz, rollouts.jsonl).
**Figure:** `figures/teacher_logit_sanity/Qwen3-4B-Base-T1.0-max512.png`.

Purpose: before generating 418,136 prompts × N=10 rollouts with stored top-20 logprobs
(+ a tail bucket), check how fast the raw Qwen3-4B-Base next-token distribution falls off.

Setup: 10 random `balanced-simple` train prompts (seed 0), N=10, temperature 1.0, top_p 1,
max 512 tokens, raw prompt path (no chat template). vLLM samples; HF (bf16, sdpa) recomputes the
full 151,936-way distribution at every response position (16,653 positions).

## Rollouts
- Mean length 167 tokens; median 83; 16% truncated at 512; 67% contain `</answer>`.
- 17/100 rollouts are empty (immediate EOS); 8 more emit the literal `<answer>...</answer>.` copied from the prompt.

## Falloff (tail mass = 1 − Σ top-k probs, per position)

| k | mean | median | p90 | p99 | P(sampled token outside top-k) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.267 | 0.150 | 0.708 | 0.908 | 0.261 |
| 5 | 0.084 | 0.0064 | 0.298 | 0.713 | 0.085 |
| 20 | 0.035 | 0.00073 | 0.108 | 0.506 | 0.035 |
| 64 | 0.018 | 0.00021 | 0.043 | 0.337 | 0.018 |
| 100 | 0.014 | 0.00013 | 0.030 | 0.279 | 0.014 |
| 1000 | 0.0040 | 0.00002 | 0.0049 | 0.088 | 0.0043 |

Entropy: mean 1.03 nats, median 0.60.

- **Heavy-tailed, concentrated in a few positions.** At the median position top-20 holds 99.93%
  of the mass, but 10.6% of positions leave >10% outside top-20 and those 10% carry 77% of the
  total missing mass. Median prob decays roughly as a power law in rank (panel a).
- **3.5% of sampled tokens fall outside top-20** (matches mean tail mass, as it should at T=1).
  These positions need the sampled token's logprob stored alongside the top-20.
- Tail mass is flat across response position; truncated and finished rollouts have the same tail
  (0.035 vs 0.034).
- **vLLM vs HF logprobs disagree** on the same top-20 tokens: per-position max |Δlogprob|
  median 0.13, p90 0.25, p99 0.47 nats (bf16 kernel differences). Matters if teacher targets are
  taken from vLLM's sampling pass rather than a separate scoring forward.

## Full run: generation + NTP-distill SFT (launched 2026-11-06)

**Generation** (`scripts/generate_distill_data.sh`, job 58226039, `kempner_requeue`, 4 H100, resumable
200 shards): all 418,136 `balanced-simple` train prompts (= exactly the prompts one GRPO epoch sees,
1633 × 256), N=10, T=1, **total cap 512 incl. prompt** (response budget 512 − len(prompt) ≈ 401–407),
top-20 (id, logprob) + sampled-token logprob per position, no filtering (empty and truncated rollouts
kept). Per-prompt seeds. Output:
`/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown/distill-data/Qwen3-4B-Base/balanced-simple-T1.0-total512-n10-top20/`.
Pilot (500 prompts, old 512-response cap): 9.3k tok/s/GPU, 165 tok/rollout, 19.9% empty, 15.5%
truncated, 6.1% exactly correct, top-20 tail mass 0.034, sampled token in top-20 96.5%.

**SFT, NTP arm** (`scripts/run_qwen3_distill_ntp_sft.sh`, jobs 58227788/91/93 = lr 3e-6/1e-5/3e-5,
afterok on generation): Qwen3-0.6B-Base, exact teacher token ids (`distill_dataset.py`, no EOS added to
truncated rollouts), 1 epoch = 16,333 steps, global batch 256 (64/GPU × 2 GPUs × 2 accum), AdamW
β=(0.9, 0.95), cosine to 0 with 3% warmup, max_length 512, save every 2000 (keep 2), resume=auto.
64/GPU is the memory ceiling on H100 at length 512: 96 OOMs on the fp32 full-vocab logits
(liger fused CE is not installed in the env). Expected ~13–14 h/run on 2 H100.

## 2026-11-10: NTP SFT loss vs the teacher-entropy floor

No val loss was logged (only the final val pass, which OOMed), so the curve is the train loss
parsed from the logs (`scripts/plot/plot_distill_ntp_loss.py` ->
`figures/Qwen3-distill-grpo/distill_ntp_loss_vs_step.png`). 1 epoch, so each batch is unseen when
scored; each logged point is one 256-rollout batch every 50 steps (noisy, ~0.92-1.22). All three
runs share the batch sequence, so the paired difference vs lr=1e-5 is low-noise.

| | nats/token |
|---|---|
| student loss, first 1000 steps (lr 3e-6 / 1e-5 / 3e-5) | 1.103 / 1.090 / 1.093 |
| student loss, last 1000 steps | 1.053 / 1.047 / 1.057 |
| **teacher NLL on its own samples** (= entropy floor; T=1, top_p=1, 602.5M tokens, 200 shards, per-shard 0.918-0.975) | **0.944** |

- Loss drops ~1.12 -> ~1.05 in the first ~1.5k steps, then plateaus while the LR cosines to 0.
- lr=1e-5 is best throughout, but the whole 10x LR range spans <=0.01 nats.
- Remaining gap to the floor = forward KL(teacher || student) on teacher samples ~ **0.10 nats/token**
  (~15 nats per rollout at ~144 response tokens/rollout). Roughly 40% of the start-to-floor gap
  was closed. Floor is from vLLM bf16 logprobs vs HF-trained student; small bias possible.

## 2026-11-10: logit-KD LR sweep, 4000 steps, constant LR (jobs 59497236/7/8)

Exps `balanced-simple-distill-kd-t4b-base-T1-total512-n10-sftlr{3e-6,1e-5,3e-5}-bs256-constant-wu0.03-seed1`
(warmup 120 steps, then flat; log every 10 steps; save 2000/4000). All three hit the SAME final-val
OOM as NTP (verl `validation_step`, 9.27 GiB lm_head alloc at micro-batch 64; the KD val path is
NOT chunked, contrary to what the pilot suggested) after `global_step_4000` was saved; cancelled
and CPU-merged with `KEEP_SHARDS=true` (merge jobs 59531702/5/8), so each exp dir has a merged HF
model AND a resumable global_step_4000 (model + optim + scheduler + dataloader).

KD and NTP see identical batches per step (same parquet order, seed, world size, batch): per-batch
Pearson r(KD kd_ce, NTP loss) = 0.990-0.993. Through step 4000 NTP's cosine LR is >= 94% of peak.
Figure: `figures/Qwen3-distill-grpo/distill_kd_vs_ntp_loss.png`
(`scripts/plot/plot_distill_kd_vs_ntp_loss.py`). Steps 3000-4000 (paired, same batches):

| lr | KD sampled-token CE | NTP CE | KD - NTP | KD objective |
|---|---|---|---|---|
| 3e-6 | 1.076 | 1.092 | -0.016 | 0.091 |
| 1e-5 | 1.066 | 1.089 | -0.023 | 0.082 |
| 3e-5 | 1.067 | 1.108 | -0.041 | 0.082 |

- KD beats NTP at every LR on NTP's own metric; vs the BEST NTP (1e-5), KD 1e-5 and 3e-5 are both ~0.022 better.
- KD 1e-5 ~= 3e-5 (tie on both CE and KD objective); 3e-6 is clearly worse.
- Absolute CE in this window sits above the epoch-average floor because steps 3000-4000 are a
  hard stretch of batches (NTP is at ~1.09 there vs ~1.05 at epoch end); compare paired numbers.
- The KD objective is far less noisy than the CE (a KL, so teacher sampling entropy drops out) and
  is still decreasing at step 4000 -> extending is reasonable; it is also the better tuning signal.

## 2026-11-11: held-out KD loss for student selection

`scripts/score_kd_heldout.py` scores merged students on the prompts that NO candidate trained
on: the SFT sampler is a seeded uniform permutation (DistributedSampler seed=1), so a run of S
steps at bs256 trained on exactly its first S×256 rows. For S=8000 that leaves 549 of 418,136
prompts with none of their 10 rollouts seen (5,490 rollouts, 776k response tokens). Shuffle
check: each tenth of the dataset (file order) is 49% seen; teacher accuracy (1.7% / 1.7% / 1.4%)
and response length (144 / 144 / 141 tokens) match across trained-on / not-trained-on /
held-out rows. Scores: `analysis_data/kd_heldout_scores.json` (per-prompt sums -> paired
bootstrap SE over prompts).

| candidate (4000 steps, constant LR) | held-out KD obj | sampled-token CE |
|---|---|---|
| lr 1e-5 | 0.08110 | 1.05928 |
| lr 3e-5 | 0.08125 (+0.00015 ± 0.00028) | 1.06032 (+0.00104 ± 0.00039) |

The two are tied on the KD objective (the final val/loss had 3e-5 ahead by 0.0008, also
within noise). 8000-step runs (cos to 0.1×peak, 200 warmup, jobs 59827445/6) are scored by
selector job 59834193 (`scripts/select_kd_student_and_grpo.sh`), which then launches GRPO
(lr 3e-6, kl 3e-3, 1 epoch) on the lowest held-out KD objective.

### 8k cosine runs and final selection (selector job 59834193)

| candidate | held-out KD | Δ vs best (±SE) | sampled CE | Δ CE (±SE) |
|---|---|---|---|---|
| sftlr3e-5-cos8k | 0.06762 | — | 1.04553 | — |
| sftlr1e-5-cos8k | 0.07491 | +0.00729 ±0.00018 | 1.05280 | +0.00727 ±0.00027 |
| sftlr1e-5-const4k | 0.08110 | +0.01348 ±0.00020 | 1.05928 | +0.01375 ±0.00030 |
| sftlr3e-5-const4k | 0.08125 | +0.01363 ±0.00021 | 1.06032 | +0.01479 ±0.00031 |

Selected sftlr3e-5-cos8k (margin ~40 SE). GRPO (lr 3e-6, kl 3e-3, 1 epoch, balanced-simple):
`balanced-simple-grpo-from-distill-kd-t4b-base-sftlr3e-5-cos8k-lr3e-6-kl3e-3-seed1`,
train 59881176, evals 59881179/59881180/59881182, plot 59881183.
