# OLMo-2-7B pretraining ladder: GT-SFT LR sweep

**Date:** 2026-08-31
**Status:** 16 SFT + 48 eval jobs submitted
**Scripts:** `scripts/download_olmo2_7b_ckpts.py|.sh`, `scripts/run_olmo2_7b_gtsft_sweep.sh`
**Companion:** `2026-08-25_olmo2-pretrain-ladder-grpo.md` (the 1B ladder)

## Goal

Extend the pretraining-budget ladder to 7B so model size can be compared against
pretraining budget. Same recipe as the 1B arms: GT answer-only SFT on
`balanced-simple`, 200 steps, batch 64, then (later) GRPO at lr 3e-6 / kl 3e-3.

## Base checkpoints

`allenai/OLMo-2-1124-7B`, four stage1 revisions — all four requested budgets exist
upstream exactly (unlike the 1B ladder, where 2014B/2999B did not):

| tag | tokens | revision |
|---|---|---|
| `pt101B` | 101B | `stage1-step24000-tokens101B` |
| `pt252B` | 252B | `stage1-step60000-tokens252B` |
| `pt500B` | 500B | `stage1-step119000-tokens500B` |
| `pt1007B` | 1007B | `stage1-step240000-tokens1007B` |

**Downloaded, not symlinked to a shared cache**: the `dam_lab` cache holds only the
0425-1B repo. 29.2 GB per revision (fp32, 6 shards) = ~117 GB total, into
`/n/netscratch/kempner_bingbin_lab/Lab/bingbin/hf-cache/hub`, exposed as
`$MODEL_DIR/OLMo-2-1124-7B-pt<N>B`. Job 43321919, **17 min**, every revision
verified (all files present, sizes and LFS sha256 matching the Hub).

Caveat: that cache is on `/n/netscratch`, which is purge-eligible. The symlinks are
only as durable as scratch.

### Architecture / tokenizer checks

All four revisions identical: `Olmo2ForCausalLM`, 32 layers, hidden 4096,
intermediate 11008, 32 heads / **32 KV heads (no GQA)**, vocab 100352, maxpos 4096,
fp32 on disk, **no `chat_template`** — so the same plain-concatenation path in
`sft_dataset.py` (verl patch 3b) that the 1B runs used.

**The 7B and 1B share a byte-identical tokenizer**: same `tokenizer.json` blob hash
`5fe172127988...`, same length 100278, and the GT SFT parquet tokenizes to identical
token IDs under both (sha `f49ab76e52b2560b` over 300 rows). So the two ladders are
comparable at the token level, and the existing GT data needed **no** regeneration
(it stores prompt/response as text; SFTDataset tokenizes at load).

## Sweep

4 arms x SFT LR {1e-5, 3e-5, 1e-4, 3e-4} = 16 runs, 200 steps each, seed 1,
reusing `sft-data/gt/balanced-simple/gt-answeronly.parquet`. Each run's SFT
checkpoint is evaluated on `balanced-simple` / `balanced5-simple` /
`balanced6-simple` -> 48 eval jobs.

### Resource decision: 4 H100s, not the repo's 8-for-7B convention

200 steps x 64 x ~120 tokens is trivial FLOPs; 7B SFT here is memory-bound. FSDP2
shards the fp32 Adam state (~98 GB: 28 master + 28 m + 28 v + 14 bf16 params) to
~25 GB/GPU on 4 GPUs. **Verified by probe**, but only after fixing `max_length` (see the OOM
section below): the retry (job 43334580, pt1007B lr3e-5) COMPLETED in 15:43, all 200
steps, train/loss 0.1136, merged to a 14 GB bf16 HF checkpoint with FSDP shards
auto-cleaned. So 4 GPUs suffice and the sweep's queue footprint is halved.
NB the merge writes bf16, so 16 SFT outputs cost ~224 GB, not ~460 GB fp32.

Graded fallbacks if a future config OOMs (none needed so far), cheapest last since
activations are negligible at 120 tokens and the memory is nearly all optimizer
state: `SFT_GPUS=8` -> `model.fsdp_config.cpu_offload=True` (frees ~56 GB) ->
`offload_params=True` -> `data.micro_batch_size_per_gpu=1`. `train_sft.sh` has no
`SFT_EXTRA_ARGS` passthrough yet; add one rather than editing the Hydra line.

Evals get `-t 06:00:00` rather than `eval.sh`'s 1:30 default: 7B generation is
slower, and the 1B no-SFT n=5 evals died on exactly that limit.

### Job IDs

- Probe (pt1007B lr3e-5): SFT 43326465; evals 43328420 / 43328423 / 43328425
  (submitted separately — the probe ran with `SKIP_EVAL=1`)
- pt101B lr{1e-5,3e-5,1e-4,3e-4}: 43328180 / 43328193 / 43328209 / 43328223
- pt252B: 43328239 / 43328251 / 43328261 / 43328267
- pt500B: 43328278 / 43328295 / 43328309 / 43328343
- pt1007B lr{1e-5,1e-4,3e-4}: 43328364 / 43328378 / 43328388

## Next

1. Score the 16 SFT checkpoints (`plot_results._score_checkpoint` for mean@1, and
   `scripts/compute_passk.py` for pass@1/pass@32 — prefer pass@k, see the 1B log for
   why mean@32 is misleading here).
2. Pick one matched LR across arms (the 1B ladder chose 3e-5; do not mix per-arm
   bests, that confounds pretraining budget with SFT LR).
3. GRPO on top at lr 3e-6 / kl 3e-3 for the 1B/7B comparison.

Verify SFT completion by **file existence**
(`sft-checkpoints/<model>/<exp>/model.safetensors*`), not job State — SFT-checkpoint
evals exit non-zero on `eval.sh`'s trailing `chmod` of a `checkpoints/<...>` path
that does not exist for SFT runs.

## The OOM that killed the first launch: `data.max_length` padding

First probe (job 43326465) **finished all 200 steps** (train/loss 0.1135) and then
OOMed on GPUs 1/2/3 trying to allocate **3.06 GiB**.

Cause: verl's `SFTDataset` pads **every** sample to `data.max_length`
(`sft_dataset.py:150-158`), which `train_sft.sh` hard-coded to **2048**. The GT
answer-only sequences are <=120 tokens, so ~94% of every batch was padding, and the
lm_head logits tensor is `micro_batch x max_length x vocab`:

    4 x 2048 x 100352 x 4 bytes = 3.06 GiB     <- exactly the failed allocation

The final validation pass died on it. So `max_length` here is **not** a truncation
ceiling, it is the dominant memory term. The 1B ladder survived the identical
3.06 GiB logits only because its optimizer state is ~8x smaller.

Patch 1c behaved as designed — `global_step_200/` (82 GB of shards) was saved
*before* validation, so the weights were recoverable — but `train_sft.sh` exits
before the merge on a non-zero exit code, so nothing was merged and the `afterok`
evals were dead. Left unfixed, all 16 cells would have done this: ~1.2 TB of
unmerged shards and zero usable checkpoints.

**Fix:** `train_sft.sh` gained `SFT_MAX_LENGTH`, **default 2048 (unchanged)** so
teacher-rollout SFT — whose responses run to the 1024-token cap — is unaffected.
The 7B sweep sets **256** (>2x headroom over the 120-token data). Side benefit: the
forward/backward is ~8x cheaper, so the sweep is much faster than budgeted.

The 82 GB of unmerged probe shards were deleted and that cell rerun, so all 16 cells
have identical provenance rather than one salvaged one.

**Applies to the 1B ladder too:** those GT-SFT runs also trained on 94% padding.
They did not OOM and padding is masked out of the loss, so the results stand — but
pass `SFT_MAX_LENGTH=256` on any rerun.

### Final job IDs (after the relaunch)

- pt1007B lr3e-5: SFT 43334580 (**COMPLETED, merged**); evals 43341105/43341107/43341109
- pt101B lr{1e-5,3e-5,1e-4,3e-4}: 43338xxx block (submitted 2026-08-31, see squeue/sacct)
- pt252B, pt500B: same block, 4 LRs each
- pt1007B lr1e-5: same block; lr1e-4: 43341083; lr3e-4: 43341091
- Queue after relaunch: 15 SFT + 48 eval (the 16th SFT already complete)

The first launch's job IDs (43326465, 43328180-43328388) were all cancelled and
should be ignored.

## Results: 4x4 sweep complete (2026-08-31)

All 16 SFT cells COMPLETED and merged; all 48 evals produced JSON. Scored with
`scripts/score_gtsft_lr_sweep.py` (routes through `compute_passk.build()`, the same
code path as the trajectory plots).

| arm | sft_lr | ID pass@1 | ID pass@32 | n=5 pass@1 | n=5 pass@32 | n=6 pass@1 | n=6 pass@32 |
|---|---|---|---|---|---|---|---|
| pt101B | 1e-5 | 0.2287 | 0.7613 | 0.0461 | 0.2985 | 0.0097 | 0.0853 |
| pt101B | **3e-5** | 0.2839 | 0.7854 | 0.0506 | 0.2866 | 0.0054 | 0.0585 |
| pt101B | 1e-4 | 0.2708 | 0.8084 | 0.0348 | 0.2281 | 0.0043 | 0.0389 |
| pt101B | 3e-4 | 0.2066 | 0.7633 | 0.0091 | 0.0626 | 0.0009 | 0.0063 |
| pt252B | 1e-5 | 0.2548 | 0.7723 | 0.0489 | 0.3145 | 0.0060 | 0.0929 |
| pt252B | **3e-5** | 0.2919 | 0.8044 | 0.0641 | 0.3152 | 0.0108 | 0.0907 |
| pt252B | 1e-4 | 0.3089 | 0.8275 | 0.0511 | 0.2781 | 0.0097 | 0.0577 |
| pt252B | 3e-4 | 0.2066 | 0.7533 | 0.0037 | 0.0504 | 0.0009 | 0.0028 |
| pt500B | 1e-5 | 0.2668 | 0.7974 | 0.0515 | 0.2909 | 0.0065 | 0.0739 |
| pt500B | **3e-5** | 0.3190 | 0.8235 | 0.0582 | 0.2879 | 0.0077 | 0.0725 |
| pt500B | 1e-4 | 0.2608 | 0.7844 | 0.0247 | 0.1442 | 0.0009 | 0.0125 |
| pt500B | 3e-4 | 0.2146 | 0.7733 | 0.0035 | 0.0225 | 0.0006 | 0.0031 |
| pt1007B | 1e-5 | 0.2648 | 0.8004 | 0.0561 | 0.3305 | 0.0097 | 0.0992 |
| pt1007B | **3e-5** | 0.3260 | 0.8185 | 0.0558 | 0.2617 | 0.0099 | 0.0631 |
| pt1007B | 1e-4 | 0.2909 | 0.7944 | 0.0123 | 0.0500 | 0.0006 | 0.0040 |
| pt1007B | 3e-4 | 0.2387 | 0.7613 | 0.0004 | 0.0121 | 0.0000 | 0.0009 |

**Matched LR: 3e-5** — best mean ID pass@1 across all four arms (0.3052 vs 0.2828 /
0.2538 / 0.2166), and best per-arm for 3 of 4 (pt252B marginally prefers 1e-4,
0.3089 vs 0.2919). Same choice as the 1B ladder, which is convenient for the
cross-size comparison. Do not mix per-arm bests — that confounds pretraining budget
with SFT LR.

Findings:

1. **ID pass@1 rises monotonically with pretraining at the matched LR**: 0.2839 ->
   0.2919 -> 0.3190 -> 0.3260 for 101B -> 252B -> 500B -> 1007B. Cleaner than the 1B
   ladder, which was non-monotonic — more arms and wider budget spacing help.
2. **High LR destroys OOD coverage, and worse the more pretrained the model.**
   At 3e-4 every arm collapses (n=5 pass@32 0.0121-0.0626 vs 0.26-0.33 at 1e-5). The
   damage at 1e-4 scales with pretraining: pt101B keeps n=5 pass@32 0.2281 but
   pt1007B falls to 0.0500 — a 6.6x loss vs its own 1e-5. Better-pretrained models
   are *more* fragile to over-training the terse answer-only format.
3. **ID pass@1 and OOD coverage prefer different LRs.** ID pass@1 peaks at 3e-5 (or
   1e-4 for pt252B) while n=5/n=6 pass@32 peak at **1e-5** for every arm
   (e.g. pt1007B n=5 0.3305 at 1e-5 vs 0.2617 at 3e-5; n=6 0.0992 vs 0.0631). If the
   downstream GRPO benefits from starting with more sampling diversity — plausible
   given the 1B result that GRPO *destroys* coverage — 1e-5 may be the better
   warmup despite lower pass@1. Worth one A/B before committing the GRPO stage.
4. ID pass@32 is ~0.75-0.83 everywhere and barely moves with LR, so it does not
   discriminate; the LR signal is in ID pass@1 and OOD coverage.

### Two operational bugs found

- One eval (`pt252B` lr1e-5 n=6, job 43340680) died with vLLM `EngineDeadError` at
  77% of generation, yet **sacct reported `COMPLETED` exit `0:0`** while writing no
  JSON. The CLAUDE.md warning about eval job State is worse than documented: it can
  be wrong in *both* directions. Trust file existence only. Resubmitted (43363203),
  COMPLETED in 6:49.
- **`eval.sh` did not recognise sharded merged checkpoints.** The guard was
  `[ -f model.safetensors ]`, but a merged 7B is `model-0000N-of-00003.safetensors` +
  `model.safetensors.index.json`. So all 48 evals ran two doomed `verl.model_merger`
  attempts (HFValidationError/OSError noise) before falling through to `eval.py`,
  and the "merged == already evaluated" heuristic silently breaks at >=7B. Fixed to
  accept either layout.

## Next

GRPO on top at lr 3e-6 / kl 3e-3 from the **3e-5** checkpoints, for the 1B-vs-7B
comparison — with the caveat in finding 3 that 1e-5 may be the better RL starting
point.

## Matching a 7B checkpoint to the 1B 4001B-token model (2026-09-01)

Question: which 7B pretraining checkpoint, after the same 200-step GT-SFT at its
best LR, matches the **1B pt4001B** post-SFT model on n=3,4 — on **both** pass@1
and pass@32?

Target (1B `pt4001B`, best LR 3e-5): **pass@1 0.2979 / pass@32 0.7984**.
Its best LR is also the 1B ladder's matched LR, so there is no
best-vs-matched ambiguity for the reference point.

Extra checkpoints downloaded for this: 126B, 151B, 202B (bracketing) plus 42B,
63B, 84B (see the mis-step note below). Swept at {1e-5, 3e-5, 1e-4} — 3e-4 dropped
because it lost on all four original arms by a wide margin.

`scripts/plot/plot_7b_vs_1b_match.py` → `figures/gtsft-lr-sweep/pass1_vs_pass32_7b_vs_1b_match.png`

| 7B arm | best LR | pass@1 | pass@32 | d pass@1 | d pass@32 | joint dist |
|---|---|---|---|---|---|---|
| 101B | 3e-5 | 0.2839 | 0.7854 | -0.0140 | -0.0130 | 0.0192 |
| **126B** | **3e-5** | **0.2869** | **0.7964** | **-0.0110** | **-0.0020** | **0.0112** |
| 151B | 3e-5 | 0.3109 | 0.8175 | +0.0130 | +0.0191 | 0.0231 |
| 202B | 3e-5 | 0.3049 | 0.8074 | +0.0070 | +0.0090 | 0.0114 |
| 252B | 1e-4 | 0.3089 | 0.8275 | +0.0110 | +0.0291 | 0.0311 |
| 500B | 3e-5 | 0.3190 | 0.8235 | +0.0211 | +0.0251 | 0.0327 |
| 1007B | 3e-5 | 0.3260 | 0.8185 | +0.0281 | +0.0201 | 0.0345 |

**Answer: 7B @ 126B tokens (`stage1-step30000-tokens126B`), SFT lr 3e-5.**

- **126B and 202B are a statistical tie** on joint distance (0.0112 vs 0.0114) and
  bracket the target from opposite sides (126B under on both, 202B over on both).
  The true crossing interpolates to ~150-190B tokens.
- 126B wins on pass@32 almost exactly (-0.0020); 202B wins on pass@1 (+0.0070).
  Prefer **126B**: its coverage match is near-exact, and it errs *below* the target,
  so a claim like "7B needs ~30x less pretraining to match 1B-at-4001B" stays
  conservative.
- Headline ratio: **4001B / 126B ~ 32x** less pretraining at 7B for equal post-SFT
  n=3,4 performance.

Caveats: single seed. `151B` breaks monotonicity (0.3109, above both neighbours),
so treat 126-202B as flat within noise rather than reading an ordering into it. The
whole seven-point spread is only 0.284-0.326 pass@1, so the optimum is shallow.

### Two process mistakes worth remembering

1. **Bracketed the wrong target first.** 42B/63B/84B were downloaded against an
   earlier target (1B pt1007B, 0.2758/0.7793), for which 7B-pt101B was already
   *above* and the crossing lay below 101B. When the target moved to pt4001B the
   crossing moved up into 101-252B and those three became strictly irrelevant — they
   were downloaded (~88 GB, still on disk, untrained) but correctly excluded from the
   sweep once flagged.
2. **Scored before the evals landed.** The first full run reported `pt151B` best
   LR = 1e-5 at pass@1 0.2608, an apparent outlier far below its neighbours. Cause:
   `plot_7b_vs_1b_match.py` *reads* `.passk` caches and does not build them, and the
   caches had been built while 151B's 3e-5/1e-4 evals were still running — so only
   one LR was visible. With complete caches 151B's best is 3e-5 at 0.3109. Lesson:
   build caches only after confirming the underlying JSONs exist, or make the reader
   build on demand.

### Update: 135B added (2026-09-01)

`stage1-step32000-tokens135B` downloaded and swept at {1e-5, 3e-5, 1e-4}:
pass@1 0.2447 / 0.2889 / 0.2768 → best lr 3e-5 at **0.2889 / 0.8024**.

New ranking by joint distance to the 1B pt4001B target (0.2979 / 0.7984):

| arm | best LR | pass@1 | pass@32 | joint dist |
|---|---|---|---|---|
| **135B** | 3e-5 | 0.2889 | 0.8024 | **0.0099** |
| 126B | 3e-5 | 0.2869 | 0.7964 | 0.0112 |
| 202B | 3e-5 | 0.3049 | 0.8074 | 0.0114 |
| 101B | 3e-5 | 0.2839 | 0.7854 | 0.0192 |
| 151B | 3e-5 | 0.3109 | 0.8175 | 0.0231 |
| 252B | 1e-4 | 0.3089 | 0.8275 | 0.0311 |
| 500B | 3e-5 | 0.3190 | 0.8235 | 0.0327 |
| 1007B | 3e-5 | 0.3260 | 0.8185 | 0.0345 |

135B is the new point estimate and has the nicest property of the three: it
**straddles** the target (pass@1 -0.0090, pass@32 +0.0040) rather than sitting
under on both (126B) or over on both (202B).

**Do not treat the ranking as resolved.** The top three span 0.0015 in joint
distance (0.0099 / 0.0112 / 0.0114) at one seed per cell; adding 135B changed the
winner without reducing the real uncertainty. The 151B anomaly (0.3109, above both
135B and 202B despite lying between them in tokens) is direct evidence that
~0.01-0.02 pass@1 swings are noise here.

Defensible claim: **the matching 7B checkpoint lies between ~126B and ~202B tokens
(~20-32x less pretraining than the 1B's 4001B)**, point estimate 135B. Further
token-budget resolution will not sharpen this — a second seed on the 126-202B arms
would.

Figures regenerated (both now include every LR, per request):
`figures/gtsft-lr-sweep/pass1_vs_pass32_7b_vs_1b_match.png` (28 7B points, colour =
tokens, marker = LR, large = arm's best LR) and `..._match_bestlr.png` (best-LR only,
8 points).
