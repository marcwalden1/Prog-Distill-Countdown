# Qwen3 base GRPO: size, rollout.n, and learning rate (2026-09-23 → 2026-09-28)

First GRPO runs on the **Qwen3 base** family, on `balanced-simple`. Five runs
across three axes: model size (0.6B / 1.7B / 4B), `rollout.n` (4 vs 8), and
learning rate (3e-6 vs 1e-6 at 1.7B).

Qwen3 needs no code changes — `transformers 4.51.1` ships `models/qwen3` and
`vllm 0.8.5.post1` registers `Qwen3ForCausalLM`. Vocab is 151,936, same as
Qwen2.5, so the default (non-Gemma) microbatch settings apply.

## Runs

All `DATA_SOURCE=balanced-simple` (so `DISABLE_CHAT_TEMPLATE=True`), kl=3e-3,
1635 steps, `train_batch_size=256`, `ppo_mini_batch_size=256` (1 optimizer step
per rollout), seed1, `save_freq=100`. All 16 checkpoints × 3 datasets evaluated
for every run (240/240 evals).

| model | lr | rollout.n | GPUs | micro/gpu | s/it | wall | train jid |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 3e-6 | 4 | 2 | 32 | 29.5 | 19.6 h | 47811563 |
| Qwen3-0.6B-Base | 3e-6 | 8 | 2 | 32 | 74.3 | 1 d 11.7 h | 47816172 |
| Qwen3-1.7B-Base | 3e-6 | 4 | 4 | 16 | ~31 | 14.2 h | 47811591 |
| Qwen3-1.7B-Base | 1e-6 | 4 | 4 | 16 | 22.6 | 14.6 h | 48380016 |
| Qwen3-4B-Base | 3e-6 | 4 | 4 | 16 | 61.8 | 23.3 h | 47999663 |

`VAL_KWARGS_N=4` is pinned on the n=8 run so its validation stays comparable
(it defaults to `rollout_n`). Confirmed: both 0.6B runs report a byte-identical
`val_before_train` mean@4 of 0.0457121370791821.

`val_before_train` mean@4 by size: 0.6B 0.0457, 1.7B 0.0657, 4B 0.0776 — base
Qwen3 emits `<answer>` from the `balanced-simple` prompt without SFT, so base
GRPO is viable here (unlike base gemma-3-270m).

## Results (unbiased pass@k, full prompt set, temp 0.6, n=32)

`analysis_data/idood_qwen3_base_grpo.csv` (240 rows), from
`scripts/export_idood_scores.py`. Figures from
`scripts/plot/plot_qwen3_size_comparison.py --preset {size,rolloutn,lr}` into
`figures/Qwen3-base-grpo/`, `figures/Qwen3-0.6B-rolloutn/` and
`figures/Qwen3-1.7B-Base/Qwen3-1.7B-lr/`.

**The `size` figures use the best available LR per model**, not a matched one:
1.7B at lr=1e-6, 0.6B and 4B at lr=3e-6 (updated 2026-09-28). lr=3e-6 collapses
OOD on the 1.7B (Finding 1), so a shared-LR figure understates it. Every series
label and the suptitle carry the LR so the figure is not misread as lr-matched;
the remaining confound is that 0.6B and 4B have no lr=1e-6 run to choose from.

| run | dataset | p@1 s100 | p@1 best | p@1 s1600 | p@32 best | p@32 s1600 |
|---|---|---|---|---|---|---|
| 0.6B lr3e-6 n4 | ID n=3,4 | 0.324 | 0.766 @1500 | 0.750 | 0.947 @1500 | 0.911 |
| 0.6B lr3e-6 n4 | OOD n=5 | 0.061 | 0.237 @1600 | 0.237 | 0.598 @1500 | 0.541 |
| 0.6B lr3e-6 n4 | OOD n=6 | 0.014 | 0.041 @1600 | 0.041 | 0.199 @1500 | 0.184 |
| 0.6B lr3e-6 n8 | ID n=3,4 | 0.453 | 0.807 @1500 | 0.714 | 0.929 @1500 | 0.894 |
| 0.6B lr3e-6 n8 | OOD n=5 | 0.098 | 0.280 @1200 | 0.202 | 0.578 @1400 | 0.456 |
| 0.6B lr3e-6 n8 | OOD n=6 | 0.019 | 0.062 @1400 | 0.023 | 0.234 @1400 | 0.098 |
| 1.7B lr3e-6 n4 | ID n=3,4 | 0.510 | 0.854 @1400 | 0.760 | 0.970 @1400 | 0.935 |
| 1.7B lr3e-6 n4 | OOD n=5 | 0.121 | 0.295 @600 | 0.146 | 0.622 @1400 | 0.465 |
| 1.7B lr3e-6 n4 | OOD n=6 | 0.029 | 0.073 @600 | 0.017 | 0.273 @600 | 0.118 |
| 1.7B lr1e-6 n4 | ID n=3,4 | 0.199 | 0.826 @1600 | 0.826 | 0.972 @1500 | 0.966 |
| 1.7B lr1e-6 n4 | OOD n=5 | 0.025 | 0.283 @1600 | 0.283 | 0.633 @1600 | 0.633 |
| 1.7B lr1e-6 n4 | OOD n=6 | 0.006 | 0.069 @1200 | 0.057 | 0.303 @1200 | 0.262 |
| 4B lr3e-6 n4 | ID n=3,4 | 0.710 | 0.900 @1300 | 0.883 | 0.988 @700 | 0.987 |
| 4B lr3e-6 n4 | OOD n=5 | 0.273 | 0.370 @800 | 0.321 | 0.724 @700 | 0.715 |
| 4B lr3e-6 n4 | OOD n=6 | 0.087 | 0.113 @500 | 0.060 | 0.394 @500 | 0.323 |

### Finding 1 (headline): the 1.7B's OOD collapse is an LR artifact

At lr=3e-6 the 1.7B peaks OOD at **step 600** and then falls below its own
step-100 checkpoint (n=6 pass@1 0.029 → 0.073 @600 → 0.017 @1600), while in-dist
keeps climbing to step 1400. At lr=1e-6, same model / data / kl / n / seed /
GPUs / micro-batch, that does not happen: OOD n=5 is **still rising at step 1600**
and the final checkpoint **Pareto-dominates** lr=3e-6 on every dataset and both
metrics:

| | ID p@1 | OOD n=5 p@1 | OOD n=6 p@1 | ID p@32 |
|---|---|---|---|---|
| lr=3e-6 @1600 | 0.760 | 0.146 | 0.017 | 0.935 |
| lr=1e-6 @1600 | **0.826** | **0.283** | **0.057** | **0.966** |

lr=1e-6 starts far slower (ID p@1 0.199 vs 0.510 at step 100) and simply never
turns over, reaching nearly the same peak ID accuracy (0.826 vs 0.854) with
roughly double the OOD.

**This supersedes the earlier "bigger overfits harder" reading.** Ranking the
three sizes at a single shared lr=3e-6 confounded scale with *how far past its
stable LR each model was*. The 1.7B was not anomalous as a model; 3e-6 is too
high for it on this task. A fair size comparison needs a per-size LR sweep,
which we do not have for 0.6B or 4B.

### Finding 2: late-training OOD decay is near-universal at lr=3e-6

Every lr=3e-6 run peaks OOD before step 1600 and gives some back, earlier the
larger the model (0.6B still rising @1600; 1.7B peaks @600; 4B @500-800). The
give-back is mild for the 4B (n=5 0.370 → 0.321) and catastrophic for the 1.7B.
The lr=1e-6 run is the only arm without it. **Do not default to reporting step
1600** — for most of these runs it is not the best checkpoint, on either metric.

### Finding 3: rollout.n=8 wins at peak, loses at step 1600

n=8 beats n=4 at each arm's own best checkpoint on all three datasets (ID 0.807
vs 0.766; n=5 0.280 vs 0.237; n=6 0.062 vs 0.041) but is worse at step 1600 on
all three (n=6 0.023 vs 0.041). It peaks earlier (1200-1500) and then drops
hard. Doubling `rollout.n` accelerates both the learning and the collapse —
consistent with n=8 being further along an effective-progress axis per step,
which also matches its ~2.5x cost per step. **Worth it only with early
stopping**; at a fixed 1600 steps it is strictly worse *and* cost ~1.8x the
wall-clock (35.7 h vs 19.6 h).

### Finding 4: the 4B's in-dist is saturating

4B in-dist pass@32 is 0.988 by step 700 and 0.987 at 1600. `balanced-simple`
n=3,4 is close to ceiling at that scale, so in-dist comparisons at 4B are
running out of headroom. Its step-100 OOD n=5 (0.273) already beats every other
run's *final* number — much of its advantage is pretrained, not earned by GRPO.

Mechanism, from the `pass1_vs_pass32` figures: in-dist all models ride nearly the
same pass@1/pass@32 curve with the headroom gap closing — GRPO converting
coverage the base model already had into first-sample accuracy. The collapsing
arms move *backwards* along that curve on n=6 late in training.

**Caveats:** single seed throughout; n=6 differences are a few points on 3,519
prompts (0.062 vs 0.041 is ~74 prompts).

## Infra fixes and gotchas this round

- **`VLLM_CACHE_ROOT` race (fixed).** vLLM's torch-compile cache defaults to
  `~/.cache/vllm` on NFS and is keyed only by the model graph hash, so two
  concurrent jobs on the same model collide:
  `OSError: [Errno 116] Stale file handle` at engine init. Killed job 47811579.
  All 11 GPU scripts now export a node-local per-job `VLLM_CACHE_ROOT`. Details
  in CLAUDE.md.
- **`eval.sh`'s 1:30 default is too short for base-model GRPO evals.** These runs
  produce long rollouts: n=5 (4620 prompts) takes ~1:50-2:20, n=6 (3519) ~1:26.
  42 of the first 96 eval tasks TIMEOUT'd. Pass `EVAL_SBATCH_ARGS="-t 6:00:00"`.
  The measured cost table in `2026-09-15_passk-grid-and-temp1.0-sweep.md` is for
  ~20-token SFT'd responses and does **not** transfer to base-model runs.
- **The GPUs are 80 GB, not 96 GB.** CLAUDE.md's hardware table says
  `nvidia_h100_80gb_hbm3 (96 GB)`; the OOM traceback reports 79.17 GiB total.
  Size memory against 80 GB.
- **OOM from response-length growth, not a constant config.** The 1.7B lr=1e-6
  run at micro=32 ran 106 steps then died in `loss.backward()`. Cause: train
  `response_length/mean` went 412 → 746 tokens between steps 50 and 100, and
  `ppo_micro_batch_size_per_gpu` is a fixed *sequence* count, so activation
  memory nearly doubled under an unchanged config. A micro-batch that fits early
  can still OOM late. `USE_DYNAMIC_BSZ=True` is the real fix (token-budgeted
  micro-batches) but changes token-mean weighting, so it breaks comparability.
- **Micro-batch must divide the normalized mini-batch.** verl asserts
  `ppo_mini_batch_size (256 after normalization) % ppo_micro_batch_size_per_gpu == 0`.
  The normalized value is `256 * rollout.n / n_gpus` = 256 here, so **only powers
  of two are legal** — `PPO_MICRO=24` fails at startup (job 48372633, 5 min).
- **verl auto-resumes.** `resume_mode` defaults to `auto` and `train_grpo.sh`
  never overrides it, so resubmitting under the same `EXP_NAME` silently resumes
  from the newest checkpoint — which after a failed run means a mixed-config
  run. Move or delete the exp dir before a clean restart.
- `kempner_requeue` preempts and requeues eval arrays; `eval.sh` has no
  skip-if-output-exists, so a preempted task redoes its generation (merged
  checkpoints are reused, so only generation time is lost).
- W&B size tag: `${MODEL_NAME##*-}` gave `Base` for `Qwen3-0.6B-Base`; now strips
  the `-Base` suffix first (`train_grpo.sh`, `plot_results.py`).

## Open

- **Per-size LR sweep** is the obvious next step — lr=1e-6 for 0.6B and 4B, to
  redo the size comparison without the LR confound (Finding 1).
- Seed 2 for anything we want to claim.
- Does lr=1e-6 also fix the n=8 collapse? (0.6B lr1e-6 n8.)
- Whether the OOD collapse tracks a response-length or entropy change
  (`plot_entropy_length.py`) — same diagnostic used for the OLMo-2 ladder. The
  length data is already in the training logs (412 → 746 over 100 steps on the
  1.7B), so this is cheap.
- Leftover: `checkpoints/Qwen3-1.7B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1.oom-micro32-step100`
  (20 GB on netscratch) from the failed micro=32 attempt — safe to delete.
