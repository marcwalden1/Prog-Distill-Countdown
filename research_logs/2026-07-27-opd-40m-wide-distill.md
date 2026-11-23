# 2026-07-27 — Offline OPD on the 40M-wide distill student (first OPD run)

First actual submission of the offline-OPD pipeline built on 2026-07-09
(`research_logs/2026-07-09-offline-opd.md`). Student is a **from-scratch
~40M-body** model, not the 0.5B originally planned.

## Setup

| Piece | Value |
|---|---|
| Student (π_ref, OPD init) | `sft-checkpoints/Qwen2.5-40M-wide/balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1` (bingbin tree) |
| Teacher | `Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1` @ **step 1600**, read-only from sdholakia's tree (`TEACHER_CHECKPOINT_DIR`) — already merged, no write to that tree |
| Teacher consistency | ✓ that student's SFT parquet is `sft-data/Qwen2.5-40M-wide/teacher-Qwen2.5-1.5B-balanced-grpo-lr3e-6-kl3e-3-seed1-n4/step_1600.parquet` (distill = teacher-final data) |
| Tokenizer | shared Qwen vocab (40M embeddings copied verbatim from Qwen2.5-0.5B); chat_template present, so `generate_opd_data.py`'s Qwen-only guard passes |
| Rollouts | 10,000 prompts × n=4 @ temp 1.0, max 1024 new tokens → ~40k rows (defaults) |
| OPD training | 1600 steps, batch 64, clip 10.0, warmup 0.1 cosine, seed 1, save/test every 50 |
| OPD LR | **two arms: 2e-6 and 2e-5** |

LR rationale: 2e-6 is the paper/default value, chosen with the 0.5B student
(SFT LR 3e-5) in mind. This student's SFT LR is 3e-4 — 10× higher — so 2e-6 may
barely move it; 2e-5 scales with that gap. Bingbin's call: run both.

Data namespace (shared by both arms — gen-data runs **once**):

    opd-data/Qwen2.5-40M-wide/student-balanced-distill-t1p5b-sftlr3e-4-steps6220-seed1-teacher-Qwen2.5-1.5B-balanced-grpo-lr3e-6-kl3e-3-seed1-step1600-n4-temp1.0-p10000/opd.parquet

## Jobs (submitted 2026-07-27)

| Stage | Arm lr=2e-6 | Arm lr=2e-5 |
|---|---|---|
| gen-data (shared) | `35491577` | — (depends on the same job) |
| OPD train | `35491578` | `35491739` |
| eval balanced / 5 / 6 | `35491579` / `35491580` / `35491583` | `35491740` / `35491742` / `35491744` |
| plot | `35491584` | `35491745` |

`EXP_NAME`s: `balanced-opd-from-distill-t1p5b-sftlr3e-4-opdlr{2e-6,2e-5}-seed1`.

Arm 1 went through `scripts/run_pipeline.sh --opd balanced-grpo-lr3e-6-kl3e-3-seed1`.
Arm 2 was hand-submitted (`train_opd.sh` + eval×3 + plot) with
`--dependency=afterok:35491577` so it reuses arm 1's parquet instead of
generating a second, differently-sampled rollout set — `run_pipeline.sh`'s
reuse check is a file-existence test, so submitting arm 2 through the pipeline
before the parquet existed would have queued a duplicate gen-data job writing
to the same path. Submission script kept only in the session scratchpad; the
recipe is the table above.

## What to compare against

- The SFT student itself (OPD step 0) — the pre-OPD baseline.
- `checkpoints/Qwen2.5-40M-wide/balanced-distill-t1p5b-sftlr3e-4-steps6220-grpo-lr1e-6-kl3e-3-seed1`
  — GRPO-on-top of the same student (job 29788968, launched 2026-07-09), the
  RL alternative to OPD from an identical starting point.

Metrics: mean@1 / mean@32 on `balanced` (ID) and `balanced5`/`balanced6` (OOD),
across the 10-point eval curve (steps 150–1600).

## Watch items

- Offline-OPD bias grows as π_θ drifts from π_ref; 1600 steps is ~10× the
  paper's proportional budget, so mid-curve regression is expected to be
  visible in the eval curve. Checkpoints every 50 steps.
- Rollouts are unfiltered by correctness (distribution matching); per-row
  grader scores are stored, so the rollout accuracy breakdown printed by the
  gen-data job is the reference for how weak π_ref is.

## Rollout stage output (40k rollouts from π_ref, temp 1.0, ID training prompts)

| Outcome | Count | Share |
|---|---|---|
| correct (1.0) | 27,873 | 69.7% |
| format-only, wrong answer (0.1) | 7,692 | 19.2% |
| no `<answer>` tag (0.0) | 4,435 | 11.1% |

π_ref is a genuinely competent policy. Not comparable to held-out eval numbers
(these are training prompts at temp 1.0, vs. unseen puzzles at temp 0.6).

## Bug found + fixed: numpy ints break the teacher-scoring stage

The first gen-data job (`35491577`) completed stage 1 and then died at the start
of stage 2:

    msgspec.ValidationError: Expected `int`, got `ext` - at `$[1][0]`
      vllm/v1/engine/core.py:521 in process_input_socket → serial_utils.py:197

**Cause.** `run_score` built its `TokensPrompt`s with `[list(x) for x in
df["prompt_ids"]]`. Parquet returns numpy arrays, so `list(...)` yields
`numpy.int64` elements. vLLM's V1 engine ships requests over zmq through
msgspec; numpy scalars get encoded via vLLM's ext hook, and the decoder on the
engine side — typed as `list[int]` — rejects them. The rollout stage is immune
because its IDs come straight from `tokenizer.encode` as Python ints. This is
also why the 2026-07-09 login-node smoke test left behind only a
`*.rollouts.parquet` under `opd-data/Qwen2.5-0.5B/smoke-opd/` — it died in the
same place; the validation that day covered the dataset/loss, not this stage.

**Fixes** (applied in the working tree; `scripts/generate_opd_data.{py,sh}` and
`research_logs/` are still untracked in git):
1. `scripts/generate_opd_data.py` — cast to Python ints:
   `[[int(t) for t in x] for x in df[...]]`. Verified against the real parquet
   on the login node: old path fails an `msgspec` `list[int]` roundtrip
   (`TypeError: Encoding objects of type numpy.int64 is unsupported`), new path
   passes.
2. `scripts/generate_opd_data.sh` — **skip stage 1 if
   `<output>.rollouts.parquet` already exists.** The score stage deletes that
   file only on success, so a stage-2 failure is now retryable without
   re-sampling — and without silently swapping in a different rollout set.

**Failure mode to know:** the dead engine thread left the job in `RUNNING` with
no progress (same zombie shape as the gemma SFT OOM note in CLAUDE.md). It would
have burned the full 8h walltime. Check gen-data logs for actual progress, not
job State.

## Bug 2: student samples token ids past the tokenizer's vocabulary

Retry `35495698` got past msgspec and then died in vLLM's input validation:

    ValueError: Token id 151821 is out of vocabulary
      vllm/v1/engine/processor.py:354 in _validate_model_input

**Cause.** Not a teacher/student tokenizer mismatch — both tokenizers are
identical (`len=151665`, `max_token_id=151664`). Qwen's *embedding matrix* is
padded to `vocab_size=151936`, so ~271 rows have no tokenizer entry but are
still in the softmax. The from-scratch student occasionally samples them; vLLM
validates input ids against `tokenizer.max_token_id` (the score stage feeds
rollout tokens back in as *input*, which is why stage 1 never tripped on this).

Extent: **32 of 40,000 rollouts (0.080%)**, 32 of 33.7M tokens (0.00009%), 30
distinct ids, all in 151,691–151,935. Affected rows skew *correct* (24/32 score
1.0) — a single stray token in an otherwise-fine response.

**Fix.** `run_score` drops rollouts containing ids `> max(teacher_tok.get_vocab().values())`
(the same quantity vLLM checks), with a loud WARNING and count. Chosen over
`skip_tokenizer_init=True` (which would bypass validation): those embedding rows
were never trained, so the teacher's logprob for them is arbitrary rather than
useful signal, and 0.08% of rows is not worth the V1-engine risk. 39,968 rows
remain.

Both fixes were dry-run against the real parquet on the login node (no GPU):
filter drops exactly 32, max surviving id 151,660 ≤ 151,664, and a real
`prompt_ids + response_ids` request survives an `msgspec` `list[int]` roundtrip.

Also pre-checked, so it won't be the third failure: sequence lengths are
prompt ≤ 165, response ≤ 1036, **total ≤ 1201** — comfortably inside the
teacher's `max_model_len=2048` and the OPD trainer's `data.max_length=2048`
(zero rows truncated).

## π_ref is degenerate: 69% of rollouts hit the length cap

Worth knowing before reading any OPD result. Response-length stats:

| Rollout class | n | median len | at/near 1024-token cap |
|---|---:|---:|---:|
| correct (1.0) | 27,873 | 1036 | 58.0% |
| format-only (0.1) | 7,692 | 1036 | 93.6% |
| no answer tag (0.0) | 4,435 | 1036 | 95.9% |
| **all** | 40,000 | 1036 | **69.1%** |

The student emits an answer and then repeats the whole
`<answer> … </answer>\n= 111 - 60\n= 51\n` block until truncation (plus
occasional junk tokens like `ágenes`). It still scores 1.0 because
`compute_score` extracts the **last** `<answer>` match. Same perseveration /
diversity-collapse family as `2026-07-08-generation-evolution-across-grpo.md`.

Implication for this experiment: most of the ~34M rollout tokens are repetition
that the teacher assigns very low probability to, so `clamp(log π_T − log π_θ,
±10)` will saturate at −10 over a large fraction of positions and the OPD
gradient will be dominated by "stop repeating" rather than "reason better".
That may be a fine thing for OPD to fix, but it is not the same experiment as
OPD on a well-behaved 0.5B student. If the curves are flat or weird, this is the
first thing to look at — candidate follow-ups: shorter `MAX_LENGTH` for the
rollouts, or repetition-aware filtering before scoring.

## Recovery

The rollouts parquet survived every attempt (28 MB, 40k rows) — only stage 2 was
ever redone.

**After bug 1:** resubmitted gen-data as `35495698` and repointed the two
pending train jobs with `scontrol update jobid=<id> Dependency=afterok:35495698`,
then cancelled the zombie. Order matters: `scancel` first would have left them
`DependencyNeverSatisfied`. That worked — but `35495698` then hit bug 2 and
**FAILED**, which tore down both chains anyway (`afterok` on a failed job
cancels all 10 dependents).

**After bug 2:** resubmitted everything from scratch — one shared gen-data job +
two full train/eval/plot chains. Recipe kept in the table below; a copy of the
submission script is in the session scratchpad, but it is short enough to
retype from the job table.

### Final job IDs (2026-07-27, third submission)

| Stage | Arm lr=2e-6 | Arm lr=2e-5 |
|---|---|---|
| gen-data (shared) | `35496222` | same job |
| OPD train | `35496223` | `35496228` |
| eval balanced / 5 / 6 | `35496224` / `35496225` / `35496226` | `35496229` / `35496230` / `35496231` |
| plot | `35496227` | `35496232` |

Superseded (do not look for results under these): `35491577` (zombie),
`35495698` (failed), `35491578`/`35491739` + their evals/plots (cancelled).

## Lessons for the next OPD run

1. **`run_pipeline.sh --opd` cannot submit a second arm sharing one dataset.**
   Its reuse check is `[ -f <final parquet> ]`, so any arm submitted before the
   first one finishes queues a duplicate gen-data job writing the same path.
   Either wait for gen-data, or hand-wire extra arms with
   `--dependency=afterok:<gendata jid>`.
2. **`afterok` makes a gen-data failure cost the whole chain.** Two failures ⇒
   two full resubmissions. For a first-ever run of a pipeline stage, consider
   submitting gen-data alone and only chaining training once the parquet exists.
3. **The score stage had never actually run before today.** The 2026-07-09
   "validated on login node" claim covered `OPDDataset` and the loss branch, not
   stage 2 — and the leftover `opd-data/Qwen2.5-0.5B/smoke-opd/*.rollouts.parquet`
   was the visible evidence of that (a successful score stage deletes it).
   Both bugs were GPU-path-only and would have been caught by scoring ~10 rows.

## Teacher logprob distribution — OPD cannot fix the perseveration

Measured on the finished parquet (33.67M scored tokens, mean logprob −0.1810):

| Response positions | mean teacher logprob | ≈ p/token |
|---|---:|---:|
| 0–100 | −0.6160 | 0.540 |
| 400+ (the repetitive tail) | −0.1154 | **0.891** |
| all | −0.1810 | 0.834 |

88.1% of all tokens have teacher logprob > −0.1; only 0.72% are below −5 and
0.333% below −10 (the clamp). **The teacher scores the repetitive tail better
than the meaningful opening** — conditioned on a repeating prefix, more
repetition is the easy prediction.

This corrects the expectation written above: the advantage does *not* saturate
at −10 over the filler. It is near zero across most of the sequence, so OPD has
almost no gradient with which to fix perseveration — the teacher likes the
repetition too. OPD's usable leverage is the first ~100 tokens, where the
teacher is genuinely uncertain (p≈0.54).

Per-score-class mean teacher logprob/token: score 1.0 → −0.2282,
0.1 → −0.3006, **0.0 → −0.1671**. The teacher is *most* comfortable with the
no-answer-tag rollouts: response quality and teacher likelihood are close to
decoupled on this dataset.

Follow-up if the curves are flat: truncate rollouts at the first complete
`<answer>` block before teacher scoring, so the frozen set isn't 69%
self-similar filler. That is a data-side change, not a knob on this run.

## Training results (both arms COMPLETED, ~36 min each on 1×H100)

624 steps/epoch (39,968 / 64), 1600 steps ≈ 2.56 epochs, ~1.12 s/step.
32 checkpoints each (`global_step_{50..1600}`).

| Arm | val/loss @50 | val/loss @1600 | train/loss @1600 |
|---|---:|---:|---:|
| lr=2e-6 (`35496223`) | −0.0962 | −0.0544 | −0.0464 |
| lr=2e-5 (`35496228`) | — | −0.0470 | −0.0360 |

Reading the loss: it is `advantage × CE` with `advantage = clamp(log π_T −
log π_θ, ±10)` and `CE = −log π_θ ≥ 0`. Rollouts come from the student, so the
student assigns higher probability to them than the teacher does → advantage
negative on average → **loss is expected to be negative**; it is roughly a
negative reverse-KL estimate, not a quality metric. Moving *toward zero* means
the student is converging on the teacher: arm 1 closed ~44% of the gap, and
lr=2e-5 ended nearer zero than lr=2e-6. Both arms trained; 2e-6 was not inert.

LR scheduler verified healthy (verl patch 1a): at step 50, arm 1 logged
`train/lr(1e-3)=0.000625` = 6.25e-7 = exactly `2e-6 × 50/160` on the
10%-warmup ramp.

## RESULTS — OPD beats GRPO-on-top on ID by a wide margin

All 60 eval tasks COMPLETED, 10/10 checkpoint coverage per arm per dataset.
Step-1600 comparison, all from the **same** SFT student:

| Setup | compute | ID n=3/4 mean@1 | ID mean@32 | n=5 mean@32 | n=6 mean@32 |
|---|---|---:|---:|---:|---:|
| SFT student (π_ref, step 0) | — | 0.5336 | 0.5291 | 0.0644 | 0.0669 |
| GRPO on top (lr1e-6 kl3e-3) | 1635 steps, 2×H100 | 0.6881 | 0.7021 | 0.0833 | 0.0844 |
| **OPD lr=2e-6** | 1600 steps, 1×H100, 36 min | 0.7222 | 0.7511 | 0.0956 | 0.0920 |
| **OPD lr=2e-5** | 1600 steps, 1×H100, 36 min | **0.7743** | **0.7892** | **0.0986** | **0.0933** |

**Both OPD arms beat GRPO-on-top on every dataset**, at roughly a quarter of the
GPU-hours (36 min × 1 GPU vs 1635 GRPO steps × 2 GPUs). Best arm is +0.086
mean@1 / +0.087 mean@32 over GRPO on ID, and +0.241 mean@1 over π_ref.

ID mean@1 curves (n=3/4):

    step        150    300    450    600    750    900   1050   1200   1400   1600
    lr=2e-6  0.5396 0.5406 0.6309 0.6971 0.7172 0.7141 0.7262 0.7252 0.7252 0.7222
    lr=2e-5  0.7272 0.7031 0.7362 0.7262 0.7452 0.7723 0.7563 0.7583 0.7613 0.7743

- **No mid-curve regression.** The 2026-07-09 worry that a 1600-step budget
  (~10× the paper's proportional budget) would drift far enough from π_ref for
  offline-OPD bias to hurt did **not** materialize: both curves rise and
  plateau, and step 1600 is at or near the max for both arms.
- **lr=2e-5 saturates fast** — 0.7272 mean@1 by step 150, already above GRPO's
  step-1600 value. Most of the gain is available in <5 minutes of training.
- lr=2e-6 needs ~600 steps to get there, confirming the LR concern that
  motivated running two arms: the default would have understated OPD badly at a
  smaller step budget.

**OOD is still ~nil, and mean@32 hides that.** n=5 mean@1 is 0.001–0.003 and
n=6 mean@1 is **exactly 0.0000 at every checkpoint of every arm**. Since raw
grader scores are 0 / 0.1 / 1.0, an OOD mean@32 of ~0.09 means "nearly every
sample earns 0.1 format credit, almost none is correct". The OOD improvements
above are gains in *format validity*, not in solving bigger puzzles. Do not
report OOD mean@32 as OOD accuracy for this model.

### Perseveration: partially dented, not fixed

Mean response length on `balanced` (cap 1024):

| Setup | length |
|---|---:|
| SFT student | 981.0 |
| GRPO on top (step 1600) | 975.0 (450→1600: 978→975) |
| OPD lr=2e-5 (step 1600) | 866.2 (plateaus ~850–866 from step 150) |

OPD cut ~12% off the length; **GRPO changed it by ~0.6%**. So the prediction
above (OPD has little gradient against repetition because the teacher scores the
repetitive tail at p≈0.89) was directionally right — the repetition largely
survives, still ~860 of a 1024 cap — but OPD did move it, and moved it far more
than GRPO did. The accuracy gain is therefore not explained by fixing
perseveration; it comes from the first ~100 tokens where the teacher is actually
informative.

(`.lengths` for the lr=2e-6 arm land when its plot job `35496227` finishes.)

### Caveats

- The GRPO-on-top baseline has only 5/10 ID checkpoints evaluated
  (450/600/900/1400/1600) — a pre-existing gap in that run's eval coverage, not
  from this work. The step-1600 comparison is apples-to-apples.
- Single seed (seed 1) throughout; no error bars.
- Eval is temp 0.6, n=32, held-out puzzles — not comparable to the temp-1.0
  training-prompt rollout accuracy reported earlier in this log.

Figures: `figures/Qwen2.5-40M-wide/balanced-opd-from-distill-t1p5b-sftlr3e-4-opdlr{2e-6,2e-5}-seed1/eval_n{34,5,6}_temp0.6.png`.

## Status

- 2026-07-27: **complete.** Two bugs fixed in the never-before-exercised teacher
  scoring stage; gen-data + both OPD arms + 60 evals all COMPLETED; plots
  running. Headline: OPD > GRPO-on-top on ID at ~4× less compute, OOD still ~0
  real accuracy.

## Group 2 (2026-07-27, later): 10 epochs × LR {1e-5, 2e-5, 5e-5}

Same frozen parquet (39,968 rows), `OPD_TRAIN_STEPS=6250` = 10.02 epochs
(624 steps/epoch), `OPD_SAVE_FREQ=OPD_TEST_FREQ=250` (25 ckpts/run instead of
125), eval array `5,15,25,35,50,65,80,95,110,125` → steps 250…6250.
Jobs: train `35571097`/`35571104`/`35571109`, all COMPLETED, ~2.2 h each.
No gen-data job — `run_pipeline.sh --opd` reused the parquet.

### ID (n=3/4) — more epochs clearly help; LR barely matters

| Setup | epochs | ID mean@1 (final) | ID mean@1 (best) | ID mean@32 (best) |
|---|---|---:|---:|---:|
| SFT student π_ref | — | 0.5336 | — | 0.5291 |
| GRPO on top | 1 (418k prompts) | 0.6881 | — | 0.7021 |
| OPD lr=2e-6 | 2.56 | 0.7222 | 0.7262 | 0.7511 |
| OPD lr=2e-5 | 2.56 | 0.7743 | 0.7743 | 0.7892 |
| **OPD lr=1e-5** | **10** | 0.7874 | 0.8024 (@5500) | 0.8144 (@4750) |
<!-- lr=1e-5 full curve (all 10 pts after the eval reruns below):
     step    250    750   1250   1750   2500   3250   4000   4750   5500   6250
     mean@1  .5587  .7392 .7412  .7573  .7773  .7783  .7964  .7823  .8024  .7874
     mean@32 .5916  .7594 .7685  .7863  .8028  .8097  .8089  .8144  .8143  .8139 -->
| **OPD lr=2e-5** | **10** | 0.7934 | 0.7984 (@4750) | **0.8216** (@4750) |
| **OPD lr=5e-5** | **10** | **0.8034** | 0.8034 (@6250) | 0.8114 (@4750) |

- **10 epochs > 2.56 epochs by ~+0.03 mean@1 / +0.032 mean@32.** Best OPD now
  beats GRPO-on-top by **+0.115 mean@1** and **+0.119 mean@32**.
- **LR is nearly irrelevant at this budget.** All three arms land within 0.016
  mean@1 / 0.013 mean@32. LR only sets the *speed*: mean@1 at step 250 is
  0.5587 / 0.7051 / 0.7212 for 1e-5 / 2e-5 / 5e-5. By ~step 2500 they converge.
- mean@32 plateaus around step 4750 for all three; mean@1 wobbles ±0.01 after.
  Nothing regresses — no offline-OPD collapse even at 10 epochs on frozen data.

### OOD — completely insensitive to epochs and LR

`balanced6` **mean@1 is exactly 0.0000 at all 30 checkpoints of all three
arms** (as in every earlier run, and in GRPO). `balanced5` mean@1 stays
0.0009–0.0035. mean@32 sits at ~0.099 (n=5) / ~0.097 (n=6), a hair above the
2.56-epoch runs, but that is format credit, not correctness. **10× the training
does nothing for OOD.**

### Perseveration rebounds slightly with longer training

Mean response length on `balanced` (cap 1024; π_ref 981.0, GRPO@1600 975.0):

| arm | @250 | @750 | @2500 | @6250 |
|---|---:|---:|---:|---:|
| lr=1e-5 | 985 | 840 | 868 | 874 |
| lr=2e-5 | 904 | 842 | 882 | 884 |
| lr=5e-5 | 854 | 872 | 877 | 883 |

All arms dip to ~840–870 early, then **creep back up** ~30–40 tokens. Longer
OPD does not further suppress the repetition — consistent with the teacher
assigning the repetitive tail p≈0.89.

### val/loss vs accuracy: decoupled again, more starkly

Final val/loss: **1e-5 → −0.0458, 2e-5 → −0.0332, 5e-5 → −0.0203**. Higher LR
gets monotonically closer to the teacher in loss terms, yet accuracy is
essentially identical across the three (0.787–0.803 mean@1). 5e-5 has 2.3×
smaller |val/loss| than 1e-5 for no accuracy gain. Further confirmation that
val/loss measures teacher agreement on π_ref's stale rollouts, not task skill —
do not use it to pick LR or checkpoints.

### Eval reliability note (3/90 tasks lost, then recovered)

All 90 eval tasks reported SLURM `COMPLETED`, but the lr=1e-5 arm was missing
3 result JSONs (`balanced` step 4000; `balanced5` steps 250, 6250). Root cause
was **not** a bad checkpoint (all 25 dirs are complete): it is
`Triton compilation failed: triton_poi_fused_cat_4` at vLLM engine init — the
same broken-triton-C-extension issue CLAUDE.md documents for gemma evals in
this conda env, here firing *intermittently* (3/90) rather than always, since
these runs share one torch.compile cache keyed on an identical model config.
Reran those 3 with the documented workaround
(`VLLM_USE_V1=0 TORCHDYNAMO_DISABLE=1 EVAL_EXTRA_ARGS="--enforce_eager"`):
jobs `35605891`, `35605892`, replot `35605893` — **all COMPLETED, coverage now
10/10 on every dataset for every arm.** The recovered points changed nothing:
lr=1e-5 `balanced` step 4000 = 0.7964/0.8089 (its curve max is still 0.8024
mean@1 @5500), `balanced5` steps 250 = 0.0017/0.0767 and 6250 = 0.0019/0.0987.
The eager path works reliably; it is just ~5–6× slower, which is irrelevant for
a handful of tasks on a 178M model.

**Reminder confirmed the hard way:** `sacct` State is not evidence an eval
produced output. Always count result JSONs.

### Stacked accuracy-bar figure (2026-07-28)

Added the OPD lr=5e-5 run (step 6250 = its final checkpoint, so the bar carries
no step annotation) to `figures/Qwen2.5-40M/accuracy_bars_stacked.png` as a third
wide bar, next to `40M-wide distill+GRPO` — both start from the *same* SFT
student, so those two bars isolate GRPO vs OPD. Regenerated with
`sbatch --account=bingbin_lab scripts/plot_accuracy_bars.sh Qwen2.5-40M`
(job `35779091`, 38 s). Two additions to `scripts/plot_accuracy_bars.py`: the new
bar, and an argv substring filter on `figures_dir` so one figure can be refreshed
without rewriting the gemma / 0.5B ones. New wrapper `scripts/plot_accuracy_bars.sh`
mirrors `plot_passk.sh` (serial_requeue, 8 CPU, 64 GB) — **note the account must be
`bingbin_lab`, not `kempner_bingbin_lab`, for serial_requeue.**

**This figure is pass@1 / pass@32, not mean@1 / mean@32** (solid = pass@1,
striped = pass@32 − pass@1). That distinction matters, because it reverses one
of the conclusions above:

| run (step) | n=3,4 pass@1/pass@32 | n=5 | n=6 |
|---|---|---|---|
| Qwen2.5-1.5B teacher (1600) | 0.851 / 0.958 | 0.316 / 0.687 | 0.058 / 0.296 |
| 40M-wide distill+GRPO (1600) | 0.688 / 0.953 | 0.002 / 0.019 | 0.000 / 0.000 |
| **40M-wide distill+OPD (6250, final)** | **0.803** / **0.944** | 0.002 / 0.021 | 0.000 / 0.000 |
| 40M-wide PD-mix+GRPO (1600) | 0.724 / 0.958 | 0.043 / **0.227** | 0.000 / 0.006 |
| 40M-deep distill+GRPO (1600) | 0.678 / 0.960 | 0.002 / 0.024 | 0.000 / 0.000 |
| 40M-deep PD-mix+GRPO (1600) | 0.743 / 0.951 | 0.061 / **0.266** | 0.000 / 0.005 |

1. **OPD has the best pass@1 of all six 40M runs** (0.803, vs 0.678–0.743 for
   every GRPO arm) — it closes most of the gap to the 1.5B teacher's 0.851.
2. **But OPD has the *lowest* pass@32 of the 40M runs** (0.944 vs 0.951–0.960).
   So OPD **sharpens**: individual samples get much better, while the set of
   problems solvable by ≥1 of 32 samples shrinks slightly. mean@32 (0.8216, far
   above GRPO's 0.7021) rewards exactly that sharpening, whereas pass@32
   rewards coverage — which is why the two metrics disagree in direction. Both
   are correct; report them together, not interchangeably.
3. **OOD is set by the distillation curriculum, not the post-training method.**
   OPD leaves n=5 essentially where distill+GRPO was (pass@32 0.021 vs 0.019)
   and n=6 at exactly 0, while **PD-mixture+GRPO reaches n=5 pass@32 0.227
   (wide) / 0.266 (deep)** — an order of magnitude better. So the earlier
   "OOD is pinned at ~0 either way" holds *only within the distill-student
   family*; it is not a statement about 40M models. OPD cannot rescue a
   distill-only student's OOD, because the frozen rollouts contain no n=5/6
   composition to learn from.

Practical consequence: the highest-value next OPD experiment is **OPD on the
PD-mixture student**, which is the arm that actually has OOD ability to
sharpen.

### Bearing on the fresh-prompt (group 1) question

10,000 prompts recycled 10× (0.80 mean@1) **beats 418,136 fresh prompts under
GRPO (0.69)** on ID, and OOD is pinned at 0 in both. So prompt *breadth* is not
what limits this student on ID, and nothing here suggests more distinct prompts
would unlock OOD. That is an argument against paying ~14 h of teacher scoring to
match RL's full 1.67M-rollout volume, and for the cheap 25,600-prompt
single-epoch variant instead.

## Suggested next steps

Priority order after the group-2 results and the pass@k finding:

1. **OPD on the PD-mixture student** (`balanced-progdistill-mixture-t1p5b-sftlr3e-4-steps6220-seed1`).
   The only 40M arm with real OOD ability (n=5 pass@32 0.227–0.266), so it is
   the only one where OPD's sharpening could plausibly improve OOD rather than
   just ID. Needs a fresh gen-data pass (different π_ref → different rollouts).
2. **Second seed** on a 10-epoch arm before treating OPD > GRPO as settled
   (everything so far is seed 1, though +0.115 pass@1 is a large margin).
3. **Truncate rollouts at the first complete `<answer>`** and regenerate: the
   one change that directly targets the 69% self-similar filler. Now more
   motivated, since longer training made length *rebound* (~840 → ~884).
4. **LR sweep is closed** — 1e-5/2e-5/5e-5 land within 0.016 mean@1 at 10
   epochs. Don't spend more compute here; spend it on epochs or on (1).
5. Remaining coverage: **40M-deep** distill + PD-mixture, and the originally
   planned **Qwen 0.5B** distill/progdistill arms from
   `2026-07-09-offline-opd.md`.

## Status (2026-07-28)

- Groups 1600-step (lr 2e-6, 2e-5) and 10-epoch (lr 1e-5, 2e-5, 5e-5) all
  **complete**, 10/10 eval coverage on all 3 datasets for all 5 runs.
- `figures/Qwen2.5-40M/accuracy_bars_stacked.png` refreshed with the OPD
  lr=5e-5 bar.
- **Fresh-prompt OPD runs (matching RL's prompt regime): PAUSED at Bingbin's
  request.** The open decision, if resumed, is whether "same data as RL" means
  one non-repeating epoch at OPD's batch 64 (25,600 prompts × 4 = 102,400
  rollouts, ~1 h gen-data) or matching GRPO's full 1.67M-rollout volume
  (418,136 prompts, batch raised to 1024, ~14 h gen-data, ~12 GB parquet).
  Group-2 evidence favors the cheap option: 10k prompts recycled 10× already
  beats GRPO's 418k fresh prompts on ID by +0.115 pass@1, so prompt breadth is
  not the binding constraint.
- Untracked in git: `scripts/generate_opd_data.{py,sh}` (bug fixes),
  `scripts/plot_accuracy_bars.sh` (new), `research_logs/`. Modified and tracked:
  `scripts/plot_accuracy_bars.py`.
