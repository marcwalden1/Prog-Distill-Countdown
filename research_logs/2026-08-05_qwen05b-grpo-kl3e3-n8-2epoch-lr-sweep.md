# Qwen2.5-0.5B GRPO: kl=3e-3, n=8 rollouts, 2 epochs, lr ∈ {1e-6, 3e-6} (2026-08-03 → 08-05)

Follow-up to `2026-08-03_idood-pareto-teacher1p5b-vs-student05b.md`. That note found the
baseline 0.5B GRPO run (`balanced-grpo-seed1`, kl=1e-3, n=4, 1 epoch) collapses to
~21-token responses with entropy → 0.014 by step ~600, dropping `<think>` entirely.
This experiment tests whether stronger KL prevents that, and whether more
data/steps helps.

## Config (both arms)

Identical except LR. `MODEL_NAME=Qwen2.5-0.5B`, `DATA_SOURCE=balanced`.

| | value | vs old baseline |
|---|---|---|
| `KL_COEF` | 3e-3 | was 1e-3 |
| `ROLLOUT_N` | 8 | was 4 |
| `VAL_KWARGS_N` | 4 (pinned) | unchanged — keeps val mean@4 comparable |
| `ENTROPY_COEFF` | 0 | unchanged (KL is the single intervention) |
| `TOTAL_EPOCHS` / `TOTAL_STEPS` | 2 / 3266 | was 1 / 1633 |
| `USE_DYNAMIC_BSZ` | True, `ppo_max_token_len_per_gpu=16384` | was fixed micro-batch 16 |
| GPUs | 4 × H100 | was 2 |

**Why total_epochs matters:** `data/balanced/train.parquet` is 418,136 rows = exactly
1633 steps at `train_batch_size=256`, and verl's `fit()` loops
`for epoch in range(trainer.total_epochs)`, stopping early only once
`global_steps >= total_training_steps`. With `total_epochs=1` a run is *data*-limited
to 1633 steps no matter how large `TOTAL_STEPS` is — the old baseline stopped at 1633,
not its configured 1635. Both must be raised. `TOTAL_EPOCHS` did not exist as a knob
before this experiment (it was hardcoded); added in `scripts/train_grpo.sh`.

**Dynamic-bsz caveat:** with `loss_agg_mode=token-mean` and `use_dynamic_bsz=False`,
verl (`dp_actor.py:416`) normalizes loss *within* each micro-batch then averages with
weight `1/G`, so unequal micro-batch token counts perturb the effective per-token
weighting. Turning on dynamic bsz is therefore a **numerics change, not just a speed
change** — it is one of four differences vs the old baseline, so the results below
cannot be attributed to KL alone.

## Efficiency findings (measured, not estimated)

Profiled the baseline (`28828883`, 1633 steps, 2 GPUs, 5:53:42): `update_actor` was
**46.8%** of wall time (2.67 h), `gen` 29.9%, ref+old_log_prob 13.1%, validation 6.4%.
Cause: `ppo_mini_batch_size` 256 × `rollout.n` 4 ÷ 2 GPUs = 512 seqs/GPU at
`PPO_MICRO_BATCH_SIZE_PER_GPU=16` → **32 grad-accum micro-batches/step**, ~3k tokens
each, `mfu/actor` 0.17.

20-step probe (n=8, 4 GPUs, dynamic bsz) vs baseline at matched early steps:

| | baseline (n=4, 2 GPU, fixed-16) | probe (n=8, 4 GPU, dynamic) |
|---|--:|--:|
| step | 20.48 s | 23.02 s |
| `gen` | 9.75 s | 9.66 s |
| `update_actor` | 7.23 s | 5.69 s |
| `mfu/actor` | 0.162 | 0.212 |
| peak GPU reserved | 95.8 GB | 81.3 GB |

Per *sequence* ~2× more efficient, but per-step wall time 12% **slower**: n=8 doubles
the work and 4 GPUs doubles throughput, so they cancel; dynamic bsz only attacks
`update_actor` + logprob, and `gen` (42% of the step) is untouched. Lesson: don't
project run time from a *run-average* step time when response length is changing —
the baseline's 11.67 s/step average was low only because it had collapsed to 21-token
responses.

## Arm 1: lr=1e-6 — COMPLETED

Job `36971637`, 3266/3266 steps, **13:28:42**, 14.72 s/it, MaxRSS 51 GB.

Held-out (full prompt set, pass@k = unbiased Chen et al. estimator):

| step | ID pass@1 | n=5 pass@1 | n=6 pass@1 | ID pass@32 | n=5 pass@32 | n=6 pass@32 |
|---|--:|--:|--:|--:|--:|--:|
| 600 | 0.1977 | 0.0513 | 0.0126 | 0.3771 | 0.1221 | 0.0423 |
| 1600 | 0.2056 | 0.0509 | 0.0137 | 0.4223 | 0.1442 | 0.0517 |
| 3250 | 0.2342 | 0.0608 | 0.0163 | 0.4213 | 0.1448 | 0.0548 |

**vs old baseline** (kl=1e-3, n=4, 1 epoch) at its step 1600 — a clean swap, every
pass@1 down and every pass@32 up, consistent across all three datasets:

| | new (3250) | old (1600) | Δ |
|---|--:|--:|--:|
| ID pass@1 | 0.234 | 0.290 | −0.056 |
| n=5 pass@1 | 0.061 | 0.093 | −0.032 |
| n=6 pass@1 | 0.016 | 0.022 | −0.006 |
| ID pass@32 | 0.421 | 0.364 | +0.057 |
| n=5 pass@32 | 0.145 | 0.123 | +0.021 |
| n=6 pass@32 | 0.055 | 0.044 | +0.011 |

Interpretation: kl=3e-3 held entropy at ~2.8 instead of collapsing to 0.014 and kept
responses at ~150–195 tokens instead of ~21, i.e. `<think>` was retained. The policy is
correspondingly **less sharpened** — any single sample is less likely correct (pass@1
down) but 32-sample coverage is broader (pass@32 up). The higher train `score/mean`
(0.42 vs 0.36) tracks the mean-over-samples quantity, i.e. pass@32, **not** pass@1 —
so train reward was not misleading, it just wasn't measuring what pass@1 measures.

**More steps bought little.** ID pass@1 is 0.198 at step 600 and 0.234 at 3250: +0.037
for 2650 additional steps. pass@32 flattens after ~1050. Most of the 13.5 h was plateau.

**Still far from the teacher** (Qwen2.5-1.5B `balanced-grpo-lr3e-6-kl3e-3-seed1`,
step 1600): ID pass@1 0.860 vs 0.234 (3.7×), n=5 pass@1 0.318 vs 0.061 (5.2×),
n=5 pass@32 0.692 vs 0.145 (4.8×). No Pareto crossing; same ID→OOD ray, teacher
much further along it.

Plots: `figures/comparison/id_vs_ood_teacher1p5b_vs_student05b_kl3e3_2ep/`,
data `analysis_data/idood_teacher1p5b_vs_student05b_kl3e3_2ep.csv`.

## Arm 2: lr=3e-6 — THREE FAILURES, currently 61% done

| # | job | start | outcome |
|---|---|---|---|
| 1 | `36971655` | fresh, 384 GB | **wedged at step 79.** Log froze 3h37m, live tqdm at 16.87 s/it, no NCCL/CUDA error, no traceback, still RUNNING holding 4 GPUs. Cancelled manually. |
| 2 | `37040294` | resumed from step 50, 384 GB | **OUT_OF_MEMORY at ~step 80** after 1:24:30. Host RAM: `MaxRSS` 383 GB vs 384 GB requested (`--mem-per-gpu=96G` × 4). Not GPU memory. |
| 3 | `37337375` | fresh, 768 GB, `GRPO_LOG_FREQ=1` | **FAILED exit 1 at step 2000/3266** after 9:32:47. Last driver output = step-2000 checkpoint save at 10:47:54, job ended 11:03:01. No traceback, no Ray/NCCL/CUDA error, log ends cleanly. **Cause: the lab `/n/holylabs` group quota hit 100% (see below).** |

### Root cause of attempt 3: `/n/holylabs` group quota exhausted

    $ quota /n/holylabs/LABS/kempner_bingbin_lab
    Disk quotas for grp kempner_bingbin_lab (gid 11454):
    Filesystem   used   quota    files      quota
    /n/holylabs  4.0Ti   4.0Ti   338024  10000000

`df` is useless here — it reports 1.4 P *filesystem* free while the group quota is
100% consumed. Evidence this is the cause:

- Attempt 3 died immediately after writing the ~5.7 GB `global_step_2000` checkpoint,
  i.e. exactly when the quota would have been crossed. Once writes fail, the log
  itself cannot record the error — which explains the traceback-free, cleanly-ended log.
- `touch logs/.wtest` → `Disk quota exceeded`.
- **Every** subsequently submitted job now dies in 1-5 s with `ExitCode 0:53` and
  **no output file created at all**, because SLURM cannot open its stdout under
  `logs/` (a symlink into `/n/holylabs`). Confirmed with a minimal
  `sbatch --wrap='hostname'` using the same partition/gres/output pattern — same failure.
  36 eval tasks for this arm failed this way twice (jobs `37418929`/`37418930`/`37418935`
  and `37419294`/`37419295`/`37419296`).

**Nothing can run until space is freed.** Usage under this repo's trees:

| path | size |
|---|--:|
| `rl-checkpoints/checkpoints/Qwen2.5-0.5B` | 594 G |
| ├─ `balanced-grpo-dynbsz-n8-lr1e-6-kl3e-3-ep2-seed1` (arm 1, **complete + fully evaluated**) | 305 G |
| └─ `balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1` (arm 2, needed for eval + resume) | 228 G |
| `rl-checkpoints/checkpoints/gemma-3-270m` | 281 G |
| `rl-checkpoints/checkpoints/Qwen2.5-40M-wide` | 228 G |
| `rl-checkpoints/sft-checkpoints` | 44 G |
| `repo-artifacts` (results/figures/logs) | 132 G |
| `.aborted-balanced-grpo-dynbsz-n8-lr3e-6-...-20260805` | 5.8 G |

Cheapest unblock: arm 1 is finished and fully evaluated, so its 65 intermediate
checkpoints (5.7 GB each) are redundant — pruning it to `global_step_3250` frees
~300 G on its own. **Not done: deleting checkpoints is the owner's call.**

### Corrections to earlier hypotheses — both wrong

1. **"Attempt 1 was a transient node wedge; just resubmit."** Wrong: it recurred.
2. **"Failures at step 79/80 are deterministic and memory-driven."** Also wrong.
   Attempt 3 sailed through step 80 and ran to 2000, and per-step
   `perf/cpu_memory_used_gb` (visible only because `GRPO_LOG_FREQ=1`) shows host RAM at
   **54.6 GB at step 1 → 65.4 GB at step 1999** — a slow ~11 GB creep, never remotely
   near 384 GB. So the 768 GB bump was not needed, and the step-79/80 coincidence
   across attempts 1–2 was just that. **Three attempts, three distinct failure modes;
   this arm is flaky for reasons not yet identified.** The lr=1e-6 arm ran the same
   config for 3266 steps cleanly, so it is not the config as such.

### Training metrics (attempt 3, per-step)

| step | cpu RAM | GPU res | resp_len | clip | entropy | train score |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 54.6 | 67.2 | 474.8 | 0.248 | 2.169 | 0.010 |
| 200 | 55.2 | 81.7 | 236.9 | 0.104 | 3.142 | 0.325 |
| 800 | 56.0 | 82.0 | 273.9 | 0.122 | 2.950 | 0.441 |
| 1400 | 61.6 | 82.0 | 301.9 | 0.134 | 2.933 | 0.465 |
| 1999 | 65.4 | 82.0 | 252.0 | 0.064 | 2.347 | 0.513 |

No entropy collapse (2.3–3.1 throughout), no length collapse (~250 tokens), and train
score **0.513 at step 1999 vs lr=1e-6's ~0.42** at comparable steps — so on the train
metric lr=3e-6 is ahead. Whether that survives to held-out pass@1 is exactly the open
question, given arm 1 showed train score tracks pass@32 rather than pass@1.

### State / how to continue

40 checkpoints intact, `global_step_{50..2000}` step 50, `latest_checkpointed_iteration.txt=2000`.
Resumable: relaunch the same command; `resume_mode: auto` reloads step 2000.
Evals for the 12 available grid steps (150…1800, 2000) submitted 2026-08-05 as jobs
`37418929` (balanced) / `37418930` (balanced5) / `37418935` (balanced6).
The pipeline's own eval/plot jobs were auto-CANCELLED when the `afterok` dependency failed.

## Infrastructure added

- `scripts/train_grpo.sh`: new `TOTAL_EPOCHS`, `ENTROPY_COEFF`, `USE_DYNAMIC_BSZ`,
  `PPO_MAX_TOKEN_LEN_PER_GPU` knobs (`total_epochs` and `entropy_coeff` were hardcoded).
- `scripts/export_idood_scores.py` / `.sh`: full-prompt-set re-scoring → tidy CSV with
  pass@1/pass@32/mean@1/mean@32 + `n_prompts`/`n_samples`, caching in its own dir so a
  read-only source tree is never mutated and a foreign (possibly capped) `.scores` is
  never trusted.
- `scripts/plot_id_vs_ood_runs.py`: ID-vs-OOD trajectories for arbitrary runs across
  result trees, reading that CSV.
- `scripts/watchdog_train_jobs.sh`: flags RUNNING train jobs whose stdout has gone
  stale (>`STALE_MIN`, default 20) with a `GRACE_MIN=40` startup exemption; alerts to
  console + `logs/watchdog-alerts.log` + `mail`. It correctly caught attempt 2 at
  23:30:04, ~44 min before that job died.
  **Known gap: it only inspects RUNNING jobs, so it cannot report a crash** — attempts
  2 and 3 both died silently as far as the watchdog was concerned. Adding
  "left the queue without reaching the final step" detection is the obvious next step.

## Gotchas worth remembering

- **SLURM `COMPLETED` ≠ eval JSON exists.** 6 of 48 evals on arm 1 wrote no JSON despite
  COMPLETED, all `OSError: [Errno 116] Stale file handle` in `filelock` (transient NFS).
  A rerun fixed 5; `balanced` step 300 failed the same way twice. Always verify by file.
- **eval.sh derives its checkpoint as `step = 50 × SLURM_ARRAY_TASK_ID`**, and its
  default array tops out at step 1600. A >1633-step run needs an explicit
  `EVAL_SBATCH_ARGS="--array=..."`; the grid used here was
  `3,6,9,12,15,18,21,24,28,32` (old grid, comparable to prior work) `+ 36,42,45,52,58,65`.
- Checkpoints are ~5.7 GB each at 4-GPU FSDP, so a 65-checkpoint run is ~305 GB.
