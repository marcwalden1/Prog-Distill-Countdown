# pass@k grid for the 7B-vs-1B set, + temp-1.0 eval sweep (2026-09-15)

Extends `figures/olmo2-ladder/7b-vs-1b-sftgrpo-lr3e-6kl3e-3/` with a pass@k x
dataset grid, and submits the temperature-1.0 evals needed for a second copy of
it. All five arms are **GRPO on top of answer-only GT-SFT students** (format
`answer` in `sft_formats.py`, ~17-token responses); no `search`-format arm is in
this set.

## Arms

| label | model | GRPO exp | SFT origin |
|---|---|---|---|
| 1B 1007B | OLMo-2-0425-1B-pt1007B | `balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1` | `balanced-simple-gtsft200-lr3e-5-seed1` |
| 1B 4001B | OLMo-2-0425-1B-pt4001B | same | same |
| 7B 126B | OLMo-2-1124-7B-pt126B | `balanced-simple-gtsft200lr3e-5-grpo-lr3e-6-kl3e-3-seed1` | `balanced-simple-gtsft200-lr3e-5-seed1` |
| 7B 135B | OLMo-2-1124-7B-pt135B | same | same |
| 7B 3896B | OLMo-2-1124-7B-pt3896B | `balanced-simple-gtsft200lr1e-5-grpo-lr3e-6-kl3e-3-seed1` | `balanced-simple-gtsft200-lr1e-5-seed1` |

pt3896B's SFT LR is 1e-5, not 3e-5 — it is the LR baked into that run's own
`EXP_NAME`, so its origin dot is the student GRPO actually started from.

## New code

- `scripts/compute_passk_multi.py` — caches the correct-count histogram
  (`hist[c]` = prompts with exactly c of N samples correct) as `.passk_multi`,
  ~1 KB per eval file. `compute_passk.py` short-circuits at the first correct
  sample so it can only ever produce pass@1 and pass@N; this grades all N.
- `scripts/plot/plot_passk_grid.py` — rows k = 1,2,4,8,16,32, columns
  n=3,4 / n=5 / n=6, one figure per temperature
  (`passk_grid_temp<T>_<tag>.png`). Unbiased estimator `1 - C(N-c,k)/C(N,k)`,
  so the k=1 row is c/32 rather than `.passk`'s first-sample pass@1 — same
  expectation, agrees to ~0.001.
- `plot_vs_step` added to `plot_olmo2_ladder_pareto.py` (pass@1 / pass@32 vs
  step, 2x3), from the previous round of this work.

## Finding: GRPO helps only for k <= 2

From `passk_grid_temp0.6_7b-vs-1b.png`. The k=1 and k=2 rows rise over training;
k=4 and above start *below* the SFT student and never recover. In-dist the
crossover is between k=2 and k=4, and it is a warmup-phase event — pass@32 falls
0.80 -> 0.49 within the first 200 steps, then is flat for 1400 more.

Underlying shape change (in-dist, pt3896B): the SFT student has 22.7% of prompts
never solved in 32 samples and 0.5% always solved; after 1600 GRPO steps that is
56.9% never / 38.7% always. GRPO converts a broad-shallow distribution into a
bimodal one.

7B pt3896B is the exception on OOD: the only arm whose n=6 pass@k rises at every
k, and still climbing at step 1600 (n=6 pass@1 0.033 vs 0.013-0.017 for the
rest). It is also the only arm that *loses* in-dist pass@1 over training
(0.437 @ s600 -> 0.429 @ s1600).

## Eval cost model (measured, corrects two earlier guesses)

Mean walltime per array task, from 286 completed `eval.sh` tasks tagged by each
log's `Saving results to` line. One array task = **one checkpoint x one
dataset** (`eval.sh` takes the step from `SLURM_ARRAY_TASK_ID`, the dataset from
`EVAL_DATASET`).

| | n=3,4 (997 prompts) | n=5 (4620) | n=6 (3519) |
|---|---|---|---|
| 1B | 4.8 min | 52.4 min | 29.8 min |
| 7B | 5.7 min | 17.4 min | 13.2 min |

Two things worth keeping:

- **Evals are 1 GPU**, not 4 or 8 — `eval.sh` requests
  `--gres=gpu:nvidia_h100_80gb_hbm3:1` and `eval.py` sets
  `tensor_parallel_size=torch.cuda.device_count()`. So GPU-hours == job-hours.
- **1B is 3x slower than 7B on n=5** despite being 7x smaller, and not because
  of GPU count (both 1 GPU). The decode-steps explanation first written here was
  **disproved** by the temp-1.0 measurements below: 1B and 7B response lengths
  are the same and neither truncates. Cause still unknown.

At temp 0.6 these evals are overhead-dominated: ~20-token responses over
`n_prompts x 32` sequences at vLLM's H100 default `max_num_seqs=1024` is only
~600 decode steps for all of n=3,4. Raising temperature was predicted to cost
real time (samples running to the cap instead of terminating) — **this did not
happen at temp 1.0**; see the measurement below. Still plausible at temp 2.0,
still unmeasured.

One temperature over all on-disk checkpoints is **~59 GPU-hours**.

## Submitted: temp 1.0 (2026-09-15)

150 array tasks, `-A kempner_bingbin_lab --array=<steps> -t 02:00:00` (n=3,4) or
`-t 06:00:00` (n=5/n=6), `EVAL_EXTRA_ARGS="--temperature 1.0"`. Temps 0.3 and
2.0 are deliberately **paused**.

`eval.sh` only *verifies* `$SLURM_JOB_ACCOUNT`, it never sets it — submitting
without `-A kempner_bingbin_lab` fails with "Invalid account or
account/partition combination" because the default account is not valid on
`kempner_requeue`.

Job IDs: GRPO arms `46613510`-`46613540` (15 array jobs), SFT students
`46613570`-`46613602` (15 single-task jobs).

### Landed 2026-09-16 — results

147 of 150 JSONs written (194 tasks COMPLETED, 15 PREEMPTED-and-requeued on
`kempner_requeue`). Figure:
`figures/olmo2-ladder/7b-vs-1b-sftgrpo-lr3e-6kl3e-3/passk_grid_temp1.0_7b-vs-1b.png`.

**The one gap (as first analysed — see the corrected tree section below):
7B pt135B step 300.** All 3 datasets failed with
`ValueError: Unrecognized model ... Should have a model_type key in its
config.json`. The checkpoint is a **partial save** from Sep 1, not merely
unmerged: `actor/` has all four `model_world_size_4_rank_*.pt` shards at full
size but no `actor/huggingface/` config dir, `optim_..._rank_2.pt` is 0 bytes
and `rank_3` is absent — the training job was killed mid-save. It was never
evaluated at temp 0.6 either. Recoverable in principle by dropping a matching
`config.json`/tokenizer into `actor/huggingface/` and re-merging; not done —
and **moot**, because this checkpoint belongs to the aborted Sep-1 run in the
default tree, not to the run the temp-0.6 curve came from (see below).

**Response length does NOT grow from 0.6 to 1.0** — the prediction in the cost
model above is wrong at this temperature. Measured on step-1600 outputs:

| | temp 0.6 | temp 1.0 |
|---|---|---|
| 7B pt3896B, n=3,4 | 36.2 chars mean, 0% missing `</answer>` | 36.2 chars, 0% |
| 1B pt1007B, n=3,4 | 36.0 chars, 0% | 36.0 chars, 0% |
| 1B pt1007B, n=5 | 41.3 chars, 0.01% | 41.3 chars, 0.03% |

Format adherence is total at temp 1.0 and the decode-step blowup does not
happen. A temp-0.3 sweep should therefore cost the same as this one; temp 2.0
remains unmeasured. This also removes the proposed explanation for why 1B is 3x
slower than 7B on n=5 — it is not response length, and the cause is still
unknown.

**Break-even k moves with temperature.** GRPO(step 1600) minus SFT, by k:

    n=3,4      k=1      k=2      k=4      k=8     k=16     k=32
    1B 1007B  0.6   +0.2261  +0.1189  +0.0035  -0.1064  -0.1984  -0.2678
    1B 1007B  1.0   +0.2802  +0.1737  +0.0383  -0.1032  -0.2233  -0.3039
    1B 4001B  0.6   +0.1792  +0.0692  -0.0423  -0.1442  -0.2310  -0.3039
    1B 4001B  1.0   +0.2441  +0.1306  -0.0042  -0.1388  -0.2537  -0.3390
    7B 3896B  0.6   +0.0896  -0.0293  -0.1410  -0.2374  -0.3196  -0.3902
    7B 3896B  1.0   +0.1545  +0.0234  -0.1186  -0.2502  -0.3569  -0.4363

The crossover sits between k=2 and k=4 at temp 0.6 and between k=4 and k=8 at
temp 1.0, on all three datasets. **But this is the SFT baseline moving, not the
GRPO endpoint.** An earlier version of this note said "raising the temperature
makes GRPO look better at small k and worse at large k", which wrongly implies
the GRPO curves shift. They barely do. Absolute pass@k, 1B 4001B, n=3,4,
temp 1.0 minus temp 0.6 on the *same* model:

    k=          1        2        4        8       16       32
    SFT    -0.0652  -0.0580  -0.0318  +0.0034  +0.0340  +0.0481
    GRPO   -0.0003  +0.0034  +0.0062  +0.0089  +0.0113  +0.0130

**The post-GRPO policy is nearly temperature-invariant** (|delta| <= 0.013 at
every k, and weakly positive), while the SFT student is strongly
temperature-sensitive: hotter sampling costs it 6.5 points of pass@1 and buys it
4.8 points of pass@32. That is what a collapsed vs. a broad output distribution
looks like — GRPO has removed enough entropy that the sampling temperature has
almost nothing left to act on.

So the honest statement is: the GRPO-minus-SFT gap is temperature-dependent
almost entirely through the baseline. Any "GRPO helps up to k=K" claim must name
the temperature, but because K depends on how the *reference* model was sampled,
not on the RL model's behaviour changing.

`passk_grid_temp0.6-1.0_7b-vs-1b.png` shows this directly — solid (0.6) and
dashed (1.0) GRPO curves lie almost on top of each other, while the filled (0.6)
and hollow (1.0) SFT origin dots at step 0 are far apart in opposite directions
at k=1 and k=32.

pt3896B remains the OOD outlier at temp 1.0: n=6 pass@32 0.0404 vs 0.025-0.028
for the 1B arms, and it is the only arm whose n=6 curve is still rising at step
1600.

### Original submission notes

When they land:

    for spec in <model exp> ...; do
      python3 scripts/compute_passk_multi.py --model $1 --exp $2 --temp 1.0
    done
    python3 scripts/plot/plot_passk_grid.py <same --run/--origin flags> --temp 1.0 \
      --tag 7b-vs-1b --set 7b-vs-1b-sftgrpo-lr3e-6kl3e-3

## Checkpoint trees — NOTHING WAS PURGED (corrected 2026-09-16)

An earlier version of this note claimed 7B pt126B/pt135B had been purged down to
2-3 checkpoints. **That was wrong.** All 8 checkpoints of both runs are intact.
There are *three* checkpoint trees in play, and different arms were evaluated
from different ones:

| arm | tree its temp-0.6 eval read | status |
|---|---|---|
| 1B pt1007B, 1B pt4001B | `/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints` (2026-08-26) | since moved into the default tree, 16 steps each |
| 7B pt3896B | same holylabs tree (2026-09-10) | since moved into the default tree, 8 steps |
| **7B pt126B, 7B pt135B** | **`/n/netscratch/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`** (2026-09-02) | **all 8 steps still there; never moved** |

Note the two netscratch paths differ only in the lab segment
(`kempner_bingbin_lab` vs `bingbin_lab`) — easy to conflate.

The default `CHECKPOINT_DIR`
(`/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown`) holds a *separate,
partial* copy of the pt126B/pt135B runs: a training job from Sep 1 18:31-21:09
that reached only steps 100/200/300 and was killed mid-save (hence the 0-byte
optimizer shard in step_300). Its step-200 shards differ from the
`kempner_bingbin_lab` tree's step-200, so it is not the same weights.

**Consequence for the first temp-1.0 submission:** it read the default tree, so
pt126B/pt135B were evaluated at steps 100/200 of the *wrong run*. Those 24 files
were quarantined and the two arms re-submitted 2026-09-16 against
`CHECKPOINT_DIR=/n/netscratch/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`,
array `2-16:2` (steps 200-1600) — jobs `46735051`-`46735065`, 48 tasks. The
1B arms, pt3896B, and all five SFT students were unaffected: their temp-1.0
evals read the same weights as their temp-0.6 evals.

**Rule to carry forward:** before reporting a checkpoint as missing, check all
three trees. CLAUDE.md already says "Runs from before the move are NOT found by
default"; this note originally ignored it.
