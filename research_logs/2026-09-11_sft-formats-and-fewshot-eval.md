# Shared SFT response formats + few-shot base eval

**Date:** 2026-09-11
**Status:** code landed, tested locally; no jobs submitted
**New:** `sft_formats.py`, `tests/test_sft_formats.py`
**Touched:** `scripts/generate_gt_sft_data.py`, `eval.py`, `scripts/eval_base.sh`,
`scripts/run_olmo2_gt_sft_ladder.sh`

## Why

GT answer-only SFT installs a ~15-token response (`<answer> ... </answer>` and
stop). `2026-08-25_olmo2-pretrain-ladder-grpo.md` measured the consequence: 32.5
chars/response after SFT+GRPO vs 2383 for no-SFT GRPO, and post-GRPO ID pass@32
collapsing onto pass@1 (0.4774 / 0.5055). With a 15-token response the only
variation across `rollout.n=4` rollouts is *which* expression is emitted in one
shot — "try → check → retry" is outside the support of the rollout distribution,
so GRPO cannot amplify it, and prompts where all 4 rollouts agree contribute a
zero advantage.

The fix is a target format whose support contains failed attempts.

## `sft_formats.py` — one definition of "what a solution looks like"

Shared by the SFT-target generator and eval's few-shot exemplars, so
`eval.py --n_shot K --shot_format X` measures exactly the format
`generate_gt_sft_data.py --format X` would train on.

| format | response (after a `Solution:` prompt) | tokens (OLMo, n=3,4) |
|---|---|---|
| `answer` | ` <answer> 14 + 27 + 9 </answer>` | 17 |
| `think_empty` | ` <think></think>\n<answer> ... </answer>` | 22 |
| `verify` | `<think> 14 + 27 = 41, 41 + 9 = 50, which is the target. </think>` | 54 |
| `search` | k rejected legal attempts, then the answer | 84 |

`search` example (real row, `--search-attempts 1-3`):

    <think>
    Try ((18 * 29) * 27) + 72 = 14166; too high.
    Try (27 * (18 + 29)) - 72 = 1197; too high.
    Try ((29 + 27) * 18) / 72 = 14; too high.
    Try (29 - 27) * 72 / 18 = 8; that is the target.
    </think>
    <answer> (29 - 27) * 72 / 18 </answer>

### Where the distractors come from

A pattern from `annotated_expressions.json` at the puzzle's size, with the
puzzle's **own** numbers plugged into a random permutation. Every distractor is
therefore a *legal* Countdown expression (each number exactly once) that simply
misses. Kept only if the value is an integer, `|value| <= 100_000`, and
`!= target` — an alternative *correct* expression must never appear as a
rejected attempt. Distinct expressions and distinct values, so no attempt
repeats a number already ruled out.

Candidate space per puzzle is ample: n=3 is 32 patterns × 6 permutations = 192
(enumerated exhaustively, ~110 wrong integer-valued candidates, 25 distinct
values); n=4 is 320 × 24 = 7680 (~3555 wrong, 252 distinct values). n=5/6 are
sampled under a flat `MAX_CANDIDATE_TRIES=600` budget. Cost: 16–23 ms/row, so
the 12.8k-row SFT warmup builds in ~4 minutes on one core.

`--distractor-strategy`: `near` (k closest misses), `random`, `mixed` (default —
half random, half near, ordered by decreasing `|value - target|` so the trace
reads as a search closing in).

Safety: rejected attempts live strictly inside `<think>`.
`grader_utils.extract_solution` takes the **last** `<answer>` tag, so the graded
answer is unchanged; the generator's `compute_score == 1.0` gate still passes on
every row (2000/2000 on a `--format search` build).

## Few-shot eval

`eval.py --n_shot K --shot_format {answer,think_empty,verify,search}`
(`--shot_seed`, `--shot_source`, `--shot_attempts`, `--shot_strategy`).

- **Fixed, seeded exemplar block** prepended to every prompt: lower variance than
  per-prompt shots, and vLLM's prefix cache makes the shared prefix nearly free.
- **Requires `--prompt_format raw`** (hard error otherwise): exemplars are plain
  prompt+response text and cannot be spliced into a chat template without
  changing what the model sees. Base models have no chat template anyway.
- **Exemplar pool:** `data/<shot_source>/train.parquet`, with `auto` mapping
  `balanced5*`/`balanced6*` → the n=3,4 train split of the same prompt variant
  (they are test-only). Exemplars are never drawn from any test set.
- Read with pyarrow row-group slices, not `load_dataset` — only K rows are
  needed from a 282 MB parquet, and it skips the HF-datasets cache that wedges on
  NFS home.
- **Results go to `<result_dir>/<K>shot-<format>/`**, a subdir, not a filename
  suffix: `plot_results.py` globs `{ds}_temp0.6_n32_*.json` under
  `global_step_*`/`round_*`/`sft_final`, so a suffix-only scheme would silently
  merge zero-shot and few-shot points into one curve.

`scripts/eval_base.sh` passes `N_SHOT` / `SHOT_FORMAT` / `SHOT_SEED` /
`SHOT_ATTEMPTS` / `SHOT_STRATEGY` / `SHOT_SOURCE` through, and also picked up
three fixes it needed: per-`$USER` model root and account (it hard-coded
mwalden's model dir, `PYTHONPATH`, and `--account=kempner_kdbrantley_lab`), and a
node-local `HF_DATASETS_CACHE`.

Suggested first sweep (the point is to pick the SFT format before spending SFT
compute):

    for fmt in answer search; do
      for k in 2 8; do
        MODEL_NAME=OLMo-2-0425-1B-pt4001B EVAL_DATASET=balanced-simple \
        N_SHOT=$k SHOT_FORMAT=$fmt sbatch scripts/eval_base.sh
      done
    done

Read pass@1 (accuracy) and pass@32 (coverage) via `scripts/compute_passk.py`, and
the mean response length — the question is whether the base model *can* be
steered into a multi-attempt format at all.

## Run-identity fix in the GT-SFT ladder

`run_olmo2_gt_sft_ladder.sh` built `sft_exp` with no format tag, so a
`GT_FORMAT=search` run would have overwritten the answer-only checkpoints and
results. The format is now in the name (`...-gtsft200-search-lr1e-5-seed1`),
with `answer` deliberately omitted so existing runs keep resolving. Same for the
data file: `gt-answeronly.parquet` (historical) vs `gt-<format>.parquet`.

## Side finding: 19.1% of `balanced6` has a broken `canonical_pattern`

812 of the 4328 n=6 canonical patterns in `annotated_expressions.json` do not use
each letter exactly once — e.g. `(A-B-C-D)*E/D` reuses `D` and never uses `F`.
That is **671/3519 rows (19.1%)** of `data/balanced6/test.parquet`. n=2/3/4/5 are
clean (0 malformed patterns, 0 affected rows in `balanced` and `balanced5`).

Consequences:

- Any GT-derived target built from n=6 (`generate_gt_sft_data.py`,
  `scripts/diagnose_gt_ntp.py`) reconstructs an equation that reuses one number
  and drops another: it scores 0.1, not 1.0. The generator's verify gate aborts
  loudly, so this cannot silently poison SFT data — but n=6 cannot be used as a
  GT-SFT source without filtering.
- Per-pattern analyses on n=6 are reading an unreliable pattern string for ~1 row
  in 5.

**The puzzles themselves are fine.** An exact subset-DP solvability check over
rationals found 40/40 sampled malformed-pattern rows solvable (and 40/40
well-formed ones), so the target is always reachable from the 6 given numbers —
the recorded *pattern* is wrong, not the puzzle. n=6 eval numbers stand.

## Tests

`python3 tests/test_sft_formats.py -v` — 10 tests, ~9 s: every format scores 1.0
on real rows from both prompt families; the open-`<think>` prompt family is
closed rather than reopened; distractors validate against the puzzle's numbers,
never equal the target, and never repeat a value; each `Try ... = v` line states
the expression's true value; traces are deterministic given a seed.

(NB `python3 -m unittest tests/<file>.py` does not work in this env — run the
file directly. The existing `tests/test_generate_sft_data.py` fails the same way,
so its docstring's command is stale too.)

---

# Results: few-shot base evals -> format choice -> SFT (2026-09-11, same day)

## Two bugs the first submissions found

1. **`eval_base.sh` had no PATH fix.** `conda activate verl` leaves `python3`
   resolving to the personal miniconda, so `eval.py` died instantly with
   `ModuleNotFoundError: transformers`. `eval.sh` has carried the two-line fix
   (`export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"`) since forever;
   `eval_base.sh` was shielded only by Marc's hard-coded `PYTHONPATH`, which this
   change removed. Fixed.
2. **Few-shot base models continue the exemplar pattern.** After answering the
   real puzzle they invent further puzzles and answer those, ~3500 chars to the
   length cap. `extract_solution` takes the LAST `<answer>` tag, so every cell
   read `pass@1 = 0.0000` — the graded answer came from a hallucinated puzzle.
   Fixes: `eval.py --stop_after_answer` (default on for `--n_shot > 0` only, so
   zero-shot evals keep their exact historical sampling), and
   `score_fewshot_evals.py` truncates at the first `</answer>` before grading —
   byte-identical to what a stop sequence produces, so JSONs written before and
   after the fix score identically and **no GPU time was lost**.

## The grid (balanced-simple, 997 prompts x 32 samples, temp 0.6)

| Model | k | format | pass@1 | pass@32 | >=2 attempts | chars |
|---|---|---|---|---|---|---|
| pt1007B | 2 | answer | 0.0080 | 0.1153 | — | 37 |
| pt1007B | 2 | search | 0.0090 | 0.1224 | 0.93 | 134 |
| pt1007B | 8 | answer | **0.0251** | **0.1725** | — | 34 |
| pt1007B | 8 | search | 0.0140 | 0.1284 | 0.96 | 134 |
| pt4001B | 2 | answer | 0.0090 | 0.1053 | — | 37 |
| pt4001B | 2 | search | 0.0080 | 0.1755 | 0.71 | 132 |
| pt4001B | 8 | answer | **0.0251** | **0.2558** | — | 36 |
| pt4001B | 8 | search | 0.0050 | 0.1184 | 0.99 | 161 |

CSV: `analysis_data/fewshot_olmo2_base.csv`. 7B arms (pt135B, pt3896B) still
running at time of writing.

**By the pre-registered rule, `search` loses**: at k=8 it trails `answer` on both
pass@1 and pass@32 on both models. Say that plainly rather than reading the one
favorable cell (pt4001B k=2 pass@32) as support.

## Why it loses: the traces are arithmetically fake

Over the models' own `Try <expr> = <value>` lines:

| | Try-lines | states the true value | uses each number once |
|---|---|---|---|
| pt1007B k=2 / k=8 | 18k / 20k | 0.039 / 0.033 | 0.304 / 0.394 |
| pt4001B k=2 / k=8 | 16k / 24k | 0.042 / 0.019 | 0.306 / 0.222 |

The models copy the *shape* of a search trace readily (71-99% make >=2 attempts)
but only 2-4% of stated values are the true value of the expression written, and
most attempts do not even use the given numbers once each. The "check" step
carries no information, so extra attempts add noise ahead of the answer. That is
a capability gap 2-8 exemplars cannot close — and exactly what SFT on 418k
verified traces supplies by construction.

## Decision

Ran SFT on **`search`**, both 1B models, matched to the existing answer-only arms
(200 steps, `SFT_LR=3e-5`, seed 1, `balanced-simple`), no GRPO chained.

Reasoning, recorded because it goes against the table: an answer-only GT-SFT arm
already exists for both models (2026-08-25), so re-running it yields no new
information; and the few-shot test cannot discriminate what SFT installs here,
because both formats sit near the floor (pass@1 0.5-2.5%) and search's entire
deficit traces to arithmetic infidelity the SFT targets fix. The honest framing
is that the few-shot eval **selected against the format's in-context version and
diagnosed why**, rather than that it endorsed search.

The control the diagnostic points at is `verify` (faithful evaluation of one
expression, no multi-attempt):

    GT_FORMAT=verify SFT_LR=3e-5 bash scripts/run_olmo2_gt_sft_ladder.sh pt1007B pt4001B

## Jobs

- GT data: `46074104` — 418,136 rows, 418,136/418,136 verified at 1.0, 121 s on 32 cores.
- SFT: `46095862` (pt1007B), `46095866` (pt4001B). Both COMPLETED, merged; final
  train/loss 0.346 / 0.329, `train/lr(1e-3)` 0.0054 -> 0.0 (scheduler horizon correct).
  `EXP_NAME=balanced-simple-gtsft200-search-lr3e-5-seed1`.
- SFT evals: `46095863/64/65` (pt1007B), `46095867/68/71` (pt4001B), x {n=3,4, n=5, n=6}.

## Operational notes

- `kempner_requeue` preempted a 1B eval 5 min in; the 7B arms were moved to
  non-preemptible `kempner_h100` (a 6 h eval has no checkpointing and can loop).
- Three 1B cells sat behind `ReqNodeNotAvail` with a start time 3 h out; moved to
  `kempner_h100`, where they ran in ~10 min each with the stop fix (vs ~30 min
  without).
- Beware `sacct`/filter regexes on these names: `/7B/` matches `pt100`**`7B`**.
  That mistake cancelled four 1B jobs, one of them mid-run. Use `/-7B-/`.

## SFT result: `search` is strictly worse than `answer` at matched budget (12/12 cells)

200 steps, batch 64, `SFT_LR=3e-5`, seed 1, `balanced-simple`. The answer-only
arm is the existing 2026-08-25 ladder run; the search arm is new. pass@1 / pass@32:

| Model | split | answer-SFT | search-SFT |
|---|---|---|---|
| pt1007B | n=3,4 | **0.2387 / 0.7733** | 0.1795 / 0.7573 |
| pt1007B | n=5   | **0.0426 / 0.2682** | 0.0305 / 0.2597 |
| pt1007B | n=6   | **0.0088 / 0.0719** | 0.0023 / 0.0415 |
| pt4001B | n=3,4 | **0.2979 / 0.7984** | 0.2257 / 0.7834 |
| pt4001B | n=5   | **0.0606 / 0.2823** | 0.0245 / 0.1820 |
| pt4001B | n=6   | **0.0099 / 0.0753** | 0.0014 / 0.0227 |

Answer-only wins every cell. Notably the *better-pretrained* model is hurt more
(pt4001B n=5 pass@32 0.2823 -> 0.1820, vs pt1007B 0.2682 -> 0.2597).

### What the search format did install

Measured on pt4001B ID outputs (400 prompts x 32 samples):

| | chars | attempts/sample | >=2 attempts | legal expressions | arithmetic correct |
|---|---|---|---|---|---|
| answer-SFT | 35 | 0.00 | 0.000 | — | — |
| search-SFT | 177 | 3.09 | 0.986 | 0.993 | 0.134 |

and **more** sampling diversity, not less — distinct final answers per prompt (of 32):

| | ID n=3,4 | n=6 |
|---|---|---|
| answer-SFT | 6.60 | 14.09 |
| search-SFT | **9.48** | **17.80** |

So the format bought exactly the exploration room it was designed for (5x the
response length, 3 legal candidates per rollout, +44% answer diversity on ID),
and still lost on every accuracy and coverage metric. **Diversity without a
working filter is noise:** the model proposes legal candidates and misreports
what 87% of them evaluate to, so the extra exploration is unsteered. (An earlier
hypothesis in this log — that the fake verification *collapses* the sampling
distribution — is wrong; it widens it.)

Not a length artifact: 100% answer-tag rate on both splits, max 369 chars against
a 1024-token cap.

### Recommendation

Do not take `search` to GRPO as the primary arm on this evidence. Two cheap
follow-ups, in order:

1. **`verify`** (~15 min/model) — targets the binding constraint directly, since
   the 13% arithmetic faithfulness is what makes the search trace useless:

       GT_FORMAT=verify SFT_LR=3e-5 bash scripts/run_olmo2_gt_sft_ladder.sh pt1007B pt4001B

2. **`search` at a larger SFT budget.** 200 steps x 64 = 12.8k examples is ~3% of
   the data, and the search target is a strictly harder mapping (it must learn to
   *evaluate*, not just to emit an answer). The 12/12 loss may be a budget
   artifact rather than a property of the format. `SFT_STEPS=600` or `1000` would
   settle it.

The one live argument for `search` that these numbers cannot refute: on the
2026-08-25 ladder, GRPO from the answer-only student collapsed ID pass@32 from
0.798 to ~0.49 while doubling pass@1. A student that starts with more rollout
diversity is what that collapse was starving for — but that is a hypothesis about
the RL step, not a result, and it does not justify preferring a student that is
6-7 pass@1 points behind before RL starts.

## Complete 16-cell few-shot grid (all four models)

`balanced-simple`, 997 prompts x 32 samples, temp 0.6. CSV:
`analysis_data/fewshot_olmo2_base.csv`.

| Model | k | format | pass@1 | pass@32 | fmt rate | tries | chars |
|---|---|---|---|---|---|---|---|
| 1B-pt1007B | 2 | answer | 0.0080 | 0.1153 | 1.000 | — | 37 |
| 1B-pt1007B | 2 | search | 0.0090 | 0.1224 | 0.998 | 1.93 | 134 |
| 1B-pt1007B | 8 | answer | 0.0251 | 0.1725 | 1.000 | — | 34 |
| 1B-pt1007B | 8 | search | 0.0140 | 0.1284 | 0.999 | 2.04 | 134 |
| 1B-pt4001B | 2 | answer | 0.0090 | 0.1053 | 0.999 | — | 37 |
| 1B-pt4001B | 2 | search | 0.0080 | 0.1755 | 0.998 | 1.71 | 132 |
| 1B-pt4001B | 8 | answer | 0.0251 | 0.2558 | 1.000 | — | 36 |
| 1B-pt4001B | 8 | search | 0.0050 | 0.1184 | 1.000 | 2.47 | 161 |
| 7B-pt135B | 2 | answer | 0.0020 | 0.0261 | 0.997 | — | 39 |
| 7B-pt135B | 2 | search | 0.0000 | 0.0251 | **0.846** | 7.22 | 409 |
| 7B-pt135B | 8 | answer | 0.0130 | 0.1575 | 0.998 | — | 39 |
| 7B-pt135B | 8 | search | 0.0030 | 0.0692 | 0.993 | 3.40 | 197 |
| 7B-pt3896B | 2 | answer | 0.0211 | 0.2568 | 1.000 | — | 34 |
| 7B-pt3896B | 2 | search | 0.0231 | 0.2568 | 0.969 | 3.42 | 204 |
| 7B-pt3896B | 8 | answer | **0.0391** | **0.3501** | 1.000 | — | 36 |
| 7B-pt3896B | 8 | search | 0.0181 | 0.3019 | 0.988 | 3.91 | 216 |

Three patterns:

1. **`answer` at k=8 is the best cell for every model**, and the margin over
   `search` at k=8 is large (pass@32 0.1725 vs 0.1284, 0.2558 vs 0.1184, 0.1575 vs
   0.0692, 0.3501 vs 0.3019).
2. **At k=2 the ordering reverses or ties** — search is ahead on pt1007B
   (0.1224 vs 0.1153) and pt4001B (0.1755 vs 0.1053) and level on pt3896B
   (0.2568 both, with higher pass@1 0.0231 vs 0.0211). So the trace helps when
   there is little else in context and hurts once enough direct demonstrations of
   the mapping are available. Worth remembering before reading any single-k
   few-shot result as a format verdict.
3. **Pretraining budget dominates everything**: 7B-pt3896B 8-shot answer (0.3501
   pass@32) vs 7B-pt135B (0.1575), and the weak 7B cannot even terminate a search
   trace reliably — format rate 0.846 at k=2, i.e. 15% of samples never emit an
   `<answer>` tag, burning the budget on 7.2 attempts. Consistent with the
   arithmetic-faithfulness story: with no working check, nothing tells the model
   when to stop trying.

Model size is not the whole story either: 7B-pt135B (0.1575) trails 1B-pt4001B
(0.2558) at the same k and format — 135B tokens of pretraining is worse than
4001B even at 7x the parameters.

---

# LR sweep (2026-09-12): the format deficit is not an LR artifact

The first search-vs-answer comparison was unfair: answer had a 3-point LR sweep
from the 08-25 ladder and search had a single inherited point (3e-5). Both
formats are now swept over {1e-5, 3e-5, 1e-4, 3e-4}, 200 steps, seed 1.

## ID n=3,4 (pass@1 / pass@32)

| lr | pt1007B answer | pt1007B search | pt4001B answer | pt4001B search |
|---|---|---|---|---|
| 1e-5 | 0.2076/0.7533 | 0.1294/0.7021 | 0.2578/0.7854 | 0.1946/0.7743 |
| 3e-5 | 0.2387/0.7733 | 0.1795/0.7573 | **0.2979/0.7984** | 0.2257/0.7834 |
| 1e-4 | **0.2758**/0.7793 | 0.2377/**0.7874** | 0.2497/0.7733 | 0.2026/0.7793 |
| 3e-4 | 0.2337/0.7503 | 0.1916/0.7623 | 0.1795/0.7091 | 0.1595/0.7342 |

Best-vs-best on OOD: n=5 answer 0.0459/0.2790 (pt1007B, lr 1e-5) and
0.0606/0.2823 (pt4001B, lr 3e-5) vs search 0.0305/0.2597 and 0.0245/0.2260;
n=6 answer 0.0088/0.0787 and 0.0099/0.0753 vs search 0.0026/0.0475 and
0.0026/0.0355.

**Verdict: with each format at its own best LR, `answer` wins 11 of 12
model x split x metric comparisons.** The lone exception is pt1007B ID pass@32
(search 0.7874 vs 0.7793 at lr 1e-4) -- coverage only, within noise. Single seed;
at 997 ID prompts the difference SE is ~0.02, so pt4001B's 0.072 gap is ~3.5 SE
while pt1007B's 0.038 is ~2 SE. OOD n=5 (4620 prompts) is tighter.

## Separate finding: SFT LR trades ID accuracy against OOD, sharply

pt4001B, answer format:

| | lr 1e-5 | 3e-5 | 1e-4 | 3e-4 |
|---|---|---|---|---|
| ID pass@1 | 0.2578 | **0.2979** | 0.2497 | 0.1795 |
| n=5 pass@32 | 0.2623 | **0.2823** | 0.0617 | **0.0009** |
| n=6 pass@32 | 0.0543 | **0.0753** | 0.0091 | 0.0000 |

Past 3e-5 OOD does not decline, it collapses (n=5 pass@32 falls 300x by 3e-4).
And the ID-best LR is not the OOD-best one: on pt1007B, ID pass@1 peaks at 1e-4
while n=5 and n=6 both peak at 1e-5. **Selecting an SFT LR on ID alone silently
destroys the OOD numbers** -- relevant to every arm in this project, not just
these. The ladder's existing 3e-5 is a good compromise; this is the first
evidence for *why*.

## GRPO on the search students (launched 2026-09-12)

lr 3e-6, kl 3e-3, PPO_MICRO_BATCH_SIZE_PER_GPU=32, seed 1, kempner_h100,
-t 30:00:00. Each student taken at its own best-ID LR:

| Model | student | train job | evals | plot |
|---|---|---|---|---|
| pt1007B | search lr1e-4 | 46193459 | 46193461/63/65 | 46193468 |
| pt4001B | search lr3e-5 | 46193480 | 46193481/93/97 | 46193498 |

Startup config verified from both logs: exp name, lr, kl, and PPO microbatch 32
(it rides sbatch --export=ALL, not an explicit ladder passthrough).

**A naming collision had to be fixed first.** `grpo_exp` carried neither the SFT
format nor the SFT LR, so both runs would have been named
`balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1` -- which already exists for
both 1B models (the answer-only GRPO arms at exactly this lr/kl). Their
checkpoints and results would have been overwritten in place. Now
`...-gtsft200[-<format>]-sftlr<LR>-grpo-...`, matching the Qwen convention. Side
effect: future answer-format GRPO runs get `-sftlr<LR>-` too, so they no longer
resolve to the historical directory names.

---

# GRPO on the search students (results, 2026-09-13)

Both runs finished (46193459 in 7:16, 46193480 in 10:31), 16 eval checkpoints x 3
splits each. Compared against the existing answer-only GRPO arms at the *same*
lr/kl. **The arms are matched on SFT LR**: pulling `actor_rollout_ref.model.path`
from W&B shows the answer arms started from `gtsft200-lr1e-4` (pt1007B) and
`gtsft200-lr3e-5` (pt4001B) -- each model's ID-best SFT LR, the same rule used to
pick the search students. (The 08-25 log's headline "GT-SFT(3e-5)" is wrong for
pt1007B; check the run config, not the prose.)

## Step 1600, pass@1 / pass@32

| run | ID n=3,4 | n=5 | n=6 |
|---|---|---|---|
| 1007B answer | 0.4774 / 0.5055 | 0.0952 / 0.1165 | 0.0128 / 0.0196 |
| 1007B search | 0.4684 / **0.6088** | 0.0271 / 0.0671 | 0.0000 / 0.0006 |
| 4001B answer | 0.4664 / 0.4945 | 0.1102 / 0.1303 | 0.0176 / 0.0230 |
| 4001B search | **0.4995** / **0.6489** | 0.0749 / **0.1489** | 0.0037 / 0.0114 |
| 1007B GRPO only | 0.1976 / 0.2056 | — (evals missing, backfill 46273793) | 0.0091 / 0.0099 |
| 4001B GRPO only | 0.3139 / 0.3250 | 0.0846 / 0.0985 | 0.0205 / 0.0247 |

**On ID the SFT-stage verdict inverts.** Pre-GRPO, answer won 11 of 12 cells;
after GRPO the search student matches or beats it: pt4001B by +0.033 pass@1 and
+0.154 pass@32, pt1007B a pass@1 tie (-0.009, inside noise) with +0.103 pass@32.

**The diversity mechanism, measured.** Coverage retained from SFT to step 1600,
and the pass@32 - pass@1 gap at 1600 (distance above y=x):

| run | ID retained | ID gap | n=5 retained | n=6 retained |
|---|---|---|---|---|
| 1007B answer | 0.65x | 0.0281 | 0.46x | 0.35x |
| 1007B search | 0.77x | **0.1404** | 0.60x | 0.10x |
| 4001B answer | 0.62x | 0.0281 | 0.46x | 0.31x |
| 4001B search | **0.83x** | **0.1494** | **0.82x** | 0.50x |

The answer arms end essentially *on* y=x -- 32 samples find ~3 prompts per 1000
that one sample missed, i.e. near-total sampling collapse. The search arms hold a
14-15 point gap. The GRPO-only arms run *along* y=x for their entire trajectory:
plain GRPO never has coverage above its accuracy at all.

**OOD still favours answer in absolute terms**, except pt4001B n=5 pass@32 (search
0.1489 vs 0.1303; at their best checkpoints, search 0.1119/0.2067 @1500 vs answer
0.1229/0.1446 @1400 -- near-equal accuracy, 43% more coverage). The deficit is
**inherited from the SFT stage, not caused by GRPO**: the search format is far more
LR-fragile OOD, so pt1007B's ID-best LR (1e-4) left its search student at n=6
pass@32 0.0057 vs the answer student's 0.0557 -- 10x worse before RL started, and
GRPO cannot recover it (ends at 0.0006). pt4001B, at the gentler 3e-5, keeps its
search student alive OOD and is the arm where search competes. This is exactly the
ID-selection trap flagged in the LR sweep section above.

Caveats: single seed; ID pass@1 differences (~0.03) are ~1.5 SE, the pass@32
differences (0.10-0.15) are ~5-7 SE.

## Figures

`figures/olmo2-ladder/` is now one folder per comparison set, with a README
mapping folder -> comparison -> tag -> regeneration command (folder names are
descriptive and deliberately *not* the `--tag`; pass `--set <folder>` to write
into an existing one):

- `1b-sftgrpo2formats/` — answer vs search, 2 models (`--tag format-grpo`)
- `1b-grpo-vs-sftgrpo2formats/` — the same plus GRPO-only (`--tag arms3`)

`plot_olmo2_ladder_pareto.py` and `plot_passk_curve.py` gained `--set`, plus
`--run-color` / `--run-linestyle` so a second dimension (SFT format) can ride on
line style instead of overloading hue.

## Next

The one configuration that could make search win on both axes: GRPO on the
pt1007B search student built at **lr 1e-5 or 3e-5** instead of 1e-4, i.e. a
student that is not already OOD-dead. `verify` remains unrun.

---

# 2026-09-14: 7B GRPO runs killed by a disk quota, not OOM; checkpoints moved

All four 7B search-GRPO runs (46344661/687/709/726) FAILED 2-2.7 h in. Diagnosis:

- They died within **90 seconds of each other** (02:43:13-02:44:42) on **four
  different healthy nodes**, at four different training steps (149, 151, 267, 300).
  A per-job resource limit cannot do that; a shared-resource failure can.
- **No traceback in any log** — they stop mid-line. Expected for this cause:
  `logs/` symlinks into the same volume, so once it was over quota Python could not
  write the traceback. An OOM would have left a CUDA OOM trace or a
  `slurmstepd: oom-kill` line. W&B memory was <60%, consistent.
- Confirmed directly: `dd` to the lab tree returned **`Disk quota exceeded`**.
  `df` showing 33% is the 2 PB aggregate `/holylabs`, not the per-lab quota — that
  is why it looked fine.

The four runs had written **572 GB** between them (~82 GB per saved step for a 7B:
FSDP shards + fp32 optimizer state, `SAVE_FREQ=100`). Deleted them (0 eval JSONs,
nothing evaluated lost); the volume became writable again immediately.

A sweep for leftovers found **zero** merged checkpoints still holding `actor/` or
top-level `*.pt` — `eval.sh`'s merge-then-clean has been working. What exists is
~756 GB of *never-evaluated* checkpoints from other runs (7B pt126B/pt135B GRPO
arms, the two no-SFT 7B runs, a 1B balanced234 run, a Qwen 0.5B ep2 run); left
alone, since deleting those destroys work rather than cruft.

## Checkpoints now live on netscratch

`CHECKPOINT_DIR` default for bingbin, in all 12 scripts that hard-coded it:

    /n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown

`checkpoints/` is real storage there; `sft-checkpoints/` (641 GB), `sft-data/` and
`opd-data/` are symlinks back to the holylabs tree, so one `CHECKPOINT_DIR` still
resolves both the SFT students being read and the RL checkpoints being written.
Verified by dry-run: the 7B GRPO launcher resolves its SFT student through the
symlink and would write under netscratch.

**Pre-move runs need `CHECKPOINT_DIR=/n/holylabs/.../rl-checkpoints` passed
explicitly** to be evaluated or plotted. netscratch is purge-eligible.

## Relaunched, held behind the migration (2026-09-14)

All four runs resubmitted unchanged except the save cadence. Each pipeline is
train + 3 eval arrays + plot; the train jobs carry
`--dependency=afterok:46411771` (the `migrate-ckpts` rsync), so nothing writes
into the destination tree until that job's verify-and-delete step has finished —
new directories at the destination would have made its whole-tree file/byte
comparison fail and left the source undeleted.

| model | SFT student | old job | train | evals (n=3,4 / 5 / 6) | plot |
|---|---|---|---|---|---|
| pt135B | search lr1e-4 | 46344661 | 46449062 | 46449093/115/126 | 46449147 |
| pt135B | search lr1e-5 | 46344687 | 46449149 | 46449150/51/52 | 46449153 |
| pt3896B | search lr3e-5 | 46344709 | 46449154 | 46449155/56/57 | 46449158 |
| pt3896B | search lr1e-5 | 46344726 | 46449159 | 46449172/86/87 | 46449188 |

Config identical to the killed attempt (verified against their logs): GRPO
lr 3e-6, kl 3e-3, seed 1, `kempner_h100`, 4 x H100, `-t 30:00:00`,
`PPO_MICRO=16`, logprob micro 16, vLLM util 0.4, tp 1, `CHECKPOINT_DIR` now
defaulting to netscratch and the SFT students resolving through the
`sft-checkpoints` symlink.

**`SAVE_FREQ=200` with `EVAL_STEP_MULT=200` and `EVAL_ARRAY=1-8`** (steps
200..1600). Halves the disk cost: 8 saved steps x ~82 GB = ~660 GB per run,
~2.6 TB across the four, against ~5.2 TB at `SAVE_FREQ=100`.

**Off-grid final checkpoint.** `total_training_steps` is 1635 for a 1-epoch run
at batch 256, and verl saves unconditionally on the last step
(`ray_trainer.py:1333`, `is_last_step or global_steps % save_freq == 0`), so each
run also writes `global_step_1635/`. It is not on the 200-grid, so the eval array
never touches it and it keeps its `actor/` shards plus fp32 optimizer state
(~82 GB x 4 = ~330 GB). Step 1635 is scientifically redundant with 1600 (35
steps), so the follow-up is to delete those four directories once the runs
finish, not to evaluate them.

Per-lab netscratch quota could not be read from the login node (no `lfs quota` —
it is NFS, and `df` reports the 4 PB aggregate at 91% full). Same blind spot that
hid the holylabs quota; a 64 MB `dd` probe succeeds today.

### Migration done, all four released (2026-09-14 15:32)

`46411771` COMPLETED in 3:22:20 — rsync 3 h 18 m, verify byte-exact
(58,462 files / 1.84 TB both sides, 0 pending transfers), source deleted and
replaced by a symlink. The four train jobs released on it immediately and are
running on separate nodes; `save_freq: 200` and `Total training steps: 1635`
confirmed in all four logs, `default_local_dir` under netscratch, no tracebacks.
First pre-train validation: pt135B search lr1e-4 at `val-core reward mean@4`
0.243.

Note on watching the rsync: `squeue`'s `TIME_LEFT` is the unused walltime of the
12 h limit, not an estimate of remaining work — it read "9 h left" at the 3 h
mark with only ~20 min of copying to go. For progress, read the log's rsync
`--info=progress2` byte counter, not the queue.

## pt3896B resumed for the last 235 steps (2026-09-17)

The two pt135B runs finished (`COMPLETED`, 17:31 and 14:23). Both **pt3896B**
runs hit `TIMEOUT` at exactly 30:00:00 with `global_step_1400` as their last
save — 235 steps short of `total_training_steps=1635`.

Resume was possible because nothing had stripped the optimizer state. `eval.sh`
merges a checkpoint and then deletes `actor/`, but the chained eval arrays
(`46449155/56/57`, `46449172/86/87`) were `CANCELLED` when the train jobs timed
out, so every `global_step_*` still had its `optim_world_size_4_rank_*.pt` and
`data.pt`. **A timed-out run whose evals already ran is not resumable** — check
for `actor/` before assuming otherwise.

The mechanism is stock verl: `trainer.resume_mode: 'auto'` (never overridden by
`train_grpo.sh`) makes `_load_checkpoint` read
`latest_checkpointed_iteration.txt`, load actor + optimizer + `data.pt`, and
continue. `del_local_ckpt_after_load: False`, so step 1400 survives the load.
Re-running the original launcher is the whole procedure — the exp names it
builds are byte-identical, so verl resumes instead of restarting:

    SAVE_FREQ=200 EVAL_STEP_MULT=200 EVAL_ARRAY=1-8 \
    GT_FORMAT=search KLS=3e-3 GRPO_LR=3e-6 \
    PPO_MICRO=16 LOGPROB_MICRO=16 GRPO_TIME=12:00:00 \
    SFT_LR=1e-5 bash scripts/run_olmo2_7b_gtsft_grpo.sh pt3896B   # and SFT_LR=3e-5

| run | old job | resume job | evals (n=3,4 / 5 / 6) | plot |
|---|---|---|---|---|
| pt3896B search sftlr1e-5 | 46449159 | 46908163 | 46908171/178/181 | 46908213 |
| pt3896B search sftlr3e-5 | 46449154 | 46908217 | 46908220/261/289 | 46908301 |

12 h walltime: step 1200→1400 took 4 h 42 m (~85 s/step), so 235 steps ≈ 5.5 h
plus `val_before_train` and model load.

### Nothing to move or rename afterwards

A resume writes `global_step_1600/` and `global_step_1635/` into the *same*
`checkpoints/<MODEL>/<EXP>/` directory under the same `global_step_N` naming —
`Checkpoint path:` in the resume log is byte-identical to the original's. The
checkpoint tree is already consolidated; there is no per-job subdirectory.

What does *not* consolidate on its own is the **training log**, and that is the
only reason `val_reward_vs_step.png` would truncate: it is parsed from
`logs/train_grpo.sh-<jid>-1.out`, and `find_training_log()` returned the first
log matching the run, i.e. the older job — so the curve stopped at 1400.

Fixed in code rather than by hand-merging logs (`gather_experiment.py`,
`plot_results.py`):

- `find_training_logs()` (plural) returns **the resume chain**: start at the
  newest matching log and walk back while each job printed verl's
  `Setting global step to N`, stopping at the job that started from scratch.
- `load_val_reward_curve()` merges `parse_step_metrics` across that chain, newer
  jobs winning on overlapping steps (a resume re-validates at N via
  `val_before_train`; the fresher number is the right one).

The chain boundary is load-bearing, not decoration. These exp names are reused
by the **abandoned** 2026-09-13 attempts (`46344726`, `46344709`, killed by the
holylabs quota at steps 100 and 150), which are a different trajectory, not an
earlier segment of this one. Naively merging every name-matching log would splice
them in. Verified: pt135B resolves to a single log and a clean 0..1600 curve with
its stale attempts excluded.

Cost control: `_resumed_from_step()` scans a bounded 8 MB prefix (the marker sits
~120 KB in) — these logs reach 100 MB and the pass-2 whole-file fallback still
only runs when no log has a config header at all.

### Outcome (2026-09-18)

Both resumes `COMPLETED` in 5:51:40 / 6:01:13 (~85 s/step held). Each exp dir
now holds all 8 grid checkpoints, `latest_checkpointed_iteration.txt` = 1600,
and the merged val-reward curve is a continuous 33 points over 0..1600:

| run | 1450 | 1500 | 1550 | 1600 |
|---|---|---|---|---|
| pt3896B search sftlr1e-5 | 0.592 | 0.573 | 0.595 | 0.596 |
| pt3896B search sftlr3e-5 | 0.621 | 0.623 | 0.634 | 0.634 |

**Correction: `global_step_1635/` is never written, so there is nothing to
delete.** The earlier prediction above was wrong. verl's unconditional last-step
save is gated on `global_steps >= total_training_steps` (1635), but
`train.parquet` is data-limited to 1633 steps at batch 256, so the condition
never becomes true — the epoch loop simply exits and the final save is 1600, on
the grid. Confirmed on all four runs, including the two pt135B ones that
completed without ever timing out: every one stops at `global_step_1600`. The
~330 GB cleanup item is moot.

**Log-reading trap hit while checking this** (now in CLAUDE.md "Known issues"
and the `parse_step_metrics` docstring): `grep -aoE "\bstep:[0-9]+" <log> | tail -1`
reported the last step as **338**. That is `timing_s/step:338.23` — seconds,
truncated at the decimal — because `\b` matches after the `/`. verl's step line
also carries `perf/time_per_step:` and `training/global_step:`. It fails
silently into a plausible number; use `parse_step_metrics()`, whose regex
requires the trailing `" - "`.
