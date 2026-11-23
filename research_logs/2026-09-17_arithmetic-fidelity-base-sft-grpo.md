# Arithmetic fidelity across base → SFT → GRPO (2026-09-17)

## Question

Rollouts contain format-valid equations paired with a wrong evaluation of
themselves, e.g. `(87 + 17 + 60) / 9 = 13`. **Is that error developed by RL, or
already present in the base model?** The proposed measurement was the NTP loss
on the value after `=`, tracked from the base model through RL.

Answered here for the OLMo-2 GT-SFT runs (`--format search`), which is the
setting where the question is well posed.

## Why `search`-format GT-SFT is the right place to ask

`sft_formats.py --format search` installs trace lines of a fixed shape:

    Try 53 - (22 * 3) = -13; too low.
    Try 3 * 22 + 53 = 119; too high.
    Try 53 + 22 + 3 = 78; that is the target.

So the arithmetic-claim positions are **specified by the format** and the correct
value at each is **uniquely determined**. Two things follow:

- There *is* a ground-truth answer token, unlike free-form CoT. In Countdown the
  `<answer>` tag holds an *expression*, not a value, so outside a trained trace
  format the `= <value>` position exists only in model-generated text whose
  prefix the measured model chose itself.
- The multiple-valid-solutions confound that made `ce_answer` uninterpretable in
  `2026-08-06_gt-ntp-distribution-diagnosis-kl3e3-2ep-seed2.md` (GT-answer CE
  rose 0.91 → 2.0 while eval mean@1 rose 0.13 → 0.43) does **not** apply: an
  expression has one value.

For contrast, the same question is close to unmeasurable on the Qwen runs. On
`Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1` the fraction of outputs showing
any `expr = value` line collapses from 0.78 at step 150 to ~0.10 for the rest of
training (consistent with `2026-07-08-generation-evolution-across-grpo.md`), so
any rollout-derived arithmetic rate is dominated by drift in *what the model
chooses to write down*.

## Two probes

1. **`scripts/probe_arith_fidelity.py`** — behavioral, reads stored eval JSONs,
   no GPU. Parses every `Try <expr> = <value>; <verdict>` line, recomputes
   `eval(expr)` with `grader_utils.evaluate_equation`, compares to the claimed
   value. `--self-test` runs the parser over ground-truth traces rendered by
   `sft_formats` and asserts zero miscomputations (**899 Try-lines from 300 GT
   traces → 0 miscomputations, 0 verdict errors**).

2. **`scripts/diagnose_arith_ntp.py`** (+ `.sh`) — teacher-forced, 1×H100.
   Needed because the behavioral probe can only score arithmetic a model chooses
   to show, and base models cannot emit the trace format without few-shot
   exemplars — a *different conditioning* from the SFT/GRPO checkpoints. Teacher
   forcing scores base, SFT and every GRPO step on byte-identical continuations.

### The metric has to be split three ways

Pooling the `= value` positions hides the effect, because the positions measure
different things:

| span | what it measures |
|---|---|
| rejected attempt (`too high`/`too low`) | **arithmetic evaluation** — clean probe |
| final line (`that is the target`) | mostly the **format prior** that a trace ends at the target |
| the `too high`/`too low` phrase | **surface coherence** — does the verdict match the model's *own* claimed value; can be right while every number is wrong |

## Behavioral results

`OLMo-2-0425-1B-pt4001B`, `balanced-simple` (997 prompts × 8 of 32 samples),
temp 0.6. SFT = `gtsft200-search-lr3e-5-seed1`; GRPO = `-sftlr3e-5-grpo-lr3e-6-kl3e-3-seed1`.
Full curve: `analysis_data/arith_fidelity_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv`.

| stage | Try/out | rejected-line miscompute | final-line miscompute | final claims `= target` | verdict error |
|---|---:|---:|---:|---:|---:|
| base 2-shot | 1.71 | 0.964 | 0.977 | 0.898 | 0.399 |
| base 8-shot | 13.60 | **0.945** | 0.928 | **0.160** | 0.635 |
| SFT (200 steps) | 3.12 | **0.954** | 0.786 | **1.000** | **0.002** |
| GRPO 100 | 3.30 | 0.964 | 0.680 | 0.992 | 0.006 |
| GRPO 400 | 4.43 | 0.957 | 0.625 | 0.975 | 0.027 |
| GRPO 800 | 6.34 | 0.956 | 0.543 | 0.973 | 0.049 |
| GRPO 1200 | 8.89 | 0.926 | 0.503 | 0.993 | 0.122 |
| GRPO 1600 | 12.52 | **0.918** | 0.493 | 0.990 | 0.156 |

Scale / pretraining-amount cross-section (same probe, `base 8-shot` and
`SFT-lr3e-5`, plus the step-1600 GRPO point where that arm exists):

| model | base 8-shot rej. | SFT rej. | GRPO-1600 rej. |
|---|---:|---:|---:|
| OLMo-2-1124-7B-pt3896B | 0.908 | **0.841** | — |
| OLMo-2-1124-7B-pt135B | 0.959 | 0.907 | 0.912 |
| OLMo-2-0425-1B-pt4001B | 0.938 | 0.900 | 0.918 |
| OLMo-2-0425-1B-pt1007B | **0.979** | 0.905 | 0.886 |

## Findings

1. **Sampled arithmetic evaluation is flat at ~0.84–0.98 error across base →
   SFT → GRPO.** It is a base-model property; neither the SFT warmup nor 1600
   steps of GRPO creates it or fixes it. (Teacher-forced finding 8 qualifies
   this: the *distribution* over the true value does move — SFT improves it,
   GRPO gives part of that back — without changing the sampled error rate.) More pretraining and more scale
   help only slightly (best: 7B-pt3896B at 0.841; worst: 1B-pt1007B base at
   0.979). **The answer to the motivating question is: already in the base
   model, not developed by RL.**

2. **What SFT installs is the "= target" assertion habit.** Base models assert
   the target on 16% of final lines (8-shot); after 200 SFT steps it is
   **100.0%**, and GRPO holds it at ~0.97–0.99. This is the mechanism behind the
   symptom: ~95% arithmetic error *composed with* a format prior that the last
   line ends at the target produces confident-looking fabrication. The base model
   has the same bad arithmetic but no habit of pinning the result to the target,
   so it does not *look* like motivated reasoning.

3. **SFT teaches the surface logic of verification near-perfectly while teaching
   none of the arithmetic.** Verdict error goes 0.635 (base 8-shot, ≈ chance) →
   **0.002** after 200 steps. The model reliably says "too high" when its own
   claimed value exceeds the target — and the claimed value is wrong 95% of the
   time. Coherent-looking traces are not evidence of computation.

4. **GRPO erodes that coherence as traces lengthen.** Verdict error climbs
   0.002 → 0.156 while `Try` lines per output go 3.1 → 12.5. Nothing in the
   1.0/0.1/0.0 reward constrains intermediate lines, so the trace is free to
   drift — the same incentive gap documented for trailing junk in
   `2026-08-06_gt-ntp-distribution-diagnosis...md` (finding 5).

5. **What GRPO actually improves is expression selection, not evaluation.**
   Final-line miscompute — does the submitted expression really hit the target —
   falls monotonically **0.786 → 0.493** over 1600 steps, while rejected-line
   miscompute does not budge. The reward acts on which expression is emitted,
   not on the model's ability to evaluate one.

6. **Wrong values are only mildly pulled toward the target, and RL reduces the
   pull.** Among miscomputed rejected lines, the fraction whose claimed value is
   closer to the target than the true value is: 0.673 (base 8-shot) → 0.634
   (SFT) → ~0.41–0.56, trending to ≈0.5, across GRPO. Caveat: there is no
   calibrated null for this statistic (a random wrong value is not 50/50 closer),
   so read it only as a relative comparison across stages.

## Caveats

- **The base rows are 8-shot**, which is a different conditioning from the
  SFT/GRPO checkpoints (and it shows: 13.6 `Try` lines/output vs 3.1 — the base
  models ramble). Base-vs-SFT is therefore confounded; the flatness across
  SFT → GRPO, which is same-conditioning, is the solid part. Closing that gap is
  exactly what the teacher-forced probe is for.
- **200 SFT steps × batch 64 = 12,800 examples is a format-installation budget,
  not a capability budget.** `sft-data/gt/balanced-simple/gt-search.parquet` has
  418,136 rows, so the SFT stage sees **3.1% of one epoch**
  (`run_olmo2_gt_sft_ladder.sh:42` says as much: "a warmup, not a
  distillation"). So "search SFT fails to teach arithmetic" is
  **not** a supported claim — that was never tested. The informative contrast is
  that the *same* budget moves the target-assertion habit 0.16 → 1.00 and
  verdict coherence 0.635 → 0.002 while leaving arithmetic error statistically
  unchanged. The budget is plainly sufficient to install a format prior; whether
  a capability-scale budget would move arithmetic is open.
- The probe only sees lines matching the trained shape. Format adherence is
  **1.000 at every stage here**, so no denominator drift, but a run whose format
  decays would need the adherence column read alongside the rates.
- 8 of the 32 stored generations per prompt are read (`--max-samples`), which is
  ~8k outputs per stage — ample for these rates, but the numbers are not
  bit-identical to a full 32-sample pass.
- **Checkpoint-tree trap (cost me a wrong claim, 2026-09-18).** I first reported
  the 7B-pt135B *answer*-format GRPO arm as "died around step 300" after reading
  only the default tree (`/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown`),
  which holds the documented partial, aborted copy (3 steps, max 300). The
  **complete** run — 8 steps, 200…1600, all merged — is in the third tree,
  `/n/netscratch/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`, exactly as
  CLAUDE.md warns. `holylabs/.../rl-checkpoints/checkpoints` is a *symlink* to
  the default tree, so checking it is not an independent check. Always resolve
  all three before concluding a run is short or missing.
- **A resumed run does not create a second folder.** The pt3896B search-sftlr3e-5
  arm ran as FAILED (46344709) → TIMEOUT (46449154, reached step 1400) →
  COMPLETED (46908217 = wandb `c9ae112u`, `Resuming from .../global_step_1400`,
  1400 → 1600 of a 1635-step target). Same `EXP_NAME`, so verl resumed in place;
  steps 200–1600 all live in the one directory. What lagged was the *evals*
  (steps 1400/1600 were evaluated before the continuation produced them), which
  is why a behavioral curve can end early while the checkpoints are complete.
- Extending any GT-derived probe to n=6 hits the known
  `annotated_expressions.json` defect (19.1% of `balanced6` rows have a
  `canonical_pattern` that does not use each letter once; see CLAUDE.md). n=3,4
  (`balanced-simple`) is clean.

## Teacher-forced probe

`scripts/diagnose_arith_ntp.py`. For each puzzle it renders the exact
`search` response GT-SFT trains on (`render_response`, byte-identical to the SFT
target) after the byte-exact eval prompt (`render_prompt`, mode="eval"; OLMo-2
has no `chat_template`, so this is the raw parquet string), recovers char spans
for each value and verdict by regex on the rendered text, and reports
per-(stage, dataset):

- `ce_reject_value` / `ce_final_value` / `ce_verdict` — CE on the three span
  classes above (`n_attempts` fixed at 3 so every probe row has one shape).
- `logp_true_slot` / `logp_target_slot` — at the **last** rejected value slot,
  the same prefix is scored twice, once forcing the true value and once forcing
  the target. Summed over the number's tokens, because OLMo's BPE splits some
  numbers (`-13` and `1000` are 2 tokens each) — "the answer token" is not a
  single token, so spans are masked by char offset, never by position.
- `p_true_norm = sigmoid(logp_true − logp_target)` — **entropy-invariant**, which
  matters because raw CE moves with global sharpening (the 2026-08-06 run
  collapsed 1.31 → 0.40 nats in 150 steps). Report this next to the CE curves.
- Drift controls at the same positions: entropy, top-1 prob, KL(ckpt ‖ base).

Ran as job `46948932` (1×H100, `kempner_requeue`, COMPLETED in 5:07). CSV:
`analysis_data/arith_ntp_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv`.

| stage | ce_reject_value | ce_verdict | ce_final_value | p_true_norm | frac_true_preferred | entropy | KL‖base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | **5.388** | 2.120 | 1.801 | **0.121** | **0.032** | 1.664 | 0 |
| SFT | **3.373** | **0.000** | **0.000** | **0.578** | 0.627 | 0.390 | 1.233 |
| GRPO 100 | 3.670 | 0.000 | 0.329 | **0.707** | 0.788 | 0.373 | 1.260 |
| GRPO 400 | 3.699 | 0.001 | 1.038 | 0.577 | 0.620 | 0.391 | 1.261 |
| GRPO 800 | 3.812 | 0.001 | 1.492 | 0.542 | 0.578 | 0.387 | 1.245 |
| GRPO 1200 | 3.746 | 0.008 | 2.092 | 0.501 | 0.505 | 0.409 | 1.221 |
| GRPO 1600 | 3.673 | 0.028 | 1.610 | **0.459** | 0.436 | 0.412 | 1.216 |

### Teacher-forced findings

7. **The base model has a strong target-pull in its distribution.** At a rejected
   attempt's value slot it assigns the **target** 7:1 odds over the true value
   (`p_true_norm` 0.121) and prefers the true value on only **3.2%** of puzzles.
   So the "write the target after `=`" prior is *pre-existing*, not created by
   SFT — which corrects what finding 2 alone would suggest. The two are
   different positions and must not be conflated: base already prefers the
   target *distributionally* at intermediate slots, while *behaviorally* it
   asserts the target on only 16% of final lines. SFT changes the second to
   100%; it does not create the first.

8. **SFT measurably reduces that target-pull, and GRPO partially restores it.**
   `p_true_norm` goes base 0.121 → SFT **0.578** → GRPO peak 0.707 (step 100) →
   **0.459** at step 1600, i.e. back to slightly preferring the target.
   `frac_true_preferred` traces the same path (0.032 → 0.627 → 0.436). So 3.1%
   of an epoch of `search` SFT *does* teach something real about evaluating
   non-target expressions — its correct evaluations of wrong expressions are not
   inert — and RL walks part of it back. **This qualifies finding 1: sampled
   arithmetic accuracy is flat across all stages, but the distribution over the
   true value is not, and the RL-direction change is a degradation.**
   `ce_reject_value` agrees (base 5.388 → SFT 3.373 → 3.67–3.96 across GRPO,
   every GRPO step worse than SFT), but on its own it is confounded by the
   entropy collapse below; `p_true_norm` is the unconfounded version and moves
   the same way, so the effect is real rather than a sharpening artifact.
9. **The two format priors are memorized exactly and then eroded.**
   `ce_verdict` and `ce_final_value` both hit **0.000** at SFT — after 12.8k
   examples the verdict phrase and the final `= target` are fully determined —
   then climb under GRPO (verdict 0.000 → 0.028, final 0.000 → 1.610, with a
   3.703 spike at step 1400 that matches that step's behavioral outlier,
   rej_miscompute 0.976). The teacher-forced verdict curve mirrors the
   behavioral verdict-error curve (0.002 → 0.156), so the two probes agree.
10. **Essentially all distribution drift happens at SFT, not during GRPO.**
   Entropy 1.664 → 0.390 and KL(ckpt‖base) 0 → 1.233 at the SFT step, then
   entropy sits at 0.37–0.42 and KL at 1.20–1.28 for all 1600 GRPO steps. This
   is unlike the Qwen base-GRPO run of `2026-08-06_gt-ntp-diagnosis...`, where
   GRPO itself did the sharpening — here GRPO redistributes inside a budget that
   SFT already spent. It is also why the raw CE panel must be read next to the
   drift figure.

**Prediction check:** the closing prediction below (that "the model has no idea"
would dominate over "overridden by the target") was **wrong for the base model** —
base is strongly target-overridden (0.121). It is right from SFT onward: the
model is roughly indifferent between the true value and the target while still
getting 95% of sampled evaluations wrong, so most of its mass is on neither.

## Cross-model replication + the answer-format control (2026-09-18)

Six arms, all GRPO lr3e-6 kl3e-3, each using the SFT LR its GRPO arm actually
started from. `answer`-format arms have **no behavioral probe** — their rollouts
are a bare `<answer>` tag with zero arithmetic claims — so only the teacher-forced
probe applies there, and the forced `search` trace is off-policy for them.

| model | fmt | SFT LR | ce_reject_value (base → SFT → 1600) | p_true_norm (base → SFT → 1600) |
|---|---|---|---|---|
| 1B-pt4001B | search | 3e-5 | 5.39 → **3.37** → 3.67 | 0.121 → **0.578** → 0.459 |
| 7B-pt135B | search | 1e-5 | 5.45 → **3.77** → 4.01 | 0.262 → **0.390** → 0.268 |
| 7B-pt3896B | search | 3e-5 | 4.88 → **2.81** → 3.35 | 0.139 → **0.712** → 0.696 |
| 1B-pt4001B | answer | 3e-5 | 5.39 → 9.93 → **13.33** | 0.121 → 0.095 → 0.185 |
| 7B-pt135B | answer | 3e-5 | 5.45 → 7.94 → **10.65** | 0.262 → 0.216 → 0.238 |
| 7B-pt3896B | answer | 1e-5 | 4.88 → 6.57 → **11.98** | 0.139 → 0.123 → 0.112 |

Figures: `figures/diagnosis/arith_{fidelity,drift}_all.png` and
`arith_ptrue_by_arm.png` — see the Figures section below.

11. **The answer-format control confirms finding 8 is caused by the `search`
    format, not by SFT in general.** `answer` SFT never demonstrates a single
    `= value`; across all three models its `p_true_norm` stays flat and low
    (0.09–0.26 at every stage, never approaching 0.5), while every `search` arm
    jumps at the SFT step. So the metric moves because the trace format
    demonstrates correct evaluation of *wrong* expressions — 12.8k examples of
    it — and not because finetuning or RL does anything to arithmetic per se.
12. **How much of the SFT gain survives GRPO tracks pretraining, and it is the
    clearest cross-model signal here.** 7B-pt3896B keeps essentially all of it
    (0.712 → 0.696); 1B-pt4001B loses about half and ends *below* indifference
    (0.578 → 0.459); 7B-pt135B loses all of it and returns to its base value
    (0.390 → 0.268 vs base 0.262). The 7B pair differs only in pretraining
    tokens (3896B vs 135B) and spans nearly the whole range, so tokens matter
    directly; 1B-pt4001B has comparable tokens to pt3896B but 1/7 the
    parameters and sits in between, so capacity matters too. **The
    RL-degradation in finding 8 is therefore not universal — it is what happens
    to under-pretrained models.**
13. **For answer-format arms, `ce_reject_value` measures format surprise, not
    arithmetic.** It rises 2–7 nats (to 10.6–13.3) while `p_true_norm` stays
    flat, i.e. mass moves away from the true value *and* the target alike —
    exactly what an answer-only model should do when forced through a `Try …`
    trace it wants to replace with an immediate `<answer>`. `ce_verdict` says
    the same (2.6–4.4 throughout, vs ~0.000 for every search arm). **Only
    `p_true_norm` is comparable across SFT formats**, because it is a ratio of
    two continuations at one identical prefix; the raw CE columns are not.

Sanity check that the checkpoints are distinct: across all six CSVs there are
**zero duplicated (ce_reject_value, p_true_norm) rows**, so no two steps in any
arm resolve to the same weights — worth re-running after any checkpoint
consolidation, since identical rows would flag a bad copy rather than a bad
analysis.

## Figures

Three, all in `figures/diagnosis/`. Each is a grid: one metric per row, one
column per SFT format, one line per model — colour encodes the model, and the
search-vs-answer columns are the control contrast.

- `arith_fidelity_all.png` — 6 rows: teacher-forced CE on the true value of a
  rejected attempt, on the verdict, and on the final value; `p_true_norm`; then
  the two behavioral rates. The behavioral rows are empty in the `answer` column
  and say so — those rollouts are a bare `<answer>` tag.
- `arith_drift_all.png` — 3 rows: predictive entropy, top-1 probability,
  KL(ckpt‖base) at the probe positions. Read the CE rows of the fidelity grid
  against this one.
- `arith_ptrue_by_arm.png` — the headline single-metric view of `p_true_norm`,
  same facets, larger.

y is shared across each row so the two formats are directly comparable. Arms use
different save grids (1B every 100, 7B every 200), so GRPO steps sit on a
numeric axis with `base` and `SFT` at fixed negative offsets behind a divider —
they are not RL steps.

`figures/diagnosis/metrics.tex` gives the precise definition of every quantity
in these figures and in the CSVs — masks, aggregation (the teacher-forced
columns are macro, per-prompt-then-over-prompts; the behavioral ones are micro,
pooled over lines), the miscompute tolerance, and which denominator each rate
uses. Compiles standalone; the `% BEGIN BODY`/`% END BODY` block is `\input`-able.

Plotters: `scripts/plot/plot_arith_grid.py --which {fidelity,drift}` (the merged
grids) and `scripts/plot/plot_arith_ptrue_by_arm.py`. `scripts/plot/plot_arith_fidelity.py`
still renders a single arm on its own (`--tag`, and omit `--behavioral` for
answer arms) if one needs a closer look; its per-arm PNGs were superseded by the
grids and deleted.

## Reproduction

```bash
# parser self-test (must be 0 miscomputations)
python3 scripts/probe_arith_fidelity.py --self-test

# behavioral curve
F=balanced-simple_temp0.6_n32_max1024.json
M=OLMo-2-0425-1B-pt4001B
python3 scripts/probe_arith_fidelity.py \
  --stage "base-2shot=results/baselines/$M/2shot-search/$F" \
  --stage "base-8shot=results/baselines/$M/8shot-search/$F" \
  --stage "sft-lr3e-5=results/$M/balanced-simple-gtsft200-search-lr3e-5-seed1/sft_final/$F" \
  --grpo-root results/$M/balanced-simple-gtsft200-search-sftlr3e-5-grpo-lr3e-6-kl3e-3-seed1 \
  --steps 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600 \
  --out analysis_data/arith_fidelity_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv

# teacher-forced diagnosis (1 H100; --account is required on kempner_* partitions)
MODELS=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models
C=/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown
BASE_MODEL=$MODELS/$M \
STAGES="base=$MODELS/$M;sft=$C/sft-checkpoints/$M/balanced-simple-gtsft200-search-lr3e-5-seed1" \
CKPT_ROOT=$C/checkpoints/$M/balanced-simple-gtsft200-search-sftlr3e-5-grpo-lr3e-6-kl3e-3-seed1 \
STEPS=100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600 \
OUT_CSV=analysis_data/arith_ntp_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv \
  sbatch --account=kempner_bingbin_lab -J arith_ntp scripts/diagnose_arith_ntp.sh

# figures (all six arms; the arm->CSV map lives in the plotter)
python3 scripts/plot/plot_arith_grid.py --which fidelity
python3 scripts/plot/plot_arith_grid.py --which drift
python3 scripts/plot/plot_arith_ptrue_by_arm.py

# one arm on its own, if a closer look is needed
python3 scripts/plot/plot_arith_fidelity.py \
  --behavioral analysis_data/arith_fidelity_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv \
  --teacher-forced analysis_data/arith_ntp_olmo2_1B_pt4001B_search_sftlr3e-5_grpo.csv \
  --out-dir figures/diagnosis --tag OLMo-2-0425-1B-pt4001B_search-sftlr3e-5 \
  --title "OLMo-2-1B-pt4001B | search GT-SFT (lr3e-5) -> GRPO lr3e-6 kl3e-3"
```

Re-run the two grid commands after the pt3896B-search evals at steps 1400/1600
land (queued as `46908220_7/_8`); the behavioral rows for that arm currently end
at 1200, and the plotters pick up whatever is in the CSVs.

## Open follow-ups

- **`verify` vs `search` as a natural experiment.** `verify` only ever
  demonstrates `<gt steps> = target`; `search` additionally demonstrates
  *correct* evaluations of wrong expressions. If the target-assertion habit is
  format-induced, `verify`-SFT should show it at least as strongly with less
  exposure to non-target evaluation. No `verify` runs exist on disk; it is a
  200-step SFT (`GT_FORMAT=verify`).
- **Is arithmetic movable at all at this scale?** Partly answered: 3.1% of an
  epoch already moves `p_true_norm` 0.121 → 0.578, so the `search` format's
  correct evaluations of wrong expressions do transfer. What is untested is
  whether a larger budget closes the remaining gap or plateaus — the sampled
  error rate did not budge at all, so distributional gain ≠ behavioral gain here.
- **A bare-context probe is NOT the right way to separate "can't compute" from
  "target is in context"** — recording the reason so it is not attempted naively.
  Dropping the target from the prompt changes much more than the target's
  presence: (a) when the target is in the prompt it is a *copyable* token, so in
  bare context `P(target)` collapses for a trivial reason (no induction-head
  copy) and `p_true_norm` jumps toward 1.0 — that measures token salience, not
  arithmetic; (b) `p_true_norm` is invariant to temperature-like rescaling only
  *within a fixed prefix*, which is why it survived the SFT entropy collapse
  (1.66 → 0.39 nats) but would not survive changing the prompt distribution,
  since the denominator's own prior moves. The confound runs in the direction
  that would manufacture the hypothesis. Better designs, if this is picked up:
  **vary the declared target while holding the expression and its true value
  fixed** (prompt type, trace format and expression all constant; the point
  `target == true value` is a free null), or **hold the prompt fixed and score
  the true value against a decoy** matched to the target in magnitude and token
  length but not named in the prompt. A contrast-free alternative is greedy
  exact-match on the value, comparable across contexts but coarse.
  Not implemented (2026-09-18 decision: not worth over-indexing on).
- **Why does GRPO undo part of the SFT gain (finding 8)?** Nothing in the
  1.0/0.1/0.0 reward scores intermediate lines, so this may be free drift — but
  it moves consistently toward the target, which is the reward-relevant token.
  Worth checking against a `TRAILING_PENALTY`-style intermediate-line penalty,
  and against the `answer`-format GRPO arm, which has no intermediate lines to
  drift.
