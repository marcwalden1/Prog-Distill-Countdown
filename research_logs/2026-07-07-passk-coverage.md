# Pass@k coverage: Qwen2.5-0.5B base vs GRPO checkpoints (2026-07-07)

## Motivation

Measure *coverage* (pass@k at growing k) of the base model vs post-RL
checkpoints, as a candidate predictor of post-RL performance (cf. Yue et al.
2025: RL mostly sharpens the base policy onto solutions it could already
sample). Also inspect raw generations to understand failure modes (format vs
arithmetic) and response-length differences.

## Setup

- Models:
  - `base` = `/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/Qwen2.5-0.5B`
  - `grpo-seed1` = `Qwen2.5-0.5B/balanced-grpo-seed1/global_step_1600`
  - `grpo-warmup` = `Qwen2.5-0.5B/balanced-grpo-warmup0.03-beta2_0.95-seed1/global_step_1600`
  - `1.5B-grpo` = `Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1/global_step_1600`
    (Shlok's canonical teacher; checkpoint under
    `/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints/`)
- Eval: `eval.py`, n=32 samples/prompt, max 1024 tokens, temps {0.6, 1.0},
  datasets {balanced (n=3,4), balanced5, balanced6}. Base-model results live in
  `results/Qwen2.5-0.5B/base/global_step_0/`.
- pass@k: unbiased estimator (Chen et al. 2021), correct = `compute_score`==1.0,
  ≤1000 prompts/file (same cap as `plot_results.py`).
- Tools (new): `scripts/plot_passk.py` (pass@k + `.passk` per-prompt-correct
  caches), `scripts/inspect_generations.py` (format taxonomy + length stats +
  sample dumps).
- Env note: run plotting with `~/miniconda3/envs/common/bin/python3`
  (verl env has no matplotlib); `inspect_generations.py` needs transformers →
  verl env python works.

## Results: pass@32 (unbiased, 32 samples)

| Model | temp | n=3,4 (in-dist) | n=5 (OOD) | n=6 (OOD) |
|---|---|---|---|---|
| base | 0.6 | 0.018 | 0.001 | 0.000 |
| base | 1.0 | 0.004 | 0.000 | 0.000 |
| grpo-seed1 | 0.6 | 0.364 | 0.184 | 0.050 |
| grpo-seed1 | 1.0 | 0.392 | 0.200 | 0.060 |
| grpo-warmup | 0.6 | 0.292 | 0.084 | 0.032 |
| grpo-warmup | 1.0 | 0.327 | 0.117 | 0.042 |

Full pass@{4,8,16,32} table: rerun `scripts/plot_passk.py` (instant via caches).
Figure: `figures/Qwen2.5-0.5B/passk/passk_grid.png`.

Findings:

1. **Base coverage is near zero everywhere** (best 1.8% pass@32, in-dist
   t0.6; exactly 0/1000 on n=6). Unlike the trained models it gets *worse*
   at temp 1.0 → its rare successes are fragile lucky completions, not a
   solving mode that exploration amplifies. GRPO at this scale created
   coverage the base did not have (counterexample to the strong
   "RL only sharpens existing coverage" claim, at least at 0.5B).
2. **grpo-seed1 strictly dominates grpo-warmup** at every k/dataset/temp;
   largest relative gap OOD n=5 (0.200 vs 0.117 at t1.0).
3. **temp 1.0 > temp 0.6** for coverage on all trained-model panels →
   standardize coverage measurements at temp 1.0.

## Why similar val mean@4 but different pass@k (seed1 vs warmup)?

From `.passk` caches at temp 1.0 on `balanced` (= the training val set):

| | mean@1 | breadth (pass@32) | depth (c/n on solved) | near-deterministic (c≥31) |
|---|---|---|---|---|
| grpo-seed1 | 0.264 | 0.392 | 0.674 | 0.035 |
| grpo-warmup | 0.211 | 0.327 | 0.646 | 0.093 |

- Real mean@1 gap (5.3 pts), mostly *breadth*; warmup is ~3x more often
  near-deterministic on its solved puzzles (sharper policy).
- val mean@4 includes the 0.1 format bonus. warmup never misses the answer
  tag (0.0% `no_answer_tag`) and has high `bad_numbers` → more 0.1 mass.
  Expected val score at t1.0: seed1 0.325 vs warmup 0.290 — format bonus
  shrinks the metric gap by ~1/3; n=4 val noise plausibly hides the rest.
- Takeaway: **val mean@4 is a poor coverage proxy** (format bonus +
  prompt-level concentration are invisible in it).

## Generation inspection (format & length)

Artifacts: `results/generation_samples/<run>_<dataset>_temp<T>.txt` (readable
samples) + `results/generation_samples/summary.tsv` (full taxonomy table).
Taxonomy mirrors `compute_score`: only a missing `<answer>` tag scores 0.0;
`bad_numbers` / `unevaluable` / `wrong_value` all get the 0.1 format score.

Key rows (`balanced`, temp 0.6; fractions over 32k samples):

| run | mean/median len | trunc@1024 | correct | wrong_value | bad_numbers | no_tag |
|---|---|---|---|---|---|---|
| base | 385 / 177 | 20.1% | 0.001 | 0.014 | 0.586 | 0.398 |
| grpo-seed1 | 27 / 21 | 0.2% | 0.290 | 0.633 | 0.020 | 0.057 |
| grpo-warmup | 90 / 101 | 0.0% | 0.210 | 0.580 | 0.210 | 0.000 |

- **base**: not a pure format failure — ~60% of samples produce an `<answer>`
  tag (copies the template, hallucinates fake extra "User:" turns), but the
  dominant failure is semantic: can't use each number exactly once
  (`bad_numbers` 59%). 20% of generations are runaway repetition loops
  truncated at the cap.
- **grpo-seed1** (median 21 tokens): `<think>` collapsed; emits an equation
  guess immediately, frequently without verifying the arithmetic (same wrong
  guess repeated across all 32 samples on some prompts). Correctness comes
  from pattern recognition; errors are mostly `wrong_value` with valid
  number usage.
- **grpo-warmup** (median ~100 tokens): the extra length is an artifact —
  it **echoes the user prompt back** (with the example answer mutated to use
  the puzzle numbers) before answering. Always emits the tag; much higher
  `bad_numbers` (21% in-dist, 49% on n=5, e.g. tacking `* 1` onto
  equations).

## Qwen2.5-1.5B post-GRPO comparison (added same day, evals 29155625/29155626)

pass@32, all four models. Figures (2026-07-08 restyle: per-point value labels,
no title, legend at bottom; regenerate with `plot_passk.py --result-root
results --point-labels all --no-title --legend-loc bottom`):

- `figures/passk/passk_all_grpo_only.png` — all 4 models
  (base / grpo-seed1 / grpo-warmup / 1.5B-grpo)
- `figures/passk/passk_3models.png` — without grpo-warmup, labels renamed to
  0.5B-Base / 0.5B-GRPO / 1.5B-GRPO (`--style-slots 0,1,3` keeps 1.5B purple)

| Model | temp | n=3,4 | n=5 | n=6 |
|---|---|---|---|---|
| 1.5B-grpo | 0.6 | 0.958 | 0.792 | 0.362 |
| 1.5B-grpo | 1.0 | 0.972 | 0.841 | 0.388 |
| grpo-seed1 (0.5B) | 1.0 | 0.392 | 0.200 | 0.060 |
| grpo-warmup (0.5B) | 1.0 | 0.327 | 0.117 | 0.042 |

Inspection (`results/generation_samples/1.5B-grpo_*.txt`):

- **Length is genuine search + degenerate repetition after success.** The
  1.5B does guess-and-check: propose `<answer>`, expand the arithmetic
  line-by-line, and if the value misses the target, propose a different
  equation (real verification + backtracking). But once a correct equation
  is found it *repeats the verified answer block in a loop* until the
  1024-token cap. Hence 56% of in-dist t0.6 samples "truncated" at the cap
  while 86% are correct — truncation is mostly post-success padding, not
  cut-off search. Scoring is robust to this because `extract_solution`
  takes the *last* answer tag.
- Length stats: in-dist mean 739 tok (t0.6) / 566 (t1.0); n=5 mean 943/877
  with 84%/65% at the cap — search runs longer on harder puzzles, and some
  OOD truncations likely do cut off search (worth a 2048-token spot-check
  before reading n=5/6 numbers as capability ceilings — note the earlier
  0.5B-progdistill 4096-token re-eval found no truncation effect, but the
  1.5B truncates far more).
- Format taxonomy (in-dist t0.6): 85.9% correct, 13.3% wrong_value, 0.6%
  bad_numbers, 0.1% no_tag — essentially perfect constraint compliance,
  unlike both 0.5B GRPO models.
- **Coverage contrast with 0.5B:** 1.5B's pass@k curves are much steeper in
  k (n=6 t1.0: 0.175→0.388 from k=4→32, >2x; 0.5B curves are nearly flat)
  — its 32 samples explore genuinely different equations, whereas the 0.5B
  models mostly resample the same guess. Sample diversity, not just
  pass@1, is what separates the model scales.
- So the length difference is *mechanistically* the story: 1.5B-grpo
  learned iterate-verify-revise (length = number of attempts); 0.5B
  grpo-seed1 learned single-shot pattern matching (21 tokens); 0.5B
  grpo-warmup learned prompt-echo + single shot (~100 tokens). The 0.5B
  models' "reasoning" carries no search, so extra samples add little
  coverage.

## Pending / next
- [ ] (declined for now / optional) full 0.1-vs-0.0 rescoring pass to pin
      down the val-mean@4 decomposition exactly — partially superseded by
      the taxonomy table above.
- [ ] Coverage metrics beyond pass@k discussed on 2026-07-07: pattern-level
      coverage (via `canonical_pattern_index`), fraction of GRPO groups with
      reward variance at (n=4, t1.0), correct-solution diversity, cognitive
      behaviors (Gandhi et al. 2025).

## Ops notes

- One eval task (base/balanced6/t0.6) died with the known vLLM
  torch.compile-cache JSONDecodeError; retried fine with `--enforce_eager`.
- Eval driver used for one-off checkpoint+temp combos: scratchpad script
  (`eval_passk.sh`, array over 3 datasets, env: CKPT/OUTDIR/TEMP/EXTRA_FLAGS,
  1×H100 kempner_requeue 3h). Recreate from this description if needed.
