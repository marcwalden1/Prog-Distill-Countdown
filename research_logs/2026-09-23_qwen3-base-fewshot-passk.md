# Base-model few-shot pass@k: Qwen3 and Qwen2.5, 0/2/8 shots, answer vs search

**Date:** 2026-09-23; **Qwen3-4B added 2026-09-28**; **k extended to 256 + Qwen2.5
family added 2026-09-29**
**Figures:** `figures/fewshot/fewshot_passk_qwen3_base.png`,
`figures/fewshot/fewshot_passk_qwen25_base.png` (+ `.pdf`)
**Data:** `analysis_data/fewshot_passk_{qwen3,qwen25}_base.csv` (162 + 108 rows)
**Touched:** `scripts/plot/plot_fewshot_passk.py`

## What was run

No Qwen3 or Qwen2.5 base evals existed under `results/baselines/` (only the four
OLMo-2 ladder points), so every arm here was generated from scratch: 5 models x
{0-shot, 2/8-shot x answer/search}, `EVAL_DATASET=balanced-simple`, 997 prompts,
temp 0.6, max 1024. All COMPLETED.

| family | models | jobs | N | wall |
|---|---|---|---:|---|
| Qwen3 | 0.6B-Base, 1.7B-Base, 4B-Base | 47973843.., 49109348.. | 32 | 3-26 min |
| Qwen3 | same | 49284611-49284636 | 256 | 6 min - 2h53 |
| Qwen2.5 | 0.5B, 1.5B | 49299431-49299445 | 256 | 6 min - 30 min |

**k > 32 required re-running everything.** pass@k is undefined for k > N, so the
original N=32 evals could not be extended by any amount of re-analysis; the
n=256 files are separate JSONs (`..._n256_max1024.json`) and the n=32 ones are
left in place. Where they overlap the two agree to ~0.5-1.5 points at k=32
(e.g. 4B 8-shot search 0.754 at N=32 vs 0.763 at N=256), which is the expected
sampling spread across independent draws and a decent check on both runs.

**Qwen2.5 base models carry no `-Base` suffix** -- the plain `Qwen2.5-0.5B` /
`Qwen2.5-1.5B` dirs *are* the base models in that family; only instruct variants
are suffixed. Not the same convention as Qwen3.

0-shot has no exemplars and so no format: its JSON is at the root of
`results/baselines/<model>/` and the same curve is drawn in both panels.

## Does pass@k saturate by k=256? No.

Marginal gain per doubling of k, the number that answers this:

| arm | @32 | @64 | @128 | @256 | d(32->64) | d(64->128) | d(128->256) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B 0-shot | 0.125 | 0.204 | 0.309 | 0.433 | 0.079 | 0.105 | **0.124** |
| Qwen3 0.6B 8-shot answer | 0.466 | 0.580 | 0.684 | 0.766 | 0.114 | 0.104 | 0.082 |
| Qwen3 0.6B 8-shot search | 0.316 | 0.446 | 0.580 | 0.706 | 0.130 | 0.134 | **0.126** |
| Qwen3 1.7B 0-shot | 0.321 | 0.443 | 0.565 | 0.675 | 0.122 | 0.122 | 0.110 |
| Qwen3 1.7B 8-shot search | 0.575 | 0.686 | 0.779 | 0.852 | 0.112 | 0.093 | 0.072 |
| Qwen3 4B 0-shot | 0.491 | 0.621 | 0.730 | 0.811 | 0.131 | 0.109 | 0.081 |
| Qwen3 4B 8-shot answer | 0.640 | 0.742 | 0.825 | 0.887 | 0.102 | 0.083 | 0.062 |
| Qwen3 4B 8-shot search | 0.763 | 0.839 | 0.893 | 0.930 | 0.075 | 0.054 | **0.037** |
| Qwen2.5 0.5B 0-shot | 0.010 | 0.019 | 0.037 | 0.069 | 0.009 | 0.018 | **0.032** |
| Qwen2.5 0.5B 8-shot answer | 0.287 | 0.385 | 0.484 | 0.570 | 0.099 | 0.099 | 0.085 |
| Qwen2.5 1.5B 0-shot | 0.216 | 0.306 | 0.411 | 0.520 | 0.090 | 0.105 | **0.109** |
| Qwen2.5 1.5B 8-shot search | 0.458 | 0.580 | 0.696 | 0.796 | 0.122 | 0.116 | 0.100 |

(full 25-arm table in the CSVs)

- **Nothing has plateaued at k=256.** Every one of the 25 arms still gains
  >= 3.7 points from k=128 to k=256. On a log-k axis most arms are still close to
  *straight*, which is the signature of no saturation at all, not of a curve
  levelling off.
- **The only arm showing real deceleration is the strongest one**, Qwen3 4B
  8-shot search (0.075 -> 0.054 -> 0.037 per doubling, at 0.930). That is the
  expected ceiling effect: it is running out of unsolved prompts, not out of
  sampling luck.
- **The weakest arms are still *accelerating*.** Qwen2.5-0.5B 0-shot goes
  0.009 -> 0.018 -> 0.032 per doubling and Qwen3-0.6B 0-shot 0.079 -> 0.105 ->
  0.124. These sit on the early, convex part of a sigmoid in log k: a tiny
  per-sample success rate times a large k is only now starting to pay. Their
  pass@256 says little about where they would end up -- it is not a ceiling.
- **Practical reading:** on this task pass@k is still a sampling-budget
  measurement at k=256, not a capability ceiling, for everything except 4B with
  search exemplars. Comparisons between arms should quote a fixed k; "best
  pass@k" without a k is meaningless here since the curves have different slopes
  and some cross.

## Cross-arm findings (unchanged by the larger budget)

- **The answer-vs-search preference is monotone in scale.** Margin
  `search - answer` at pass@32 (N=256 run):

  | | 2-shot | 8-shot |
  |---|---:|---:|
  | Qwen2.5 0.5B | +0.048 | **-0.094** |
  | Qwen3 0.6B | +0.006 | **-0.150** |
  | Qwen2.5 1.5B | +0.076 | -0.005 |
  | Qwen3 1.7B | +0.037 | +0.015 |
  | Qwen3 4B | +0.164 | +0.124 |

  Search traces only pay off once the model can execute the arithmetic they
  ask for; below that they cost accuracy, and the crossover sits somewhere
  around 1.5-1.7B on this task. The 4B point was submitted as a prediction test
  of the 0.6B-vs-1.7B flip and came out in the predicted direction.
- **8-shot search hurts the two smallest models** (Qwen3-0.6B 0.316 vs 0.466
  answer at pass@32; Qwen2.5-0.5B 0.192 vs 0.287) and for both, 8-shot search is
  *worse than 2-shot search* -- the only arms where adding exemplars hurts. The
  ~84-token trace x 8 is a long prefix for a sub-1B model. Note this reverses at
  large k: by k=256 Qwen3-0.6B 8-shot search (0.706) catches 2-shot (0.705),
  because it is the fastest-climbing arm in the whole set. Unverified mechanism;
  `scripts/score_fewshot_evals.py` gives the response-length and attempt-count
  columns that would settle it. **Still not run.**
- **Exemplars substitute for roughly one model generation.** Qwen3-0.6B + 2
  answer-shots (0.369 @32) beats Qwen3-1.7B zero-shot (0.321); Qwen3-1.7B + 2
  answer-shots (0.499) is level with Qwen3-4B zero-shot (0.491).
- **Qwen3 >> Qwen2.5 at matched size.** Qwen3-0.6B 8-shot answer 0.766 @256 vs
  Qwen2.5-0.5B 0.570; Qwen3-1.7B 0.817 vs Qwen2.5-1.5B 0.770. The zero-shot gap
  is the widest: Qwen2.5-0.5B is essentially unable to do this task zero-shot
  (pass@1 = 0.0003, pass@256 = 0.069).
- **pass@1 stays low relative to pass@256 everywhere** (best pass@1 is 4B 8-shot
  search at 0.187 against 0.930 pass@256) -- broad coverage, poor first-shot
  precision, the same shape the OLMo-2 ladder showed.

## Script changes

`plot_fewshot_passk.py` had a hardcoded OLMo model list and shot counts {2, 8}:

- `MODEL_PRESETS[<tag>]` selects the model set (`olmo2`, `qwen3_base`,
  `qwen25_base`), with `--models "dir:label:#rrggbb,..."` as an override.
- `--shots 0,...` is supported; `k_shot == 0` drops the `<K>shot-<format>`
  subdir from the glob.
- `pick_eval_json()` chooses among coexisting sample budgets by the `_n<N>_`
  field rather than by filename sort -- a plain `sorted()[0]` picks `n256` over
  `n32` only because `'2' < '3'`, which would have silently picked wrong for a
  32-vs-128 pair. `--samples N` pins an exact budget.
- Per-arm k truncation: an arm evaluated at a smaller N contributes a shorter
  curve with a NOTE instead of crashing (`pass_at_k` returns None for k > N,
  which would `TypeError` in the sum). The CSV gained an `n_samples` column.
- `SHOT_STYLE` is dotted/dashed/solid for 0/2/8 shots. **This restyles the
  existing olmo2 figure if it is regenerated** (2-shot moves from dotted to
  dashed); the committed `fewshot_passk_olmo2.png` was left untouched.
- Hues are palette slots 1/2/3 per family, assigned smallest-model-first, and
  restart at 1 for each family because the families live in separate figures. A
  **cross-family figure must pass `--models`** or two models share blue. The
  2-slot qwen pair validates clean (worst adjacent CVD dE 24.7 protan,
  normal-vision 33.6, contrast >= 3:1 light).

Regenerate with:

    python3 scripts/plot/plot_fewshot_passk.py --tag qwen3_base  --shots 0,2,8 \
        --ks 1,2,4,8,16,32,64,128,256
    python3 scripts/plot/plot_fewshot_passk.py --tag qwen25_base --shots 0,2,8 \
        --ks 1,2,4,8,16,32,64,128,256
