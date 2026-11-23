# ID-vs-OOD Pareto: 1.5B teacher GRPO vs 0.5B base GRPO (2026-08-03)

## Runs compared (both 10/10 checkpoints, all 3 datasets)

| Arm | Model | EXP_NAME | Eval tree |
|---|---|---|---|
| teacher GRPO | Qwen2.5-1.5B | `balanced-grpo-lr3e-6-kl3e-3-seed1` | `/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results` |
| student GRPO | Qwen2.5-0.5B | `balanced-grpo-seed1` | `results/` (own tree) |

x = ID `balanced` (n=3,4), y = OOD `balanced5` (n=5) / `balanced6` (n=6); one point
per checkpoint at steps {150…1600}, **pass@1 and pass@32**.

## Metric

Unbiased pass@k (Chen et al. 2021), matching `scripts/plot_passk.py`: mean over
prompts of `1 - C(n-c, k) / C(n, k)`, with `c` = samples scoring exactly 1.0 out of
`n`=32. So pass@1 = mean `c/n`, and pass@32 = fraction of prompts with ≥1 correct
sample. The CSV also carries the older `mean_at_1` / `mean_at_32` columns for
continuity.

**pass@32 ≠ mean@32.** mean@32 averages the raw `compute_score` (0 / 0.1 / 1.0), so
it grants partial credit for valid-but-wrong format; a floor-bound model can score
~0.10 on n=6 with essentially no correct answers. pass@32 counts only correct
answers, and is the larger number wherever the model actually solves anything
(teacher n=5 step 1600: mean@32 `0.386` vs pass@32 `0.692`). pass@1 ≈ mean@1 as
expected (both ~"is a single sample correct"), differing only by the estimator.

## Gotcha: mixed capped / full-set `.scores`

The 1.5B run's `.scores` in sdholakia's read-only tree are **pre-2026-07-13,
1000-prompt-capped** (denominators divide 1000; mtime Apr 23), while the 0.5B ones
are full-set. Mixing them inflates the teacher's OOD numbers and distorts the
comparison — exactly the failure mode in
`research_logs/2026-07-13_fix-maxprompts-cap-fullset-regen.md`. The 1.5B copy in
`results_sdholakia/` covers only Qwen2.5-0.5B and gemma-3-270m, so there was no
full-set cache to reuse.

Effect at step 1600 (teacher, mean@1, the metric the stale caches hold): n=5 `0.405`
capped → **`0.317`** full set (−0.088); n=6 `0.072` → **`0.055`** (−0.017).
Head-slice over-states, as documented.

New scripts re-score the full prompt set and cache in their own dir, so a
read-only source tree is never mutated and a foreign `.scores` is never trusted:

- `scripts/export_idood_scores.py` (+ `scripts/export_idood_scores.sh`, SLURM,
  16 CPUs — the login node has 1) → tidy CSV
  `analysis_data/idood_teacher1p5b_vs_student05b.csv`
  (60 rows: 2 runs × 3 datasets × 10 steps; pass@1/pass@32/mean@1/mean@32 plus
  `n_prompts` and `n_samples` recorded, so a future cap bug is visible in the data).
- `scripts/plot_id_vs_ood_runs.py` → `figures/comparison/id_vs_ood_teacher1p5b_vs_student05b/`
  (`grid.png` = 2×2 grid, plus the 4 panels standalone). Colors from repo `colors.py`.

Validation: independently reproduces the existing full-set caches, e.g. 0.5B n=6
step 1600 `0.0227 / 0.1118`, and the teacher's n=5 step-1600 full-set value agrees
with bingbin's own re-eval of that step (`0.3162` vs `0.3169`, different eval JSON).

## Findings

**No crossing, no trade-off frontier.** Unlike the distill-vs-progdistill
comparisons, the two arms lie on essentially the *same* ID→OOD ray; the teacher
simply travels much further along it. Both arms improve ID and OOD together for the
whole run — there is no regime here where one buys OOD by giving up ID.

Final checkpoint (step 1600), full set:

| Arm | ID pass@1 | n=5 pass@1 | n=6 pass@1 | ID pass@32 | n=5 pass@32 | n=6 pass@32 |
|---|--:|--:|--:|--:|--:|--:|
| teacher 1.5B | 0.860 | 0.318 | 0.059 | 0.965 | 0.692 | 0.292 |
| student 0.5B | 0.290 | 0.093 | 0.022 | 0.364 | 0.123 | 0.044 |

- The 0.5B run's **entire trajectory** ends roughly where the 1.5B run's *first*
  evaluated checkpoint (step 150: ID pass@1 0.337, n=5 0.081) already sits. Its star
  and the teacher's open circle nearly coincide in the pass@1 panels.
- Teacher n=5 pass@1 climbs 0.081 → 0.318; the last ~5 checkpoints bunch tightly
  (ID 0.79–0.86, n=5 0.26–0.32), i.e. late GRPO buys OOD at almost no ID cost.
- **The pass@32 gap is far wider than pass@1 suggests.** On n=5 the teacher ends at
  0.692 vs the student's 0.123 (5.6×), and on n=6 0.292 vs 0.044 (6.6×) — versus
  ~3.4× and ~2.6× on pass@1. The teacher's *coverage* of OOD puzzles is the bigger
  separation; the student almost never has a correct answer anywhere in its 32
  samples on n=6.
- **n=6 pass@1 is near-floor for both arms** (≤ 0.066) and non-monotone — the
  teacher peaks at step 1200 (0.066) above its step-1600 value (0.059). Don't read
  n=6 pass@1 ordering as signal at this resolution; n=6 pass@32 is much better
  behaved (monotone-ish 0.108 → 0.292) and is the panel to use for n=6.
- **0.5B transient dip at step 300**: pass@32 drops to ID 0.074 / n=5 0.003 from
  0.104 / 0.021 at step 150, recovering by step 450 (ID 0.260 / n=5 0.065). Real
  early-GRPO instability, not a plotting artifact. Under mean@32 this looked far more
  dramatic (ID 0.031 vs 0.130) because the collapse was mostly *format* validity;
  pass@32 shows the correctness cost was smaller.

## Reproduce

```bash
EXPORT_RUNS="teacher GRPO (Qwen2.5-1.5B)=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results::Qwen2.5-1.5B::balanced-grpo-lr3e-6-kl3e-3-seed1
student GRPO (Qwen2.5-0.5B)=results::Qwen2.5-0.5B::balanced-grpo-seed1" \
EXPORT_OUT=analysis_data/idood_teacher1p5b_vs_student05b.csv \
  sbatch scripts/export_idood_scores.sh          # ~45 s

~/.conda/envs/verl/bin/python3 scripts/plot_id_vs_ood_runs.py \
  --csv analysis_data/idood_teacher1p5b_vs_student05b.csv \
  --tag teacher1p5b_vs_student05b --out-dir comparison
```
