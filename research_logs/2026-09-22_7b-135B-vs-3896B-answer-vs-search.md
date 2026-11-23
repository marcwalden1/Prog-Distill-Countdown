# OLMo-2-7B pt135B vs pt3896B: answer-format vs search-format GT-SFT, across GRPO

**Date:** 2026-09-22
**Script:** `scripts/plot/plot_olmo7b_format_arms.py`
**Figures:** `figures/olmo2-ladder/7b-135B-vs-3896B-formats/{id_vs_ood,pass1_vs_pass32}.png`
**Data:** `analysis_data/olmo7b_format_arms_passk.csv` (108 rows)

## Arms

GRPO config is matched across all four: lr 3e-6, kl 3e-3, seed 1, steps 200..1600.
Only the SFT stage differs. Step 0 of each curve is that arm's `sft_final`.

| model | format | SFT LR | GRPO exp |
|---|---|---|---|
| pt135B | answer | 3e-5 | `balanced-simple-gtsft200lr3e-5-grpo-lr3e-6-kl3e-3-seed1` |
| pt135B | search | 1e-4 | `balanced-simple-gtsft200-search-sftlr1e-4-grpo-lr3e-6-kl3e-3-seed1` |
| pt3896B | answer | 1e-5 | `balanced-simple-gtsft200lr1e-5-grpo-lr3e-6-kl3e-3-seed1` |
| pt3896B | search | 3e-5 | `balanced-simple-gtsft200-search-sftlr3e-5-grpo-lr3e-6-kl3e-3-seed1` |

**LR selection: highest ID pass@1 at step 1600.** Forced for the answer arms (one
GRPO run each). For the search arms it was a real choice, and *not a neutral one* —
the losing LR (1e-5 in both models) has roughly double the OOD coverage:

| arm | SFT LR | ID p@1 / p@32 | n=5 p@1 / p@32 |
|---|---|---|---|
| pt135B search | **1e-4** | **0.4915 / 0.6389** | 0.0277 / 0.0857 |
| pt135B search | 1e-5 | 0.3992 / 0.4594 | **0.0907 / 0.1374** |
| pt3896B search | **3e-5** | **0.5948 / 0.7633** | 0.0630 / 0.1387 |
| pt3896B search | 1e-5 | 0.5537 / 0.7362 | **0.0851 / 0.2074** |

`--all-lrs` draws the runner-ups too. The same ID-vs-OOD LR split was already
recorded for answer-format SFT in `2026-08-31_olmo2-7b-gtsft-lr-sweep.md` (finding 3).

## Findings (step 1600 unless noted)

1. **Format decides what GRPO can keep, more than pretraining budget does.**
   Answer-format arms collapse coverage: ID pass@32 0.802 -> 0.501 (pt135B) and
   0.834 -> 0.443 (pt3896B), i.e. GRPO ends *below* where SFT started while pass@1
   roughly doubles. Search-format arms hold it: 0.784 -> 0.639 (pt135B) and
   0.845 -> 0.763 (pt3896B). In the pass@1-vs-pass@32 figure the answer arms dive
   toward the diagonal; the search arms travel right along the top.
2. **Search-format buys ID and sells OOD.** At matched GRPO, search beats answer on
   ID pass@1 in both models (0.4915 vs 0.4704; 0.5948 vs 0.4293) but is 3-4x worse
   on n=5 pass@1 (0.0277 vs 0.1095; 0.0630 vs 0.1331) and essentially dead on n=6
   (<=0.001 vs 0.018-0.031). The search arms shown here start OOD-damaged at step 0 —
   that is the high-LR effect above, not something GRPO did.
3. **Pretraining budget shifts the whole trajectory right, not up.** pt3896B leads
   pt135B on ID at every step within a format, but the two answer arms land on
   almost the same n=5 pass@32 (0.147 vs 0.130) — OOD coverage after GRPO is
   roughly budget-independent, matching the 7B-vs-1B convergence in
   `plot_olmo_7b_vs_1b_grpo.py`.
4. **Only the answer arms improve OOD at all.** n=5 pass@1 rises 0.050 -> 0.110
   (pt135B) and 0.066 -> 0.133 (pt3896B); both search arms end at or below their
   SFT origin on every OOD metric.

## Notes

- pass@k caches for the four search-GRPO runs did not exist and were built here
  (`scripts/compute_passk.py`, 96 eval JSONs). pass@32 is coverage (>=1 of 32
  samples exactly correct), *not* the mean@32 in `.scores`.
- Single seed per cell; the 2026-08-31 log estimates ~0.01-0.02 pass@1 of noise on
  these arms, so within-format ID gaps below that should not be read as ordering.
