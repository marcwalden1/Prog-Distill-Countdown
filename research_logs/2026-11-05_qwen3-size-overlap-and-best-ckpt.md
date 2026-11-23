# Qwen3-Base GRPO across sizes: which problems each model solves, final vs best checkpoint (2026-11-04 → 2026-11-05)

A follow-up analysis of the size sweep in `2026-09-23_qwen3-base-grpo-size-rolloutn-lr.md`.
No new training or evals were run; everything comes from existing eval JSONs.

**Runs compared** (all `balanced-simple`, kl=3e-3, 1 epoch, `rollout.n=4`, seed1):

| label | run |
|---|---|
| 0.6B | `Qwen3-0.6B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1` |
| 1.7B | `Qwen3-1.7B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1` |
| 4B   | `Qwen3-4B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1` |

> **Correction (2026-11-05).** The first version of this note, and the commit that added it, used
> the opposite learning rates: 0.6B at lr 1e-6, and 1.7B/4B at lr 3e-6. That selection was a
> mistake. All figures, `analysis_data/qwen3_size_passk.csv` and
> `analysis_data/qwen3_venn_counts_step1600.json` have been regenerated for the selection above.
> The old selection's headline finding, that the final 1.7B checkpoint is below the 0.6B on n=5
> and n=6, came from the 1.7B lr=3e-6 run declining late in training. It does not hold for the
> selection above.

**Eval setup:** temp 0.6, 32 samples per problem, 1024-token cap, full test sets
(n=3,4: 997; n=5: 4620; n=6: 3519).

## 1. Which problems each model solves (step 1600)

Figures, all in `figures/Qwen3-base-grpo/venn/`:
- `venn_*.png`: classic Venn diagram, with counts in every region.
- `euler_*.png`: area-proportional Euler diagram. Circle areas and pairwise overlaps are exact. The
  3-way overlap is approximate, but its error is under 1% of N in every panel. All panels in a
  column share one scale.
- `upset_*.png`: UpSet plot, one bar per exclusive region.

Each figure shows three definitions of "solved": sample #1 correct (= the repo's mean@1), at least
16/32 samples correct, and pass@32 (any of 32 samples correct).

Region counts are in `analysis_data/qwen3_venn_counts_step1600.json`. Keys are membership bits in
the order (0.6B, 1.7B, 4B), so e.g. `"101"` means solved by 0.6B and 4B but not 1.7B. Selected
counts:

| definition | test set | all 3 | only 4B | only 0.6B | only 1.7B | 0.6B∧4B, not 1.7B | 1.7B∧4B, not 0.6B | none |
|---|---|---|---|---|---|---|---|---|
| sample #1 | n=3,4 | 675 | 72 | 10 | 24 | 41 | 117 | 42 |
| sample #1 | n=5 | 571 | 686 | 146 | 227 | 247 | 407 | 2235 |
| sample #1 | n=6 | 25 | 265 | 51 | 82 | 43 | 84 | 2951 |
| pass@32 | n=3,4 | 901 | 22 | 1 | 2 | 6 | 60 | 5 |
| pass@32 | n=5 | 2075 | 529 | 89 | 119 | 240 | 636 | 837 |
| pass@32 | n=6 | 453 | 627 | 62 | 99 | 115 | 354 | 1792 |

Findings:
- **Every size beats the next smaller one on every test set and definition.** For example, sample
  #1 accuracy on n=5 is 23.1% / 28.3% / 41.4% for 0.6B / 1.7B / 4B.
- **n=3,4: the smaller models' solved sets sit almost entirely inside the 4B's.** Under pass@32,
  only 3 problems are solved by 0.6B or 1.7B but not by 4B.
- **Out of distribution (n=5, n=6), the 4B is not a superset.** Under pass@32, 303 n=5 problems
  (6.6%) and 178 n=6 problems (5.1%) are solved by 0.6B or 1.7B but not by 4B.
- **The 1.7B's solved set overlaps more with the 4B's than the 0.6B's does.** On n=5 under
  sample #1, 407 problems are solved by 1.7B and 4B but not 0.6B, vs 247 solved by 0.6B and 4B but
  not 1.7B. Under pass@32 the gap is 636 vs 240.

## 2. Accuracy vs size: final vs best checkpoint

Figure: `figures/Qwen3-base-grpo/size_curve_pass1_pass32.png` (rows: pass@1, pass@32; columns:
n=3,4 / n=5 / n=6). The solid line is the final checkpoint (step 1600). The dashed line is the
**oracle** best of the 16 checkpoints (steps 100–1600), chosen separately for each model, test set
and metric on the test set itself, so it is an upper bound rather than a checkpoint you could
select without peeking. pass@1 is the unbiased estimate (mean over the 32 samples), not sample #1.

Data: `analysis_data/qwen3_size_passk.csv` (240 rows: 3 runs × 3 test sets × 16 steps, plus the
2-epoch run below). All 240 match `pass_at_1`/`pass_at_32` in
`analysis_data/idood_qwen3_base_grpo.csv` exactly.

Values are % accuracy. Each cell is "final → best (@step)", or just the final value when the best
checkpoint is step 1600.

| metric | model | n=3,4 | n=5 | n=6 |
|---|---|---|---|---|
| pass@1 | 0.6B (lr 3e-6) | 75.0 → 76.6 (@1500) | 23.7 | 4.1 |
| pass@1 | 1.7B (lr 1e-6) | 82.6 | 28.3 | 5.7 → 6.9 (@1200) |
| pass@1 | 4B (lr 1e-6) | 90.9 → 91.0 (@1400) | 40.7 → 40.7 (@1400) | 11.9 → 12.3 (@900) |
| pass@32 | 0.6B (lr 3e-6) | 91.1 → 94.7 (@1500) | 54.1 → 59.8 (@1500) | 18.4 → 19.9 (@1500) |
| pass@32 | 1.7B (lr 1e-6) | 96.6 → 97.2 (@1500) | 63.3 | 26.2 → 30.3 (@1200) |
| pass@32 | 4B (lr 1e-6) | 99.2 | 75.3 | 44.0 |

Findings:
- **Accuracy increases with size on every panel, at both the final and the best checkpoint.**
- **The final checkpoint is close to the best one.** pass@1 is within 1.6 pts of the best
  everywhere. pass@32 is within 4.1 pts, except for the 0.6B on n=5 (54.1 vs 59.8), where the best is
  at step 1500, one checkpoint before the end. That gap is probably
  checkpoint-to-checkpoint noise rather than a trend, but it isn't checked here.
- The best checkpoints are late (steps 900–1500). There is no early out-of-distribution peak like
  the one in the old lr=3e-6 1.7B/4B runs.

### 2b. Adding the 0.6B lr=1e-6 2-epoch run

Figure: `figures/Qwen3-base-grpo/size_curve_pass1_pass32_2epoch.png`. This is the same plot with
`Qwen3-0.6B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-ep2-seed1` overlaid next to the 0.6B position.
The orange star is the final checkpoint and the orange dot is the oracle best of its 32 checkpoints
(steps 100–3200). The run trained for 3266 steps, but step 3200 is the last checkpoint on the
100-step save grid, so the star is step 3200. See also
`2026-10-11_qwen3-0.6b-lr1e-6-epoch2-plateau.md`.

**Caveat: orange and blue differ in both learning rate and epochs.** The only 2-epoch run is at lr
1e-6, while the blue 0.6B is now the lr 3e-6 run. A lr=3e-6 2-epoch 0.6B run does not exist.

| metric | test set | 0.6B lr3e-6 1 ep: final | 0.6B lr1e-6 2 ep: @1600 | 0.6B lr1e-6 2 ep: final @3200 | 0.6B lr1e-6 2 ep: best |
|---|---|---|---|---|---|
| pass@1 | n=3,4 | 75.0 | 72.4 | 73.1 | 76.9 @2700 |
| pass@1 | n=5 | 23.7 | 21.7 | 21.3 | 23.8 @2700 |
| pass@1 | n=6 | 4.1 | 4.4 | 3.6 | 4.7 @1100 |
| pass@32 | n=3,4 | 91.1 | 92.8 | 90.9 | 93.9 @1900 |
| pass@32 | n=5 | 54.1 | 57.4 | 51.1 | 59.0 @1100 |
| pass@32 | n=6 | 18.4 | 23.2 | 16.9 | 25.1 @1100 |

Findings:
- **Two epochs at lr 1e-6 do not beat one epoch at lr 3e-6 at the final checkpoint**, on any
  panel. The 2-epoch run's oracle best roughly matches the lr=3e-6 run's final checkpoint on
  pass@1, and beats it on n=5/n=6 pass@32 (59.0 vs 54.1; 25.1 vs 18.4).
- **What the second epoch itself adds** (2-epoch run, step 3200 vs step 1600): pass@1 is about
  flat, and pass@32 drops, mostly out of distribution (n=5 57.4→51.1; n=6 23.2→16.9).
- The 2026-10-11 log found that two identical-config runs (lr 1e-6, 1 vs 2 epochs, which are the
  same configuration through step 1633) differ by about 5 pts at step 1600 from sampling
  nondeterminism alone. Gaps of that size between single-seed runs in this section are within
  noise.

## Reproduce

    python3 scripts/plot/qwen3_venn_data.py            # per-sample correctness @ STEP (default 1600), ~30 s
    PY=~/.conda/envs/verl/bin/python3                    # base python3 has no numpy
    $PY scripts/plot/plot_qwen3_venn.py                  # venn_*.png
    $PY scripts/plot/plot_qwen3_venn_sized.py            # euler_*.png, upset_*.png
    python3 scripts/plot/qwen3_size_curve_data.py      # all ckpts (16; 32 for the 2-ep run), ~10 min first time, cached after
    $PY scripts/plot/plot_qwen3_size_curve.py            # size_curve_pass1_pass32.png
    $PY scripts/plot/plot_qwen3_size_curve.py --ep2      # size_curve_pass1_pass32_2epoch.png
    $PY scripts/plot/plot_qwen3_length_hist.py           # lengths/length_hist_nontrunc_step1600.png (tokenizes, ~10+ min first time)
    $PY scripts/plot/plot_qwen3_length_hist.py --include-truncated   # lengths/length_hist_step1600.png
    $PY scripts/plot/plot_qwen3_tries_hist.py            # lengths/tries_hist_nontrunc_step1600.png (needs length_hist caches)
