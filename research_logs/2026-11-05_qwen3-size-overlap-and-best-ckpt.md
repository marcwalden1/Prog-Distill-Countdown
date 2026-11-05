# Qwen3-Base GRPO across sizes: which problems each model solves, final vs best checkpoint (2026-11-04 → 2026-11-05)

A follow-up analysis of the size sweep in `2026-09-23_qwen3-base-grpo-size-rolloutn-lr.md`.
No new training or evals were run; everything comes from existing eval JSONs.

**Runs compared** (all `balanced-simple`, kl=3e-3, 1 epoch, `rollout.n=4`, seed1). Each model
uses its better learning rate from that log:

| label | run |
|---|---|
| 0.6B | `Qwen3-0.6B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1` |
| 1.7B | `Qwen3-1.7B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1` |
| 4B   | `Qwen3-4B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1` |

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
| sample #1 | n=3,4 | 569 | 108 | 8 | 28 | 66 | 136 | 67 |
| sample #1 | n=5 | 236 | 716 | 229 | 160 | 317 | 204 | 2676 |
| sample #1 | n=6 | 10 | 154 | 80 | 33 | 30 | 16 | 3187 |
| pass@32 | n=3,4 | 862 | 36 | 0 | 3 | 20 | 66 | 9 |
| pass@32 | n=5 | 1562 | 817 | 110 | 100 | 495 | 431 | 1048 |
| pass@32 | n=6 | 236 | 546 | 114 | 53 | 257 | 98 | 2187 |

Findings:
- **n=3,4: the smaller models' solved sets sit almost entirely inside the 4B's.** Under pass@32,
  only 4 problems are solved by 0.6B or 1.7B but not by 4B.
- **Out of distribution (n=5, n=6), the 4B is not a superset.** Under pass@32, 110 n=5 problems and
  114 n=6 problems are solved only by the 0.6B.
- **The final 1.7B checkpoint is worse than the 0.6B on n=5 and n=6.** Its sample #1 accuracy is
  14.8% vs 18.7% on n=5 and 1.9% vs 3.7% on n=6. The 0.6B's solved set overlaps more with the
  4B's than the 1.7B's does: 317 vs 204 problems on n=5 under sample #1.

## 2. Accuracy vs size: final vs best checkpoint

Figure: `figures/Qwen3-base-grpo/size_curve_pass1_pass32.png` (rows: pass@1, pass@32; columns:
n=3,4 / n=5 / n=6). The solid line is the final checkpoint (step 1600). The dashed line is the
**oracle** best of the 16 checkpoints (steps 100–1600), chosen separately for each model, test set
and metric on the test set itself, so it is an upper bound rather than a checkpoint you could
select without peeking. pass@1 is the unbiased estimate (mean over the 32 samples), not sample #1.

Data: `analysis_data/qwen3_size_passk.csv` (144 rows). These numbers match `pass_at_1`/`pass_at_32`
in `analysis_data/idood_qwen3_base_grpo.csv` exactly on all 144 rows.

Values are % accuracy. Each cell is "final → best (@step)", or just the final value when the best
checkpoint is step 1600.

| metric | model | n=3,4 | n=5 | n=6 |
|---|---|---|---|---|
| pass@1 | 0.6B | 66.9 | 19.0 | 3.9 |
| pass@1 | 1.7B | 76.0 → 85.4 (@1400) | 14.6 → 29.5 (@600) | 1.7 → 7.3 (@600) |
| pass@1 | 4B | 88.3 → 90.0 (@1300) | 32.1 → 37.0 (@800) | 6.0 → 11.3 (@500) |
| pass@32 | 0.6B | 88.6 → 88.7 (@1400) | 48.1 → 49.1 (@1100) | 18.0 → 19.6 (@600) |
| pass@32 | 1.7B | 93.5 → 97.0 (@1400) | 46.5 → 62.2 (@1400) | 11.8 → 27.3 (@600) |
| pass@32 | 4B | 98.7 → 98.8 (@700) | 71.5 → 72.4 (@700) | 32.3 → 39.4 (@500) |

Findings:
- **The 0.6B (lr 1e-6) is near its best at step 1600.** The only gaps are ≤1.6 pts of pass@32.
- **The 1.7B (lr 3e-6) declines late in training.** Its final checkpoint is non-monotone in size
  (below the 0.6B on n=5 and n=6), but its best checkpoints are monotone: n=5 pass@1 peaks at
  29.5% at step 600 and ends at 14.6%. So the dip in the final-checkpoint curve is mostly late
  out-of-distribution degradation at lr 3e-6, not a capacity effect. The cause is not checked here;
  a lower learning rate or KL change for 1.7B would test it.
- **With the oracle best checkpoint, accuracy increases with size on every panel.** The 4B also
  peaks early on n=6 (step 500) and loses 5–7 pts by step 1600.
- **The out-of-distribution peaks (steps 500–800) come well before the in-distribution peaks
  (steps 1300–1400)** for 1.7B and 4B. Choosing a checkpoint on n=3,4 would therefore miss most
  of the n=5/n=6 headroom.

## Reproduce

    python3 scripts/plot/qwen3_venn_data.py            # per-sample correctness @ STEP (default 1600), ~30 s
    PY=~/.conda/envs/verl/bin/python3                    # base python3 has no numpy
    $PY scripts/plot/plot_qwen3_venn.py                  # venn_*.png
    $PY scripts/plot/plot_qwen3_venn_sized.py            # euler_*.png, upset_*.png
    python3 scripts/plot/qwen3_size_curve_data.py      # all 16 steps, ~6 min first time (cached after)
    $PY scripts/plot/plot_qwen3_size_curve.py            # size_curve_pass1_pass32.png
