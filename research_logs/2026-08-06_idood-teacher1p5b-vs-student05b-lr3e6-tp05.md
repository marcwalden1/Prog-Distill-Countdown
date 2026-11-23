# ID-vs-OOD Pareto: 1.5B teacher GRPO vs 0.5B GRPO with trailing penalty (tp0.5) — 2026-08-06

Fourth run of the teacher-vs-student ID/OOD comparison (method + gotchas in
`research_logs/2026-08-03_idood-pareto-teacher1p5b-vs-student05b.md`); this one
swaps in the **trailing-penalty 0.5** 2-epoch dyn-bsz student at lr3e-6, seed1.

## Runs

| Arm | Model | EXP_NAME | Eval tree | Ckpts |
|---|---|---|---|---|
| teacher GRPO | Qwen2.5-1.5B | `balanced-grpo-lr3e-6-kl3e-3-seed1` | sdholakia's `results/` (read-only) | 10 (150…1600) |
| student GRPO | Qwen2.5-0.5B | `balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-tp0.5-ep2-seed1` | `results/` (own tree) | 14 (150…3250) |

Steps 1050/1200/1600 were missing the ID (`balanced`) eval JSON; backfilled with
`MODEL_NAME=... EVAL_DATASET=balanced sbatch --account=kempner_bingbin_lab
--array=21,24,32 scripts/eval.sh` (job 37536459) before exporting.
(Note: `eval.sh` no longer pins `--account` in its `#SBATCH` header, so a direct
sbatch on `kempner_requeue` needs `--account=kempner_bingbin_lab` on the CLI.)

- CSV: `analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_tp05.csv` (72 rows)
- Figures: `figures/comparison/id_vs_ood_teacher1p5b_vs_student05b_lr3e6_tp05/` (`grid.png` + 4 panels)
  (2×2 grid + 4 standalone panels)

## Final checkpoints, full set

| Arm (final step) | ID pass@1 | n=5 pass@1 | n=6 pass@1 | ID pass@32 | n=5 pass@32 | n=6 pass@32 |
|---|--:|--:|--:|--:|--:|--:|
| teacher 1.5B (1600) | 0.860 | 0.318 | 0.059 | 0.965 | 0.692 | 0.292 |
| student 0.5B tp0.5 (3250) | 0.422 | 0.111 | 0.026 | 0.472 | 0.167 | 0.056 |

## Observations

- Same shape as every previous teacher/student comparison: one shared ID→OOD ray,
  the teacher travels much further; no trade-off frontier.
- vs the no-penalty seed2 sibling (`.../lr3e-6-kl3e-3-ep2-seed2`, final 0.424 /
  n=5 0.125 / n=6 0.033 pass@1): the trailing penalty ends at essentially the same
  ID (0.422) with slightly lower OOD (n=5 0.111, n=6 0.026). Within run-to-run
  noise — the penalty neither helps nor hurts accuracy here (it targets response
  junk, not correctness). Different seeds, so not a controlled A/B.
- Second epoch again buys ID slowly (pass@1 0.375 @1600 → 0.422 @3250) with a
  noisy plateau (dip at 2000, 2800).
- n=6 pass@1 near-floor and non-monotone for both arms — read n=6 via pass@32.

## Reproduce

```bash
# backfill missing ID evals (steps 1050/1200/1600), then:
EXPORT_RUNS="teacher GRPO (Qwen2.5-1.5B)=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results::Qwen2.5-1.5B::balanced-grpo-lr3e-6-kl3e-3-seed1
student GRPO (Qwen2.5-0.5B)=results::Qwen2.5-0.5B::balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-tp0.5-ep2-seed1" \
EXPORT_OUT=analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_tp05.csv \
  sbatch scripts/export_idood_scores.sh          # jobs 37536674 + 37537078

~/.conda/envs/verl/bin/python3 scripts/plot_id_vs_ood_runs.py \
  --csv analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_tp05.csv \
  --tag teacher1p5b_vs_student05b_lr3e6_tp05 --out-dir comparison
```
