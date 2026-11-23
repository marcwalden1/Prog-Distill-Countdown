# ID-vs-OOD Pareto: 1.5B teacher GRPO vs 0.5B GRPO (dynbsz n8, lr3e-6, kl3e-3, 2ep, seed2) — 2026-08-06

Third run of the teacher-vs-student ID/OOD comparison (method + gotchas in
`research_logs/2026-08-03_idood-pareto-teacher1p5b-vs-student05b.md`); this one
swaps in the **seed2** 2-epoch dyn-bsz student at lr3e-6.

## Runs

| Arm | Model | EXP_NAME | Eval tree | Ckpts |
|---|---|---|---|---|
| teacher GRPO | Qwen2.5-1.5B | `balanced-grpo-lr3e-6-kl3e-3-seed1` | sdholakia's `results/` (read-only) | 10 (150…1600) |
| student GRPO | Qwen2.5-0.5B | `balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2` | `results/` (own tree) | 16 (150…3250) |

Full-set re-scoring (teacher rows all cache hits from the 2026-08-03 export).

- CSV: `analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_seed2.csv` (78 rows)
- Figures: `figures/comparison/id_vs_ood_teacher1p5b_vs_student05b_lr3e6_seed2/` (`grid.png` + 4 panels)
  (2×2 grid + 4 standalone panels)

## Final checkpoints, full set

| Arm (final step) | ID pass@1 | n=5 pass@1 | n=6 pass@1 | ID pass@32 | n=5 pass@32 | n=6 pass@32 |
|---|--:|--:|--:|--:|--:|--:|
| teacher 1.5B (1600) | 0.860 | 0.318 | 0.059 | 0.965 | 0.692 | 0.292 |
| student 0.5B seed2 (3250) | 0.424 | 0.125 | 0.033 | 0.454 | 0.172 | 0.069 |

## Observations

- Same picture as the seed1 lr3e-6 comparison: both arms travel roughly the same
  ID→OOD ray, the teacher just goes much further. No trade-off frontier.
- The second epoch keeps buying ID slowly (pass@1 0.365 @1600 → 0.424 @3250) with
  mild OOD gains, but the trajectory is noisy/plateauing past ~step 2100
  (0.394 → 0.370 → 0.424); the n=6 pass@1 panel is near-floor and non-monotone —
  read n=6 via pass@32 as before.
- Unlike the base-bsz seed1 student (whose whole trajectory ended near the
  teacher's step-150 point), this student's final ID pass@1 (0.424) does clear the
  teacher's first checkpoint (0.337). Its OOD n=5 (0.125) sits between teacher
  steps 150 (0.081) and 300 (0.172) — still far from the teacher's frontier.

## Reproduce

```bash
EXPORT_RUNS="teacher GRPO (Qwen2.5-1.5B)=/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results::Qwen2.5-1.5B::balanced-grpo-lr3e-6-kl3e-3-seed1
student GRPO (Qwen2.5-0.5B)=results::Qwen2.5-0.5B::balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2" \
EXPORT_OUT=analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_seed2.csv \
  sbatch scripts/export_idood_scores.sh          # job 37511188, ~1 min

~/.conda/envs/verl/bin/python3 scripts/plot_id_vs_ood_runs.py \
  --csv analysis_data/idood_teacher1p5b_vs_student05b_lr3e6_kl3e3_seed2.csv \
  --tag teacher1p5b_vs_student05b_lr3e6_seed2 --out-dir comparison
```
