# Fix MAX_PROMPTS=1000 plotting cap; regenerate all affected plots on full set (2026-07-13)

## The bug

Plotting scored only the **first 1000 prompts** of each eval JSON, via
`MAX_PROMPTS = 1000` + `data[:MAX_PROMPTS]` in three scripts:
`plot_results.py`, `scripts/plot_passk.py`, `scripts/inspect_generations.py`.
`eval.py` itself never subsampled — the full 4620 (n=5) / 3519 (n=6) prompts ×
32 samples were always on disk; only the scoring/plotting truncated.

`data/balanced{5,6}/test.parquet` is **ordered by `canonical_pattern_index`**, so
the first 1000 rows are a fixed, low-index (easier) slice (patterns 0–104 of 556
for n=5), not a random sample. n=3,4 (`balanced`, ~997 prompts) is under the cap →
unaffected. `.lengths`/response-length paths don't cap → unaffected.

### Quantified bias (first-1000 head vs full; Δ = full − head)

| run / step | dataset | Δ mean@1 | Δ mean@32 |
|---|---|--:|--:|
| Qwen progdistill-mix / 1050 | n=5 | −0.125 | −0.102 |
| gemma mixture kl3e-4 / 300  | n=5 | −0.047 | −0.045 |
| Qwen warmup / 1600          | n=5 | +0.001 | −0.004 |
| Qwen warmup / 1600          | n=6 | −0.003 | −0.003 |

Head **over-states** accuracy, always. Magnitude scales with model competence on
that set: a mid-accuracy model (~0.4–0.5 on n=5) exposes the difficulty gradient
(first-quintile m@32 0.52 vs later 0.34) → up to ~10–12 pt inflation; a
floor-bound model (~0.12) shows ~0. n=6 bias ≤0.02 (near floor). Because the bias
differs per run, it distorts *relative* comparisons (e.g. distill vs progdistill
Pareto on n=5).

## Fix

Set `MAX_PROMPTS = None` (guarded slice) in all three scripts → full-set scoring
by default. Downstream scripts inherit the fix via `plot_results._score_checkpoint`
/ `plot_passk` imports or by reading the `.scores`/`.passk` caches.

## Cache invalidation

`.scores`/`.passk` are mtime-keyed and record no prompt count, so stale capped
caches would be silently reused. Deleted all `balanced5`/`balanced6` `.scores` +
`.passk` in `results/` (213+30) and `results_sdholakia/` (101+84). Kept `balanced`
(provably full) and all `.lengths`.

## sdholakia data copy

Several aggregate scripts read sdholakia's trees (`.../sdholakia/results` and
`.../repo-artifacts/RL-skill-comp/results`), read-only. Copied the 7 referenced
exp dirs (eval files only; json/scores/passk/lengths, ~21 GB, no checkpoints) into
`results_sdholakia/` (symlink → `repo-artifacts/RL-skill-comp/results_sdholakia`),
and repointed `SD`/`RA`/`_SDH` in `plot_accuracy_bars.py`,
`plot_accuracy_bars_best_sft_models.py`, `plot_gemma_hparam_lines.py`,
`plot_id_vs_ood_gemma_kl3e4.py`.

## Regeneration (full set; W&B skipped; SLURM `serial_requeue`, `--account=bingbin_lab`)

Each affected `X.png` renamed to `X_deprecated.png` before regeneration.

- **Per-run `eval_n5`/`eval_n6`** (24 runs): `scripts/regen_plot_results_fullset.sh`
  → 24 `plot_results.sh` jobs, `WANDB_MODE=disabled`. 48 fresh plots.
- **Aggregate JSON-based**: `scripts/regen_aggregate_plots_fullset.sh` →
  `plot_accuracy_bars`, `plot_accuracy_bars_40m`, `plot_accuracy_bars_best_sft_models`,
  `plot_gemma_hparam_lines`, `plot_id_vs_ood_gemma_kl3e4`, `plot_id_vs_ood`,
  `plot_id_vs_ood_40m` (all OK).
- **pass@k**: `scripts/regen_passk_plots_fullset.sh` → `passk_all_grpo_only`,
  `passk_3models`, `passk_grpo_only_3models` (invocations reconstructed from
  `2026-07-07-passk-coverage.md`; `passk_grpo_only_3models` composition inferred).
  `passk_{grid,by_step}_n64.png` are in-dist (balanced) only → **unaffected**,
  restored (were over-deprecated).

63 deprecated figures now have fresh full-set replacements.

## CSV-backed plots (follow-up 2026-07-13)

- **gemma `id_vs_ood_pass{1,32}_n{5,6}_lr1e-6.png`** — NOW REGENERATED. The 4 runs
  the plot needs (distill/progdistill × kl3e-3/kl3e-4) are exactly the copied
  `results_sdholakia/gemma-3-270m` dirs. `scripts/regen_gemma_pareto_csv.py`
  rebuilds `analysis_data/gemma270m_pareto_eval_progression.csv` from their
  full-set `.scores`; `plot_id_vs_ood_gemma_csv.py` repointed at that local CSV
  (DEPRECATED banner removed) → 4 fresh full-set plots.
- **`plot_gemma_hparam_lines_sanity.py`** — still deprecated-only; it's a pure
  cross-check of the JSON-based `plot_gemma_hparam_lines.py` (already regenerated),
  so no separate replacement needed.
- **Qwen `comparison/id_vs_ood_{distill,progdistill}.png`** — REGENERATED (2026-07-14)
  for the 6 of 10 runs whose full eval JSONs exist. Downloaded balanced+balanced5
  n32 JSONs from HF `marcwalden/rl-skill-comp-evals` → `hf-evals-qwen05b/`
  (`scripts/rebuild_qwen05b_fig7_csv_fullset.py` re-scores full and writes
  `analysis_data/qwen05b_fig7_pareto_fullset.csv`, 118 rows, 3 distill + 3
  progdistill). NB: HF's own `.scores` are the OLD capped values — we re-scored the
  JSONs, not reused them. The 4 `balanced-grpo-from-*` runs have no JSONs on HF (or
  anywhere on-cluster) and are dropped. `id_vs_ood_combined.png` is not produced by
  the current script (main() emits only per-method) → stays deprecated-only.

## Final ID-vs-OOD status

13 fresh (full-set): 6 Qwen-0.5B per-run `ID_vs_OOD.png`, 2 Qwen-40M `id_vs_ood.png`,
gemma `id_vs_ood_kl3e-4.png`, 4 gemma `id_vs_ood_pass*`. Remaining stale-only: 2
Qwen-40M `Qwen40M_*_id_vs_ood.png` (superseded by `id_vs_ood.png`, filename change)
and the 3 Qwen comparison plots above.
- **Old-naming orphans** (11): older filenames/locations that current scripts no
  longer emit (e.g. `Qwen40M_*_id_vs_ood.png` → now `id_vs_ood.png`;
  per-shape `Qwen2.5-40M-{wide,deep}/accuracy_bars_stacked.png` → now combined
  `Qwen2.5-40M/accuracy_bars_stacked.png`). Each is superseded by a fresh
  current-named figure.
