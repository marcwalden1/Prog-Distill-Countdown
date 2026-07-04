# Pareto eval data

This directory stores small, git-tracked analysis tables used to compare ID and
OOD eval performance for distillation experiments.

`qwen05b_grpo_on_top_pareto_eval.csv` is generated from cached eval score files
under `results/Qwen2.5-0.5B`. Each row is one `(run, checkpoint, eval_dataset)`
measurement. The plotting task pivots rows as:

- x-axis: `eval_dataset=balanced`, `n_label=n=3/4`, `split=id`
- y-axis: `eval_dataset=balanced5`, `n_label=n=5`, `split=ood`
- y-axis: `eval_dataset=balanced6`, `n_label=n=6`, `split=ood`
- metric: either `mean_at_1` or `mean_at_32`

Regenerate with:

```bash
python3 scripts/export_pareto_eval_data.py
```

`qwen05b_distill_grpo_run_registry.csv` records the Qwen 0.5B distill/progdistill+GRPO
runs we currently care about, including their canonical checkpoint root and
eval root. `qwen05b_distill_grpo_eval_inventory.csv` is the detailed
per-step/per-dataset inventory for the same runs.

`qwen05b_seed1_distill_eval_backfill_jobs.csv` records the SLURM eval array
jobs submitted on 2026-05-25 to fill missing seed1 vanilla-distill evals for
`sftlr=3e-5` and `sftlr=3e-6`.

Regenerate those inventory files with:

```bash
python3 scripts/inventory_qwen05b_distill_grpo_runs.py
```

Current default scope:

- model: `Qwen2.5-0.5B`
- methods: `distill`, `progdistill`
- post-SFT GRPO hyperparams: `grpo_lr=1e-6`; includes both `grpo_kl=3e-4`
  and `grpo_kl=3e-3` where those runs exist

## Gemma-3-270M GRPO sweep

`pareto_run_registry.csv` lists **every** Gemma-3-270M distill/progdistill+GRPO run
present locally, not just the sftlr=1e-4 winners. SFT LR *was* searched for Gemma:
distill over `{3e-6, 1e-5, 3e-5, 1e-4, 3e-4}`, progdistill over
`{3e-6, 3e-5, 1e-4, 3e-4}` (GRPO lr and KL also varied). `sftlr=1e-4, grpo_lr=1e-6,
KL=3e-3` is the *selected* best per method, not an a-priori fixed value.

`gemma_grpo_sweep_eval.csv` is the per-`(run, checkpoint, dataset)` eval table for
that full sweep, sourced from the **local** `.scores`/`.lengths` caches under
`results/gemma-3-270m`. Columns mirror `pareto_eval_progression.csv`
(`mean_at_1` = binary mean@1, `mean_at_32` = **shaped** mean@32 incl. the 0.1
format floor). For binary OOD *coverage* use `gemma_pregrpo_coverage_binary.csv` /
`gemma_pregrpo_coverage_passk.csv` instead.

**Provenance caveat:** the Gemma rows in `pareto_eval_progression.csv` (the curated
4-run canonical table) were generated from Marc's netscratch copy
(`/n/netscratch/kdbrantley_lab/...`, no longer accessible) and differ slightly from
the local `.scores` (different eval-sampling instance; e.g. distill kl3e-3 balanced
step 150 = 0.800 there vs 0.828 locally). `gemma_grpo_sweep_eval.csv` is therefore a
separate, internally-consistent local table rather than an extension of that file —
do not mix the two in one plot.

Regenerate the registry sweep rows + the sweep eval table with:

```bash
python3 scripts/refresh_gemma_grpo_sweep.py
```

## Session analysis tables (coverage hypothesis, 2026-06)

- `gemma_pregrpo_coverage_binary.csv`, `gemma_pregrpo_coverage_passk.csv` — pre-GRPO
  KD/PD SFT-student coverage across sftlr (binary mean@k and pass@k, all splits).
- `gemma_postgrpo_kl3e-3_lr1e-6_binary.csv` — post-GRPO binary mean@k across sftlr at
  the shared kl3e-3/lr1e-6 setting.
- `gemma_beststable_traj_binary.csv` — binary trajectories, shared best-stable pair.
- `gemma_bestpermethod_traj_binary.csv` — binary trajectories, each method's own best.
