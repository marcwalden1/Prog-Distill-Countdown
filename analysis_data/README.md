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

`qwen05b_distill_grpo_run_registry.csv` records the Qwen 0.5B distill+GRPO
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
- post-SFT GRPO hyperparams: `grpo_lr=1e-6`, `grpo_kl=3e-4`
