# 2026-07-08 — From-scratch ~40M students: distill / progdistill / mixture × GRPO

## Models

Built with `scripts/make_scratch_model.py` (new): stock `Qwen2ForCausalLM`, config
inherited from Qwen2.5-0.5B (hidden 896, 14 heads / 2 KV, head_dim 64, vocab 151,936,
tied embeddings, bf16) except depth/FFN. Body randomly initialized (seed 1,
initializer_range 0.02); **embedding table copied verbatim from Qwen2.5-0.5B**
(trainable, lm_head tied). Saved under
`/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/`; provenance in each
model dir's `scratch_init.json`.

| Model | Layers | FFN | Body params | Total (tied 136.1M emb) |
|---|---|---|---|---|
| `Qwen2.5-40M-wide` | 5 | 2432 | 41.9M | 178.0M |
| `Qwen2.5-40M-deep` | 8 | 1216 | 40.9M | 177.0M |

WandB size tags: `40M-wide` / `40M-deep` (train_grpo.sh / train_sft.sh / plot_results.py
special-case `*-40M-*` model names).

## Teacher + data

- Teacher: Shlok's `Qwen2.5-1.5B / balanced-grpo-lr3e-6-kl3e-3-seed1`
  (10/10 merged checkpoints, read-only via new `TEACHER_CHECKPOINT_DIR` env var →
  `/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints`).
- SFT parquets: reused Shlok's
  `sft-data/Qwen2.5-0.5B/teacher-Qwen2.5-1.5B-balanced-grpo-lr3e-6-kl3e-3-seed1-n4/`
  (metadata match: n=4, no filter/truncate, max_length 1024; `step_final` is a symlink
  to `step_1600`). Copied (106 MB) into bingbin's
  `rl-checkpoints/sft-data/Qwen2.5-40M-{wide,deep}/<same namespace>/`.
  **Zero gen-data jobs needed.** Every parquet: 39,856 rows.

## Experiment matrix (12 chains, submitted 2026-07-08)

2 models × {distill, progdistill, progdistill_mixture} × SFT LR {1e-5, 3e-5}, each
`run_pipeline.sh --<mode> <teacher> --GRPO`.

- SFT: **"all data" budget = 6,220 steps** (batch 64). progdistill: 622/round × 10
  (1 epoch of each round's parquet); mixture: 1 epoch of pooled 398,560 rows;
  distill: compute-matched ~10 epochs of teacher-final data. Chosen over the old
  1600-step budget (which covered only ~26% of the pool). Warmup ratio 0.1, seed 1,
  wandb log freq 50 (`SFT_LOG_FREQ=50`).
- GRPO: lr 1e-6, KL 3e-3, 1635 steps, rollout n=4, 2×H100, save/test every 50,
  train-metric wandb logging every 50 steps (`GRPO_LOG_FREQ=50`, needs the new
  `ray_trainer.py` `trainer.log_freq` patch).
- EXP_NAME pattern: `balanced-<mode>-t1p5b-sftlr<lr>-steps6220-grpo-lr1e-6-kl3e-3-seed1`.

### Job IDs (SFT → GRPO → eval×3 → plot)

| Model | Mode | SFT LR | SFT | GRPO | Evals | Plot |
|---|---|---|---|---|---|---|
| wide | distill | 1e-5 | 29429709 | 29429710 | 29429711/12/13 | 29429714 |
| wide | progdistill | 1e-5 | 10 rounds (…29429728) | 29429729 | 29429730/31/32 | 29429733 |
| wide | mixture | 1e-5 | 29429734 | 29429737 | 29429739/40/42 | 29429744 |
| wide | distill | 3e-5 | 29429745 | 29429746 | 29429747/48/49 | 29429750 |
| wide | progdistill | 3e-5 | 10 rounds (…29429772) | 29429773 | 29429776/78/80 | 29429781 |
| wide | mixture | 3e-5 | 29429785 | 29429786 | 29429801/02/06 | 29429807 |
| deep | distill | 1e-5 | 29429809 | 29429810 | 29429823/24/25 | 29429826 |
| deep | progdistill | 1e-5 | 10 rounds (…29429956) | 29429957 | 29429958/59/60 | 29429961 |
| deep | mixture | 1e-5 | 29429962 | 29429963 | 29429964/65/66 | 29429967 |
| deep | distill | 3e-5 | 29429968 | 29429969 | 29429970/71/72 | 29429973 |
| deep | progdistill | 3e-5 | 10 rounds (…29429985) | 29429986 | 29429987/88/89 | 29429994 |
| deep | mixture | 3e-5 | 29429995 | 29430000 | 29430001/05/06 | 29430009 |

108 jobs total. Full submission log archived at the time of launch; per-round
progdistill SFT job IDs are in SLURM (`sacct --name=train_sft.sh`).

## Update (2026-07-08, later): pending GRPO legs cancelled

Per Bingbin: cancelled the 8 GRPO jobs that hadn't started, plus their dependent
eval/plot jobs (which could never run once the GRPO dependency was gone). The 4
GRPO jobs already RUNNING were kept, with their evals/plots intact:

- **Kept (running)**: wide-distill sftlr1e-5 (29429710), wide-distill sftlr3e-5
  (29429746), wide-mixture sftlr1e-5 (29429737), wide-mixture sftlr3e-5 (29429786).
- **Cancelled (pending)**: 29429729, 29429773 (wide progdistill ×2); 29429810,
  29429969 (deep distill ×2); 29429963, 29430000 (deep mixture ×2); 29429957,
  29429986 (deep progdistill ×2) — plus their 24 eval arrays and 8 plot jobs.

All SFT jobs were left untouched and continue to run; the affected chains will
still produce their final SFT checkpoints (progdistill: `round_1600`), just no
GRPO/eval on top for now.

## Update (2026-07-08, later): sftlr1e-5 GRPO runs cancelled too

Per Bingbin: also cancelled the two running GRPO legs on sftlr=1e-5 SFT
checkpoints — wide-distill (29429710, ~3.4 h in) and wide-mixture (29429737,
~3.1 h in) — plus their eval/plot jobs. Their partial GRPO checkpoints (up to
~global_step_350-400) remain on disk under
`checkpoints/Qwen2.5-40M-wide/balanced-<mode>-t1p5b-sftlr1e-5-steps6220-grpo-lr1e-6-kl3e-3-seed1/`.
**Still running:** the two sftlr=3e-5 GRPO legs (wide-distill 29429746,
wide-mixture 29429786) with evals/plots intact — now the only GRPO in flight.

## Update (2026-07-08, later): SFT-only sweep at higher LRs

Per Bingbin: 8 additional **SFT-only** chains (no GRPO), 2 models ×
{distill, progdistill_mixture} × SFT LR {1e-4, 3e-4}. Same everything else
(6,220 steps, teacher data reuse, seed 1, log freq 50). Each chain = SFT +
SFT-final eval ×3 datasets + plot. EXP_NAME pattern:
`balanced-<mode>-t1p5b-sftlr<lr>-steps6220-seed1` (no grpo suffix).

| Model | Mode | SFT LR | SFT | Evals | Plot |
|---|---|---|---|---|---|
| wide | distill | 1e-4 | 29519524 | 29519525/26/27 | 29519528 |
| wide | mixture | 1e-4 | 29519544 | 29519548/52/54 | 29519556 |
| wide | distill | 3e-4 | 29519557 | 29519558/59/60 | 29519561 |
| wide | mixture | 3e-4 | 29519562 | 29519563/64/65 | 29519566 |
| deep | distill | 1e-4 | 29519567 | 29519568/69/70 | 29519572 |
| deep | mixture | 1e-4 | 29519575 | 29519579/82/84 | 29519587 |
| deep | distill | 3e-4 | 29519589 | 29519591/93/94 | 29519595 |
| deep | mixture | 3e-4 | 29519596 | 29519619/21/30 | 29519632 |

## Update (2026-07-09): SFT LR sweep results + 1e-3 extension

Final `val/loss` (seeded 500-batch subsample of each run's own training data;
comparable within a column) for the distill / progdistill_mixture arms:

| Setup | 1e-5 | 3e-5 | 1e-4 | 3e-4 |
|---|---|---|---|---|
| wide-distill | 0.2829 | 0.1284 | 0.0641 | **0.0507** |
| wide-mixture | 0.8609 | 0.3465 | 0.1855 | **0.1673** |
| deep-distill | 0.3426 | 0.1515 | 0.0692 | **0.0554** |
| deep-mixture | 0.6479 | 0.3316 | 0.1945 | **0.1692** |

3e-4 won every setup (monotonic in LR), so Bingbin extended the sweep with
**sftlr=1e-3** before picking GRPO candidates (GRPO launch on winners is on hold
until these finish). Submitted 2026-07-09, same 6,220-step SFT-only chains:

| Model | Mode | SFT | Evals | Plot |
|---|---|---|---|---|
| wide | distill | 29769804 | 29769806/19/22 | 29769823 |
| wide | mixture | 29769824 | 29769825/27/28 | 29769844 |
| deep | distill | 29769847 | 29769849/50/51 | 29769852 |
| deep | mixture | 29769853 | 29769854/55/56 | 29769857 |

Next step (standing): once these finish, pick lowest val loss per setup across
all 5 LRs and launch GRPO (lr=1e-6, kl=3e-3) + eval + plot on the 4 winners.

### Update (2026-07-09, later): wide GRPO launched on Bingbin's manual pick

Bingbin selected the **wide sftlr=3e-4** students for GRPO directly (without
waiting for the 1e-3 sweep). Launched with lr=1e-6, kl=3e-3, standard 1635 steps:

| Setup | GRPO | Evals | Plot |
|---|---|---|---|
| wide-distill sftlr=3e-4 | 29788968 | 29788969/70/71 | 29788972 |
| wide-mixture sftlr=3e-4 | 29788973 | 29788974/75/76 | 29788977 |

Deep arms: GRPO not yet launched — awaiting the 1e-3 SFT results and Bingbin's pick.

### Update (2026-07-09, later): 1e-3 results + deep GRPO launched

sftlr=1e-3 completed clean but was worse than 3e-4 in all four setups
(wide-distill 0.0641, wide-mixture 0.1701, deep-distill 0.0704, deep-mixture
0.1703) — **3e-4 is the confirmed val-loss winner everywhere**; the LR curve
bottoms out at 3e-4. Per Bingbin, launched GRPO (lr=1e-6, kl=3e-3) on the two
deep 3e-4 students, completing GRPO-on-best for all 4 setups:

| Setup | GRPO | Evals | Plot |
|---|---|---|---|
| deep-distill sftlr=3e-4 | 29792151 | 29792152/53/54 | 29792155 |
| deep-mixture sftlr=3e-4 | 29792156 | 29792157/58/59 | 29792160 |

## Smoke tests (both COMPLETED, exit 0)

- SFT `29425780`: 20 steps on step_150 parquet, wide model — loss 6.1→, merge+cleanup OK.
  2m52s; ~2.38 s/step.
- GRPO `29425801`: 60 steps, wide model — vLLM rollout + weight sync fine with the
  custom dims, checkpoints saved. Val reward 0.0 as expected (random body). 33m.

## Expectations / notes

- Eval only hits final checkpoints per pipeline defaults (GRPO chains: 10-step eval
  array over global_step_150..1600 of the **GRPO** stage; SFT quality itself is only
  evaluated indirectly). For SFT-stage curves, submit manual evals later.
- The from-scratch bodies start at reward ~0; GRPO success depends entirely on the
  SFT stage producing format-valid answers.
- `verl/verl/trainer/ppo/ray_trainer.py` gained an optional `trainer.log_freq`
  (default 1 = stock): throttles per-step train metrics; val-carrying steps and the
  last step always log. Captured in `patches/local-fixes-vs-083da9ab.patch`
  (file newly created; `apply_verl_patches.sh` referenced it but it was missing
  from this checkout).
