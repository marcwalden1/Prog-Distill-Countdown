# CLAUDE.md — RL-skill-comp

## What this repo is

Research code for the paper **"How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown"** (arXiv:2512.01775, NeurIPS 2025 workshops). It trains LLMs with GRPO (or PPO) on the **Countdown task**: given a target number and N input numbers, generate an arithmetic equation using all N numbers exactly once that equals the target, using +, −, ×, ÷.

---

## Cluster environment

- **Cluster:** Harvard Kempner (SLURM)
- **Conda env:** verl (Miniforge3/26.1.0-fasrc01)
- **Python:** 3.10
- **GPUs:** nvidia_h100_80gb_hbm3 (96 GB) or A100 (80 GB) depending on user partition
  - 0.5B / 1.5B / 3B: 4 GPUs per job
  - 7B: 8 GPUs per job

> Each user has their own partition, account, and storage paths. See your personal ~/.claude/CLAUDE.md.

---

## Key paths

| What | Path |
|---|---|
| Training data | data/balanced/train.parquet (282 MB) |
| Test data (n=3,4) | data/balanced/test.parquet (~997 rows, deduplicated) |
| Test data (n=5) | data/balanced5/test.parquet |
| Test data (n=6) | data/balanced6/test.parquet |
| Raw puzzles | data/countdown_{size}_pattern_{idx}.json |
| SLURM logs | logs/%x-%A-%a.out |
| WandB run dir (per-job) | /tmp/wandb_${SLURM_JOB_ID}/wandb/ (local; runs sync live in online mode) |
| Models | USER-SPECIFIC — see personal CLAUDE.md |
| Checkpoints | USER-SPECIFIC — see personal CLAUDE.md |

**bingbin's checkpoint tree moved 2026-09-14** (holylabs hit its lab quota and killed
four 7B GRPO runs mid-training). `CHECKPOINT_DIR` now defaults to
`/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown`, where `checkpoints/` is real
storage and `sft-checkpoints/`, `sft-data/`, `opd-data/` are symlinks back to
`/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints/` (small inputs; the
1.7 TB of RL checkpoints is what needed to move). Consequences:

- **Runs from before the move are NOT found by default.** To eval/plot one, pass
  `CHECKPOINT_DIR=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`.
- **There is a THIRD tree:** `/n/netscratch/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`
  (note: `kempner_bingbin_lab`, not `bingbin_lab` — the two netscratch paths differ
  only in that segment). The Sept-2026 OLMo-2-7B pt126B/pt135B GRPO runs live there
  and were never moved. Worse, the default tree holds a *partial, aborted* copy of
  those same `EXP_NAME`s (a Sep-1 job killed mid-save at step 300) with different
  weights — so reading the default tree gives you a different run, not an error.
  **Before concluding a checkpoint is missing or purged, check all three trees**, and
  confirm which one a run's existing evals used:
  `grep -m1 -oE "/n/[^ ]*rl-checkpoints|/n/netscratch/bingbin_lab/[^ ]*" logs/eval.sh-<jid>-<task>.out`
- **netscratch is purge-eligible.** Anything there is scratch; merge, evaluate, and
  copy out what matters.
- A 7B GRPO run costs ~82 GB per saved step (FSDP shards + fp32 optimizer state),
  so ~1.3 TB per 16-checkpoint run. Budget before launching several in parallel;
  `SAVE_FREQ=200` halves it.

**Available models:**
- Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B
- Llama-3.2-3B (requires tokenizer_config.json override)
- Qwen2.5-40M-wide / Qwen2.5-40M-deep — from-scratch ~40M-body students, built by
  `scripts/make_scratch_model.py` (see "From-scratch ~40M models" below)

---

## Training: scripts/train_grpo.sh

### Launching

    export MODEL_NAME=Qwen2.5-1.5B   # required
    sbatch scripts/train_grpo.sh

    # Override array for multiple seeds:
    sbatch --array=1-3 scripts/train_grpo.sh

    # 7B needs 8 GPUs:
    sbatch --gres=gpu:nvidia_h100_80gb_hbm3:8 --array=1-3 -t 30:00:00 scripts/train_grpo.sh

### Environment variables (all optional overrides)

| Var | Default | Meaning |
|---|---|---|
| MODEL_NAME | Qwen2.5-1.5B | Model to train |
| PROJECT_DIR | $PWD | Repo root |
| CHECKPOINT_DIR | USER-SPECIFIC | Checkpoint base dir — set in personal CLAUDE.md |
| DATA_SOURCE | balanced | Data directory under data/ |
| EXP_NAME | {data_source}-grpo-seed{SLURM_ARRAY_TASK_ID} | Experiment name |
| MAX_LENGTH | 1024 | Max response length |
| CHECKPOINT_PATH | {checkpoint_dir}/checkpoints/{model_name} | Output dir |
| GRPO_LOG_FREQ | 50 | wandb/console cadence for per-step train metrics; needs verl patch 5 below |
| SAVE_FREQ | 100 | Checkpoint save cadence (steps). **Must match eval.sh's `EVAL_STEP_MULT`** |
| TEST_FREQ | 50 | Validation cadence (steps); independent of SAVE_FREQ |

### Key hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | GRPO (algorithm.adv_estimator=grpo) |
| Learning rate | 1e-6 |
| Train batch size | 256 |
| Rollout samples per prompt | 4 (rollout.n=4) |
| Validation samples per prompt | 4 (val_kwargs.n=4) |
| Val sampling | do_sample=True, temperature=1.0 |
| KL loss | use_kl_loss=True, coef=0.001, type=low_var_kl |
| KL in reward | False |
| Advantage normalization | False (norm_adv_by_std_in_grpo=False) |
| Entropy coeff | 0 |
| Loss aggregation | token-mean |
| Save freq | every 100 steps (`SAVE_FREQ`) |
| Test (val) freq | every 50 steps (`TEST_FREQ`) |
| Total epochs | 1 |
| Gradient checkpointing | True |
| Strategy | FSDP2 |
| Generation engine | vLLM, gpu_memory_utilization=0.6 |
| Prompt max length | 1024 |
| Response max length | 1024 |

### WandB

- Mode: **online by default** (`WANDB_MODE=online`). Compute nodes have outbound HTTPS; runs stream live to wandb.ai during training. Override with `WANDB_MODE=offline ...` if a node is air-gapped, then `wandb sync /tmp/wandb_<jobid>/wandb/offline-run-*` after.
- `WANDB_DIR` is pinned to `/tmp/wandb_${SLURM_JOB_ID}` (NFS-backed home busts wandb-core's 30s `ServicePollForTokenError`). The local dir disappears with the node, so for online runs the live URL is the durable artifact.
- Project: prog_distill (hard-coded in `trainer.project_name`)
- Run name: {MODEL_NAME}-{exp_name} (e.g. Qwen2.5-1.5B-balanced-grpo-seed1)
- Tags: stage (`grpo`/`sft`/`opd`, plus `DISTILL_MODE`), model **size** (`0.5B`, `270m`,
  `40M-wide`), **model family** (`Qwen2.5`, `Qwen3`, `OLMo-2`, `gemma-3`) and `data_source`.
  The family tag (added 2026-09-22) is what you filter on to separate Qwen3 from Qwen2.5 /
  OLMo — the size tag deliberately drops the family, and the run name isn't tag-filterable.
  It is derived from `MODEL_NAME` by `_model_family()` in `train_{grpo,sft,opd}.sh` (leading
  segments up to the first size-like or 4-digit-date segment) and is appended **even when a
  launcher presets `WANDB_TAGS`** (the `run_olmo2_*.sh` scripts do). `plot_results.py` has the
  same rule as `model_family()` for the eval runs it logs; keep the two in sync.
  Backfill for existing runs: `scripts/backfill_wandb_tags.py --entity <e> [--dry-run]`
  (all 112 `bingbin/prog_distill` runs were backfilled on 2026-09-22: 110 `OLMo-2`, 2 `Qwen3`).
- Eval scores can be uploaded to wandb separately via `scripts/upload_evals_to_wandb.py` — reads `results/{model}/{exp}/global_step_*/*.{scores,lengths}` and logs per-step scalars.

### Checkpoints

Saved to {output_dir}/{exp_name}/global_step_{N}/ every 100 steps (`SAVE_FREQ`).

**The save cadence and the eval cadence are coupled.** `eval.sh` evaluates
`global_step_$((EVAL_STEP_MULT * SLURM_ARRAY_TASK_ID))`, so a checkpoint that
isn't on the `EVAL_STEP_MULT` grid is never evaluated — it just accumulates on
disk with its optimizer shards intact. Both default to 100; `run_pipeline.sh`
warns at submission if `SAVE_FREQ != EVAL_STEP_MULT`.

---

## Training: scripts/train_ppo.sh

Same structure as GRPO but:
- algorithm.adv_estimator=gae (standard PPO with critic)
- Has a separate critic model (critic.model.path, critic.optim.lr=1e-5)
- Default model: Qwen2.5-3B
- Smaller mini batch (64 vs 256)

---

## From-scratch ~40M models (2026-07-08)

`scripts/make_scratch_model.py` builds small students that slot into every pipeline
stage unchanged (stock `Qwen2ForCausalLM`; HF/verl/vLLM all load them natively):
config inherited from Qwen2.5-0.5B (hidden 896, 14 heads / 2 KV, vocab 151,936, tied
embeddings) except depth/FFN; body randomly initialized (seeded); **embedding table
copied verbatim from the donor** (trainable). Provenance in `<model dir>/scratch_init.json`.

    python3 scripts/make_scratch_model.py --donor $MODEL_DIR/Qwen2.5-0.5B \
        --preset wide --out $MODEL_DIR/Qwen2.5-40M-wide --seed 1

| Preset | Layers | FFN | Body params | Total |
|---|---|---|---|---|
| `wide` | 5 | 2432 | 41.9M | 178.0M |
| `deep` | 8 | 1216 | 40.9M | 177.0M |

Notes:
- WandB size tag for `*-40M-*` model names is `40M-<shape>` (e.g. `40M-wide`) —
  special-cased in `train_grpo.sh`, `train_sft.sh`, `plot_results.py`.
- These models train fine with the default (non-Gemma) microbatch settings.
- Base GRPO directly on a scratch model gets ~0 reward (random body can't produce
  `<answer>` format) — always SFT first.
- The 2026-07-08 experiment matrix (2 models × {distill, progdistill, mixture} ×
  SFT LR {1e-5, 3e-5}, SFT budget 6,220 steps = one pass over all 10 teacher
  parquets, GRPO lr=1e-6 kl=3e-3) is logged in
  `research_logs/2026-07-08_40m-scratch-students.md`.

---

## Offline on-policy distillation (OPD, added 2026-07-09)

Implements "Lightning OPD" (arXiv:2604.13010) on top of an existing SFT student:
rollouts are sampled **once** from the SFT student (π_ref), the teacher is
queried **once** for per-token logprobs over those rollouts, and the student
then trains on the frozen dataset with the policy-gradient surrogate
`loss_t = clamp(log π_T − log π_θ, ±OPD_CLIP) · (−log π_θ)` (teacher term
precomputed, student term live; implemented as advantage-weighted CE).
**Teacher consistency:** the OPD teacher must be the SAME checkpoint that
generated the student's SFT data (for progdistill students, the final-round
= step-1600 teacher). Teacher and student must share a tokenizer (Qwen family
only; base-format tokenizers are rejected by `generate_opd_data.py`).

    MODEL_NAME=Qwen2.5-0.5B \
    EXP_NAME=balanced-opd-from-distill-sftlr3e-5-lr2e-6-seed1 \
    TEACHER_MODEL_NAME=Qwen2.5-1.5B TEACHER_STEP=1600 \
    TEACHER_CHECKPOINT_DIR=<tree holding the teacher checkpoints> \
    OPD_STUDENT_PATH=<merged SFT student checkpoint dir> \
    bash scripts/run_pipeline.sh --opd <teacher-EXP_NAME>

Submits gen-data (1 job: rollout stage then teacher-scoring stage, two python
processes) → `train_opd.sh` (1600 steps, batch 64, save_freq=100 →
`checkpoints/{MODEL}/{EXP}/global_step_{100..1600}`, GRPO-style layout) →
standard eval array × 3 datasets → plot. `--GRPO` is not supported after `--opd`.

Key pieces:
- `scripts/generate_opd_data.py` / `.sh` — rollouts store exact token IDs (no
  retokenize drift); teacher scored via vLLM `prompt_logprobs=0` with
  `enable_prefix_caching=False` (V1 engine rejects the combination). Output
  parquet at `$CHECKPOINT_DIR/opd-data/<MODEL>/<namespace>/opd.parquet` with
  columns prompt/prompt_ids/response_ids/teacher_logprobs/response/score/
  rollout_logprob, namespace
  `student-<tag>-teacher-<T_MODEL>-<T_EXP>-step<S>-n<N>-temp<T>-p<PROMPTS>`.
- `opd_dataset.py` (repo root) — injected via verl's `data.custom_cls`, so no
  verl-side dataset patch. Same batch contract as SFTDataset + a
  `teacher_logprobs` tensor aligned to input positions.
- OPD loss branch in `verl/trainer/fsdp_sft_trainer.py` (**local verl patch 6**,
  gated on `+data.opd=True`; see patch table below).

Env knobs (defaults): `OPD_TRAIN_STEPS=1600` (compute-matched to the SFT
stage; note the paper used ~10× fewer OPD steps relative to SFT, so watch the
eval curve for saturation/regression past the early checkpoints),
`OPD_LR=2e-6`, `OPD_CLIP=10.0`, `OPD_ROLLOUT_TEMP=1.0` (paper: 0.8),
`OPD_MAX_PROMPTS=10000`, `N_RESPONSES=4` (→ 40k rollouts ≈ 2.5 epochs at 1600
steps; raise to 8–10 for ~1 epoch), `OPD_SEED`, `OPD_WARMUP_RATIO=0.1`
(cosine, repo SFT convention; paper used constant LR), `OPD_VAL_MAX_BATCHES=100`,
`OPD_SAVE_FREQ=100` (matches `EVAL_STEP_MULT`), `OPD_TEST_FREQ=50`.

Reference student/teacher checkpoints for the Qwen 0.5B experiments are listed
in `sft-checkpoints/best_models.md` (students + 1.5B teacher live under
sdholakia's tree — pass it as `TEACHER_CHECKPOINT_DIR`). `OPD_STUDENT_PATH` is
the merged HF dir: the distill student's exp dir directly, the progdistill
student's `round_1600/` subdir.

---

## Data pipeline

### Puzzle sizes and pattern counts

| Size | Patterns | Samples/pattern | Total |
|---|---|---|---|
| n=3 | 18 | 4000 | 72k |
| n=4 | 96 | 4000 | 384k |
| n=5 | 558 | 10 | 5580 |
| n=6 | 4328 | 1 | 4328 |

### Regenerating data

    for i in {0..17};   do python generate_puzzles.py --puzzle_size 3 --pattern_index ${i} --num_data 4000; done
    for i in {0..95};   do python generate_puzzles.py --puzzle_size 4 --pattern_index ${i} --num_data 4000; done
    for i in {0..557};  do python generate_puzzles.py --puzzle_size 5 --pattern_index ${i} --num_data 10;   done
    for i in {0..4327}; do python generate_puzzles.py --puzzle_size 6 --pattern_index ${i} --num_data 1;    done
    python preprocess_balanced.py

### Train/test split (preprocess_balanced.py)

- n=3,4: first 10 samples/pattern -> test, rest -> train
- Test set deduplicated by (sorted(nums), target) -> ~997 unique puzzles
- n=5,6: test-only sets (balanced5/, balanced6/)

### Parquet schema

    prompt: [{"role": "user", "content": "<prompt text>"}]
    ability: "math"
    reward_model: {"style": "rule", "ground_truth": <target int>}
    extra_info: {
      "numbers": [list of ints],
      "puzzle_size": int,
      "canonical_pattern_index": int,
      "canonical_pattern": str  (e.g. "((A-B)*C)+D")
    }

### Prompt format (template_type='base')

    A conversation between User and Assistant...
    User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
    Assistant: Let me solve this step by step.
    <think>

---

## Reward function: grader_utils.py

compute_score(data_source, solution_str, ground_truth, extra_info, format_score=0.1, score=1.0, trailing_penalty=0.0)

- Extracts last <answer>...</answer> match from model output
- Validates all numbers used exactly once
- Evaluates equation safely (restricted eval)
- Returns: 1.0 (correct), 0.1 (valid format, wrong answer), 0.0 (no answer / invalid)
- **Trailing-junk penalty (2026-08-05, opt-in):** `trailing_penalty` deducts that much
  reward per 1000 chars of non-whitespace content after the final `</answer>` (floored
  at 0.0). Rationale: the answer comes from the *last* `<answer>` tag, so junk after a
  correct answer (repeated `</think>` spam to the length cap) is otherwise reward-neutral
  and gets reinforced by token-mean GRPO. Default 0.0 = historical reward — eval scoring
  (`plot_results.py` etc.) is unchanged. Enable per-run at train time via
  `TRAILING_PENALTY=<x>` (read by `train_grpo.sh`/`train_ppo.sh`, passed as
  `+custom_reward_function.reward_kwargs.trailing_penalty`; sbatch env inheritance
  carries it through `run_pipeline.sh`). Put it in `EXP_NAME` (e.g. `-tp0.5-`) since it
  changes the reward function.

---

## SFT response formats + few-shot eval: `sft_formats.py` (2026-09-11)

One definition of "what a solution looks like", shared by the GT-SFT target
generator and eval's few-shot exemplars — so `eval.py --n_shot K --shot_format X`
measures exactly the format `generate_gt_sft_data.py --format X` would train on.

| format | response after a `Solution:` prompt | tokens (n=3,4) |
|---|---|---|
| `answer` | ` <answer> 14 + 27 + 9 </answer>` (historical GT target) | 17 |
| `think_empty` | vacuous `<think></think>` scaffold | 22 |
| `verify` | `<think>` evaluates the GT expression step by step | 54 |
| `search` | `<think>` rejects k legal-but-wrong expressions, then answers | 84 |

`search` exists because answer-only SFT installs a ~15-token response, leaving
GRPO no room to sample a second attempt (see
`research_logs/2026-08-25_olmo2-pretrain-ladder-grpo.md`: 32.5 chars/response,
post-GRPO pass@32 collapsing onto pass@1). Distractors are patterns from
`annotated_expressions.json` at the puzzle's size with the puzzle's own numbers
in a random permutation — legal expressions (each number once) that miss.
Alternative *correct* expressions are excluded. Rejected attempts stay inside
`<think>`, and `extract_solution` takes the **last** `<answer>` tag, so the
graded answer is unchanged and the generator's `compute_score == 1.0` gate still
passes on every row.

    # SFT data
    python3 scripts/generate_gt_sft_data.py --data-source balanced-simple \
        --output <path>.parquet --format search --search-attempts 1-3 \
        --distractor-strategy mixed        # mixed | near | random

    # few-shot base eval (answer-only vs search-trace exemplars)
    MODEL_NAME=OLMo-2-0425-1B-pt4001B EVAL_DATASET=balanced-simple \
        N_SHOT=4 SHOT_FORMAT=search sbatch scripts/eval_base.sh

Few-shot notes:
- `--n_shot` **requires the raw prompt path** (hard error otherwise): exemplars
  are plain prompt+response text and cannot be spliced into a chat template.
- Exemplars are a fixed, seeded block (vLLM prefix-caches it). Pool is
  `data/<shot_source>/train.parquet`; `auto` maps `balanced5*`/`balanced6*` to
  the n=3,4 train split of the same prompt variant, since those are test-only.
  Exemplars never come from a test set.
- Results land in `<result_dir>/<K>shot-<format>/`, a **subdir** — not a filename
  suffix, which `plot_results.py`'s `{ds}_temp0.6_n32_*.json` glob would merge
  into the zero-shot curve.
- `run_olmo2_gt_sft_ladder.sh` now puts the format in `EXP_NAME`
  (`...-gtsft200-search-lr1e-5-seed1`); `answer` is omitted so existing runs keep
  resolving.

Tests: `python3 tests/test_sft_formats.py -v` (run the file directly —
`python3 -m unittest tests/<file>.py` is broken in this env).

## Running the pipeline: scripts/run_pipeline.sh

**Always use this to launch a new experiment.** Submits train → eval → plot_results as a single SLURM dependency chain — no manual follow-up needed.

```bash
# Standard run
MODEL_NAME=Qwen2.5-1.5B EXP_NAME=balanced-grpo-seed1 bash scripts/run_pipeline.sh

# With KL coef override
MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-grpo-kl1e-4-seed1 KL_COEF=0.0001 bash scripts/run_pipeline.sh

# 7B (needs 8 GPUs)
MODEL_NAME=Qwen2.5-7B EXP_NAME=balanced-grpo-seed1 \
  TRAIN_SBATCH_ARGS="--gres=gpu:nvidia_h100_80gb_hbm3:8 -t 30:00:00" \
  bash scripts/run_pipeline.sh
```

This submits:
1. `train_grpo.sh` — trains the model
2. `eval.sh` × 3 datasets (`balanced`, `balanced5`, `balanced6`) — each as a 10-task array job, all pending on train
3. `plot_results.sh` — pending on all 3 eval jobs completing

**Scripts that are NEVER run:** `analyze.sh`, `analyze.py`, `gather_experiment.py`. Tables are produced manually on request.

---

## Progressive distillation

Distills a trained GRPO teacher into a student across one round per teacher
checkpoint: (gen-data-from-teacher-at-step-X → SFT-student), for X ∈
`TEACHER_STEPS`. Entry point: `scripts/run_pipeline.sh --progdistill <teacher-EXP_NAME>`.

`TEACHER_STEPS` defaults to **{200, 400, 600, 800, 1000, 1200, 1400, 1600}** — 8
rounds at every 200 steps. This changed with `SAVE_FREQ=100`: the historical
10-round list {150, 300, …, 1600} was 150-spaced and 6 of its steps are not
multiples of 100, so teachers trained at the current cadence never save them. Ten
rounds cannot be evenly spaced on a 100-step grid over 1600 steps; every-200 is
the aligned equivalent.

Total SFT budget = `len(TEACHER_STEPS) × PROGDISTILL_STEPS_PER_ROUND`, so the
default is now 8 × 160 = **1280**, not 1600. Use
`PROGDISTILL_STEPS_PER_ROUND=200` to restore a 1600-step budget, and pass
`TEACHER_STEPS="150 300 450 600 750 900 1050 1200 1400 1600"` to distill from a
legacy teacher saved every 50 steps.

### Prerequisites

Teacher must have a checkpoint at every step in `TEACHER_STEPS`. Verify:

    ls -d $CHECKPOINT_DIR/checkpoints/$TEACHER_MODEL_NAME/<teacher-EXP_NAME>/global_step_{200,400,600,800,1000,1200,1400,1600} 2>/dev/null | wc -l
    # expect: 8   (the number of entries in TEACHER_STEPS)

### Standard command (cross-size: 1.5B teacher → 0.5B student)

    MODEL_NAME=Qwen2.5-0.5B \
    EXP_NAME=balanced-progdistill-teacher-<teacher-tag>-sftlr<lr>-seed1 \
    TEACHER_MODEL_NAME=Qwen2.5-1.5B \
    SFT_LR=1e-5 \
    bash scripts/run_pipeline.sh --progdistill <teacher-EXP_NAME>

Submits **23 jobs** in one dependency chain:
- 10 × `generate_sft_data.sh` (H100, 4h, run in parallel as GPUs free up)
- 10 × `train_sft.sh` (sequential, each depends on prior round's SFT + its own gen-data)
- 3 × `eval.sh` (one per dataset: `balanced`, `balanced5`, `balanced6`) depending on round_1600 SFT
- 1 × `plot_results.sh` depending on all 3 evals

**Eval only hits `round_1600` by default.** For a curve across rounds, manually submit eval jobs on the intermediate SFT job IDs after submission.

### Key env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_NAME` | Qwen2.5-1.5B | student |
| `TEACHER_MODEL_NAME` | `$MODEL_NAME` | set only for cross-size distillation |
| `EXP_NAME` | required for reproducibility | put LR / teacher-tag in the name |
| `SFT_LR` | 1e-5 | applies to all 10 rounds, no per-round override |
| `PROGDISTILL_STEPS_PER_ROUND` | 160 | SFT steps per round |
| `N_RESPONSES` | 4 | teacher samples per prompt during gen-data |
| `TRUNCATE` | false | if true, keep only the final-attempt span of each teacher response |
| `SFT_DATA_NAMESPACE` | derived | override; default is `teacher-<TEACHER_MODEL_NAME>-<TEACHER_EXP_NAME>-n<N_RESPONSES>[-truncate]` |
| `FILTER_CORRECT_ONLY` | false | keep only score==1.0 responses |
| `DATA_SOURCE` | balanced | `data/<dir>/train.parquet` as prompts |
| `TEACHER_CHECKPOINT_DIR` | `$CHECKPOINT_DIR` | read teacher checkpoints from another (read-only) tree, e.g. another user's; outputs still go to `CHECKPOINT_DIR` |

### Variants

    # Progdistill + GRPO on top of round_1600
    ... --progdistill <teacher-EXP_NAME> --GRPO

    # Same-size (self) progdistill
    MODEL_NAME=Qwen2.5-1.5B TEACHER_MODEL_NAME=Qwen2.5-1.5B ...

    # Tune per-round step budget
    PROGDISTILL_STEPS_PER_ROUND=100 ...

### What lands on disk

    $CHECKPOINT_DIR/sft-data/<MODEL_NAME>/teacher-<TEACHER_MODEL_NAME>-<TEACHER_EXP_NAME>-n<N_RESPONSES>[-truncate]/step_{150..1600}.parquet
    $CHECKPOINT_DIR/sft-checkpoints/<MODEL_NAME>/<EXP_NAME>/round_{150..1600}/
        └── model.safetensors + configs + tokenizer  (merged HF, ~1 GB for 0.5B)
    results/<MODEL_NAME>/<EXP_NAME>/round_1600/{balanced,balanced5,balanced6}_*.json
    figures/<MODEL_NAME>/<EXP_NAME>/*.png

Each round saves only at its final step (`save_freq=-1` in `train_sft.sh`); if a round is killed mid-training it restarts from scratch. Use non-preemptible `kempner_h100` to avoid this. Intermediate FSDP shards are merged to HF format and deleted automatically.

### Sanity checks after submission

    squeue -u $USER                                                  # confirm 23 jobs queued
    tail -f logs/generate_sft_data.sh-<first-gendata-jid>-*.out      # watch first round's data gen
    ls $CHECKPOINT_DIR/sft-checkpoints/<MODEL_NAME>/<EXP_NAME>/      # round_150 appears ~30-60m after SFT starts

### Required local verl patch

Stock verl's `_build_model_optimizer` (in `verl/verl/trainer/fsdp_sft_trainer.py`) sizes the LR scheduler off `steps_per_epoch * total_epochs`. Each progdistill round runs `total_epochs=9999` + `total_training_steps=160`, so the scheduler horizon balloons to ~6M steps and warmup never finishes — `SFT_LR` has no observable effect and the student barely trains. The 5-line fix to make `_build_model_optimizer` prefer `trainer.total_training_steps` when set is documented under **Local verl + vLLM patches** below; reapply it on any fresh checkout. Diagnose at runtime by tailing `logs/train_sft.sh-<jid>-*.out` and checking `train/lr(1e-3)`: with the patch it peaks near `SFT_LR * 1e3`, otherwise orders of magnitude lower.

---

## Evaluation: scripts/eval.sh + eval.py

- **Step evaluated = `EVAL_STEP_MULT` (default 100) × `SLURM_ARRAY_TASK_ID`.**
- Default array: `1-16` → steps 100, 200, …, 1600 (every checkpoint of a default 1600-step run at `SAVE_FREQ=100`)
- Merges FSDP shards first (`verl.model_merger merge`), then deletes `actor/`, `critic/`, `data.pt` — so **a merged checkpoint (flat `model.safetensors`, no `*.pt`) is one that has been evaluated**, and one still holding `actor/` or top-level `*.pt` has not
- **Exception: the run's FINAL checkpoint keeps `actor/`, `critic/`, `data.pt`** (added 2026-09-29,
  `KEEP_LAST_OPTIMIZER=true` by default). eval.sh compares `eval_step` against
  `latest_checkpointed_iteration.txt` in the exp dir and skips the delete on a match, so the last
  step retains its fp32 optimizer state and dataloader position and a continuation can be a real
  `resume_mode=auto` resume rather than a weights-only warm start. Without it a continuation
  restarts Adam from zero, re-warms the LR, restarts the dataloader, and — because the warm-start
  weights are passed as `actor_rollout_ref.model.path` — **re-anchors the KL term on those weights
  instead of the base model**. Consequences: (a) the "merged == evaluated" heuristic above does not
  hold for the final step, so test for the eval JSON there; (b) resubmitting the same `EXP_NAME`
  now genuinely auto-resumes, so move or delete the exp dir for a clean restart; (c) costs one
  step's optimizer state on disk (~10 GB for 0.6B, ~82 GB for 7B). Set `KEEP_LAST_OPTIMIZER=false`
  for the old behaviour.
- A step with no checkpoint on disk logs and exits 0, so a deliberately wide array is safe and the `afterok` plot dependency still fires
- Runs `eval.py` with vLLM, n=32 samples/prompt, temperature=0.6
- Output: `results/{model}/{exp}/global_step_{N}/{dataset}_temp{T}_n{N}_max{M}.json`; caps other
  than 1024 go in a `global_step_{N}/max{M}/` subdir. `plot_results.py --max_tokens M` reads that
  subdir; `plot_results.sh` derives `M` from `--max_tokens` in `EVAL_EXTRA_ARGS`.
- **Random n=5/n=6 subsets (2026-10-17):** `OOD_SUBSAMPLE=1000 bash scripts/run_pipeline.sh` evaluates,
  scores and plots n=5/n=6 on a seeded uniform-random 1000-prompt subset (~4× cheaper; n=3,4 stays
  full). Subsets are separate dataset dirs `data/<ds>-r1000/` (seed 0, source row indices in
  `subsample.json`), built by `scripts/make_eval_subsample.py --source <ds> --n 1000`, so results,
  figures (`eval_n5_r1000_*.png`) and wandb keys (`n5_r1000`) never mix with full-set ones. Built so
  far: `balanced{5,6}{,-simple}-r1000`. Never subsample with a head slice (files are pattern-sorted).

### The array is a submission-time property, not a property of the script

The `#SBATCH --array` line in `eval.sh` is only a **default**. A CLI `--array` (or
`EVAL_ARRAY` via `run_pipeline.sh`, or `SBATCH_ARRAY_INX`) overrides it, and
nothing records the override afterwards. **Never infer which steps a past run
needed by reading `eval.sh`** — use `sacct -j <jid> -X -o JobID` (array indices ×
`EVAL_STEP_MULT`) and what is already in `results/{model}/{exp}/`. Getting this
wrong once cost the step-1800/2000 checkpoints of the two `dynbsz-*-ep2` runs
during a disk cleanup: they had been evaluated under a widened array (indices
36, 40) that the script's default did not mention.

Runs longer than 1600 steps need a wider array — the default does **not** cover them:

    # TOTAL_EPOCHS=2 (~3266 steps)
    EVAL_ARRAY=1-32 MODEL_NAME=... EXP_NAME=... bash scripts/run_pipeline.sh

    # Re-evaluate a legacy run saved at the old 50-step cadence
    EVAL_STEP_MULT=50 EVAL_ARRAY=3,6,9,12,15,18,21,24,28,32 \
      MODEL_NAME=... EXP_NAME=... sbatch scripts/eval.sh

---

## Plotting: scripts/plot/plot_results.sh + plot_results.py

- CPU-only job on `serial_requeue`, 8 CPUs, 64GB RAM, 1h time limit
- Produces **5 plots** per run in `figures/{model}/{exp}/`:
  - `val_reward_vs_step.png` — val reward mean@4 vs. training step (from log)
  - `eval_n34.png` — mean@1 + mean@32 vs. checkpoint (n=3,4)
  - `eval_n5.png` — same for n=5
  - `eval_n6.png` — same for n=6
  - `response_length.png` — mean response length (tokens) vs. checkpoint, all 3 datasets
- Uses `.scores` and `.lengths` cache files to avoid recomputing unchanged results
- Scoring uses `Pool(8)` for parallelism and scores the **full** prompt set (`MAX_PROMPTS = None`).
  The old 1000-prompt cap was a bug: `balanced{5,6}` test files are sorted by
  `canonical_pattern_index`, so a head slice is an easier subset that overstated n=5 accuracy by
  up to ~12 pts (fixed 2026-07-13, see `research_logs/2026-07-13_fix-maxprompts-cap-fullset-regen.md`)

---

## verl framework

Pinned fork of volcengine/verl at commit 083da9ab130efa2dc284eeb821a3edd6ce570fe3.
Entry point: python3 -m verl.trainer.main_ppo (used for both GRPO and PPO).

Key deps: torch==2.6.0, transformers==4.51.1, vllm==0.8.5.post1, flash-attn==2.7.4.post1, flashinfer-python==0.2.2

---

## Common operations

    # Check job status (replace with your username)
    squeue -u $USER

    # Cancel a job
    scancel <jobid>

    # Submit 0.5B job
    MODEL_NAME=Qwen2.5-0.5B sbatch scripts/train_grpo.sh

    # Submit with multiple seeds
    MODEL_NAME=Qwen2.5-1.5B sbatch --array=1-3 -t 10:00:00 scripts/train_grpo.sh

    # Tail a training log
    tail -f logs/train_grpo.sh-<jobid>-1.out

    # Upload eval mean@1 / mean@32 / response length to wandb (one offline run; sync at end)
    python3 scripts/upload_evals_to_wandb.py \
        --results-dir results/{MODEL_NAME}/{EXP_NAME} \
        --model {MODEL_NAME} --exp {EXP_NAME}

---

## Batch failure policy (MANDATORY for any multi-task submission, added 2026-10-12)

**Why:** on 2026-10-11 a 192-task long-cap eval batch lost ~88 × 8 GPU-hours: 87
tasks ran to an 8 h TIMEOUT that was visible after the first few, and 2 died on a
bad node (`holygpu8a17602`, `CUDA error: unspecified launch failure`) while
reporting COMPLETED. Nothing was watching.

**Rule.** After submitting more than one job or array task (eval arrays, seed
sweeps, `run_pipeline.sh` chains), watch the batch. **The moment any task fails,
hold every still-queued task in the batch, then triage before anything else
runs.** Holding is cheap and reversible; letting a broken config drain the queue
is not.

Tool: `scripts/triage_jobs.py <jobid> [<jobid> ...]` (stdlib only, ~1 s). It
lists every bad task with its node, a failure class and the log line, catches
"COMPLETED but no result JSON", and projects an ETA for RUNNING eval tasks from
`eval.py`'s `Saved k/N prompts` lines. Exit 1 = something needs attention.

    # watcher: arm right after submitting (background Bash / Monitor, ~15 min cadence)
    until ! python3 scripts/triage_jobs.py $JIDS >/dev/null; do sleep 900; done
    python3 scripts/triage_jobs.py --hold $JIDS     # hold queued tasks + print triage

Then act on the class:

| Class | Signature | Action |
|---|---|---|
| `node` | `unspecified launch failure`, ECC / Xid, `GPU is lost`, `No CUDA GPUs are available`, NCCL system error, `NODE_FAIL`, NFS I/O errors | Add the node to `#SBATCH --exclude=` in the job script (with a comment: date, job id, error) **and** to the pending tasks: `triage_jobs.py --exclude <node> --release $JIDS`. Resubmit the failed task(s) with the same args. No need to ask. |
| `oom` | `CUDA out of memory` / `OutOfMemoryError` | Halve the micro-batch knobs and resubmit the held tasks (cancel + resubmit; a pending job's env can't be edited). GRPO: `PPO_MICRO_BATCH_SIZE_PER_GPU`, `ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`, `REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`; PPO/SFT/OPD: the `*micro_batch_size_per_gpu` overrides via `EXTRA_ARGS`. Micro-batch only changes memory, not the update — **never** touch `ppo_mini_batch_size`, `train_batch_size`, or `rollout.n`. No need to ask. `eval.py` (vLLM) has no micro-batch knob, so an eval OOM is `other`. |
| `timeout`, `will-timeout` | walltime hit, or projected ETA > walltime | **Other**: the budget is wrong. Hold, report numbers (fraction done, projected hours), propose options, wait. |
| `hostmem`, `other` | host-RAM OOM, Python errors, bad paths, anything unrecognized | **Pause and tell the user.** Keep the tasks held; resume only after agreeing on the change. |

**GPU-idle watchdog (added 2026-11-06).** The main GPU command of `train_{grpo,ppo,sft,opd}.sh`,
`eval{,_base}.sh`, `generate_{sft,opd,distill}_data.sh` runs under `scripts/with_gpu_watchdog.sh`:
if every GPU in the job is at 0% utilization for 20 consecutive minutes (after a 15-min startup
grace), the command is killed (exit 143) and the job FAILs. Motivation: a vLLM V1 engine thread
died on a numpy-int seed, the main process waited forever, and four H100 jobs sat idle ~4h45m
until a cluster admin cancelled them. Knobs: `WATCHDOG_IDLE_MIN`, `WATCHDOG_GRACE_MIN` (raise
for a job with a long legitimate CPU-only phase *inside* the wrapped command), `GPU_WATCHDOG=0`
to disable. Checkpoint merges run outside the wrapper. A kill logs `[gpu_idle_watchdog] ...`.

Notes:
- Holding does not stop RUNNING tasks. If running siblings are clearly doomed
  (same timeout projection, same deterministic error), say so and ask before
  cancelling them — don't cancel running work unilaterally.
- Before fanning out a new *kind* of job (new cap, new model size, new dataset),
  check the first task's progress early (`triage_jobs.py` prints an ETA) — or run
  one pilot task — rather than discovering the walltime at hour 8.
- `eval.sh` now exits with `eval.py`'s status, so crashes show as FAILED; and
  `eval.py` saves progress in `<result>.json.partial.jsonl` every `--save_every`
  (128) prompts and resumes from it, so a resubmitted/requeued eval loses at most
  one chunk.

## Known issues / history

- **Never read a training step out of a log with `grep "step:[0-9]*"`
  (found 2026-09-18).** verl logs each step as one long line whose *other*
  metric keys also end in `step`, carrying seconds rather than step numbers:

      ... - timing_s/step:338.23 - perf/time_per_step:264.45 - training/global_step:1600 - ...

  `grep -aoE "\bstep:[0-9]+" <log> | tail -1` returns **338** on a run whose last
  step was 1600 — `\b` matches after the `/` in `timing_s/step:`, and the float is
  truncated at the decimal point into a plausible-looking step number. Drop the
  `\b` and `time_per_step:`/`global_step:` match too. There is no failure signal;
  you just get a wrong number. Use `parse_step_metrics()` from
  `gather_experiment.py`, whose regex requires the trailing `" - "` delimiter that
  pins the match to the real step field (`\bstep:(\d+) - `). Same rule for
  "how far did this run get" checks — read `latest_checkpointed_iteration.txt`
  or `ls global_step_*` instead of grepping.

- **19.1% of `data/balanced6/test.parquet` has a broken `canonical_pattern`
  (found 2026-09-11).** 812 of the 4328 n=6 canonical patterns in
  `annotated_expressions.json` do not use each letter exactly once — e.g.
  `(A-B-C-D)*E/D` reuses `D` and never uses `F` — affecting 671/3519 rows. n=2/3/4/5
  are clean. Consequence: any GT-derived target built from n=6
  (`generate_gt_sft_data.py`, `scripts/diagnose_gt_ntp.py`) reconstructs an equation
  that reuses one number and drops another, scoring 0.1 instead of 1.0 (the
  generator's verify gate aborts loudly, so it cannot silently poison SFT data), and
  per-pattern analyses on n=6 read an unreliable pattern string for ~1 row in 5.
  **The puzzles themselves are fine** — an exact subset-DP solvability check found
  40/40 sampled affected rows still solvable from their 6 numbers, so n=6 eval
  numbers stand. Only the recorded pattern is wrong.
- Test set deduplication fix (2025-03): Original test set had duplicate prompts (same puzzle presented multiple times). Fixed in preprocess_balanced.py by deduplicating on (sorted(nums), target). Current test set: ~997 unique puzzles.
- val_kwargs.n: Was previously 7; corrected to 4 to match rollout.n=4.
- Buggy runs: Any checkpoints from before the deduplication fix should be discarded.
- Gemma-3-270m GRPO can OOM in actor/ref log-prob computation even though the model is smaller than Qwen2.5-0.5B. Cause: Gemma's vocab is much larger (262k vs Qwen 152k), so the lm_head logits tensor during `compute_log_prob` is huge. The failed runs `gemma-270m-balanced-grpo-lr1e-6-kl3e-3-seed1` and `gemma-270m-balanced-grpo-lr1e-6-kl1e-2-seed1` died after step 1 trying to allocate ~31 GiB at `modeling_gemma3.py:958`. For Gemma reruns, keep experiment hyperparams fixed and only lower memory microbatch knobs via `EXTRA_ARGS`: `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`. Do not bake these into defaults; Qwen runs should use the normal script defaults unless explicitly overridden.
- Gemma-3-270m SFT reliably OOMs in the **final validation pass** (same large-vocab cause as the GRPO note above: `cross_entropy` over the 262k-vocab logits). Training completes and — thanks to local patch 1c (save-before-validate) — `global_step_1600/` is saved *before* validation runs, so the trained weights are intact. But the OOM traceback leaves torchrun hung on teardown: the SLURM job sits at `RUNNING` indefinitely (zombie), never reaches `train_sft.sh`'s merge step, and blocks the whole `afterok` eval/plot chain. Observed on the mixture-SFT run `gemma-3-270m/balanced-progdistill-mixture-...-sftlr1e-4-steps1600-seed1` (2026-07-09, job 29825314). **Prevention:** for Gemma SFT pass a small `SFT_VAL_MAX_BATCHES` (e.g. `4`) — the val pass is only a logged loss number, so a tiny cap avoids the OOM and lets the run merge itself. **Salvage after the fact** (weights are fine): `scancel` the zombie, then merge the shards to HF with `scripts/merge_sft_ckpt.sh` (`STEP_DIR=<...>/global_step_1600 TARGET_DIR=<exp dir> sbatch scripts/merge_sft_ckpt.sh`), and launch GRPO with `MODEL_PATH=<exp dir>`.
- **SFT final-validation OOM on Qwen3 (KD and NTP) — stale logits pinned by an omegaconf
  reference cycle (root-caused + fixed 2026-11-10).** `_compute_loss_and_backward` called
  `self.config.data.get(<missing key>)` (`opd`, `kd`, `kd_chunk`) per micro-batch. omegaconf
  raises and catches a `ConfigKeyError` internally; the caught exception's traceback holds the
  omegaconf frames, whose `f_back` pins the loss function's frame, and with it the
  `(64, 512, 151936)` bf16 logits (9.27 GiB). They are freed only when Python's cyclic GC runs,
  so 1–7 stale copies piled up and validation OOMed at ~70 GiB on some runs but not others (GC
  timing). Fix: the four flags are read once in `FSDPSFTTrainer.__init__`
  (`self._use_opd/_use_kd/_kd_chunk/_opd_clip`); after it, memory after every val batch is
  4.8 GiB flat (probe 59567081 vs 14–42 GiB before, 59564092). **Rule: never call
  `DictConfig.get()` on a possibly-missing key inside a per-step function that holds large
  tensors** — hoist it to `__init__`. Memory instrumentation stays in the file, inert unless
  `VERL_MEM_PROBE=1` (`_mem_probe`, `_mem_probe_referrers`).
- **`SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats` — poisoned
  `~/.triton/cache` (root cause, fixed 2026-08-25).** Triton's kernel cache defaults to
  `~/.triton/cache`, which is shared by *every* Python interpreter that ever ran triton
  from that home dir. A `cuda_utils` C extension built by a different CPython **imports
  fine** and then raises this `SystemError` the moment a kernel is launched
  (`triton/compiler/compiler.py` `_init_handles` → `driver.active.utils.load_binary`),
  because Python ≥3.10 requires `PY_SSIZE_T_CLEAN` for `#` format codes. On bingbin's
  home dir the cache held an untagged `cuda_utils.so` from Aug 2025 plus a
  `cuda_utils.cpython-313-*.so`, while the `verl` env is Python 3.10.20 / triton 3.2.0.
  This is **not** specific to inductor or to gemma: it breaks *any* triton kernel launch,
  and which runs survive depends only on which cache entry they resolve. It killed every
  OLMo-2-1B GRPO run at step 1 in flash-attn's fused cross-entropy
  (`verl/utils/torch_functional.py:86 logprobs_from_logits` →
  `logprobs_from_logits_flash_attn`), a path taken for *all* models — vLLM rollout and
  `val_before_train` had already succeeded. `TORCHDYNAMO_DISABLE`/`--enforce_eager` do
  **not** help here; they only suppress the inductor trigger.
  **Fix (applied):** every GPU job script now exports a node-local, per-job
  `TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-$$}"` (same pattern as `WANDB_DIR` /
  `HF_DATASETS_CACHE`), so the extension is always built by the interpreter that uses it —
  `train_grpo.sh`, `train_ppo.sh`, `train_sft.sh`, `train_opd.sh`, `eval.sh`,
  `eval_base.sh`, `generate_sft_data.sh`, `generate_opd_data.sh`, `diagnose_gt_ntp.sh`,
  `smoke_gemma.sh`. Verified on an H100: identical code fails with the shared cache and
  passes with a fresh `TRITON_CACHE_DIR` (probe job 41839649). A one-off
  `rm -rf ~/.triton/cache` also clears it, but the per-job dir prevents recurrence.
  The gemma-eval workaround below predates this diagnosis and should no longer be needed.
- **`OSError: [Errno 116] Stale file handle` at vLLM engine init — shared
  `~/.cache/vllm` torch-compile cache (found 2026-09-22, fixed).** vLLM's
  `VLLM_CACHE_ROOT` defaults to `~/.cache/vllm` (NFS home) and its
  `torch_compile_cache/<graph-hash>/` is keyed **only by the model graph**, not by
  job. Two concurrent jobs compiling the *same* model therefore write into the same
  directory; one job's rename into the cache invalidates the other's open handle and
  it dies in `triton CompiledKernel.__init__` -> `BackendCompilerFailed:
  OSError: [Errno 116] Stale file handle`, during `actor_rollout_init_model` (~7 min
  in, before step 1). Hit the third of three Qwen3-0.6B GRPO jobs launched together
  (job 47811579; the two 0.6B jobs shared graph hash `23afc4795d`). The 1.7B job in
  the same batch was unaffected — different hash, no collision. This is almost
  certainly the same root cause as the historical `JSONDecodeError: Extra data` in
  `torch._inductor.remote_cache` / the vLLM compile cache that was worked around
  with `--enforce_eager`.
  **Fix (applied):** every GPU job script now exports a node-local, per-job
  `VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"` immediately after the
  `TRITON_CACHE_DIR` block (same 11 scripts). Cost is one ~1-2 min recompile per
  job. NB **sbatch snapshots the batch script at submit time**, so already-pending
  jobs (e.g. an eval array chained by `run_pipeline.sh` before the patch) still run
  the old copy — cancel and resubmit them to pick up a script fix.
- Gemma-3-270m **eval** crashes at vLLM engine init in some conda envs even with `--enforce_eager`. On bingbin's env (2026-07-15) every gemma eval died with `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats` → `torch._dynamo.exc.BackendCompilerFailed: backend='inductor'` while inductor tried to compile+load a triton kernel (the fused RMSNorm `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0`). Root cause is a **broken triton C-extension in that conda env** — any torch.compile→inductor→triton path fails; Qwen evals happen to avoid the compiled op so they work, gemma's RMSNorm forces it. `--enforce_eager` alone is *not* enough (vLLM's V1 engine still runs a compile pass), and V0 (`VLLM_USE_V1=0`) still inductor-compiles the RMSNorm. **Fix:** submit gemma evals with `VLLM_USE_V1=0 TORCHDYNAMO_DISABLE=1 EVAL_EXTRA_ARGS="--enforce_eager"` — fully-eager, no triton. This runs correctly but is ~5-6× slower than the compiled path (fully-eager decode). Budget accordingly: n=3,4 (997 prompts ×32) ≈ 55 min; n=5 (4620) ≈ 4-5 h; n=6 (3519) ≈ 3-4 h — the default `eval.sh` 1:30 walltime TIMEOUTs on n=5/n=6, so pass `-t 08:00:00` and prefer a non-preemptible partition (`kempner_h100`) for the big sets. NB: SFT-only eval jobs finish `eval.py` (JSON written) then exit non-zero on `eval.sh`'s trailing `chmod` of the nonexistent `checkpoints/<...>` path (SFT lives under `sft-checkpoints/`) — sacct shows `FAILED`/exit 1 but **the JSON is valid**; verify by file existence, not job State.
- `EXTRA_ARGS` vs `EVAL_EXTRA_ARGS`. `EXTRA_ARGS` is Hydra-style (`key=value`) and is consumed by `train_grpo.sh`/`train_ppo.sh` only. `eval.sh` reads `EVAL_EXTRA_ARGS` (argparse flags for `eval.py`). Previously both scripts read `EXTRA_ARGS`, so setting it for gemma training silently propagated through `run_pipeline.sh` into the eval jobs (eval.py argparse → `unrecognized arguments` → all 30 evals failed in ~1 min each). Always set the gemma microbatch knobs via `EXTRA_ARGS`; only set `EVAL_EXTRA_ARGS` if you actually have argparse flags to pass to `eval.py`.

---

## Local verl + vLLM patches

The `verl/` checkout is **not** tracked by this repo. Several edits live only in the working tree (Marc keeps them on a `local-fixes` branch in his verl checkout; Shlok's checkout has them as uncommitted edits in `main`). Re-apply on any fresh clone — without these, gemma-3-270m training is fully broken and progdistill SFT silently runs at LR ≈ 0.

Verify the patches are present on your tree with:

    cd verl && git diff --stat
    # expect: fsdp_sft_trainer.py, fsdp_utils.py, rl_dataset.py, sft_dataset.py, fsdp_vllm.py, ray_trainer.py
    # (the combined diff is checked in as patches/local-fixes-vs-083da9ab.patch;
    #  apply on a fresh checkout with: bash scripts/apply_verl_patches.sh)
    cat $(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/model_executor/models/gemma3.py | grep -n 'register_buffer.*normalizer'
    # expect: persistent=False on the line

### 1. `verl/trainer/fsdp_sft_trainer.py` — SFT scheduler-total fix + seed wiring

Two edits in this file:

**a) LR scheduler horizon.** In `_build_model_optimizer`, prefer `trainer.total_training_steps` over `steps_per_epoch * total_epochs` so the cosine LR schedule actually completes during the configured run (mirrors the pattern in `fit()`):

    if self.config.trainer.get("total_training_steps", None) is not None:
        self.total_steps = int(self.config.trainer.total_training_steps)
    else:
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

Without this, progdistill SFT runs `total_epochs=9999 + total_training_steps=160` and the scheduler horizon balloons to ~6M steps → effective LR ≈ `2.6e-4 × SFT_LR`. Symptom: `train/lr(1e-3)` in wandb stays orders of magnitude below `SFT_LR * 1000`.

**b) Seed wiring.** Stock verl's SFT trainer ignores any seed config: `DistributedSampler(...)` defaults to `seed=0` and `torch.manual_seed` is never called. The patch reads `data.seed` from config and threads it into both. Required for reproducible SFT runs across `--array=1-N` seed sweeps.

    seed = int(self.config.data.get("seed", 0))
    torch.manual_seed(seed)
    self.train_sampler = DistributedSampler(
        self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True, seed=seed
    )

`scripts/train_sft.sh` passes `+data.seed=${SFT_SEED}` and `scripts/run_pipeline.sh` forwards `SFT_SEED` from env (default 1).

**c) Final-validation cap + save-before-validate (added 2026-07-07).** `train_sft.sh` points `data.val_files` at the same parquet(s) as `data.train_files`, and with `test_freq=-1` the trainer's only validation pass fires on the last step (`is_last_step` in `fit()`). For mixture SFT the "val set" is the full ~400k-row train mixture → ~100k silent forward passes (hours) after the final step, during which the job looks hung and, worse, ran **before** `save_checkpoint` — a walltime kill mid-validation lost the whole run (observed on jobs 29107881/29107889, 2026-07-07: wandb stopped at step 1600, jobs kept "Running" for hours). Two changes in `fit()`:

- The `save_checkpoint` block is moved **before** the validation block, so the last-step checkpoint is on disk before the (potentially long) final validation starts.
- The validation loop honors optional `trainer.max_val_batches` (`<0` = no cap, i.e. stock behavior) and shows a tqdm bar. `scripts/train_sft.sh` passes `+trainer.max_val_batches=${SFT_VAL_MAX_BATCHES:-500}` (500 micro-batches × 4 = 2000 samples; set `SFT_VAL_MAX_BATCHES=-1` for a full pass).
- The val `DistributedSampler` is switched from `shuffle=False` to `shuffle=True, seed=data.seed` so the capped pass is a seeded random subsample of the whole mixture rather than the head of the first parquet (which would be earliest-teacher-step data only).

### 2. `verl/utils/fsdp_utils.py` — FSDP wrap-policy: skip missing classes instead of raising

`get_fsdp_wrap_policy` raises if any class listed in `_no_split_modules` isn't found on the loaded model. Some HF model classes (e.g. `Gemma3ForCausalLM`) declare vision-tower modules in `_no_split_modules` that don't exist on the text-only variant — so the wrap-policy raises before training starts. Match HF Trainer's behavior and skip missing classes; only raise if **none** of them are found:

    transformer_cls_to_wrap = set()
    for layer_class in fsdp_transformer_layer_cls_to_wrap:
        transformer_cls = get_module_class_from_name(module, layer_class)
        if transformer_cls is not None:
            transformer_cls_to_wrap.add(transformer_cls)
    if not transformer_cls_to_wrap:
        raise Exception("Could not find the transformer layer class to wrap in the model.")

### 3. `verl/utils/dataset/rl_dataset.py` + `verl/utils/dataset/sft_dataset.py` — chat_template fallback

Both datasets call `tokenizer.apply_chat_template(...)` unconditionally. Gemma-3-270m's tokenizer has no `chat_template`, so this raises `ValueError: Cannot use apply_chat_template()...`. The repo's parquet files are already in `template_type='base'` format (the prompt is the literal string the model sees), so when no chat template exists, fall through to plain concatenation:

    if getattr(tokenizer, "chat_template", None):
        raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # Base models (e.g. gemma-3-270m): parquet content is already final.
        raw_prompt = "".join(m["content"] for m in messages)

Three call sites: `rl_dataset.py:doc2len`, `rl_dataset.py:__getitem__` (raw_prompt assembly), and `sft_dataset.py:__getitem__`.

### 4. `verl/workers/sharding_manager/fsdp_vllm.py` — gemma3 normalizer-buffer restore

The most subtle and costly bug to diagnose. **Symptom:** every gemma-3-270m GRPO run produces multilingual token salad in vLLM rollout (`val_reward = 0.0` from step 0, response length pinned to the 1024 cap). The same checkpoint generates correctly via `eval.py` (standalone vLLM with `load_format=auto`) and via HF `generate(...)`.

**Root cause.** verl's GRPO default is `actor_rollout_ref.rollout.load_format=dummy_dtensor` (set in `verl/trainer/config/ppo_trainer.yaml`). vLLM's `DummyModelLoader.load_model` calls `initialize_dummy_weights(model)`, which iterates `model.state_dict()` and overwrites floating-point tensors with random values in `[-1e-3, 1e-3]`. **Persistent buffers are included in `state_dict`.** vLLM 0.8.5's `gemma3.py` registers `normalizer = sqrt(hidden_size) ≈ 25.25` as a persistent buffer (no `persistent=False`), so it gets clobbered to a random ~0 value. verl's subsequent `update_params → model.load_weights(...)` only writes parameters, never buffers, so `normalizer ≈ 0` permanently. Forward pass: `embed_tokens(x) * normalizer ≈ 0` → uniform-random logits → token salad.

Qwen2.5 has no equivalent buffer in its vLLM model class, so it's unaffected — every Qwen GRPO works, every gemma GRPO breaks.

**Fix (verl side).** In `update_params`, after `model.load_weights(...)`, walk the model and restore any 1-element `normalizer` tensor to `sqrt(module.config.hidden_size)`:

    if not self.base_sync_done:
        import torch as _torch
        for _mod in model.modules():
            _norm = getattr(_mod, "normalizer", None)
            if isinstance(_norm, _torch.Tensor) and _norm.numel() == 1:
                _cfg = getattr(_mod, "config", None)
                if _cfg is not None and hasattr(_cfg, "hidden_size"):
                    _norm.data.fill_(float(_cfg.hidden_size) ** 0.5)

**Fix (vLLM side, canonical).** Open `~/.conda/envs/verl/lib/python3.10/site-packages/vllm/model_executor/models/gemma3.py` and find `register_buffer("normalizer", torch.tensor(normalizer))` near line 372. Add `persistent=False`:

    self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)

Both patches are kept — the vLLM edit is the canonical fix; the verl edit is defense-in-depth in case the conda env is reinstalled and the vLLM patch is lost.

**Verify:** run `repro_v2.py` on an H100 (`sbatch run_repro.sh`). Both `RUN A` (load_format=auto) and `RUN B` (load_format=dummy + push HF state_dict) must report `model.normalizer  shape=()  mean=+2.5250e+01` and produce a coherent `<answer>` for the test prompt. Pre-fix, RUN B shows mean ~`-7.4e-04` and outputs token salad.

### 5. `verl/trainer/ppo/ray_trainer.py` — `trainer.log_freq` throttle (added 2026-07-08)

Stock verl logs GRPO/PPO train metrics to wandb/console **every step**. The patch adds
an optional `trainer.log_freq` (default 1 = stock): per-step train metrics only log
when `global_steps % log_freq == 0`; steps carrying validation metrics and the final
step always log. `train_grpo.sh` passes `+trainer.log_freq=${GRPO_LOG_FREQ:-50}`.
Validation cadence is unaffected (still `test_freq=50`), so `plot_results.py`'s
val-reward parsing keeps working.

### 6. `verl/trainer/fsdp_sft_trainer.py` — offline-OPD loss branch (added 2026-07-09)

In `_compute_loss_and_backward`, when `+data.opd=True`, the per-token CE loss
is reweighted into the Lightning-OPD policy-gradient surrogate: per-token CE is
`−log π_θ(a_t|s_t)`, so `advantage = clamp(teacher_logprobs + CE.detach(),
±data.opd_clip)` equals `clamp(log π_T − log π_θ, ±clip)` with stop-gradient,
and the loss becomes `advantage * CE`. The `teacher_logprobs` batch entry comes
from `opd_dataset.py` (repo root, injected via `data.custom_cls` — not a verl
patch). Raises `NotImplementedError` under sequence parallel. Stock SFT is
untouched when `data.opd` is unset. Used by `scripts/train_opd.sh`.

### Re-applying patches on a fresh checkout

| Patch | File | Purpose |
|---|---|---|
| 1a | `verl/verl/trainer/fsdp_sft_trainer.py` (`_build_model_optimizer`) | SFT LR scheduler horizon |
| 1b | `verl/verl/trainer/fsdp_sft_trainer.py` (sampler init) | SFT seed wiring |
| 1c | `verl/verl/trainer/fsdp_sft_trainer.py` (`fit()` last-step block) | save-before-validate + `trainer.max_val_batches` cap |
| 2 | `verl/verl/utils/fsdp_utils.py` (`get_fsdp_wrap_policy`) | FSDP wrap-policy: skip missing classes |
| 3a | `verl/verl/utils/dataset/rl_dataset.py` | chat_template fallback (RL dataset, 2 sites) |
| 3b | `verl/verl/utils/dataset/sft_dataset.py` | chat_template fallback (SFT dataset) |
| 4 verl | `verl/verl/workers/sharding_manager/fsdp_vllm.py` (`update_params`) | gemma3 normalizer buffer restore |
| 4 vLLM | `<conda env>/lib/python3.10/site-packages/vllm/model_executor/models/gemma3.py` line ~372 | `persistent=False` on normalizer buffer |
| 5 | `verl/verl/trainer/ppo/ray_trainer.py` (`fit()` logging) | optional `trainer.log_freq` throttle for per-step train metrics |
| 6 | `verl/verl/trainer/fsdp_sft_trainer.py` (`_compute_loss_and_backward`) | offline-OPD loss branch (`data.opd`) |
| 7 | `verl/verl/trainer/fsdp_sft_trainer.py` (`_build_model_optimizer`) | `optim.lr_scheduler=constant` (constant LR after optional warmup; `SFT_LR_SCHEDULER=constant` in `train_sft.sh`) |
| 8 | `verl/verl/trainer/fsdp_sft_trainer.py` (`kd_token_loss`, `_compute_loss_and_backward`, `training_step`) | logit-distillation loss branch (`+data.kd=True`): bucketed forward KL to stored teacher top-k + sampled-token bucket + tail; logs `train/kd_ce`. Batch from `DistillKDDataset` (`distill_dataset.py`), sidecars from `scripts/convert_distill_topk.py` |
| 9 | `verl/verl/trainer/fsdp_sft_trainer.py` (`_build_model_optimizer`) | optional cosine floor `+optim.min_lr_ratio=r`: linear warmup from 0, cosine to r×peak (`SFT_MIN_LR_RATIO` in `train_sft.sh`, `MIN_LR_RATIO` in `run_qwen3_distill_kd_sft.sh`); unset = stock cosine |

---

## Current experiment handoff (May 25, 2026)

This section is a compact memory dump from the May 2026 Gemma/Qwen distillation and GRPO work. Prefer it over older chat memory when continuing the experiment.

### Repo / branch state

- Branch: `my-feature`.
- Latest pushed local changes as of May 22/25 include:
  - `Tag training runs and link artifact dirs`
  - `Default GRPO jobs to two H100s`
  - `Tag W&B runs and harden Gemma GRPO jobs`
- The remote branch moved while pushing; local commits were rebased on top of Marc's remote commits and the user successfully pushed.
- Important code changes now in the repo:
  - `scripts/train_grpo.sh` defaults GRPO training to **2 H100s** instead of 4.
  - For Gemma GRPO, `scripts/train_grpo.sh` automatically uses Gemma-safe microbatches:
    - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
    - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16`
    - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16`
  - `WANDB__SERVICE_WAIT` default is 300 seconds to reduce W&B startup failures.
  - `plot_results.py`, `scripts/train_sft.sh`, and `scripts/train_grpo.sh` tag W&B runs by stage/mode/model size where possible.
  - `scripts/backfill_wandb_tags.py` exists for W&B tag backfill, but mutating some older Qwen runs failed with 403 permissions.
  - `scripts/run_pipeline.sh` passes `EVAL_EXTRA_ARGS` through to eval jobs.

### W&B / auth notes

- W&B project/entity: `bingbin/prog_distill` (entity default; override with `WANDB_ENTITY=<entity>`).
- `wandb status` may show `api_key: null` even after login because credentials are stored in `~/.netrc`; that is normal for this CLI.
- Backfill script command:

    env -u WANDB_API_KEY /n/home06/sdholakia/.conda/envs/verl/bin/python3 scripts/backfill_wandb_tags.py --quiet

- If the backfill emits 403s, the key is valid but lacks write access to those runs. Have the run owner/admin grant write permission or run the script from the owner account.

### Official Qwen student choices

After checking W&B directly:

| Qwen student | Official SFT LR | Evidence |
|---|---:|---|
| progdistill | `1e-6` | finished run `Qwen2.5-0.5B-balanced-progdistill-sftlr1e-6-grpo-lr1e-6-kl3e-4-seed1` (`2oeru148`) and eval `48cfb75b` |
| vanilla distill | `3e-5` | clean finished/evaled candidate `Qwen2.5-0.5B-balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1`; previous same-name run is archived locally as `...-seed1-obsolete` |

The Qwen progdistill student that was uploaded/used for Marc to recreate GRPO was the `sftlr=1e-6` student. It was uploaded as a subfolder into `https://huggingface.co/shlokdho/qwen2.5-1.5b-countdown-teacher`, not a new repo.

### Pure Gemma SFT student results

These are final pure distill/progdistill student evals for `gemma-3-270m`, from local `.scores` files under `results/gemma-3-270m`. Each cell is `mean@1 / mean@32`.

**Vanilla distill**

| SFT LR | n=3/4 | n=5 | n=6 |
|---:|---:|---:|---:|
| `3e-6` | `0.1043 / 0.1961` | `0.0200 / 0.1196` | `0.0010 / 0.1000` |
| `1e-5` | `0.4183 / 0.4687` | `0.0890 / 0.1667` | `0.0020 / 0.0990` |
| `3e-5` | `0.5697 / 0.6003` | `0.1250 / 0.1854` | `0.0030 / 0.0846` |
| `1e-4` | `0.6309 / 0.6672` | `0.1890 / 0.2487` | `0.0050 / 0.0885` |
| `3e-4` | `0.6630 / 0.6656` | `0.2060 / 0.2588` | `0.0100 / 0.0911` |

**Progressive distill**

| SFT LR | n=3/4 | n=5 | n=6 |
|---:|---:|---:|---:|
| `3e-6` | `0.1174 / 0.1860` | `0.0130 / 0.1013` | `0.0010 / 0.0869` |
| `3e-5` | `0.5647 / 0.6061` | `0.1670 / 0.2353` | `0.0060 / 0.0963` |
| `1e-4` | `0.6038 / 0.6241` | missing | missing |
| `3e-4` | `0.6058 / 0.6207` | `0.1600 / 0.2361` | `0.0100 / 0.0915` |

Interpretation used for supervisor discussion: `sftlr=1e-4` is defensible for both Gemma students because it is best on progdistill n=3/4 and very close to the best distill setting, giving an apples-to-apples student comparison.

### Important missing eval / failed eval

~~Progdistill `sftlr=1e-4` n=5 and n=6 are still missing~~ **Resolved 2026-06-04.** Evals rerun with `--enforce_eager` (bypasses vLLM compile cache; see `eval.py`). Final scored values (all 4620 / 3519 prompts, same metric as `plot_results.py`):

| | n=5 mean@1 | n=5 mean@32 | n=6 mean@1 | n=6 mean@32 |
|---|---|---|---|---|
| progdistill `sftlr=1e-4` | `0.1229` | `0.1909` | `0.0043` | `0.0863` |

`.scores` cache files written to `results/gemma-3-270m/balanced-progdistill-sftlr1e-4-seed1/global_step_1600/`. Original failure: `JSONDecodeError: Extra data` in `torch._inductor.remote_cache` / vLLM torch compile cache — fixed permanently by adding `--enforce_eager` flag to `eval.py` and passing `EVAL_EXTRA_ARGS="--enforce_eager"` at submission.

### Base Gemma GRPO held-out evals

Base Gemma GRPO validation mean@4 clustered around 0.10, but held-out eval mean@1 was flat zero. Final held-out results:

| GRPO lr | KL | n=3/4 | n=5 | n=6 |
|---:|---:|---:|---:|---:|
| `1e-6` | `3e-4` | `0.0000 / 0.0896` | `0.0000 / 0.0893` | `0.0000 / 0.0903` |
| `1e-6` | `1e-3` | `0.0000 / 0.0432` | `0.0000 / 0.0415` | `0.0000 / 0.0415` |
| `1e-6` | `3e-3` | `0.0000 / 0.0427` | `0.0000 / 0.0422` | `0.0000 / 0.0423` |
| `3e-6` | `3e-4` | `0.0000 / 0.0421` | `0.0000 / 0.0411` | `0.0000 / 0.0336` |
| `3e-6` | `1e-3` | `0.0000 / 0.0336` | `0.0000 / 0.0371` | `0.0000 / 0.0365` |
| `3e-6` | `3e-3` | `0.0000 / 0.0454` | `0.0000 / 0.0385` | `0.0000 / 0.0433` |
| `1e-5` | `3e-4` | `0.0000 / 0.0952` | `0.0000 / 0.0949` | `0.0000 / 0.0949` |
| `1e-5` | `1e-3` | `0.0000 / 0.0620` | `0.0000 / 0.0610` | `0.0000 / 0.0605` |
| `1e-5` | `3e-3` | `0.0000 / 0.0446` | `0.0000 / 0.0446` | `0.0000 / 0.0477` |

Cause of train/eval divergence: training validation was giving mostly format reward under teacher-forced sampling/validation conditions, while held-out eval at temp 0.6 produced format-valid wrong answers. Do not present validation mean@4 as held-out success.

### Student GRPO selected completed points

Main Gemma student-GRPO comparison uses `sftlr=1e-4` students and GRPO `lr=1e-6, kl=3e-4` as a conservative shared setting motivated by base-Gemma sweeps where `kl=3e-4` was consistently strong. Completed W&B/local selected metrics:

| Student | GRPO lr | KL | n=3/4 mean@32 | n=5 mean@32 | n=6 mean@32 |
|---|---:|---:|---:|---:|---:|
| distill `sftlr=1e-4` | `1e-6` | `3e-4` | `0.8985` | `0.2602` | `0.1010` |
| progdistill `sftlr=1e-4` | `1e-6` | `3e-4` | `0.8701` | `0.2943` | `0.1020` |

Other completed student-GRPO points from Marc/W&B:

| Student | GRPO lr | KL | n=3/4 mean@32 | n=5 mean@32 | n=6 mean@32 |
|---|---:|---:|---:|---:|---:|
| distill `sftlr=1e-4` | `1e-6` | `3e-3` | `0.8830` | `0.3111` | `0.1034` |
| progdistill `sftlr=1e-4` | `1e-6` | `3e-3` | `0.8517` | `0.3674` | `0.1051` |

### Student-GRPO training/rerun status from May 22

The failed/partial student-GRPO sweep was resubmitted with 2 H100s, 12h walltime, Gemma-safe microbatches, evals, and plots chained. Train job IDs:

- `14547738` progdistill sftlr1e-4 lr1e-5 kl1e-3
- `14547766` progdistill sftlr1e-4 lr1e-5 kl3e-3
- `14547803` distill sftlr1e-4 lr1e-6 kl1e-3
- `14547831` progdistill sftlr1e-4 lr3e-6 kl3e-4
- `14547866` progdistill sftlr1e-4 lr3e-6 kl3e-3
- `14547892` distill sftlr1e-4 lr1e-5 kl3e-3
- `14547929` distill sftlr1e-4 lr3e-6 kl3e-4 early400
- `14547981` distill sftlr1e-4 lr3e-6 kl1e-3
- `14548007` distill sftlr1e-4 lr3e-6 kl3e-3
- `14548026` progdistill sftlr1e-4 lr1e-6 kl1e-3
- `14548041` progdistill sftlr1e-4 lr1e-5 kl3e-4

If continuing after May 25, first check `sacct`/`squeue` for these IDs and inspect logs before resubmitting.

### Hugging Face handoff for Marc

Gemma students used for GRPO were uploaded as subfolders to `shlokdho/qwen2.5-1.5b-countdown-teacher`. Marc's agent should download those subfolders, point GRPO `MODEL_PATH` / checkpoint path at the downloaded HF model directories, and run the same `lr=1e-6, kl=3e-4` GRPO pipeline with eval+plot chaining.

Qwen 0.5B vanilla-distill `sftlr=1e-5` seed1 SFT student exists locally at:

`/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints/sft-checkpoints/Qwen2.5-0.5B/balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr1e-5-seed1`

It was uploaded to Hugging Face on 2026-05-25 as:

`https://huggingface.co/shlokdho/qwen2.5-1.5b-countdown-teacher/tree/main/qwen2.5-0.5b-distill-sftlr1e-5-seed1`

Use that HF subfolder for Marc's Qwen 0.5B distill `sftlr=1e-5` GRPO-on-top run at `grpo_lr=1e-6`, `grpo_kl=3e-4`, seed1.

### Qwen 0.5B ID/OOD Pareto data status

The current Qwen plotting task is to compare ID `balanced` (n=3/4) on the x-axis against OOD `balanced5` (n=5) and `balanced6` (n=6) on the y-axis, separately for `mean@1` and `mean@32`, for distill/progdistill pre-GRPO and GRPO-on-top checkpoints.

Tracked analysis tables live in `analysis_data/`:

- `qwen05b_grpo_on_top_pareto_eval.csv`: score rows used for plotting.
- `qwen05b_distill_grpo_run_registry.csv`: run/checkpoint/eval roots.
- `qwen05b_distill_grpo_eval_inventory.csv`: per-step eval availability.
- `qwen05b_seed1_distill_eval_backfill_jobs.csv`: SLURM eval arrays submitted for missing seed1 distill evals.

Canonical Qwen 0.5B distill+GRPO checkpoint root for the seed1/seed21 inventory:

`/n/netscratch/kempner_bingbin_lab/Lab/sdholakia/archive/rl-checkpoints/checkpoints/Qwen2.5-0.5B`

When using `scripts/eval.sh` against that archive, pass:

`CHECKPOINT_DIR=/n/netscratch/kempner_bingbin_lab/Lab/sdholakia/archive/rl-checkpoints`

Seed1 vanilla-distill eval backfill submitted on 2026-05-25:

| Job ID | Run | Dataset | Steps |
|---:|---|---|---|
| `15572281` | `sftlr3e-5` | `balanced` | `750,900,1050,1200,1400,1600` |
| `15572285` | `sftlr3e-5` | `balanced5` | all standard steps |
| `15572287` | `sftlr3e-5` | `balanced6` | all standard steps |
| `15572288` | `sftlr3e-6` | `balanced` | all standard steps |
| `15572290` | `sftlr3e-6` | `balanced5` | all standard steps |
| `15572293` | `sftlr3e-6` | `balanced6` | all standard steps |

After those jobs complete, rerun:

```bash
python3 scripts/inventory_qwen05b_distill_grpo_runs.py
python3 scripts/export_pareto_eval_data.py
```

Then commit the refreshed CSVs if the eval JSONs/scores landed under `results/Qwen2.5-0.5B`.

### Storage / archive notes

- Qwen2.5-0.5B checkpoint trees were copied/archived toward net scratch; older cleanup/archive jobs were used. Re-check lab vs netscratch before deleting anything.
- Do not assume Slurm `COMPLETED` means eval JSON exists; always check `results/.../*.json` and `.scores`, especially for Gemma evals.

---

## HuggingFace backup (2026-06-04)

Harvard cluster access was removed on 2026-06-04. Everything important was backed up to HuggingFace under `marcwalden`. Starting fresh on a new cluster: download what you need from the repos below.

### Trained model checkpoints (step_1600 only)

| HF repo | Contents |
|---|---|
| `marcwalden/qwen2.5-0.5b-countdown` | 7 Qwen2.5-0.5B GRPO runs (distill/progdistill across sftlr ∈ {1e-5,1e-6,3e-5} and kl ∈ {3e-4,3e-3}), each as a subfolder named by `EXP_NAME` |
| `marcwalden/qwen2.5-1.5b-countdown` | 2 Qwen2.5-1.5B GRPO runs: `balanced-grpo-kl3e-3-lr3e-6-seed1` (canonical teacher) and `shlok-best-1point5-model` |
| `marcwalden/gemma-3-270m-countdown` | 5 Gemma-3-270m GRPO runs: 4 student-GRPO (distill/progdistill × kl3e-4/kl3e-3) + 1 base GRPO |
| `marcwalden/gemma-3-270m-distill-sftlr1e-4-seed1` | Gemma distill sftlr=1e-4 SFT student (round_1600) |
| `marcwalden/gemma-3-270m-progdistill-sftlr1e-4-seed1-round_1600` | Gemma progdistill sftlr=1e-4 SFT student (round_1600) |
| `marcwalden/qwen2.5-0.5b-distill-sftlr1e-5-seed1` | Qwen2.5-0.5B distill sftlr=1e-5 SFT student |
| `marcwalden/qwen2.5-0.5b-progdistill-sftlr1e-6-seed1` | Qwen2.5-0.5B progdistill sftlr=1e-6 SFT student |
| `marcwalden/rl-skill-comp-sft-students` | All 6 SFT round_1600 students as subfolders (Qwen2.5-0.5B × 5 sftlr variants + Gemma progdistill sftlr=1e-4) |
| `shlokdho/qwen2.5-1.5b-countdown-teacher` | Shlok's 1.5B teacher + Qwen2.5-0.5B distill sftlr=1e-5 seed1 student (subfolders) |

Download a checkpoint for inference/eval:
```bash
huggingface-cli download marcwalden/qwen2.5-0.5b-countdown \
  --include "balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed1/*" \
  --local-dir ./checkpoints/Qwen2.5-0.5B
```

### Eval results and data (HF dataset)

`marcwalden/rl-skill-comp-evals` contains:

| Path in dataset | Contents |
|---|---|
| `Qwen2.5-0.5B/` | All eval JSONs + `.scores`/`.lengths` caches for every Qwen 0.5B GRPO run at all 10 steps |
| `Qwen2.5-1.5B/` | Same for Qwen 1.5B runs |
| `gemma-3-270m/` | Same for all Gemma runs (GRPO + pure SFT students) |
| `baselines/` | Baseline eval JSONs |
| `data/` | Train/test parquet files (`balanced/`, `balanced5/`, `balanced6/`) — the exact puzzle splits used |
| `sft-data/` | Teacher rollout parquets for all SFT rounds (81 MB) |
| `figures/` | All Pareto plots and training curves |

Download just the scores (tiny, sufficient for plotting/analysis):
```bash
huggingface-cli download marcwalden/rl-skill-comp-evals --repo-type dataset \
  --include "**/*.scores" --include "**/*.lengths" --local-dir ./results
```

Download everything:
```bash
huggingface-cli download marcwalden/rl-skill-comp-evals --repo-type dataset \
  --local-dir ./results
```

### Code

- RL training code: GitHub `marcwalden1/Prog-Distill-Countdown` (branch `my-feature`)
- `analysis_data/` CSVs (Qwen 0.5B Pareto run registry + inventory) are git-tracked in the repo

### Metric definitions (for reading `.scores` files)

`.scores` files are JSON arrays `[step, mean@1, mean@32]` where:
- **mean@1** = fraction of prompts where `outputs[0]` is exactly correct (binary, averaged over prompts)
- **mean@32** = average raw `compute_score()` over all 32 outputs per prompt, then averaged over prompts (scores are 0.0 / 0.1 / 1.0 — not pass@32)

## Data backups (2026-06-04)
Harvard cluster access removed. All data backed up:
| What | Where |
|---|---|
| Fine-tuned Gemma models | HF `marcwalden/gemma-3-270m-distill-sftlr1e-4-seed1`, `marcwalden/gemma-3-270m-progdistill-sftlr1e-4-seed1-round_1600` |
| Fine-tuned Qwen models | HF `marcwalden/qwen2.5-0.5b-distill-sftlr1e-5-seed1`, `marcwalden/qwen2.5-0.5b-progdistill-sftlr1e-6-seed1` |
| EoSS experiment results | HF dataset `marcwalden/eoss-results` |
| This code repo | GitHub `marcwalden1/RL-skill-comp` |

---

## Pareto-domination analysis: distill→GRPO vs progdistill→GRPO (2026-06-09)

Question: does **progdistill→GRPO** Pareto-dominate **distill→GRPO** — better on
*both* in-dist (`balanced`, n=3,4) and OOD (`balanced5`/`balanced6`, n=5,6) —
holding GRPO hyperparams (sftlr, lr, kl) fixed? Plan: `~/.claude/plans/swift-wibbling-toast.md`.

New script **`scripts/plot/plot_pareto_2d.py`** (run locally with
`/n/home06/mwalden/.conda/envs/verl/bin/python3`, reads only the tiny `.scores`
caches — no sbatch/OOM risk):

```bash
python3 scripts/plot/plot_pareto_2d.py \
  --result-dir /n/netscratch/kdbrantley_lab/Lab/mwalden/rl-skill-comp-results/results \
  --model <Qwen2.5-0.5B|gemma-3-270m> \
  --distill-exp <EXP_NAME> --progdistill-exp <EXP_NAME> --tag <label>
```
Produces `figures/<model>/pareto/pareto_balanced{5,6}_<tag>.png` (one trajectory
per arm, points = checkpoints) plus a per-step + overall dominance verdict
printed to stdout.

### Qwen2.5-0.5B (matched GRPO kl=3e-4, lr=1e-6; sftlr differs: distill=3e-5, progdistill=1e-6)

No Pareto domination either direction. Plots:
`figures/Qwen2.5-0.5B/pareto/pareto_balanced{5,6}_sftlr1e-6.png` (and other
`sftlr*`/`kl3e-3` tags from the in-progress sftlr=1e-5 matched sweep, tasks #48/#49).

Item 4 (4096-token re-eval of progdistill→GRPO `step_1600` on n=5/n=6, to check
for truncation at the 1024-token cap): **closed, no meaningful difference** —
OOD numbers aren't truncation-limited. Jobs cancelled.

### Gemma-3-270m, kl3e-3 (sftlr=1e-4, GRPO lr=1e-6 — fully matched, both arms 9-10/10 ckpts)

No Pareto domination either direction, but a clear **cross-arm trade-off
frontier**: progdistill→GRPO trades in-dist accuracy (0.75-0.85) for higher OOD
n=5 accuracy (0.20-0.31); distill→GRPO sits at higher in-dist (0.84-0.89) but
lower OOD n=5 (0.20-0.245). At step 1600, progdistill (0.848/0.300) nearly
matches distill's in-dist (0.862) while beating it on OOD n=5 by +0.056 — close
to dominance. n=6 shows the same direction, smaller/noisier. Plots:
`figures/gemma-3-270m/pareto/pareto_balanced{5,6}_kl3e-3.png`.

### Gemma-3-270m, kl3e-4 (sftlr=1e-4, GRPO lr=1e-6 — same arms, other KL) — IN PROGRESS

Checkpoints exist for all 32 steps on both arms, but most evals were missing.
Launched 2026-06-09 (36 eval array tasks, `kempner_requeue`, throttled to ~2
concurrent by `QOSMaxNodePerUserLimit` — expect several hours):
- `20707044`/`20707047`/`20707048`: distill arm, full 10-step eval ×
  {balanced, balanced5, balanced6}
- `20707072`: progdistill arm, missing `balanced` steps 150/450/600/1050
- `20707083`: progdistill arm, missing `balanced6` steps 300/1050

Once `.scores` coverage matches kl3e-3 (10/9/10 distill, 10/10/10 progdistill),
run:
```bash
python3 scripts/plot/plot_pareto_2d.py --model gemma-3-270m \
  --distill-exp balanced-grpo-from-distill-lr1e-6-kl3e-4-seed1 \
  --progdistill-exp balanced-grpo-from-progdistill-lr1e-6-kl3e-4-seed1 --tag kl3e-4
```
