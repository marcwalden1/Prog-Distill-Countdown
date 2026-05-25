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

**Available models:**
- Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B
- Llama-3.2-3B (requires tokenizer_config.json override)

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
| Save/test freq | every 50 steps |
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
- Eval scores can be uploaded to wandb separately via `scripts/upload_evals_to_wandb.py` — reads `results/{model}/{exp}/global_step_*/*.{scores,lengths}` and logs per-step scalars.

### Checkpoints

Saved to {output_dir}/{exp_name}/global_step_{N}/ every 50 steps.

---

## Training: scripts/train_ppo.sh

Same structure as GRPO but:
- algorithm.adv_estimator=gae (standard PPO with critic)
- Has a separate critic model (critic.model.path, critic.optim.lr=1e-5)
- Default model: Qwen2.5-3B
- Smaller mini batch (64 vs 256)

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

compute_score(data_source, solution_str, ground_truth, extra_info, format_score=0.1, score=1.0)

- Extracts last <answer>...</answer> match from model output
- Validates all numbers used exactly once
- Evaluates equation safely (restricted eval)
- Returns: 1.0 (correct), 0.1 (valid format, wrong answer), 0.0 (no answer / invalid)

---

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

Distills a trained GRPO teacher into a student across 10 rounds. Each round: (gen-data-from-teacher-at-step-X → SFT-student), for X ∈ {150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600}. Each round runs `PROGDISTILL_STEPS_PER_ROUND` SFT steps (default 160 → total SFT budget 1600). Entry point: `scripts/run_pipeline.sh --progdistill <teacher-EXP_NAME>`.

### Prerequisites

Teacher must have checkpoints at all 10 required steps. Verify:

    ls -d $CHECKPOINT_DIR/checkpoints/$TEACHER_MODEL_NAME/<teacher-EXP_NAME>/global_step_{150,300,450,600,750,900,1050,1200,1400,1600} 2>/dev/null | wc -l
    # expect: 10

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

- Default array: `3,6,9,12,15,18,21,24,28,32` → evaluates checkpoints at steps 150, 300, 450, 600, 750, 900, 1050, 1200, 1400, 1600
- Merges FSDP shards first (`verl.model_merger merge`)
- Runs `eval.py` with vLLM, n=32 samples/prompt, temperature=0.6
- Output: `results/{model}/{exp}/global_step_{N}/{dataset}_temp{T}_n{N}_max{M}.json`

---

## Plotting: scripts/plot_results.sh + plot_results.py

- CPU-only job on `serial_requeue`, 8 CPUs, 64GB RAM, 1h time limit
- Produces **5 plots** per run in `figures/{model}/{exp}/`:
  - `val_reward_vs_step.png` — val reward mean@4 vs. training step (from log)
  - `eval_n34.png` — mean@1 + mean@32 vs. checkpoint (n=3,4)
  - `eval_n5.png` — same for n=5
  - `eval_n6.png` — same for n=6
  - `response_length.png` — mean response length (tokens) vs. checkpoint, all 3 datasets
- Uses `.scores` and `.lengths` cache files to avoid recomputing unchanged results
- Scoring uses `Pool(8)` for parallelism; capped at 1000 prompts per file for n=5,6

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

## Known issues / history

- Test set deduplication fix (2025-03): Original test set had duplicate prompts (same puzzle presented multiple times). Fixed in preprocess_balanced.py by deduplicating on (sorted(nums), target). Current test set: ~997 unique puzzles.
- val_kwargs.n: Was previously 7; corrected to 4 to match rollout.n=4.
- Buggy runs: Any checkpoints from before the deduplication fix should be discarded.
- Gemma-3-270m GRPO can OOM in actor/ref log-prob computation even though the model is smaller than Qwen2.5-0.5B. Cause: Gemma's vocab is much larger (262k vs Qwen 152k), so the lm_head logits tensor during `compute_log_prob` is huge. The failed runs `gemma-270m-balanced-grpo-lr1e-6-kl3e-3-seed1` and `gemma-270m-balanced-grpo-lr1e-6-kl1e-2-seed1` died after step 1 trying to allocate ~31 GiB at `modeling_gemma3.py:958`. For Gemma reruns, keep experiment hyperparams fixed and only lower memory microbatch knobs via `EXTRA_ARGS`: `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`. Do not bake these into defaults; Qwen runs should use the normal script defaults unless explicitly overridden.
- `EXTRA_ARGS` vs `EVAL_EXTRA_ARGS`. `EXTRA_ARGS` is Hydra-style (`key=value`) and is consumed by `train_grpo.sh`/`train_ppo.sh` only. `eval.sh` reads `EVAL_EXTRA_ARGS` (argparse flags for `eval.py`). Previously both scripts read `EXTRA_ARGS`, so setting it for gemma training silently propagated through `run_pipeline.sh` into the eval jobs (eval.py argparse → `unrecognized arguments` → all 30 evals failed in ~1 min each). Always set the gemma microbatch knobs via `EXTRA_ARGS`; only set `EVAL_EXTRA_ARGS` if you actually have argparse flags to pass to `eval.py`.

---

## Local verl + vLLM patches

The `verl/` checkout is **not** tracked by this repo. Several edits live only in the working tree (Marc keeps them on a `local-fixes` branch in his verl checkout; Shlok's checkout has them as uncommitted edits in `main`). Re-apply on any fresh clone — without these, gemma-3-270m training is fully broken and progdistill SFT silently runs at LR ≈ 0.

Verify the patches are present on your tree with:

    cd verl && git diff --stat
    # expect: fsdp_sft_trainer.py, fsdp_utils.py, rl_dataset.py, sft_dataset.py, fsdp_vllm.py
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

### Re-applying patches on a fresh checkout

| Patch | File | Purpose |
|---|---|---|
| 1a | `verl/verl/trainer/fsdp_sft_trainer.py` (`_build_model_optimizer`) | SFT LR scheduler horizon |
| 1b | `verl/verl/trainer/fsdp_sft_trainer.py` (sampler init) | SFT seed wiring |
| 2 | `verl/verl/utils/fsdp_utils.py` (`get_fsdp_wrap_policy`) | FSDP wrap-policy: skip missing classes |
| 3a | `verl/verl/utils/dataset/rl_dataset.py` | chat_template fallback (RL dataset, 2 sites) |
| 3b | `verl/verl/utils/dataset/sft_dataset.py` | chat_template fallback (SFT dataset) |
| 4 verl | `verl/verl/workers/sharding_manager/fsdp_vllm.py` (`update_params`) | gemma3 normalizer buffer restore |
| 4 vLLM | `<conda env>/lib/python3.10/site-packages/vllm/model_executor/models/gemma3.py` line ~372 | `persistent=False` on normalizer buffer |

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

- W&B project/entity: `progressive_distill/prog_distill`.
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

Progdistill `sftlr=1e-4` n=5 and n=6 are **still missing** despite Slurm marking jobs completed:

- `14445270_1` attempted `balanced5`.
- `14445271_1` attempted `balanced6`.
- Both logs show vLLM engine initialization failed before result JSONs were written.
- Failure root cause in the log: Torch/vLLM compile cache load failed with `JSONDecodeError: Extra data` inside `torch._inductor.remote_cache` / vLLM torch compile cache.
- The plot job `14445272` completed, but it only plotted n=3/4 and skipped n=5/n=6 because the result JSONs did not exist.
- To fill these rows, resubmit those two evals with a clean/isolated Torch/vLLM compile cache or eager/no-compile settings if available through `EVAL_EXTRA_ARGS` / environment.

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
