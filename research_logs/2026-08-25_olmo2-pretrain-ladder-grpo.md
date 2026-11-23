# OLMo-2-1B pretraining ladder × GRPO (lr=3e-6, kl=3e-3)

**Date:** 2026-08-25
**Status:** set up, not yet submitted
**Launcher:** `scripts/run_olmo2_pretrain_ladder.sh`

## Question

How does the amount of *pretraining* a base model has seen change what RL
post-training can induce on Countdown? Prior work here varied the student
(0.5B / 1.5B / 40M-scratch) and the distillation recipe; this varies pretraining
budget with architecture and parameter count held fixed, using OLMo-2's public
stage1 checkpoint series.

## Arms

Three points on the `allenai/OLMo-2-0425-1B` stage1 curve, one GRPO run each:

| Ladder tag | Pretraining tokens | HF revision |
|---|---:|---|
| `pt1007B` | 1007B | `stage1-step480000-tokens1007B` |
| `pt2937B` | 2937B | `stage1-step1400000-tokens2937B` |
| `pt4001B` | 4001B (final) | `stage1-step1907359-tokens4001B` |

`pt2937B` stands in for the originally-intended 2999B point: the shared cache
(below) has no step-1430000 revision, and we chose not to re-download. Stage1
revisions actually present: 0, 1, 21, 42, 63, 126, 252, 504, 1007, 1469, 2098,
2937, 4001 B tokens — a 2014B point is likewise unavailable (nearest 2098B).

Everything downstream is held fixed: batch 256, `rollout.n=4`, 1 epoch = 1633
steps, save/eval every 100 steps. Only `LR=3e-6` and `KL_COEF=3e-3` are set
explicitly.

Prompts: **`template_type="base_simple"`** → `DATA_SOURCE=balanced-simple`, i.e.
train on `data/balanced-simple/train.parquet` and eval on
`balanced-simple` / `balanced5-simple` / `balanced6-simple` (same puzzles and
splits as the canonical parquets — 418,136 / 997 / 4,620 / 3,519 rows — with the
scaffold-free prompt from `scripts/make_prompt_variant_data.py`; see
`2026-08-09_base-simple-prompt-variant.md`). Row counts match the canonical
data, so the default 1-epoch / `TOTAL_STEPS=1635` budget still yields 1633 steps.

The `-simple` suffix is load-bearing and propagates automatically:
`run_pipeline.sh` derives the n=5/n=6 eval sets from it (`_ds_suffix`),
`train_grpo.sh` sets `disable_chat_template=True` for `*-simple*`, and `eval.py`
picks the raw prompt path when `"-simple" in eval_dataset`. Max prompt length is
105 tokens, far inside the 1024 cap.

Naming: `MODEL_NAME=OLMo-2-0425-1B-pt<tokens>B`,
`EXP_NAME=balanced-simple-grpo-lr3e-6-kl3e-3-seed1`. No `-raw-` tag (unlike the
Qwen `balanced-simple-raw-grpo-...` runs, where it disambiguated an archived
chat-wrapped attempt); OLMo has no chat template, so the ambiguity cannot arise.

## Base weights come from a shared read-only cache

The base models are **not** downloaded. They are read out of
`/n/netscratch/dam_lab/Everyone/olmo2_checkpoints/hub` (a standard HF cache
owned by `brachit`, group `dam_lab`, mode `r-x` for us), via one symlink per
ladder point in `$MODEL_DIR`:

    $MODEL_DIR/OLMo-2-0425-1B-pt1007B -> <shared hub>/snapshots/eda09f03...
    $MODEL_DIR/OLMo-2-0425-1B-pt2937B -> <shared hub>/snapshots/c4e2f33a...
    $MODEL_DIR/OLMo-2-0425-1B-pt4001B -> <shared hub>/snapshots/9d3e4365...

`train_grpo.sh` and `plot_results.py` both resolve a base model as
`$MODEL_DIR/$MODEL_NAME`, so the symlinks make the shared snapshots work with no
special-casing. Symlinking the *snapshot dir* is safe: its per-file symlinks are
relative to the real directory, so they resolve through the link.

Verified upstream-identical: ref commit SHAs match the Hub, and each blob's
filename (which in an HF cache *is* its content sha256) matches the Hub's LFS
hash.

**Gotcha:** the cache is not writable by us, so an *online* `from_pretrained`
against it dies with `PermissionError at .../hub/.locks`. Reading via the
snapshot path (what the symlinks do) or with `local_files_only=True` is fine.
Two caveats on the shared snapshots: they are fp32 on disk (~5.9 GB each), and
the 1007B snapshot lacks `merges.txt` / `vocab.json` (harmless — `tokenizer.json`
is present, so the fast tokenizer loads).

## Compatibility checks (all pass, verl env)

- vLLM 0.8.5.post1 registers `Olmo2ForCausalLM` — rollout engine works.
- `Olmo2ForCausalLM._no_split_modules == ['Olmo2DecoderLayer']` — FSDP wrap
  policy resolves (local verl patch 2 not even needed here).
- `_supports_flash_attn_2 == True`, and `apply_monkey_patch` falls through to the
  generic `transformers.integrations.flash_attention` patch for non-VLM
  decoder-only models, so `use_remove_padding=True` is fine.
- **No** persistent float buffers in vLLM's `olmo2.py`, so the gemma3
  `normalizer`-clobbering bug under `load_format=dummy_dtensor` (local verl
  patch 4) does **not** apply.
- `max_position_embeddings=4096` ≥ prompt 1024 + response 1024.

## Prompt rendering is identical across the three arms

The comparison here is *within* the OLMo ladder, and the arms are controlled at
the byte level. Verified:

- Every non-weight file is the **same blob** in the shared cache for all three
  snapshots — `config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `generation_config.json`. The only difference
  between the arms is the two safetensors shards, i.e. the weights.
- Over 200 real `data/balanced-simple/train.parquet` rows through
  `render_prompt(..., disable_chat_template=True)`, all three arms give the same
  rendered-text SHA (`48cf676c…`) and the same token-ID SHA (`7abde602…`).

Separately (and only relevant if these are ever compared against the *Qwen*
base-simple runs): OLMo-2 base ships no `chat_template`, so the raw and
chat-template code paths coincide for it — `render_prompt` returns the identical
string either way. `disable_chat_template=True` is therefore a no-op here rather
than a behavior change, but it is set explicitly so the intent is recorded and so
the setting matches the Qwen `*-simple` runs.

## How to run

    bash scripts/run_olmo2_pretrain_ladder.sh --dry-run   # inspect submissions
    bash scripts/run_olmo2_pretrain_ladder.sh             # all three points
    bash scripts/run_olmo2_pretrain_ladder.sh pt1007B     # one point

Each point submits the standard `run_pipeline.sh` chain: train → 3 eval arrays
(`balanced`, `balanced5`, `balanced6`) → plot. Seed rides on the sbatch array
(`--array=$SEED-$SEED`), because `train_grpo.sh` derives `+data.seed` from
`SLURM_ARRAY_TASK_ID`.

## Results

(pending)

## First launch failed at step 1 — poisoned `~/.triton/cache` (resolved)

Both first-attempt runs died after ~9-11 min, `FAILED` exit 1:

| Job | Arm | Failure |
|---|---|---|
| 41825059 | `pt4001B` | `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats` |
| 41825338 | `pt2937B` | same |

Config in both logs was correct (lr 3e-6, kl 3e-3, `balanced-simple`, chat template
disabled, right model path). vLLM rollout and `val_before_train` **succeeded** — the
logs contain OLMo's pre-RL generations. The crash was at step 1's actor log-prob:

    ray_trainer.py:1199   old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
      dp_actor.py:198     log_probs = logprobs_from_logits(...)
      torch_functional.py:86 -> logprobs_from_logits_flash_attn
        flash_attn/ops/triton/cross_entropy.py:196  cross_entropy_fwd_kernel[...]
          triton/compiler/compiler.py:390 _init_handles
            driver.active.utils.load_binary(...)   -> SystemError

Root cause: `~/.triton/cache` is shared across CPython versions. It held an untagged
`cuda_utils.so` from Aug 2025 and a `cuda_utils.cpython-313-*.so`, while the verl env
is Python 3.10.20 / triton 3.2.0. The stale extension *imports* fine and fails when
`load_binary` parses its arguments — which is why the probe shows
`PASS driver.active.utils loaded` next to `FAIL` on the kernel launch.

Not model-specific and not inductor-specific: `logprobs_from_logits` is on the hot path
for every model, so this was luck-of-the-cache, not an OLMo incompatibility. It is the
same underlying breakage as the gemma-eval note in CLAUDE.md, but that note's
`TORCHDYNAMO_DISABLE` / `--enforce_eager` workaround does not cover this path.

Probe (job 41839649, one H100, same node, same interpreter, only `TRITON_CACHE_DIR`
differing):

    A) ~/.triton/cache      FAIL  flash-attn triton cross_entropy: SystemError: PY_SSIZE_T_CLEAN...
                            FAIL  verl logprobs_from_logits: SystemError: PY_SSIZE_T_CLEAN...
    B) /tmp/triton_<jobid>  PASS  flash-attn triton cross_entropy: loss[0]=12.7431
                            PASS  verl logprobs_from_logits: shape=(2, 32)

Fix: every GPU job script now exports `TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-$$}"`
(node-local, per-job — same pattern as `WANDB_DIR` / `HF_DATASETS_CACHE`). 10 scripts
patched; see CLAUDE.md "Known issues" for the full list.

No checkpoints were written (died at step 1), and the dependent eval/plot jobs were
auto-cancelled on `DependencyNeverSatisfied`, so there is nothing to clean up before
relaunching.

## Ground-truth SFT warmup before GRPO (added 2026-08-25)

The raw OLMo-2 base checkpoints look too weak for RL to get traction: GRPO needs
some probability mass on well-formed `<answer>` outputs before the reward can do
anything. So each ladder point now gets a **short SFT pass on the canonical
solutions** first.

New pieces:

- `scripts/generate_gt_sft_data.py` — builds the GT SFT parquet with **no GPU and
  no teacher model**: the canonical solution is already in the data. Reuses
  `build_gt_equation` from `scripts/diagnose_gt_ntp.py` (substitute
  `extra_info.numbers` in order into `extra_info.canonical_pattern`) and gates the
  whole file on `grader_utils.compute_score` — any row not scoring 1.0 aborts the
  build. Measured: 3000/3000 exact (n=3 498/498, n=4 2502/2502).
- `scripts/run_olmo2_gt_sft_ladder.sh` — per arm: SFT → eval the SFT checkpoint;
  `--with-grpo` additionally chains GRPO (lr 3e-6, kl 3e-3) on the merged SFT dir.

**Target format: answer-only.** GT data has no reasoning trace and none is
invented, so the response is just the answer tag. GRPO is left free to discover
reasoning:

    prompt   : ...Return the final expression inside <answer>...</answer>. ...
               Solution:
    response :  <answer> 14 + 27 + 9 </answer><|endoftext|>

`generate_gt_sft_data.py --format think_empty` is the alternative (keeps a vacuous
`<think></think>` scaffold) if answer-only turns out to suppress response length
under RL. The generator also handles the canonical `template_type='base'` prompts,
which end *inside* an open `<think>` — there the response starts with `</think>`
rather than opening a second one.

Mechanics verified against verl's `SFTDataset` contract: with no chat template the
prompt string is used verbatim and `tokenizer.eos_token` is appended to the
response, joined with **no separator** — so the response carries its own leading
space. Token budget is tiny: prompt 105, response 15, total max 120 vs
`data.max_length=2048`.

Settings: `SFT_STEPS=200` (batch 64 → ~12.8k examples, ~3% of the 418k-row set —
a warmup, not a distillation), `SFT_LR=1e-5`, seed 1, all three arms.
`SFT_LR` is a guess carried over from the Qwen students; if 200 steps barely moves
`mean@1`, raising it is the first knob to try.

Caveat when reading job status: SFT-checkpoint evals exit non-zero on `eval.sh`'s
trailing `chmod` of the nonexistent `checkpoints/<...>` path (SFT lives under
`sft-checkpoints/`), so sacct shows FAILED while the JSON is valid — verify by
file existence. Nothing is chained `afterok` on those eval jobs for this reason;
the GRPO leg depends on the SFT job itself.

## Results: GT-SFT(3e-5, 200 steps) -> GRPO, 1007B vs 4001B (2026-08-26)

Both GRPO runs finished at step 1600, 16 eval checkpoints each × 3 datasets:
- `pt1007B` — wandb `s38ct7j2`
- `pt4001B` — wandb `jmqm4apz`

`pt2937B` has **no** GRPO run — only these two arms exist, so the middle of the
ladder is missing and "monotone in pretraining tokens" cannot be checked yet.

Plots: see the pass@k section below. The mean@1/mean@32 figures originally made
here were **deleted** — mean@32 turned out to be `0.1 + 0.9 * mean@1` (see the
mean@32 decomposition section), so it carried no information beyond mean@1 while
inflating the apparent numbers. `scripts/plot/plot_olmo2_ladder_pareto.py` is now
pass@k-only and reads the `.passk` caches.

| | ID n=3,4 m@1 | m@32 | n=5 m@1 | m@32 | n=6 m@1 | m@32 |
|---|---|---|---|---|---|---|
| 1007B SFT (pre-GRPO) | 0.2387 | 0.3270 | 0.0426 | 0.1389 | 0.0088 | 0.1066 |
| 1007B GRPO step 1600 | **0.4774** | **0.5305** | 0.0952 | 0.1838 | 0.0128 | 0.1112 |
| 4001B SFT (pre-GRPO) | 0.2979 | 0.3563 | 0.0606 | 0.1499 | 0.0099 | 0.1087 |
| 4001B GRPO step 1600 | 0.4664 | 0.5175 | **0.1102** | **0.1994** | **0.0176** | **0.1150** |

Findings:

1. **GRPO roughly doubles ID mean@1** on both arms (0.24→0.48, 0.30→0.47) and
   more than doubles OOD n=5 mean@1. The GT-SFT warmup did its job: there was
   enough well-formed output for the reward to act on.
2. **More pretraining buys OOD, not ID.** At step 1600 the *less*-pretrained arm
   is marginally ahead in-distribution (0.4774 vs 0.4664) while 4001B leads on
   both OOD sets (n=5 0.1102 vs 0.0952; n=6 0.0176 vs 0.0128). In the ID-vs-OOD
   plane the 4001B trajectory sits **above** 1007B's for the whole run, i.e. at
   matched ID accuracy it generalizes further. This is the interesting result and
   it is consistent across mean@1 and mean@32.
3. **The ID advantage of 4001B at SFT time is spent by GRPO.** 4001B starts ahead
   on ID (0.298 vs 0.239) and ends slightly behind — RL closes the ID gap while
   the OOD gap persists.
4. `mean1_vs_mean32` is tightly linear and well above y=x for every checkpoint of
   both arms, and the two arms lie on essentially the **same** line — the
   mean@32/mean@1 relationship looks like a property of the task+decoder, not of
   the pretraining budget. Arms differ only in how far along that line they get.
5. n=6 stays near the floor (m@1 ≤ 0.018) with m@32 ≈ 0.11, i.e. mostly format
   credit (0.1). Do not read n=6 mean@32 as partial success.

Caveats: single seed per arm, so the ID ordering at step 1600 (0.4774 vs 0.4664,
a 0.011 gap) is well inside plausible seed noise — do not claim less pretraining
helps ID. The OOD gap is larger and holds across all 16 checkpoints, which is
much stronger evidence. `pt2937B` GRPO is still needed for a real ladder.

### Correction to the earlier SFT LR table

The SFT-LR table logged during the LR sweep was produced by an ad-hoc scoring
script with two defects: it capped OOD at 1000 prompts (the repo's
`plot_results._score_checkpoint` uses `MAX_PROMPTS=None`, i.e. all 4620/3519) and
it got some mean@1 cells wrong (e.g. pt4001B/3e-5 ID 0.2849 vs the correct
0.2979). Recomputed with the repo scorer, and now cached as `.scores`:

| arm | sft_lr | ID m@1 | m@32 | n=5 m@1 | m@32 | n=6 m@1 | m@32 |
|---|---|---|---|---|---|---|---|
| pt1007B | 1e-5 | 0.2076 | 0.2944 | 0.0459 | 0.1351 | 0.0077 | 0.1066 |
| pt1007B | 3e-5 | 0.2387 | 0.3270 | 0.0426 | 0.1389 | 0.0088 | 0.1066 |
| pt1007B | 1e-4 | 0.2758 | 0.3462 | 0.0390 | 0.1377 | 0.0051 | 0.1055 |
| pt2937B | 1e-5 | 0.2297 | 0.3063 | 0.0348 | 0.1336 | 0.0054 | 0.1038 |
| pt2937B | 3e-5 | 0.2808 | 0.3394 | 0.0416 | 0.1408 | 0.0077 | 0.1069 |
| pt2937B | 1e-4 | 0.2287 | 0.3245 | 0.0292 | 0.1263 | 0.0074 | 0.1049 |
| pt4001B | 1e-5 | 0.2578 | 0.3288 | 0.0368 | 0.1382 | 0.0063 | 0.1048 |
| pt4001B | 3e-5 | 0.2979 | 0.3563 | 0.0606 | 0.1499 | 0.0099 | 0.1087 |
| pt4001B | 1e-4 | 0.2497 | 0.3208 | 0.0095 | 0.1089 | 0.0011 | 0.1007 |

The conclusion is unchanged: **3e-5** is best on ID for pt2937B/pt4001B and best
on OOD n=5 for both, and `1e-4` is actively harmful OOD at the top of the ladder
(pt4001B n=5 m@1 0.0095). Only pt1007B prefers 1e-4 on ID, and it is worst OOD
there — so 3e-5 remains the right matched choice for the ladder.

## pass@32 vs mean@32: GRPO collapses coverage (2026-08-26)

`mean@32` and `pass@32` are **not** the same metric, and on these runs they tell
opposite stories. Definitions (repo's own, `plot_results._score_item`):

- `mean@32` — mean raw `compute_score()` over 32 samples, scores ∈ {0, 0.1, 1.0}.
  A well-formed wrong answer earns 0.1, so this blends format credit with
  correctness. **Not a pass rate.**
- `pass@32` — fraction of prompts with **≥1** exactly-correct sample out of 32.
  Pure coverage. New: `scripts/compute_passk.py` writes `.passk` caches
  (`[step, pass@1, pass@32, n_prompts]`); plot via
  `plot_olmo2_ladder_pareto.py`. Figures in `figures/olmo2-ladder/<set>/`
  (one folder per comparison set, see that directory's README; these live in
  `1b-grpo-vs-sftgrpo/`): `id_vs_ood_pass1_<tag>.png`,
  `id_vs_ood_pass32_<tag>.png`, `pass1_vs_pass32_<tag>.png`, each with a `-zoom` variant (omit `--sft-exp`) that
  drops the SFT origin so the GRPO trajectories are legible — with the origin in
  frame the pass@32 panels are mostly empty space, since SFT sits at ID 0.77-0.80
  and every GRPO checkpoint at 0.45-0.51.

  Marker key (per run colour): **circled dot** = first RL checkpoint (step 100),
  **star** = final RL checkpoint (step 1600), **plain dot** = SFT, pre-GRPO. Small
  dots are intermediate checkpoints; step numbers are annotated at both ends. Each
  axis is scaled to its own data range (x and y ranges are independent), so `y = x`
  in the pass@1-vs-pass@32 panels appears only where the ranges overlap.

`pass@1 == mean@1` by construction (verified: both 0.2387 for pt1007B SFT).

| | ID pass@1 | ID pass@32 | n=5 pass@32 | n=6 pass@32 |
|---|---|---|---|---|
| 1007B SFT (pre-GRPO) | 0.2387 | **0.7733** | 0.2682 | 0.0719 |
| 1007B GRPO s1600 | 0.4774 | **0.5055** | 0.1165 | 0.0196 |
| 4001B SFT (pre-GRPO) | 0.2979 | **0.7950** | 0.2820 | 0.0750 |
| 4001B GRPO s1600 | 0.4664 | **0.4945** | 0.1303 | 0.0230 |

**GRPO trades coverage for reliability, and the trade is severe.** ID pass@32
falls 0.77 → 0.51 (1007B) and 0.80 → 0.49 (4001B) while pass@1 roughly doubles.
OOD is worse in relative terms: n=5 pass@32 drops ~2.2x (0.27→0.12), n=6 ~3.3x
(0.072→0.020). After GRPO the trajectories sit almost **on** y=x — 32 samples find
barely anything the first sample missed (ID 0.4774 → 0.5055) — i.e. near-total
diversity collapse. This is invisible in `mean@32`, which *rose* over training
because the 0.1 format term dominates it.

Corrections to the earlier entry in this log:

- "GRPO improved things on both metrics" holds for mean@1/mean@32 only. In
  coverage terms GRPO is a large regression, and the pre-GRPO SFT checkpoint is
  the better model if you can afford 32 samples and a verifier.
- The claim that `mean1_vs_mean32` being "tightly linear ... a property of the
  task+decoder" is an artifact: mean@32 is dominated by the 0.1 format credit, so
  it tracks format rate, not sampling behavior. The pass@k plot is the one to read.

What survives unchanged: **more pretraining still buys OOD, not ID.** 4001B leads
on OOD in coverage too (n=5 pass@32 0.1303 vs 0.1165; n=6 0.0230 vs 0.0196) while
trailing slightly on ID — same direction as the mean@k plots.

Open question for the ladder: the entropy collapse is the obvious knob to attack
(higher `ENTROPY_COEFF`, larger `KL_COEF`, or fewer GRPO steps). Whether more
pretraining *protects* against coverage collapse is worth a plot of
pass@32-vs-step per arm; both arms lose ~1.6x on ID, so on this evidence it does
not.

### mean@32 decomposition: why it looked so much better than mean@1

`compute_score` returns 1.0 / 0.1 / 0.0, so
`mean@32 = p_correct + 0.1 * p_wellformed_wrong` while `mean@1 = p_correct`. After
the GT-SFT warmup essentially every sample is well-formed
(`p_correct + p_wellformed_wrong ~= 1`, measured 0.98-1.01 on every dataset), which
collapses the metric to

    mean@32 ~= 0.1 + 0.9 * mean@1

Verified over all 102 checkpoint x dataset points: mean residual -0.0002,
max |residual| 0.0121. So mean@32 was a monotone affine rescaling of mean@1 with a
0.1 floor, not extra successes. Worst case n=6: mean@1 0.0128 vs mean@32 0.1112,
where 0.098 of the 0.111 (~88%) is format credit for well-formed wrong answers.

This retires two earlier readings in this log: the "mean@1 vs mean@32 is tightly
linear and shared across arms" observation (the line is just y = 0.1 + 0.9x, so it
had to be), and any use of mean@32 as an accuracy number. Use **pass@1** for
accuracy and **pass@32** for coverage. mean@32 remains useful only as a format
check: `(mean@32 - mean@1)/0.1 + mean@1 ~= 1` means nothing is unparseable.

## Adding the no-SFT arm: GRPO vs SFT+GRPO (2026-08-27)

Third run finished — wandb `0rbpje5m` =
`OLMo-2-0425-1B-pt1007B-balanced-simple-grpo-lr3e-6-kl3e-3-seed1`, i.e. **pt1007B
with no SFT warmup** (from `run_olmo2_pretrain_ladder.sh`). Same lr/kl/data as the
SFT+GRPO arms, so it isolates the warmup. Figures retagged `ladder`
(+ `ladder-zoom`); labels are `1007B GRPO`, `1007B SFT+GRPO`, `4001B SFT+GRPO`.

| run | ID pass@1 | ID pass@32 | n=5 pass@1 | n=5 pass@32 | n=6 pass@1 | n=6 pass@32 |
|---|---|---|---|---|---|---|
| 1007B GRPO (no SFT) | 0.1976 | 0.2056 | — | — | 0.0091 | 0.0099 |
| 1007B SFT+GRPO | **0.4774** | **0.5055** | 0.0952 | 0.1165 | 0.0128 | 0.0196 |
| 4001B SFT+GRPO | 0.4664 | 0.4945 | **0.1102** | **0.1303** | **0.0176** | **0.0230** |

**The GT-SFT warmup is worth ~2.4x final ID pass@1** (0.198 -> 0.477) at matched
GRPO budget and hyperparameters, and the no-SFT arm is worse on n=6 as well
(0.0091 vs 0.0128). Starting from ~0 pass@1, plain GRPO does climb — the reward is
not dead without SFT — but it plateaus far lower within 1600 steps. The original
hunch that the raw OLMo bases are "too weak to be RLed" is confirmed in the
practical sense: RL alone gets a fraction of the way.

Response length is the mechanism and the operational gotcha: at step 1600 the
no-SFT arm averages **2383 chars** per response (max 11351) versus **32.5 chars**
for SFT+GRPO — a 73x difference. GT-SFT installed a terse answer-only format; plain
GRPO rambles to near the length cap.

### Missing n=5 evals for both no-SFT arms

`balanced5-simple` is absent for the no-SFT runs: eval array `41841449` had **14 of
16 tasks TIMEOUT at 1:30:00** (`eval.sh`'s default walltime); only tasks 3 and 6
finished, which is why `pt4001B` no-SFT has just steps 300/600 and `pt1007B` has
none. Direct consequence of the 73x response length above: 4620 prompts x 32
samples of ~2.4k chars does not fit in 1:30. n=3,4 (997 prompts) and n=6 (3519,
capped at 1024 tokens) did fit.

Re-run with a longer walltime, e.g.:

    CHECKPOINT_DIR=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints \
    MODEL_NAME=OLMo-2-0425-1B-pt1007B \
    EXP_NAME=balanced-simple-grpo-lr3e-6-kl3e-3-seed1 \
    EVAL_DATASET=balanced5-simple \
      sbatch --array=1-16 -t 08:00:00 --partition=kempner_h100 \
             --account=kempner_bingbin_lab scripts/eval.sh
    # pt4001B: steps 300/600 already exist -> --array=1,2,4,5,7-16

Until then the n=5 panels carry an explicit "no data: 1007B GRPO" annotation
(`plot_olmo2_ladder_pareto.py` warns at load and labels the panel) so an absent
curve is never mistaken for an overlapping one.

Also note: `pt4001B` **no-SFT** GRPO exists too (ID + n=6 complete, 16 steps each)
but is not in the current figures — adding it would complete a 2x2 of
{1007B, 4001B} x {GRPO, SFT+GRPO}.

Script change: `--sft-exp` (one value for all runs) was replaced by repeatable
`--origin LABEL=SFT_EXP`. The old flag would have attached the SFT origin dot to
the no-SFT run, since it shares the pt1007B model directory.

## Full 2x2: {1007B, 4001B} x {GRPO, SFT+GRPO} (2026-08-27)

Added the `pt4001B` no-SFT arm. Figures regenerated via the new wrapper
`scripts/run_olmo2_ladder_plots.sh` (tags `ladder`, `ladder-zoom`).

Final checkpoint, pass@1 / pass@32:

| run | ID n=3,4 | n=5 | n=6 |
|---|---|---|---|
| 1007B GRPO | 0.1976 / 0.2056 | — | 0.0091 / 0.0099 |
| 4001B GRPO | 0.3139 / 0.3250 | 0.0587 / 0.0708 *(s600)* | 0.0205 / 0.0247 |
| 1007B SFT+GRPO | **0.4774 / 0.5055** | 0.0952 / 0.1165 | 0.0128 / 0.0196 |
| 4001B SFT+GRPO | 0.4664 / 0.4945 | **0.1102 / 0.1303** | 0.0176 / 0.0230 |

The 2x2 changes the reading of the pretraining effect:

1. **Without SFT, pretraining matters a lot on ID**: 0.1976 -> 0.3139 pass@1
   (+59%). **With SFT it does not**: 0.4774 vs 0.4664, i.e. the 1007B arm is
   marginally ahead. The earlier "more pretraining buys OOD, not ID" conclusion was
   drawn from the SFT+GRPO pair alone and is really a statement about that pair —
   the ID benefit of pretraining is present, but the GT-SFT warmup **substitutes**
   for it.
2. **The warmup helps the weaker base more**: +0.280 ID pass@1 for 1007B
   (0.198 -> 0.477) vs +0.153 for 4001B (0.314 -> 0.466). A clear interaction: GT
   supervision compensates for missing pretraining.
3. **On n=6 the warmup is not uniformly good.** For 4001B, no-SFT GRPO *beats*
   SFT+GRPO (pass@1 0.0205 vs 0.0176; pass@32 0.0247 vs 0.0230), while for 1007B it
   is the reverse (0.0091 vs 0.0128). So for a well-pretrained base, the answer-only
   SFT format may cost the hardest OOD split — consistent with the terse format
   suppressing the longer derivations n=6 needs (no-SFT responses average 2383
   chars vs 32.5).
4. OOD ordering by pretraining survives within each training recipe: 4001B > 1007B
   on n=6 in both the GRPO pair (0.0205 vs 0.0091) and the SFT+GRPO pair
   (0.0176 vs 0.0128).

Caveats: single seed everywhere; n=5 is missing entirely for 1007B GRPO and has
only steps 300/600 for 4001B GRPO, so **no n=5 claim spans the no-SFT arms**. The
`4001B GRPO` n=5 cell above is step 600, not 1600, and is marked `@s600` in the
table and `partial: 4001B GRPO (2/16 ckpts)` on the figure panel. Fill the gap with
the two `-t 08:00:00` eval resubmits recorded in the previous section before
drawing any n=5 conclusion about the warmup.

Plot-honesty changes made for this: the star legend now reads "last available RL
checkpoint" (it is s600, not s1600, for the partial arm), `_coverage_note()`
annotates each panel with absent and partial runs, and the console table appends
`@s<step>` whenever a cell is not from the run's final step.

## Second-epoch continuation of pt4001B no-SFT GRPO (2026-08-31)

**Job 43440872** — `OLMo-2-0425-1B-pt4001B`,
`EXP_NAME=balanced-simple-grpo-lr3e-6-kl3e-3-cont1600-seed1`. Extends the no-SFT
arm (`balanced-simple-grpo-lr3e-6-kl3e-3-seed1`, job 41841443) by a second pass
over `balanced-simple`, same lr=3e-6 / kl=3e-3 / rollout.n=4 / batch 256.

### Why a warm-start and not `TOTAL_EPOCHS=2`

A native verl resume is **impossible on these runs**: `eval.sh` merges the FSDP
shards and then deletes `actor/`, `critic/`, `data.pt`, so all 16 checkpoints of
both pt4001B arms (and both pt1007B arms) hold merged HF weights only — no
optimizer state, no dataloader state. `trainer.resume_mode=resume_path` has
nothing to load. The only way to reuse the epoch-1 compute is to warm-start:

    MODEL_PATH=<...>/balanced-simple-grpo-lr3e-6-kl3e-3-seed1/global_step_1600
    TOTAL_EPOCHS=1 TOTAL_STEPS=1635    # a second 1633-step pass

Cost decided it. Epoch 1 took **21h01m** (46.3 s/step all-in over 1633 steps),
so `TOTAL_EPOCHS=2` from scratch is ~42h against `kempner_h100`'s 48h cap — only
~13% margin. The warm-start is ~21h and fits comfortably.

**How this differs from a true 2-epoch run** (all three are real, none is fatal):

1. *KL reference moves.* verl builds the ref policy from
   `actor_rollout_ref.model.path`, so KL is now anchored to the step-1600 policy
   rather than the original base. Measured to be a **small** perturbation:
   `actor/kl_loss` sat at ~1.0e-3 per token, **flat across the entire run**
   (mean 9.97e-04, final 1.01e-03, `kl_coef=3e-3`) — the step-1600 policy is
   still very close to the base in KL.
2. *AdamW moments reset* to zero (verl's LR schedule is `warmup_style: constant`,
   `lr_warmup_steps_ratio: 0.0`, so the **LR schedule restart is a no-op** —
   a flat 3e-6 throughout either way).
3. *Data order replays epoch 1.* `+data.seed=$SLURM_ARRAY_TASK_ID`=1 and a fresh
   `RandomSampler`, so the second pass sees the identical batch sequence rather
   than a fresh epoch-2 shuffle.
4. Steps 1601–1633 of epoch 1 are skipped (last save is at 1600).

Do not describe this in a writeup as a single 2-epoch run; it is
"GRPO round 2 with a re-anchored reference."

### Evidence that a second epoch is worth running

ID (n=3,4) was **still improving at step 1600** — the final checkpoint is the
run's best and a fresh +0.020 jump:

| step | 1000 | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 |
|---|---|---|---|---|---|---|---|
| ID pass@1 | 0.2808 | 0.2678 | 0.2668 | 0.2879 | 0.2849 | 0.2939 | **0.3139** |

Slope +0.0061 pass@1 / 100 steps over 1000–1600 (+0.0075 over 600–1600), no
flattening. Training-side `val-core/reward/mean@4` agrees and is denser:
0.3649 @1550 → **0.3816 @1600**, its maximum.

**OOD n=6 is flat, though** — slope −0.00014 / 100 steps over 600–1600, best at
step 1300 (0.0207) vs final 0.0205. Expect the second epoch to buy ID and not
n=6.

Not a degenerate climb: entropy is *rising* slightly (3.233 → 3.460, +0.0024 /
100 steps) and response length is flat (drift −1.2 tok/100 steps, range
486–625, clip_ratio flat ~0.33–0.47). The flat response length is also why the
~21h/epoch projection carries over.

> Log-reading gotcha: `perf/time_per_step` in the log averages **161.7 s**, but
> `log_freq` and `test_freq` are both 50, so every logged step is a validation
> step carrying ~115 s of `timing_s/testing`. The real all-in rate is 46.3 s/step
> (21h01m / 1633 from `sacct`). Do not size walltime off the logged value.

### Submission

    MODEL_NAME=OLMo-2-0425-1B-pt4001B \
    EXP_NAME=balanced-simple-grpo-lr3e-6-kl3e-3-cont1600-seed1 \
    MODEL_PATH=$CK/checkpoints/OLMo-2-0425-1B-pt4001B/balanced-simple-grpo-lr3e-6-kl3e-3-seed1/global_step_1600 \
    DATA_SOURCE=balanced-simple LR=3e-6 KL_COEF=3e-3 \
    TOTAL_EPOCHS=1 TOTAL_STEPS=1635 \
    TRAIN_SBATCH_ARGS="-t 30:00:00" EVAL_SBATCH_ARGS="-t 08:00:00" \
    bash scripts/run_pipeline.sh

Jobs: train `43440872`, eval `43440903` (n=3,4) / `43440923` (n=5) /
`43440926` (n=6), plot `43440938`. `EVAL_SBATCH_ARGS="-t 08:00:00"` is
load-bearing — the no-SFT arm's ~2.4k-char responses TIMEOUT `eval.sh`'s
1:30 default on n=5/n=6 (this is what killed eval array 41841449).

`EVAL_ARRAY` is left at eval.sh's default 1-16 = steps 100..1600, correct here
because the continuation is itself a 1633-step run.

### n=5 backfill of the epoch-1 baseline — submitted 2026-09-01

**Job 43690606**, `--array=1,2,4,5,7-16` (14 tasks, the missing steps),
`-t 08:00:00`, `kempner_h100`. Fills `balanced5-simple` for
`balanced-simple-grpo-lr3e-6-kl3e-3-seed1` so the continuation has an epoch-1
n=5 curve to compare against.

Safe to run against already-merged checkpoints: `eval.sh:127` short-circuits the
`verl.model_merger` step when `model.safetensors` (or `.index.json`) is already
present, so the stripped-and-merged checkpoints are loaded directly.

#### Backfill completed — all 14 tasks COMPLETED, 1:37-1:54 each

`balanced5-simple` for the epoch-1 baseline is now **16/16**. The original
1:30 limit was short by only ~10-25 min, which is why the two shortest tasks
(steps 300, 600 at 1:21 / 1:29) survived and the other 14 did not.

`.passk` caches were **not** produced by `eval.sh` — only the raw JSONs — so
`scripts/compute_passk.py --model OLMo-2-0425-1B-pt4001B --exp
balanced-simple-grpo-lr3e-6-kl3e-3-seed1` was run afterward to fill them.
Without that step the backfill looks complete on disk while
`plot_olmo2_ladder_pareto.py` still renders a 2-point n=5 curve.

#### New finding: n=5 was ALSO still climbing at the end of epoch 1

The full curve (previously only steps 300/600 were known):

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|
| n=5 pass@1 | 0.0002 | 0.0242 | 0.0364 | 0.0368 | 0.0379 | 0.0587 | 0.0639 | 0.0660 |

| step | 900 | 1000 | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 |
|---|---|---|---|---|---|---|---|---|
| n=5 pass@1 | 0.0708 | 0.0710 | 0.0567 | 0.0684 | 0.0712 | 0.0712 | 0.0710 | **0.0846** |

Slope +0.00258 pass@1 / 100 steps over 1000-1600, and the **final checkpoint is
the maximum**. So at the end of epoch 1 the three splits differ:

| split | slope 1000-1600 (pass@1 /100 steps) | still improving? |
|---|---|---|
| ID n=3,4 | +0.00605 (0.2808 -> 0.3139) | yes |
| OOD n=5 | +0.00258 (0.0710 -> 0.0846) | **yes** |
| OOD n=6 | +0.00040 (0.0176 -> 0.0205) | no, flat |

The step-1500 -> 1600 jump is more solid on n=5 than on ID, because n=5 has
4.6x the prompts: n=5 +0.0136 = **3.3 SE** (SE 0.0041 on 4620 prompts), ID
+0.0200 = only **1.4 SE** (SE 0.0147 on 997). Read the ID final-step jump as
noise; the n=5 one is marginally real.

This corrects the earlier note in this log that read the no-SFT arm's n=5 as
"partial: steps 300/600 only" and declined to draw any conclusion. It also means
the pre-launch case for a second epoch was stronger than the ID curve alone
suggested — though see the epoch-2 plateau below.

### Open gap (resolved by the above): the epoch-1 baseline had almost no n=5

`balanced5-simple` exists for the epoch-1 run at **steps 300 and 600 only** — in
particular **not at step 1600**, the exact point the continuation branches from.
So there is currently no n=5 anchor for the epoch-1 endpoint and no n=5
comparison across the two epochs is possible. Backfill with:

    CHECKPOINT_DIR=/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints \
    MODEL_NAME=OLMo-2-0425-1B-pt4001B \
    EXP_NAME=balanced-simple-grpo-lr3e-6-kl3e-3-seed1 \
    EVAL_DATASET=balanced5-simple \
      sbatch --array=1,2,4,5,7-16 -t 08:00:00 --partition=kempner_h100 \
             --account=kempner_bingbin_lab scripts/eval.sh

### Warm-start verified (2026-08-31)

`actor_rollout_ref.model.path` and `tokenizer_path` both resolve to
`.../balanced-simple-grpo-lr3e-6-kl3e-3-seed1/global_step_1600` in the job's
config dump, and `val_before_train` confirms the weights actually loaded rather
than silently falling back to the base:

| | val-core reward mean@4 |
|---|---|
| continuation, step 0 (`val_before_train`) | **0.38114** |
| epoch-1 final, step 1600 | 0.38160 |
| raw pt4001B base, for contrast | 0.0208 |

A 0.00046 difference — sampling noise on 997 prompts x 4 samples. The
continuation starts exactly where epoch 1 ended.

### The continuation died silently at step ~1535 (2026-09-01)

Job 43440872 `FAILED` with exit **120:0** after 21:33:27, at step ~1535/1633
(94%). Step 1500 saved cleanly at 20:40; the process stopped ~30 min later with
**no traceback anywhere in the log**. Ruled out:

| Suspect | Evidence against |
|---|---|
| Walltime | 21:33 of a 30:00 limit |
| Cgroup OOM | MaxRSS 39.9 GB of 192 GB requested (epoch 1 peaked at 38.2 GB and completed) |
| Global disk | holylabs had 1.4 P free |
| Node fault | `holygpu8a15204` healthy; no co-located job died at the same time |

**Leading hypothesis: out of disk, node-local.** Exit 120 is CPython's
"error flushing sys.stdout at exit", which is what a full filesystem produces.
The check that was *not* done (and can no longer be done — the state died with
the job) is the compute node's `/tmp`, which holds `WANDB_DIR`, `RAY_TMPDIR`,
`HF_DATASETS_CACHE` and `TRITON_CACHE_DIR` — and on a 4-GPU node hosting two
concurrent 2-GPU jobs that is the most plausible culprit. `lfs quota` does not
work on holylabs (NFS), so a per-lab quota could not be ruled out either.
**Unconfirmed.** If it recurs, capture `df -h /tmp` from inside the job.

Note the exp dir reached **254 GB** while unmerged: 15 checkpoints x ~17 GB
(fp32 model 5.9 GB + AdamW moments 11.8 GB, sharded world_size=2) plus 1535
rollout jsonl dumps. Merging drops each to ~3 GB, reclaiming ~210 GB.

### Recovering it: eval-vs-resume is a one-way door

The dependent evals were auto-cancelled on `DependencyNeverSatisfied`, so they
never ran — which means they never stripped the checkpoints. All 15 (steps
100-1500) retained `actor/` (`model`/`optim`/`extra_state` shards for
world_size=2) plus `data.pt`, so with `resume_mode: auto` and
`latest_checkpointed_iteration.txt=1500` the run was natively resumable — unlike
the epoch-1 run.

**But eval merges `actor/` to `model.safetensors` and then deletes it**, which is
exactly what made epoch 1 unresumable. So evaluating and resuming are mutually
exclusive, and eval is the irreversible direction. Resume must come first if
both are wanted; a resume must also keep the **same world size** (2 GPUs), since
the shards are `world_size_2`.

A resume was submitted (job 43828365) but sat `PENDING` on
`QOSMaxGRESPerUser` behind 4 unrelated running jobs and 254 pending tasks, and
was then **cancelled** in favour of evaluating the 15 existing checkpoints.

### Evals of steps 100-1500 (submitted 2026-09-02)

    43920290  balanced-simple    array 1-15
    43920299  balanced5-simple   array 1-15
    43920302  balanced6-simple   array 1-15

`-t 08:00:00`, `kempner_h100`. After these land, run
`scripts/compute_passk.py --model OLMo-2-0425-1B-pt4001B --exp
balanced-simple-grpo-lr3e-6-kl3e-3-cont1600-seed1` and then
`scripts/run_olmo2_ladder_plots.sh` — the wrapper now carries the run as
**`4001B GRPO ep2`** via `CONT_EXP`. Its step numbers restart at 100, so `s100`
on that trajectory is global step ~1700 of the combined run.

Extending past step 1500 later now requires a fresh `MODEL_PATH` warm-start from
the merged step-1500 checkpoint — a *third* KL re-anchoring. Worth stating
explicitly in any writeup.

### Results: the second epoch bought nothing (2026-09-03)

All 45 eval tasks `COMPLETED` (15 checkpoints x 3 datasets), `.passk` built, and
the run is in the ladder figures as **`4001B GRPO ep2`** (regenerated
`figures/olmo2-ladder/1b-grpo-vs-sftgrpo/{id_vs_ood_pass1,id_vs_ood_pass32,pass1_vs_pass32}_{ladder,ladder-zoom}.png`).

Endpoint comparison — epoch-1 final (s1600) vs epoch-2 final (s1500):

| split | n | ep1 pass@1 | ep2 pass@1 | delta | significance |
|---|---:|---|---|---|---|
| ID n=3,4 | 997 | 0.3139 | 0.3210 | +0.0070 | +0.5 SE |
| OOD n=5 | 4620 | 0.0846 | 0.0814 | -0.0032 | -0.8 SE |
| OOD n=6 | 3519 | 0.0205 | 0.0176 | -0.0028 | -1.3 SE |

**Every change is inside +/-1.3 SE — no measurable movement on any split**, and
the two OOD sets are nominally *down*.

The whole second pass is flat, not just its endpoint. ID pass@1 oscillates
0.307-0.338 across all 15 checkpoints:

| step | 100 | 300 | 500 | 700 | 900 | 1100 | 1300 | 1500 |
|---|---|---|---|---|---|---|---|---|
| ID pass@1 | 0.3270 | 0.3300 | 0.3300 | 0.3380 | 0.3180 | 0.3139 | 0.3270 | 0.3210 |

Slope (ID pass@1 per 100 steps):

| | full pass | second half |
|---|---|---|
| epoch 1 | +0.01620 | +0.00660 |
| epoch 2 | **+0.00043** | **+0.00053** |

i.e. epoch 2's rate is ~12x lower than epoch 1's *second half* and ~38x lower
than epoch 1 overall — indistinguishable from zero. This confirms the
`val-core/reward/mean@4` plateau observed live during training (+0.0009/100
steps over steps 50-1150 vs +0.0053 in epoch 1's tail).

Note that epoch 2's **first** checkpoint (s100, i.e. global step ~1700) is
already 0.3270, above epoch 1's 0.3139 final, and nothing after it improves on
that. Whatever tiny offset exists appeared immediately and is more consistent
with the optimizer-moment reset / KL re-anchoring than with further learning.

**Conclusion: GRPO on the no-SFT pt4001B arm had saturated by the end of epoch 1.**
The pre-launch extrapolation — ID still climbing at +0.0061/100 steps and n=5 at
+0.0026/100, both with the final checkpoint at the run's maximum — did **not**
carry into a second pass. A still-rising tail on the last few checkpoints of a
1633-step run was not sufficient evidence that more data would help; the apparent
slope was largely the final-checkpoint noise spike (the ID s1500->s1600 jump was
only 1.4 SE, as noted above).

Caveats: single seed; epoch 2 stopped at step 1500 rather than 1633 (the run died
at ~1535 and was evaluated rather than resumed), so it is 133 steps shorter than a
full pass — immaterial given the flatness. The KL reference for epoch 2 is the
epoch-1 step-1600 policy, not the base.
