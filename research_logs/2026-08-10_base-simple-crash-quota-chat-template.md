# base-simple run: step-550 crash, quota post-mortem, chat-template finding (2026-08-10)

Follow-up to `research_logs/2026-08-09_base-simple-prompt-variant.md`.

## What happened

Job 37949320 (`balanced-simple-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1`) FAILED at
step ~550/3266 after 3h55m. Log ends mid-print, no traceback (I/O kill). Cause:
**holylabs lab quota hit 100%** ~22:45-22:52 on 08-09. Contributors: the run's own
FSDP checkpoints (5.6G per unevaluated save, every 50 steps ≈ 11G/h) + a 30G
HF-cache rsync into the same quota (mistake — cache move destination should never
share a quota with active training). Deleting the 30G restored writability,
i.e. the lab quota lives within ~30G of full in steady state.

## Findings

1. **All Qwen runs in this repo are chat-template-wrapped.** Qwen2.5 *base*
   tokenizers ship a `chat_template`, so verl's `rl_dataset.py` and `eval.py`
   (via `prompt_utils.py`) wrap every prompt as
   `system\nYou are a helpful assistant.\nuser\n<parquet prompt>`.
   Verified identical in old runs' rollout dumps (seed2) — consistent across
   train/eval and across experiments, so comparisons remain valid, but no Qwen
   run has ever seen a truly raw prompt. Relevant when interpreting
   "base-simple": the minimal prompt still sits inside a chat frame.
   (gemma-3-270m has no chat_template → truly raw; cross-family prompt-format
   comparisons are therefore confounded.)
2. **base-simple learns, but slower than the R1-style prompt** (rollout dumps,
   2048 rollouts/step, correct-rate): step150 0.156 vs 0.174 (old seed2),
   step550 0.264 vs 0.383. Generations show heavy pretraining noise (fake
   tracebacks, chat parroting) absent under the dialogue+`<think>` scaffold.
3. Per-step rollout JSONL dumps (`trainer.rollout_data_dir`) land in the
   checkpoint dir (`{1..550}.jsonl`) — useful for mid-run score forensics:
   `[json.loads(l)["score"] for l in open(f"{ckpt}/{step}.jsonl")]`.

## Remediation / current state

- Run copied (63G) to netscratch and resumed from step 550 with
  `CHECKPOINT_DIR=/n/netscratch/kempner_bingbin_lab/Lab/bingbin/rl-checkpoints`
  (train job **37996462**, eval 37996472-4, plot 37996475). verl
  `resume_mode=auto` picks up `latest_checkpointed_iteration.txt`.
  90-day purge applies — archive keepers to holylabs/HF after eval.
- `~/.cache/huggingface` (35G) moving to
  `/n/netscratch/kempner_bingbin_lab/Lab/bingbin/hf-cache` with a symlink left
  at the original path (home was 100% full; holylabs could not take it).
- Original holylabs run dir left in place; delete after the netscratch resume
  is confirmed healthy to free ~63G of lab quota.
- Longer term: holylabs holds ~300G/run of unmerged FSDP shards for finished
  runs (unevaluated steps keep `actor/` shards; merged HF is 1.2G). A
  merge-and-trim pass would recover most of the lab quota.

## Update (later 2026-08-10): raw-prompt restart

Per bingbin: `base-simple` should mean scaffold-free. Implemented
`disable_chat_template` end-to-end:
- verl `rl_dataset.py` (local patch 3 extension): `+data.disable_chat_template=True`
  blanks `tokenizer.chat_template` at dataset init → all template checks fall
  through to raw concatenation (incl. the overlong-prompt filter).
- `train_grpo.sh`: auto-True for `DATA_SOURCE` matching `*-simple*`
  (override: `DISABLE_CHAT_TEMPLATE`); logs "Chat template: disabled=...".
- `eval.py --prompt_format {auto,chat,raw}` (default auto = raw for `*-simple`
  datasets) via `prompt_utils.render_prompt(..., disable_chat_template=)`.
- `patches/local-fixes-vs-083da9ab.patch` regenerated.

Chat-wrapped run (jobs 37996462 + evals) CANCELLED at ~step 700; its
checkpoints remain under `balanced-simple-grpo-...` (netscratch + holylabs
copies), no held-out evals. Raw restart from step 0:
`balanced-simple-raw-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed1`, train job
**38150829** (evals 38150830-2 with EVAL_ARRAY=1-32, plot 38150833),
CHECKPOINT_DIR on netscratch. First-rollout raw-prompt verification is
automated (watcher greps 1.jsonl for the wrapper string).

## Final results (2026-08-11): raw run complete

Train 38150829 COMPLETED (3266 steps, 16h). Evals 32/32 ckpts x 3 datasets.
Full-set export: `analysis_data/idood_qwen05b_simple_raw.csv` (96 rows).
Figures: `figures/Qwen2.5-0.5B/balanced-simple-raw-grpo-.../` (plot job needed a
resubmit — `--dataset_suffix -simple` argparse bug, fixed to `=` form in
`scripts/plot/plot_results.sh`). WandB: run c2236f25.

Final checkpoint (step 3200) vs R1-style prompt baselines (full set):

| Run | ID p@1 | ID p@32 | n5 p@1 | n5 p@32 | n6 p@1 | n6 p@32 |
|---|--:|--:|--:|--:|--:|--:|
| raw simple (3200) | 0.390 | 0.397 | 0.108 | 0.118 | 0.028 | 0.032 |
| R1-style seed2 (3250) | 0.424 | 0.454 | 0.125 | 0.172 | 0.033 | 0.069 |
| R1-style tp0.5 (3250) | 0.422 | 0.472 | 0.111 | 0.167 | 0.026 | 0.056 |

Takeaways:
- Scaffold-free costs a little accuracy (ID p@1 −0.03) and was still climbing
  at end of epoch 2 — scaffold buys speed more than ceiling.
- **Diversity collapse is the real difference**: pass@32 ~ pass@1 for the raw
  run (ID 0.397 vs 0.390) while R1-style keeps a large sampling gain (n6 2x).
  Mean response length 114 tok (vs ~1024-cap thinking traces). The raw-prompt
  model converged to short, near-deterministic answers at temp 0.6.

## ID-vs-OOD vs 1.5B teacher (2026-08-11)

`figures/comparison/id_vs_ood_teacher1p5b_vs_student05b_rawsimple/` (grid + 4
panels) from `analysis_data/idood_teacher1p5b_vs_student05b_rawsimple.csv`
(teacher rows reused from the 2026-08-03 export; raw-simple rows with dataset
names normalized `balanced*-simple` -> `balanced*` — same puzzles, so axes are
comparable; prompts differ by design). Same picture as every 0.5B arm: shared
ID->OOD ray, teacher travels far further; raw-simple's 32-ckpt trajectory ends
(ID 0.390, n5 0.108 p@1) below the teacher's step-300 point, and its pass@32
panels sit nearly on the pass@1 diagonal (diversity collapse, cf. above).
