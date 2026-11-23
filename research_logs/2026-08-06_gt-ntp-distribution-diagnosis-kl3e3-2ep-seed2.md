# Checkpoint diagnosis: GT-answer NTP loss + distribution drift across GRPO (2026-08-06)

Run diagnosed: `Qwen2.5-0.5B/balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2`
(the lr=3e-6 arm of the 2026-08-05 kl3e-3/n8/2-epoch sweep, relaunched as seed2;
completed all 3266 steps). Question: how do (1) the NTP loss on the ground-truth
answer and (2) the model's predictive distributions change over training?

## Method

`scripts/diagnose_gt_ntp.py` (new; submitted via `scripts/diagnose_gt_ntp.sh`,
job 37540828, 1×H100, ~50 min). For each of the 16 merged eval-grid checkpoints
(150…1600, 1800, 2100, 2250, 2600, 2900, 3250) plus the **base model as step 0**,
teacher-force the oracle continuation

    </think>\n<answer> {GT equation} </answer>

after the byte-exact eval prompt (`prompt_utils.render_prompt`, mode="eval").
The GT equation is `canonical_pattern` with `extra_info.numbers` substituted
**in order** — verified to reproduce the target exactly on all 5617
balanced/balanced5 test rows. Prompts: all 997 ID (`balanced`) + 500 fixed-seed
OOD (`balanced5`).

Metrics per (step, dataset), computed at the continuation token positions:
CE on the full continuation and on the equation tokens alone (macro/micro/p50/p90),
predictive entropy, top-1 prob, full-vocab KL(ckpt‖base), and
p(`<|endoftext|>`)+p(`<|im_end|>`) at the position right after `</answer>`
("stop prob" — the trailing-junk probe).

Artifacts:
- CSV: `analysis_data/gt_ntp_diagnosis_qwen05b_n8_lr3e-6_kl3e-3_ep2_seed2.csv`
- Figures: `figures/Qwen2.5-0.5B/balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2/diagnosis/gt_ntp_{metrics,vs_eval}.png`
  (plotter: `scripts/plot_gt_ntp_diagnosis.py`)

## Key numbers

| step | ce_answer ID | ce_answer OOD | entropy ID | top1 ID | KL‖base ID | stop-prob ID | eval mean@1 ID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 0.914 | 0.879 | 1.309 | 0.649 | 0 | 0.831 | — |
| 150 | 1.554 | 1.934 | 0.402 | 0.892 | 0.745 | 0.971 | 0.128 |
| 600 | 1.737 | 2.252 | 0.494 | 0.875 | 0.641 | 0.925 | 0.305 |
| 1600 | 1.802 | 2.355 | 0.552 | 0.852 | 0.634 | 0.836 | 0.361 |
| 2900 | 2.018 | 2.564 | 0.582 | 0.857 | 0.625 | 0.468 | 0.370 |
| 3250 | 1.711 | 2.276 | 0.604 | 0.844 | 0.622 | 0.817 | 0.427 |

## Findings

1. **GT-answer NTP loss *rises* while eval accuracy rises.** ce_answer goes
   0.91 → ~1.7–2.0 (ID) and 0.88 → ~2.3–2.7 (OOD) over training, while eval
   mean@1 climbs 0.13 → 0.43 (ID). GRPO does not concentrate mass on the
   canonical solution — it concentrates on the policy's *own* solution
   modes/formatting, and the canonical equation becomes *less* likely than
   under the base model. Caveat: Countdown puzzles admit many valid solutions,
   so this is mode-selection, not degradation — but it means GT-NTP loss is
   **anti-correlated** with capability here and should not be used as a proxy
   metric for this run.
2. **The p90 tail grows much faster than the median** (ID p90 0.9 → ~4.1 at
   step 2900 vs p50 → ~1.8): for a subset of puzzles the policy strongly
   disprefers the canonical solution shape, rather than a uniform drift.
3. **Almost all distribution sharpening happens in the first 150 steps**, then
   slowly *relaxes*: entropy collapses 1.31 → 0.40 nats by step 150, then
   creeps back up to 0.60 by 3250; top-1 prob mirrors it (0.65 → 0.89 → 0.84).
   KL(ckpt‖base) jumps to ~0.74 at step 150 and then stays pinned at ~0.61–0.66
   for the remaining 3100 steps — the kl=3e-3 anchor holds total drift roughly
   constant while the policy redistributes *within* that budget. (Teacher-forced
   entropy here is much lower than the ~2.8 rollout entropy in the training log
   because the conditioning is the oracle continuation, not free generation.)
4. **ID and OOD drift identically in distribution space** (entropy/top1/KL
   curves nearly overlap), but ce_answer rises ~0.5 nats *more* on OOD: the
   solution-mode concentration is fit to ID structure, consistent with the
   flat-ish OOD eval accuracy (mean@1 0.02 → 0.12 plateauing after ~1400).
5. **Stop-prob after `</answer>` is unstable across training** — oscillating
   0.47–0.97 (base: 0.83) with deep dips at steps 900, 1400, and especially
   2900 (0.47 ID / 0.53 OOD). With the 1.0/0.1/0.0 reward, everything after a
   correct `</answer>` is reward-neutral, so nothing pins this quantity down —
   direct evidence for the trailing-junk incentive gap that motivated the
   trailing-penalty run (`...-tp0.5-ep2-seed1`, launched 2026-08-05). When
   comparing, expect the tp0.5 arm to hold stop-prob high and stable instead.

## Reproduction

    CKPT_ROOT=$CHECKPOINT_DIR/checkpoints/Qwen2.5-0.5B/balanced-grpo-dynbsz-n8-lr3e-6-kl3e-3-ep2-seed2 \
    STEPS=0,150,300,450,600,750,900,1050,1200,1400,1600,1800,2100,2250,2600,2900,3250 \
    OUT_CSV=analysis_data/gt_ntp_diagnosis_qwen05b_n8_lr3e-6_kl3e-3_ep2_seed2.csv \
    sbatch --account=kempner_bingbin_lab scripts/diagnose_gt_ntp.sh

Restart-safe (skips (step, dataset) rows already in the CSV). Only steps with a
merged `model.safetensors` are usable; the eval pipeline's merge step provides
them, so run evals first (or merge manually) for any additional steps.
