#!/usr/bin/env python3
"""
Canonical registry of the Qwen2.5-0.5B *SFT-from-a-1.5B-teacher* runs and the
Qwen2.5-1.5B GRPO runs they are compared against.

Every 0.5B student here was SFT'd on rollouts from the same 1.5B teacher,
`Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1` — verified from the sft-data
namespaces on disk, all of which are
`teacher-Qwen2.5-1.5B-balanced-grpo-lr3e-6-kl3e-3-seed1[-n4][-truncate]`.

Result trees (a run's evals live in exactly one of these; where a run appears in
two, the entry below names the tree with the fuller step coverage):

    BING    results                  -> .../bingbin/repo-artifacts/RL-skill-comp/results
    SDHOL_A results_shlok            -> .../sdholakia/repo-artifacts/RL-skill-comp/results
    SDHOL_B /n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results

Consumed by scripts/export_pass1_pass32_sft_grpo.sh (builds the EXPORT_RUNS
specs) and scripts/plot/plot_pass1_vs_pass32_sft_grpo.py (curve identity, color
and marker). `label` is the join key end to end — NOT exp_name, because
`balanced-grpo-lr3e-6-kl1e-3-seed1` exists as two different 1.5B runs owned by
two different people.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TREES = {
    "BING": "results",
    "SDHOL_A": "results_shlok",
    "SDHOL_B": "/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results",
}

TEACHER = "Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1"

# --- 0.5B students: SFT from the 1.5B teacher, then GRPO -------------------------
# (label, tree, model, exp_name, method, sft_lr, grpo_kl, seed, owner, note)
STUDENTS = [
    # -- vanilla distill (one SFT pass on the teacher's step-1600 rollouts) --------
    ("distill sft3e-6 kl3e-4 s1",  "SDHOL_B", "balanced-distill-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed1",
     "distill", "3e-6", "3e-4", 1, "Shlok", ""),
    ("distill sft3e-6 kl3e-4 s21", "SDHOL_A", "balanced-distill-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed21",
     "distill", "3e-6", "3e-4", 21, "Shlok", ""),
    ("distill sft1e-5 kl3e-4 s21", "SDHOL_A", "balanced-distill-sftlr1e-5-grpo-lr1e-6-kl3e-4-seed21",
     "distill", "1e-5", "3e-4", 21, "Shlok", ""),
    ("distill sft3e-5 kl3e-3 s1",  "SDHOL_A", "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-3-seed1",
     "distill", "3e-5", "3e-3", 1, "Shlok", ""),
    ("distill sft3e-5 kl3e-4 s1",  "SDHOL_B", "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1",
     "distill", "3e-5", "3e-4", 1, "Shlok", "the CLAUDE.md 'official' vanilla-distill student"),
    ("distill sft3e-5 kl3e-4 s1-v2", "SDHOL_B", "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1-v2",
     "distill", "3e-5", "3e-4", 1, "Shlok", "re-run of the row above"),
    ("distill sft3e-5 kl3e-4 s21", "SDHOL_A", "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed21",
     "distill", "3e-5", "3e-4", 21, "Shlok", ""),
    # -- progressive distill (one SFT round per teacher checkpoint) ----------------
    ("progdistill sft1e-6 kl3e-4 s1", "SDHOL_B", "balanced-progdistill-sftlr1e-6-grpo-lr1e-6-kl3e-4-seed1",
     "progdistill", "1e-6", "3e-4", 1, "Shlok", "the CLAUDE.md 'official' progdistill student"),
    ("progdistill sft1e-6 kl1e-3 n8 s1", "SDHOL_B", "balanced-progdistill-sftlr1e-6-grpo-lr1e-6-kl1e-3-n8-seed1",
     "progdistill", "1e-6", "1e-3", 1, "Shlok", "rollout.n=8; 7/10 `balanced` steps"),
    ("progdistill sft3e-6 kl3e-3 s1", "SDHOL_B", "balanced-progdistill-sftlr3e-6-grpo-lr1e-6-kl3e-3-seed1",
     "progdistill", "3e-6", "3e-3", 1, "Shlok", ""),
    ("progdistill sft3e-6 kl3e-3 s21", "SDHOL_A", "balanced-progdistill-sftlr3e-6-grpo-lr1e-6-kl3e-3-seed21",
     "progdistill", "3e-6", "3e-3", 21, "Shlok", ""),
    ("progdistill sft3e-6 kl3e-4 s1", "SDHOL_B", "balanced-progdistill-sftlr3e-6-grpo-lr1e-6-kl3e-4-seed1",
     "progdistill", "3e-6", "3e-4", 1, "Shlok", ""),
    ("progdistill sft3e-6 kl1e-3 n8 s1", "SDHOL_B", "balanced-progdistill-sftlr3e-6-grpo-lr1e-6-kl1e-3-n8-seed1",
     "progdistill", "3e-6", "1e-3", 1, "Shlok", "rollout.n=8"),
    ("progdistill sft1e-5 kl3e-3 s1", "SDHOL_B", "balanced-progdistill-sftlr1e-5-grpo-lr1e-6-kl3e-3-seed1",
     "progdistill", "1e-5", "3e-3", 1, "Shlok", ""),
    ("progdistill sft1e-5 kl3e-3 s21", "SDHOL_A", "balanced-progdistill-sftlr1e-5-grpo-lr1e-6-kl3e-3-seed21",
     "progdistill", "1e-5", "3e-3", 21, "Shlok", ""),
    ("progdistill sft3e-5 kl3e-3 s1", "SDHOL_A", "balanced-progdistill-sftlr3e-5-grpo-lr1e-6-kl3e-3-seed1",
     "progdistill", "3e-5", "3e-3", 1, "Shlok", ""),
    ("progdistill sft3e-5 kl3e-3 s21", "SDHOL_A", "balanced-progdistill-sftlr3e-5-grpo-lr1e-6-kl3e-3-seed21",
     "progdistill", "3e-5", "3e-3", 21, "Shlok", ""),
    # -- progdistill-mixture (SFT on the pooled rounds, then GRPO) -----------------
    ("mixture sft3e-5 kl1e-3 s1", "BING",
     "balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl1e-3-seed1",
     "mixture", "3e-5", "1e-3", 1, "bingbin", "9/10 `balanced` steps (step 450 missing)"),
    ("mixture sft3e-5 kl3e-3 s1", "BING",
     "balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl3e-3-seed1",
     "mixture", "3e-5", "3e-3", 1, "bingbin", ""),
]

# --- the pre-GRPO SFT students, evaluated once (the origin of each GRPO curve) ---
# (label, tree, exp_name, method, sft_lr, owner)
SFT_ONLY = [
    ("SFT-only distill sft1e-5", "SDHOL_B",
     "balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr1e-5-seed1", "distill", "1e-5", "Shlok"),
    ("SFT-only distill sft3e-5", "SDHOL_B",
     "balanced-distill-teacher-t1p5b-lr3e-6-kl3e-3-step1600-n4-sftlr3e-5-seed1", "distill", "3e-5", "Shlok"),
    ("SFT-only progdistill sft1e-6", "SDHOL_B",
     "balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr1e-6-seed1", "progdistill", "1e-6", "Shlok"),
    ("SFT-only progdistill sft3e-6", "SDHOL_B",
     "balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr3e-6-seed1", "progdistill", "3e-6", "Shlok"),
    ("SFT-only progdistill sft1e-5", "SDHOL_B",
     "balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr1e-5-seed1", "progdistill", "1e-5", "Shlok"),
    ("SFT-only progdistill sft3e-5", "SDHOL_B",
     "balanced-progdistill-t1p5b-lr3e-6-kl3e-3-s0p5b-n4-sftlr3e-5-seed1", "progdistill", "3e-5", "Shlok"),
    ("SFT-only mixture sft1e-5", "BING",
     "balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr1e-5-steps1600-seed1", "mixture", "1e-5", "bingbin"),
    ("SFT-only mixture sft3e-5", "BING",
     "balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr3e-5-steps1600-seed1", "mixture", "3e-5", "bingbin"),
    ("SFT-only mixture sft1e-4", "BING",
     "balanced-progdistill-mixture-teacher-t1p5b-lr3e-6-kl3e-3-sftlr1e-4-steps1600-seed1", "mixture", "1e-4", "bingbin"),
]

# --- 1.5B GRPO runs -------------------------------------------------------------
# `have_scores` = already present in a tracked analysis_data/idood_*.csv under that
# label, so the exporter skips it. The rest are new.
# (label, tree, exp_name, lr, kl, owner, have_scores)
TEACHERS = [
    ("1.5B lr3e-6 kl3e-3 (teacher)", "SDHOL_B", "balanced-grpo-lr3e-6-kl3e-3-seed1",
     "3e-6", "3e-3", "Shlok", "teacher GRPO (Qwen2.5-1.5B)"),
    ("1.5B lr3e-6 kl1e-3 (bingbin)", "BING", "balanced-grpo-lr3e-6-kl1e-3-seed1",
     "3e-6", "1e-3", "bingbin", "1.5B lr3e-6 kl1e-3 (Qwen2.5-1.5B)"),
    ("1.5B lr6e-6 kl3e-3 (bingbin)", "BING", "balanced-grpo-lr6e-6-kl3e-3-seed1",
     "6e-6", "3e-3", "bingbin", "1.5B lr6e-6 kl3e-3 (Qwen2.5-1.5B)"),
    ("1.5B warmup/beta2 (bingbin)", "BING", "balanced-grpo-warmup0.03-beta2_0.95-seed1",
     "1e-6", "1e-3", "bingbin", "warmup GRPO (Qwen2.5-1.5B)"),
    ("1.5B lr1e-6 kl1e-3 (Shlok)", "SDHOL_B", "balanced-grpo-lr1e-6-kl1e-3-seed1",
     "1e-6", "1e-3", "Shlok", None),
    ("1.5B lr3e-6 kl1e-3 (Shlok)", "SDHOL_B", "balanced-grpo-lr3e-6-kl1e-3-seed1",
     "3e-6", "1e-3", "Shlok", None),
    ("1.5B lr3e-6 kl3e-4 (Shlok)", "SDHOL_B", "balanced-grpo-lr3e-6-kl3e-4-seed1",
     "3e-6", "3e-4", "Shlok", None),
    ("1.5B lr3e-7 kl1e-3 (Shlok)", "SDHOL_B", "balanced-grpo-lr3e-7-kl1e-3-seed1",
     "3e-7", "1e-3", "Shlok", None),
    ("1.5B lr3e-7 kl3e-3 (Shlok)", "SDHOL_B", "balanced-grpo-lr3e-7-kl3e-3-seed1",
     "3e-7", "3e-3", "Shlok", None),
    ("1.5B lr3e-7 kl3e-4 (Shlok)", "SDHOL_B", "balanced-grpo-lr3e-7-kl3e-4-seed1",
     "3e-7", "3e-4", "Shlok", None),
]


def tree_path(tag):
    p = TREES[tag]
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def export_specs():
    """`label=result_dir::model::exp` for everything not already exported."""
    out = []
    for label, tree, exp, *_ in ((s[0], s[1], s[2]) for s in STUDENTS):
        out.append((label, f"{label}={tree_path(tree)}::Qwen2.5-0.5B::{exp}"))
    for label, tree, exp, *_ in SFT_ONLY:
        out.append((label, f"{label}={tree_path(tree)}::Qwen2.5-0.5B::{exp}"))
    for label, tree, exp, _lr, _kl, _own, have in TEACHERS:
        if have is None:
            out.append((label, f"{label}={tree_path(tree)}::Qwen2.5-1.5B::{exp}"))
    return out


if __name__ == "__main__":
    import sys
    specs = export_specs()
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(specs))
    else:
        for _, s in specs:
            print(s)
