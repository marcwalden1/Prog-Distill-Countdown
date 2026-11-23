#!/usr/bin/env python3
"""Regenerate analysis_data/gemma270m_pareto_eval_progression.csv from the
FULL-set .scores caches of the 4 gemma distill/progdistill+GRPO runs copied into
results_sdholakia/ (kl3e-3 / kl3e-4, sftlr1e-4, grpo-lr1e-6). Replaces the old CSV
that was rolled up from the capped (first-1000-prompt) scores. Columns match what
plot_id_vs_ood_gemma_csv.py consumes (method, grpo_kl, dataset, step, mean_at_1,
mean_at_32) plus provenance columns.
"""
import csv, glob, json, os, re

ROOT = "results_sdholakia/gemma-3-270m"
DIRS = [
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-3-seed1",
    "gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1",
    "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-3-seed1",
    "gemma-270m-balanced-progdistill-teacher-Qwen1.5B-sftlr1e-4-grpo-lr1e-6-kl3e-4-seed1",
]
DATASETS = ["balanced", "balanced5", "balanced6"]
OUT = "analysis_data/gemma270m_pareto_eval_progression.csv"


def meta(dirname):
    method = "progdistill" if "progdistill" in dirname else "distill"
    kl = re.search(r"-kl([0-9e.\-]+)-seed", dirname).group(1)
    return method, kl


def main():
    rows = []
    for d in DIRS:
        method, kl = meta(d)
        for ds in DATASETS:
            for sp in glob.glob(os.path.join(ROOT, d, "global_step_*",
                                             f"{ds}_temp0.6_n32_max1024.json.scores")):
                step, m1, m32 = json.load(open(sp))
                lp = sp.replace(".scores", ".lengths")
                mlen = json.load(open(lp))[1] if os.path.exists(lp) else ""
                rows.append(dict(model="gemma-3-270m", method=method, sft_lr="1e-4",
                                 grpo_lr="1e-6", grpo_kl=kl, seed="1", exp_name=d,
                                 dataset=ds, step=step, mean_at_1=m1, mean_at_32=m32,
                                 mean_response_length=mlen))
    rows.sort(key=lambda r: (r["method"], r["grpo_kl"], r["dataset"], r["step"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["model", "method", "sft_lr", "grpo_lr", "grpo_kl", "seed", "exp_name",
            "dataset", "step", "mean_at_1", "mean_at_32", "mean_response_length"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
