#!/usr/bin/env python3
"""Rebuild the Qwen-0.5B fig7 ID-vs-OOD CSV with FULL-set mean@1/mean@32.

Takes the original qwen05b_fig7_pareto/pareto_eval_progression.csv (whose
method/sft_lr/grpo_kl/exp_name columns are correctly parsed but whose
mean_at_1/mean_at_32 were computed on the OLD first-1000-prompt cap), keeps only
the runs whose full eval JSONs are available locally (downloaded from HF), and
overwrites mean_at_1/mean_at_32 by re-scoring the full JSON. Only balanced +
balanced5 rows are kept (the only datasets plot_id_vs_ood_comparison.py uses).

Writes analysis_data/qwen05b_fig7_pareto_fullset.csv.
"""
import csv, glob, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader_utils import compute_score

ORIG = ("/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/repo-artifacts/"
        "RL-skill-comp/analysis_data/qwen05b_fig7_pareto/pareto_eval_progression.csv")
DL = sys.argv[1] if len(sys.argv) > 1 else "hf-evals-qwen05b"   # HF snapshot root
OUT = "analysis_data/qwen05b_fig7_pareto_fullset.csv"
KEEP_DATASETS = {"balanced", "balanced5"}


def score_item(args):
    outputs, target, nums = args
    if not outputs:
        return None
    extra = {"numbers": nums}
    m1 = float(compute_score(None, outputs[0], target, extra, verbose=False) == 1.0)
    scores = [compute_score(None, o, target, extra, verbose=False) for o in outputs]
    return m1, sum(scores) / len(scores)


def score_json(path):
    data = json.load(open(path))
    items = [(it["outputs"], it["target"], it["nums"]) for it in data if it.get("outputs")]
    with Pool(8) as pool:
        scored = [s for s in pool.map(score_item, items) if s is not None]
    n = len(scored)
    return n, sum(s[0] for s in scored) / n, sum(s[1] for s in scored) / n


def main():
    rows = list(csv.DictReader(open(ORIG)))
    out_rows, skipped, done = [], set(), 0
    for r in rows:
        if r["dataset"] not in KEEP_DATASETS:
            continue
        exp, ds, step = r["exp_name"], r["dataset"], r["step"]
        jp = os.path.join(DL, "Qwen2.5-0.5B", exp, f"global_step_{step}",
                          f"{ds}_temp0.6_n32_max1024.json")
        if not os.path.exists(jp):
            skipped.add(exp)
            continue
        n, m1, m32 = score_json(jp)
        r["mean_at_1"], r["mean_at_32"] = m1, m32
        out_rows.append(r)
        done += 1
        print(f"  scored {exp} {ds} step{step}: n={n} m@1={m1:.4f} m@32={m32:.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    kept = sorted({r["exp_name"] for r in out_rows})
    print(f"\nWrote {OUT}: {done} rows, {len(kept)} runs")
    for e in kept:
        print("  KEPT   ", e)
    for e in sorted(skipped - set(kept)):
        print("  SKIP(no full JSON) ", e)


if __name__ == "__main__":
    main()
