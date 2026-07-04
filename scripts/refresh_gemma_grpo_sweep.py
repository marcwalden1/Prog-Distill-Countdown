#!/usr/bin/env python3
"""Refresh the Gemma-3-270M GRPO sweep analysis tables from local score caches.

Writes analysis_data/gemma_grpo_sweep_eval.csv (per run/checkpoint/dataset eval,
sourced from local .scores/.lengths) and appends any missing Gemma distill/
progdistill+GRPO runs to analysis_data/pareto_run_registry.csv. Idempotent:
existing registry rows are preserved and de-duplicated by
(model, method, sft_lr, grpo_lr, grpo_kl, seed).

mean_at_1  = binary mean@1 (.scores[1]); mean_at_32 = shaped mean@32 (.scores[2]).
"""
import os, re, glob, csv, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = ["results/gemma-3-270m", "results/results/gemma-3-270m"]
DATASETS = ["balanced", "balanced5", "balanced6"]


def parse(d):
    m = re.search(
        r"sftlr([0-9e\-\.]+?)-grpo-lr([0-9e\-\.]+?)-kl([0-9e\-\.]+?)-"
        r"(?:(early400)-)?seed(\d+)", d)
    if not m or "grpo" not in d or "distill" not in d:
        return None
    sftlr, glr, kl, early, seed = m.groups()
    method = "progdistill" if "progdistill" in d else "distill"
    return dict(method=method, sft_lr=sftlr, grpo_lr=glr, grpo_kl=kl,
                seed=seed, early=bool(early), dirname=d)


def collect():
    runs = {}
    for root_rel in ROOTS:
        root = os.path.join(REPO, root_rel)
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            info = parse(d)
            if not info:
                continue
            steps = {}
            for ds in DATASETS:
                for sc in glob.glob(os.path.join(
                        root, d, "global_step_*",
                        f"{ds}_temp0.6_n32_max1024.json.scores")):
                    step = int(re.search(r"global_step_(\d+)", sc).group(1))
                    try:
                        arr = json.load(open(sc))
                    except Exception:
                        continue
                    ln, mlen = sc[:-7] + ".lengths", ""
                    if os.path.exists(ln):
                        try:
                            mlen = json.load(open(ln))[1]
                        except Exception:
                            pass
                    steps.setdefault(step, {})[ds] = (arr[1], arr[2], mlen)
            key = (info["method"], info["sft_lr"], info["grpo_lr"],
                   info["grpo_kl"], info["seed"], info["early"])
            if key not in runs or len(steps) > len(runs[key]["steps"]):
                runs[key] = dict(info=info, steps=steps, result_dir=root_rel)
    return runs


def write_sweep(runs):
    path = os.path.join(REPO, "analysis_data", "gemma_grpo_sweep_eval.csv")
    cols = ["model", "method", "sft_lr", "grpo_lr", "grpo_kl", "seed",
            "early400", "exp_name", "dataset", "step", "mean_at_1",
            "mean_at_32", "mean_response_length"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for _, r in sorted(runs.items()):
            i = r["info"]
            for step in sorted(r["steps"]):
                for ds in DATASETS:
                    if ds not in r["steps"][step]:
                        continue
                    m1, m32, mlen = r["steps"][step][ds]
                    w.writerow(dict(
                        model="gemma-3-270m", method=i["method"],
                        sft_lr=i["sft_lr"], grpo_lr=i["grpo_lr"],
                        grpo_kl=i["grpo_kl"], seed=i["seed"],
                        early400=i["early"], exp_name=i["dirname"],
                        dataset=ds, step=step, mean_at_1=m1, mean_at_32=m32,
                        mean_response_length=mlen))
    print("wrote", path, "|", len(runs), "runs")


def update_registry(runs):
    path = os.path.join(REPO, "analysis_data", "pareto_run_registry.csv")
    existing = list(csv.DictReader(open(path)))
    cols = list(existing[0].keys())
    have = {(r["model"], r["method"], r["sft_lr"], r["grpo_lr"],
             r["grpo_kl"], r["seed"]) for r in existing}
    added = 0
    for _, r in sorted(runs.items()):
        i = r["info"]
        key = ("gemma-3-270m", i["method"], i["sft_lr"], i["grpo_lr"],
               i["grpo_kl"], i["seed"])
        if key in have:
            continue
        steps = sorted(s for s in r["steps"] if "balanced" in r["steps"][s])
        if not steps:
            continue
        existing.append(dict(
            model="gemma-3-270m", method=i["method"], sft_lr=i["sft_lr"],
            grpo_lr=i["grpo_lr"], grpo_kl=i["grpo_kl"], seed=i["seed"],
            exp_name=i["dirname"], n_checkpoints=len(steps),
            checkpoint_steps=",".join(map(str, steps)),
            result_dir=r["result_dir"]))
        have.add(key)
        added += 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in existing:
            w.writerow({k: row.get(k, "") for k in cols})
    print("registry: +{} rows, {} total".format(added, len(existing)))


if __name__ == "__main__":
    runs = collect()
    write_sweep(runs)
    update_registry(runs)
