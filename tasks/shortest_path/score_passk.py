"""Unbiased pass@k for shortest-path eval JSONs (written by eval.py).

    python3 tasks/shortest_path/score_passk.py \
        --results results/baselines/Qwen3-0.6B-Base/max512/spath-simple_temp0.6_n128_max512.json \
                  results/baselines/Qwen3-0.6B-Base/max512/spath5-simple_temp0.6_n128_max512.json ... \
        --out analysis_data/spath_base_passk.csv

Dataset is inferred from the JSON filename (<dataset>_temp...), the model from
the path component after results/baselines/ (or pass --model). Rows of eval.py's
output list are aligned with data/<dataset>/test.parquet by position.

Per prompt: c = #samples with reward 1.0 out of n. pass@k = 1 - C(n-c, k)/C(n, k)
(Chen et al. 2021), averaged over prompts, for k = 1, 2, 4, ..., n. Grouped by
distance d and by shortest-path multiplicity (unique vs multiple). Also reports
empty rate (blank output), format rate (reward >= 0.1) and pass@k when only the FIRST <answer> is graded
(diagnostic for base models that keep generating after answering; the training
reward grades the LAST one).
"""

import argparse
import json
import os
import re
import sys
from math import comb

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import compute_score  # noqa: E402


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def first_answer_only(text):
    i = text.find("</answer>")
    return text if i < 0 else text[: i + len("</answer>")]


def label_from_path(p):
    """results/baselines/<model>/... -> '<model>';
    results/<model>/<exp>/global_step_N/... -> '<model>/<exp>/stepN'."""
    m = re.search(r"results/baselines/([^/]+)/", p)
    if m:
        return m.group(1)
    m = re.search(r"results/([^/]+)/([^/]+)/global_step_(\d+)/", p)
    if m:
        return f"{m.group(1)}/{m.group(2)}/step{m.group(3)}"
    return "?"


def score_file(path, model):
    ds = os.path.basename(path).split("_temp")[0]
    df = pd.read_parquet(f"data/{ds}/test.parquet")
    with open(path) as f:
        recs = json.load(f)
    assert len(recs) == len(df), f"{path}: {len(recs)} records vs {len(df)} parquet rows"
    rows = []
    for rec, (_, r) in zip(recs, df.iterrows()):
        gt, info = r["reward_model"]["ground_truth"], r["extra_info"]
        outs = rec["outputs"]
        sc = [compute_score(None, o, gt) for o in outs]
        sc_first = [compute_score(None, first_answer_only(o), gt) for o in outs]
        rows.append({
            "model": model, "dataset": ds, "idx": int(info["index"]),
            "distance": int(info["distance"]),
            "num_shortest_paths": int(info["num_shortest_paths"]),
            "n": len(outs), "c": sum(s == 1.0 for s in sc),
            "c_first": sum(s == 1.0 for s in sc_first),
            "fmt": sum(s >= 0.1 for s in sc) / len(outs),
            "empty": sum(not o.strip() for o in outs) / len(outs),
        })
    return rows


def summarize(per_prompt, ks):
    out = []
    df = pd.DataFrame(per_prompt)
    df["paths"] = df["num_shortest_paths"].map(lambda x: "unique" if x == 1 else "multi")
    for group_cols in (["model", "dataset", "distance"], ["model", "dataset", "distance", "paths"]):
        for keys, g in df.groupby(group_cols):
            keys = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row = {"paths": "all", **keys, "n_prompts": len(g), "empty_rate": g["empty"].mean(),
                   "format_rate": g["fmt"].mean()}
            for k in ks:
                row[f"pass@{k}"] = g.apply(lambda x: pass_at_k(x.n, x.c, k), axis=1).mean()
            for k in (1, ks[-1]):
                row[f"first_ans_pass@{k}"] = g.apply(lambda x: pass_at_k(x.n, x.c_first, k), axis=1).mean()
            out.append(row)
    return pd.DataFrame(out).sort_values(["model", "dataset", "distance", "paths"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="CSV path for the summary table")
    args = ap.parse_args()

    per_prompt = []
    for p in args.results:
        m = args.model or label_from_path(p)
        per_prompt += score_file(p, m)
    n = min(r["n"] for r in per_prompt)
    ks = [k for k in (2 ** i for i in range(20)) if k <= n]
    summ = summarize(per_prompt, ks)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        summ.to_csv(args.out, index=False)
        pd.DataFrame(per_prompt).to_csv(args.out.replace(".csv", "_per_prompt.csv"), index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
