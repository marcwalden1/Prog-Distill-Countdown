"""Per-prompt correctness for Qwen3 GRPO runs at a given step -> json cache."""
import json, sys, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from grader_utils import compute_score

RUNS = {
    "0.6B": "Qwen3-0.6B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
    "1.7B": "Qwen3-1.7B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
    "4B":   "Qwen3-4B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
}
DS = ["balanced-simple", "balanced5-simple", "balanced6-simple"]
STEP = int(os.environ.get("STEP", 1600))

def score_row(row):
    gt = row["target"]; extra = {"numbers": row["nums"]}
    s = [compute_score(None, o, gt, extra, verbose=False) for o in row["outputs"]]
    return [1 if x == 1.0 else 0 for x in s]

if __name__ == "__main__":
    out = {}
    with Pool(8) as p:
        for name, run in RUNS.items():
            for ds in DS:
                f = f"results/{run}/global_step_{STEP}/{ds}_temp0.6_n32_max1024.json"
                rows = json.load(open(f))
                corr = p.map(score_row, rows, chunksize=32)
                out[f"{name}|{ds}"] = {"keys": [f"{sorted(r['nums'])}|{r['target']}" for r in rows],
                                       "correct": corr}
                print(name, ds, len(rows), sum(c[0] for c in corr)/len(rows), flush=True)
    json.dump(out, open(f"analysis_data/qwen3_venn_step{STEP}.json", "w"))
