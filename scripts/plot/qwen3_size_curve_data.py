"""Per-checkpoint pass@1 (mean over 32 samples) and pass@32 (any of 32 correct) for the
Qwen3-Base GRPO size sweep (kl3e-3, 1 epoch, rollout n=4). Writes analysis_data/qwen3_size_passk.csv.
Per-problem #correct (out of 32) is cached in analysis_data/.qwen3_size_cache/ so reruns are instant."""
import json, os, sys, csv
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from grader_utils import compute_score

RUNS = {
    "0.6B": "Qwen3-0.6B-Base/balanced-simple-grpo-lr1e-6-kl3e-3-seed1",
    "1.7B": "Qwen3-1.7B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
    "4B":   "Qwen3-4B-Base/balanced-simple-grpo-lr3e-6-kl3e-3-seed1",
}
DS = ["balanced-simple", "balanced5-simple", "balanced6-simple"]
STEPS = range(100, 1601, 100)
CACHE = "analysis_data/.qwen3_size_cache"

def n_correct(row):
    extra = {"numbers": row["nums"]}
    return sum(compute_score(None, o, row["target"], extra, verbose=False) == 1.0 for o in row["outputs"])

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    rows_out = []
    with Pool(8) as p:
        for name, run in RUNS.items():
            for ds in DS:
                for step in STEPS:
                    f = f"results/{run}/global_step_{step}/{ds}_temp0.6_n32_max1024.json"
                    c = f"{CACHE}/{run.replace('/', '__')}__{step}__{ds}.json"
                    if os.path.exists(c) and os.path.getmtime(c) > os.path.getmtime(f):
                        nc = json.load(open(c))
                    else:
                        data = json.load(open(f))
                        assert all(len(r["outputs"]) == 32 for r in data), f
                        nc = p.map(n_correct, data, chunksize=32)
                        json.dump(nc, open(c, "w"))
                    N = len(nc)
                    rows_out.append({"model": name, "run": run, "dataset": ds, "step": step, "N": N,
                                     "pass@1": sum(nc) / (32 * N), "pass@32": sum(x > 0 for x in nc) / N})
                    print(name, ds, step, f"{rows_out[-1]['pass@1']:.4f} {rows_out[-1]['pass@32']:.4f}", flush=True)
    with open("analysis_data/qwen3_size_passk.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
