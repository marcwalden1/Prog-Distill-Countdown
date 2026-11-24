"""Generate the shortest-path task parquets.

    python3 tasks/shortest_path/make_data.py            # defaults below

Writes (schema mirrors data/balanced-simple, so train_grpo.sh / eval.py work as-is):
    data/spath-simple/train.parquet    d in {2,3,4}, --train-per-d each
    data/spath-simple/test.parquet     d in {2,3,4}, --test-per-d each  (ID; GRPO val set)
    data/spath5-simple/test.parquet    d = 5, --test-per-d              (OOD)
    data/spath6-simple/test.parquet    d = 6, --test-per-d              (OOD)

The "-simple" suffix makes train_grpo.sh and eval.py use the literal prompt
string (no chat template), like the Countdown *-simple variants.

Every split uses the same graph distribution (N ~ U{12..16}, k-out graph with
out-degree ~ U{1,2,3}); only the conditioned distance d differs. Per puzzle:
sample a graph, collect all (s, t) with shortest distance exactly d, reject the
graph if there are none, else pick one pair uniformly. Labels are N distinct
random capital letters, edges are listed in shuffled order. Puzzles are unique
across all splits (test generated first, train skips test keys).
"""

import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphs import OUT_DEGREES, bfs, random_kout_graph, sample_shortest_path  # noqa: E402

DATA_SOURCE = "shortest_path"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Prompt design (2026-11-22/23, see research_logs/2026-11-22_shortest-path-task.md):
# - ends with "Solution:\n": without the trailing newline Qwen3-0.6B-Base emitted
#   EOS immediately on 53% of samples (0% with it).
# - node labels are single capital letters (one token each, " K"), not 0-99 ints
#   (2-3 tokens each).
# - the example path uses 3 letters NOT in this graph, so copying it can never be a
#   correct path; spaces inside <answer> keep the first node from merging into ">K".
# - the rule line uses lowercase a/b so it cannot name a node in the graph.
PROMPT_TEMPLATE = (
    "Solve the shortest path problem.\n"
    "Directed edges: {edges}\n"
    "Find a shortest path from node {source} to node {target}.\n"
    "\n"
    "Rules:\n"
    "- An edge a -> b can only be traversed from a to b.\n"
    "- The path must start at {source} and end at {target}.\n"
    "- Use as few edges as possible.\n"
    "- You may use <think>...</think> for scratch work.\n"
    "- Return the path inside <answer>...</answer> as nodes joined by ->. "
    "For example, <answer> {example} </answer>.\n"
    "\n"
    "Solution:\n"
)


def render_prompt(edges, source, target, example_nodes):
    return PROMPT_TEMPLATE.format(
        edges=", ".join(f"{a} -> {b}" for a, b in edges), source=source, target=target,
        example=" -> ".join(example_nodes))


def make_puzzle(d, rng, n_min, n_max):
    """Return one puzzle dict at shortest distance d (rejection-sampled)."""
    while True:
        n = rng.randint(n_min, n_max)
        adj = random_kout_graph(n, rng)
        pairs = []
        for s in range(n):
            dist, _ = bfs(adj, s)
            pairs += [(s, t) for t, dd in dist.items() if dd == d]
        if pairs:
            break
    s, t = rng.choice(pairs)
    labels = rng.sample(LETTERS, n)
    unused = sorted(set(LETTERS) - set(labels))
    example_nodes = rng.sample(unused, 3)
    edges = [(labels[u], labels[v]) for u in adj for v in adj[u]]
    rng.shuffle(edges)
    ladj = {}
    for a, b in edges:
        ladj.setdefault(a, []).append(b)
    S, T = labels[s], labels[t]
    dist, cnt = bfs(ladj, S)
    assert dist[T] == d
    return {
        "source": S, "target": T, "distance": d, "num_nodes": n,
        "edges": edges, "num_shortest_paths": cnt[T], "example_nodes": example_nodes,
        "ref_path": sample_shortest_path(ladj, S, T, rng),
    }


def key(p):
    return (tuple(sorted(p["edges"])), p["source"], p["target"])


def to_row(p, split, index):
    prompt = render_prompt(p["edges"], p["source"], p["target"], p["example_nodes"])
    gt = {"source": p["source"], "target": p["target"], "distance": p["distance"],
          "edges_src": [a for a, _ in p["edges"]], "edges_dst": [b for _, b in p["edges"]]}
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "graph",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split, "index": index, "distance": p["distance"],
            "source": p["source"], "target": p["target"],
            "num_nodes": p["num_nodes"], "num_edges": len(p["edges"]),
            "num_shortest_paths": p["num_shortest_paths"],
            # One uniformly-random shortest path, for debugging / future SFT only.
            # The reward never uses it (any shortest path scores 1.0).
            "ref_path": p["ref_path"],
        },
    }


def generate(ds, per_d, split, rng, seen, args):
    rows = []
    for d in ds:
        k = 0
        while k < per_d:
            p = make_puzzle(d, rng, args.n_min, args.n_max)
            if key(p) in seen:
                continue
            seen.add(key(p))
            rows.append(to_row(p, split, len(rows)))
            k += 1
    return rows


def write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    df = pd.DataFrame([r["extra_info"] for r in rows])
    summ = df.groupby("distance").agg(
        n=("index", "size"), nodes=("num_nodes", "mean"), edges=("num_edges", "mean"),
        nsp_mean=("num_shortest_paths", "mean"),
        unique_frac=("num_shortest_paths", lambda x: (x == 1).mean()))
    print(f"\n{path}  ({len(rows)} rows)\n{summ.round(3).to_string()}")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--train-ds", default="2,3,4")
    ap.add_argument("--ood-ds", default="5,6")
    ap.add_argument("--train-per-d", type=int, default=20000)
    ap.add_argument("--test-per-d", type=int, default=300)
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.n_max <= len(LETTERS) - 3, "need 3 unused letters for the prompt example"

    train_ds = [int(x) for x in args.train_ds.split(",")]
    ood_ds = [int(x) for x in args.ood_ds.split(",")]
    rng = random.Random(args.seed)
    seen = set()

    # Test splits first so train dedups against them.
    write(generate(train_ds, args.test_per_d, "test", rng, seen, args),
          f"{args.out_root}/spath-simple/test.parquet")
    for d in ood_ds:
        write(generate([d], args.test_per_d, "test", rng, seen, args),
              f"{args.out_root}/spath{d}-simple/test.parquet")
    write(generate(train_ds, args.train_per_d, "train", rng, seen, args),
          f"{args.out_root}/spath-simple/train.parquet")

    meta = {**vars(args), "out_degrees": list(OUT_DEGREES), "graph_model": "k-out",
            "labels": "letters", "prompt_template": PROMPT_TEMPLATE}
    with open(f"{args.out_root}/spath-simple/generation.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
