"""Behavioural stats of shortest-path generations (eval.py JSONs), per model and d.

    python3 tasks/shortest_path/analyze_generations.py \
        --group 0.6B results/Qwen3-0.6B-Base/<exp>/global_step_234/max512/*.json \
        --group 4B   results/Qwen3-4B-Base/<exp>/global_step_234/max512/*.json \
        --tokenizer /path/to/Qwen3-0.6B-Base --out analysis_data_spath/gen_stats.csv

Per sample:
  tokens        response length (tokenizer), at_cap = tokens >= cap-2
  n_ans         number of complete <answer>...</answer> blocks
  first_ok      the FIRST answer is a shortest path      last_ok = the graded (LAST) one
  any_ok        some answer block is a shortest path
  flip_bad      first_ok and not last_ok   (talked itself out of a right answer)
  flip_good     not first_ok and last_ok   (self-corrected)
  edge_claims   "X -> Y" pairs in the text OUTSIDE answer blocks; claim_real = share that are graph edges
  loop          graded answer revisits a node; excess = len(path)-1-d for valid too-long walks
"""

import argparse
import collections
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_errors import classify, load_results  # noqa: E402
from reward import parse_path  # noqa: E402

ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CHAIN = re.compile(r"\b([A-Z])((?:\s*->\s*[A-Z]\b)+)")
DOUBT = re.compile(r"not (?:the )?(?:shortest|correct|optimal|valid)|however|let'?s try|let me|re-?evaluate|"
                   r"double-check|mistake|recheck|optimi[sz]e", re.I)


def is_sp(path, gt, edges):
    return (path is not None and len(path) == int(gt["distance"]) + 1 and path[0] == str(gt["source"])
            and path[-1] == str(gt["target"]) and all((a, b) in edges for a, b in zip(path, path[1:])))


def sample_stats(text, gt):
    edges = set(zip(map(str, gt["edges_src"]), map(str, gt["edges_dst"])))
    answers = [parse_path(a.strip()) for a in ANS.findall(text)]
    oks = [is_sp(p, gt, edges) for p in answers]
    outside = ANS.sub(" ", text)
    claims = []
    for m in CHAIN.finditer(outside):
        nodes = [m.group(1)] + re.findall(r"[A-Z]", m.group(2))
        claims += list(zip(nodes, nodes[1:]))
    cls, _ = classify(text, gt)
    last = answers[-1] if answers else None
    d = int(gt["distance"])
    return dict(
        cls=cls, n_ans=len(answers), first_ok=bool(oks and oks[0]), last_ok=bool(oks and oks[-1]),
        any_ok=any(oks), n_distinct=len({tuple(p) for p in answers if p}),
        edge_claims=len(claims), claim_real=sum(c in edges for c in claims),
        doubt=bool(DOUBT.search(text)),
        loop=bool(last and len(set(last)) < len(last)),
        excess=(len(last) - 1 - d) if (last and cls == "too_long") else None,
        chars=len(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL", "JSON"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for g in args.group:
        label, paths = g[0], g[1:]
        texts, recs = [], []
        for ds, text, gt in load_results(paths):
            r = sample_stats(text, gt)
            r.update(model=label, d=int(gt["distance"]))
            recs.append(r)
            texts.append(text)
        lens = []
        for i in range(0, len(texts), 2048):  # chunked: keep only lengths, not the id lists
            lens += [len(x) for x in tok(texts[i:i + 2048], add_special_tokens=False)["input_ids"]]
        for r, n in zip(recs, lens):
            r["tokens"] = n
            r["at_cap"] = n >= args.cap - 2
        rows += recs
        print(f"{label}: {len(recs)} samples", file=sys.stderr)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out.replace(".csv", ".parquet"))

    df["flip_bad"] = df.first_ok & ~df.last_ok
    df["flip_good"] = ~df.first_ok & df.last_ok
    df["lost_right"] = df.any_ok & ~df.last_ok
    agg = df.groupby(["model", "d"]).agg(
        acc=("last_ok", "mean"), first_acc=("first_ok", "mean"), any_acc=("any_ok", "mean"),
        flip_bad=("flip_bad", "mean"), flip_good=("flip_good", "mean"), lost_right=("lost_right", "mean"),
        tokens=("tokens", "mean"), tok_p90=("tokens", lambda x: x.quantile(.9)), at_cap=("at_cap", "mean"),
        n_ans=("n_ans", "mean"), multi_ans=("n_ans", lambda x: (x > 1).mean()),
        doubt=("doubt", "mean"),
        claims=("edge_claims", "sum"), claim_real=("claim_real", "sum"))
    agg["claim_real_frac"] = agg.claim_real / agg.claims.clip(lower=1)
    agg = agg.drop(columns=["claim_real"]).reset_index()
    agg.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string(index=False))

    # accuracy conditioned on response shape
    print("\nlast_ok by #answer blocks (all d pooled, then d>=5):")
    df["n_ans_b"] = df.n_ans.clip(upper=4)
    for sub, name in ((df, "all"), (df[df.d >= 5], "d>=5")):
        t = sub.groupby(["model", "n_ans_b"]).agg(share=("last_ok", "size"), acc=("last_ok", "mean"))
        t["share"] = t.share / sub.groupby("model").size()
        print(name); print(t.round(3).unstack(0).to_string())
    print("\nno_answer: share at the token cap")
    na = df[df.cls == "no_answer"]
    print(na.groupby(["model", "d"]).at_cap.agg(["size", "mean"]).round(3).unstack(0).to_string())
    print("\ntoo_long: share revisiting a node, mean excess hops")
    tl = df[df.cls == "too_long"]
    print(tl.groupby(["model", "d"]).agg(n=("loop", "size"), loop=("loop", "mean"),
                                          excess=("excess", "mean")).round(2).unstack(0).to_string())


if __name__ == "__main__":
    main()
