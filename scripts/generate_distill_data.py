"""Teacher rollouts + top-k logprobs for offline distillation (NTP and logit KD).

Samples N responses per prompt from the teacher with vLLM and stores, for every
response token, the teacher's logprob of the sampled token and its top-k
(token id, logprob) pairs. The tail bucket for logit distillation is
1 - sum(exp(topk_logprobs)) and is computed at train time, not stored.

Prompts are split into `--num_shards` contiguous shards of the train parquet;
this process handles shards `rank, rank + world, ...` and writes
`<out_dir>/shard_<i>.parquet` atomically, skipping shards that already exist,
so a killed/requeued job resumes where it left off.

Columns (one row per rollout; token-level lists are exact vLLM token ids, no
retokenization):
  prompt_idx       int64          row index in the train parquet
  sample_idx       int16          0..N-1
  prompt_ids       list<int32>
  response_ids     list<int32>    includes the EOS token when finish_reason=stop;
                                  len(prompt_ids) + len(response_ids) <= --max_total_len
  sampled_logprobs list<float32>  teacher logprob of response_ids[t]
  topk_ids         list<int32>    flat, len = k * len(response_ids), row-major by position
  topk_logprobs    list<float32>  flat, same layout, sorted by rank within a position
  finish_reason    string         stop | length
  response         string         decoded response (skip_special_tokens)
  score            float32        grader_utils.compute_score (0 / 0.1 / 1)
"""
import argparse, os, sys, time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from prompt_utils import render_prompt  # noqa: E402
from grader_utils import compute_score  # noqa: E402


def shard_bounds(n_rows, num_shards, i):
    edges = np.linspace(0, n_rows, num_shards + 1).astype(int)
    return int(edges[i]), int(edges[i + 1])  # python ints: vLLM V1 msgspec rejects numpy ints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "data/balanced-simple/train.parquet"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_total_len", type=int, default=512,
                    help="cap on prompt + response tokens; each prompt gets max_tokens = cap - len(prompt)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--num_shards", type=int, default=200)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--only_shards", type=str, default="", help="comma list; overrides rank/world")
    ap.add_argument("--max_prompts_per_shard", type=int, default=0, help="pilot: truncate each shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.only_shards:
        shards = [int(s) for s in a.only_shards.split(",")]
    else:
        shards = list(range(a.rank, a.num_shards, a.world))
    todo = [s for s in shards if not os.path.exists(os.path.join(a.out_dir, f"shard_{s:04d}.parquet"))]
    print(f"[rank {a.rank}] {len(shards)} shards assigned, {len(todo)} to do", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(a.data, columns=["prompt", "reward_model", "extra_info", "data_source"])
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=a.max_total_len, max_logprobs=a.topk, seed=a.seed,
              enable_prefix_caching=True)

    k = a.topk
    for s in todo:
        t0 = time.time()
        lo, hi = shard_bounds(len(df), a.num_shards, s)
        if a.max_prompts_per_shard:
            hi = min(hi, lo + a.max_prompts_per_shard)
        sub = df.iloc[lo:hi]
        prompts = [render_prompt(tok, list(p), mode="eval", disable_chat_template=True)
                   for p in sub["prompt"]]
        # Per-prompt params: response budget = total cap - prompt length, and a
        # per-prompt seed (vLLM gives the n children seed+0..n-1), so the random
        # draws are independent across prompts and do not depend on sharding.
        p_lens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
        sps = [SamplingParams(temperature=a.temperature, top_p=1.0, top_k=-1, n=a.n,
                              max_tokens=int(a.max_total_len - pl), logprobs=k,
                              seed=int(a.seed * 10**8 + (lo + j) * 16), detokenize=False)
               for j, pl in enumerate(p_lens)]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        t_gen = time.time() - t0

        cols = {c: [] for c in ["prompt_idx", "sample_idx", "prompt_ids", "response_ids",
                                "sampled_logprobs", "topk_ids", "topk_logprobs", "finish_reason"]}
        n_bad = 0
        for j, o in enumerate(outs):
            p_ids = np.asarray(o.prompt_token_ids, dtype=np.int32)
            for c in o.outputs:
                r_ids = np.asarray(c.token_ids, dtype=np.int32)
                assert len(p_ids) + len(r_ids) <= a.max_total_len, (len(p_ids), len(r_ids))
                R = len(r_ids)
                samp = np.empty(R, np.float32)
                tids = np.empty((R, k), np.int32)
                tlps = np.empty((R, k), np.float32)
                for t, d in enumerate(c.logprobs):
                    samp[t] = d[int(r_ids[t])].logprob
                    items = sorted(d.items(), key=lambda kv: kv[1].rank)[:k]
                    if len(items) < k:
                        n_bad += 1
                        items = items + [items[-1]] * (k - len(items))
                    tids[t] = [i for i, _ in items]
                    tlps[t] = [v.logprob for _, v in items]
                cols["prompt_idx"].append(lo + j)
                cols["sample_idx"].append(0)  # filled below
                cols["prompt_ids"].append(p_ids)
                cols["response_ids"].append(r_ids)
                cols["sampled_logprobs"].append(samp)
                cols["topk_ids"].append(tids.reshape(-1))
                cols["topk_logprobs"].append(tlps.reshape(-1))
                cols["finish_reason"].append(c.finish_reason)
        cols["sample_idx"] = [i % a.n for i in range(len(cols["sample_idx"]))]
        texts = tok.batch_decode(cols["response_ids"], skip_special_tokens=True)
        scores = []
        for pi, txt in zip(cols["prompt_idx"], texts):
            row = df.iloc[pi]
            gt = row["reward_model"]["ground_truth"]
            ei = row["extra_info"]
            scores.append(float(compute_score(row["data_source"], txt, gt, extra_info=ei, verbose=False)))

        table = pa.table({
            "prompt_idx": pa.array(cols["prompt_idx"], pa.int64()),
            "sample_idx": pa.array(cols["sample_idx"], pa.int16()),
            "prompt_ids": pa.array(cols["prompt_ids"], pa.list_(pa.int32())),
            "response_ids": pa.array(cols["response_ids"], pa.list_(pa.int32())),
            "sampled_logprobs": pa.array(cols["sampled_logprobs"], pa.list_(pa.float32())),
            "topk_ids": pa.array(cols["topk_ids"], pa.list_(pa.int32())),
            "topk_logprobs": pa.array(cols["topk_logprobs"], pa.list_(pa.float32())),
            "finish_reason": pa.array(cols["finish_reason"], pa.string()),
            "response": pa.array(texts, pa.string()),
            "score": pa.array(scores, pa.float32()),
        })
        path = os.path.join(a.out_dir, f"shard_{s:04d}.parquet")
        pq.write_table(table, path + ".tmp", compression="zstd")
        os.replace(path + ".tmp", path)
        n_tok = int(sum(len(r) for r in cols["response_ids"]))
        dt = time.time() - t0
        print(f"[rank {a.rank}] shard {s}: prompts {lo}-{hi} rollouts {len(texts)} "
              f"resp_tokens {n_tok} gen {t_gen:.0f}s total {dt:.0f}s "
              f"({n_tok / dt:.0f} tok/s) mean_score {np.mean(scores):.3f} "
              f"truncated {np.mean([f == 'length' for f in cols['finish_reason']]):.3f} "
              f"short_topk {n_bad}", flush=True)


if __name__ == "__main__":
    main()
