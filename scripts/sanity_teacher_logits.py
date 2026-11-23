"""Sanity check: how fast does a teacher's next-token distribution fall off?

Samples N rollouts for a handful of train prompts with vLLM (temperature 1,
same raw prompt path as GRPO on *-simple data), then re-runs the teacher with HF
to get the FULL next-token distribution at every response position. Reports how
much probability mass the top-k tokens hold, how often the sampled token falls
outside the top-k, and cross-checks vLLM's top-k logprobs against HF's.

Output: <out_dir>/sanity.npz (per-position sorted top-1000 probs, entropy, ranks,
...) + summary.json + rollouts.jsonl. Plotting is a separate CPU step.
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_utils import render_prompt  # noqa: E402

KS = [1, 2, 5, 10, 20, 50, 100, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/balanced-simple/train.parquet")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vllm_topk", type=int, default=20)
    ap.add_argument("--keep_rank", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(a.data)
    rows = df.sample(n=a.n_prompts, random_state=a.seed).reset_index()
    prompts = [render_prompt(tok, list(p), mode="eval", disable_chat_template=True)
               for p in rows["prompt"]]

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.5,
              max_logprobs=a.vllm_topk, seed=a.seed, enable_prefix_caching=False)
    sp = SamplingParams(temperature=a.temperature, top_p=1.0, top_k=-1, n=a.n,
                        max_tokens=a.max_tokens, logprobs=a.vllm_topk, seed=a.seed)
    outs = llm.generate(prompts, sp)

    rollouts = []
    for i, o in enumerate(outs):
        p_ids = list(o.prompt_token_ids)
        for c in o.outputs:
            rollouts.append(dict(
                prompt_idx=int(rows["index"][i]), prompt_ids=p_ids,
                response_ids=list(c.token_ids), text=c.text,
                finish_reason=c.finish_reason,
                vllm_topk=[{int(t): float(lp.logprob) for t, lp in d.items()}
                           for d in c.logprobs]))
    del llm
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    top_probs, entropy, rank, pos, seq_id, vdiff = [], [], [], [], [], []
    with torch.no_grad():
        for s, r in enumerate(rollouts):
            ids = torch.tensor([r["prompt_ids"] + r["response_ids"]], device="cuda")
            L, R = len(r["prompt_ids"]), len(r["response_ids"])
            if R == 0:
                continue
            logits = model(ids).logits[0, L - 1:L - 1 + R].float()  # predicts response tokens
            logp = torch.log_softmax(logits, -1)
            p = logp.exp()
            sp_, si = p.sort(-1, descending=True)
            top_probs.append(sp_[:, :a.keep_rank].cpu().numpy().astype(np.float32))
            entropy.append((-(p * logp).nan_to_num().sum(-1)).cpu().numpy())
            resp = torch.tensor(r["response_ids"], device="cuda")
            # rank of sampled token (0 = argmax)
            samp_lp = logp.gather(1, resp[:, None])
            rank.append((logp > samp_lp).sum(-1).cpu().numpy())
            pos.append(np.arange(R)); seq_id.append(np.full(R, s))
            # vLLM vs HF on the vLLM-reported top-k tokens
            for t, d in enumerate(r["vllm_topk"]):
                tids = torch.tensor(list(d.keys()), device="cuda")
                hf = logp[t, tids].cpu().numpy()
                vdiff.append(np.abs(hf - np.array(list(d.values()))).max())

    tp = np.concatenate(top_probs); ent = np.concatenate(entropy)
    rk = np.concatenate(rank); ps = np.concatenate(pos); sid = np.concatenate(seq_id)
    np.savez_compressed(os.path.join(a.out_dir, "sanity.npz"), top_probs=tp, entropy=ent,
                        rank=rk, pos=ps, seq_id=sid, vllm_hf_maxdiff=np.array(vdiff))

    cum = np.cumsum(tp, -1)
    summ = dict(model=a.model, n_prompts=a.n_prompts, n=a.n, max_tokens=a.max_tokens,
                temperature=a.temperature, n_rollouts=len(rollouts), n_positions=int(len(ent)),
                resp_len_mean=float(np.mean([len(r["response_ids"]) for r in rollouts])),
                frac_truncated=float(np.mean([r["finish_reason"] == "length" for r in rollouts])),
                frac_has_answer=float(np.mean(["</answer>" in r["text"] for r in rollouts])),
                entropy_mean=float(ent.mean()), entropy_median=float(np.median(ent)),
                vllm_hf_maxdiff_median=float(np.median(vdiff)),
                vllm_hf_maxdiff_p99=float(np.percentile(vdiff, 99)))
    for k in KS:
        tail = 1 - cum[:, k - 1]
        summ[f"tail_mass_beyond_top{k}"] = dict(
            mean=float(tail.mean()), median=float(np.median(tail)),
            p90=float(np.percentile(tail, 90)), p99=float(np.percentile(tail, 99)),
            max=float(tail.max()))
        summ[f"frac_sampled_outside_top{k}"] = float((rk >= k).mean())
    json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.out_dir, "rollouts.jsonl"), "w") as f:
        for r in rollouts:
            r = dict(r); r.pop("vllm_topk")
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
