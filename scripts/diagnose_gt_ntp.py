"""Checkpoint diagnosis: NTP loss on the ground-truth answer + predictive-distribution
drift across GRPO training.

For each checkpoint (merged HF dir under <ckpt-root>/global_step_<N>; step 0 = the
base model) and each test dataset, teacher-forces the oracle continuation

    </think>\n<answer> {GT equation} </answer>

after the byte-exact eval prompt (prompt_utils.render_prompt, mode="eval" — same
ChatML rendering vLLM sees in eval.py). The GT equation is reconstructed by
substituting extra_info.numbers *in order* into canonical_pattern (verified exact
on all 5617 balanced/balanced5 test rows).

Per (step, dataset) row written to the output CSV:
  ce_full_macro/micro   — mean per-token CE over the whole continuation
                          (macro = per-prompt mean of means; micro = global token mean)
  ce_answer_macro       — CE restricted to the equation tokens inside <answer>...</answer>
  ce_answer_p50/p90     — per-prompt distribution of the answer CE
  entropy_macro         — mean predictive entropy (nats) at continuation positions
  top1_macro            — mean top-1 probability at continuation positions
  kl_base_macro         — mean full-vocab KL(ckpt || base) at continuation positions
  stop_prob_macro       — p(<|endoftext|>) + p(<|im_end|>) at the position right
                          after " </answer>" (the anti-trailing-junk signal)

Caveat for interpretation: Countdown puzzles can have many valid solutions; the
canonical equation is *one* of them, so ce_answer tracks concentration on GT-like
solutions, not correctness per se.

Restart-safe: existing (step, dataset) rows in --out are skipped.

Usage (see scripts/diagnose_gt_ntp.sh):
    python3 scripts/diagnose_gt_ntp.py \
        --ckpt-root $CHECKPOINT_DIR/checkpoints/Qwen2.5-0.5B/<EXP_NAME> \
        --base-model $MODEL_DIR/Qwen2.5-0.5B \
        --steps 0,150,300,...,3250 \
        --datasets balanced=997,balanced5=500 \
        --out analysis_data/gt_ntp_diagnosis_<tag>.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_utils import render_prompt

STOP_TOKEN_STRS = ["<|endoftext|>", "<|im_end|>"]


def build_gt_equation(pattern, nums):
    """Substitute numbers (in order) into the canonical pattern; space like the
    prompt's few-shot example, e.g. '(A-B)*C' -> '(53 - 22) * 3'."""
    eq = pattern
    for letter, n in zip("ABCDEF", nums):
        eq = eq.replace(letter, str(n))
    return re.sub(r"\s*([+\-*/])\s*", r" \1 ", eq)


def load_examples(data_dir, dataset, n_prompts, tokenizer, seed=0):
    df = pd.read_parquet(os.path.join(data_dir, dataset, "test.parquet"))
    if n_prompts and n_prompts < len(df):
        df = df.sample(n=n_prompts, random_state=seed).reset_index(drop=True)
    examples = []
    for _, row in df.iterrows():
        prompt_text = render_prompt(tokenizer, list(row["prompt"]), mode="eval")
        eq = build_gt_equation(row["extra_info"]["canonical_pattern"],
                               list(row["extra_info"]["numbers"]))
        pre, post = "</think>\n<answer> ", " </answer>"
        cont_text = pre + eq + post
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        enc = tokenizer(cont_text, add_special_tokens=False, return_offsets_mapping=True)
        c_ids = enc["input_ids"]
        # equation-token mask via char offsets into cont_text
        eq_lo, eq_hi = len(pre), len(pre) + len(eq)
        ans_mask = [(lo < eq_hi and hi > eq_lo) for lo, hi in enc["offset_mapping"]]
        examples.append({"p_ids": p_ids, "c_ids": c_ids, "ans_mask": ans_mask})
    return examples


@torch.no_grad()
def logits_at_continuation(model, batch, device):
    """Forward a batch; return (per-sample list of [n_cont, V] fp32 logits at
    continuation positions, [B, V] fp32 logits at the position after </answer>)."""
    seqs = [ex["p_ids"] + ex["c_ids"] for ex in batch]
    maxlen = max(len(s) for s in seqs)
    pad_id = model.config.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
    cont_logits, stop_logits = [], []
    for i, ex in enumerate(batch):
        lp, lc = len(ex["p_ids"]), len(ex["c_ids"])
        # token at position t is predicted from logits at t-1
        cont_logits.append(out.logits[i, lp - 1 : lp + lc - 1].float())
        stop_logits.append(out.logits[i, lp + lc - 1].float())
    return cont_logits, torch.stack(stop_logits)


def diagnose_model(model, base_model, examples, stop_ids, device, batch_size):
    per_prompt = {k: [] for k in
                  ["ce_full", "ce_answer", "entropy", "top1", "kl_base", "stop_prob"]}
    tok_ce_sum = tok_count = 0.0
    for lo in range(0, len(examples), batch_size):
        batch = examples[lo : lo + batch_size]
        cont_logits, stop_logits = logits_at_continuation(model, batch, device)
        base_cont, _ = logits_at_continuation(base_model, batch, device)
        stop_probs = F.softmax(stop_logits, dim=-1)[:, stop_ids].sum(-1)
        for i, ex in enumerate(batch):
            logits = cont_logits[i]                        # [n_cont, V]
            targets = torch.tensor(ex["c_ids"], device=device)
            logp = F.log_softmax(logits, dim=-1)
            ce = -logp.gather(1, targets[:, None]).squeeze(1)          # [n_cont]
            p = logp.exp()
            ent = -(p * logp).sum(-1)
            base_logp = F.log_softmax(base_cont[i], dim=-1)
            kl = (p * (logp - base_logp)).sum(-1)
            amask = torch.tensor(ex["ans_mask"], device=device)
            per_prompt["ce_full"].append(ce.mean().item())
            per_prompt["ce_answer"].append(ce[amask].mean().item())
            per_prompt["entropy"].append(ent.mean().item())
            per_prompt["top1"].append(p.max(-1).values.mean().item())
            per_prompt["kl_base"].append(kl.mean().item())
            per_prompt["stop_prob"].append(stop_probs[i].item())
            tok_ce_sum += ce.sum().item()
            tok_count += len(ex["c_ids"])
    row = {f"{k}_macro": float(np.mean(v)) for k, v in per_prompt.items()}
    row["ce_full_micro"] = tok_ce_sum / tok_count
    row["ce_answer_p50"] = float(np.percentile(per_prompt["ce_answer"], 50))
    row["ce_answer_p90"] = float(np.percentile(per_prompt["ce_answer"], 90))
    row["n_prompts"] = len(examples)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated; 0 = base model")
    ap.add_argument("--datasets", default="balanced=997,balanced5=500",
                    help="name=n_prompts[,name=n_prompts...]; n=0 keeps all")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    steps = [int(s) for s in args.steps.split(",")]
    datasets = [(kv.split("=")[0], int(kv.split("=")[1])) for kv in args.datasets.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKEN_STRS]
    assert all(i is not None and i >= 0 for i in stop_ids), stop_ids

    print("Tokenizing datasets...", flush=True)
    examples = {name: load_examples(args.data_dir, name, n, tokenizer, args.seed)
                for name, n in datasets}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16).to(device).eval()

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["step"], prev["dataset"]))
        print(f"Resuming: {len(done)} (step, dataset) rows already in {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for step in steps:
        todo = [(name, exs) for name, exs in examples.items() if (step, name) not in done]
        if not todo:
            continue
        if step == 0:
            model = base_model
        else:
            path = os.path.join(args.ckpt_root, f"global_step_{step}")
            if not os.path.exists(os.path.join(path, "model.safetensors")):
                print(f"SKIP step {step}: no merged model.safetensors at {path}", flush=True)
                continue
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16).to(device).eval()
        for name, exs in todo:
            row = {"step": step, "dataset": name}
            row.update(diagnose_model(model, base_model, exs, stop_ids, device, args.batch_size))
            pd.DataFrame([row]).to_csv(args.out, mode="a", index=False,
                                       header=not os.path.exists(args.out) or os.path.getsize(args.out) == 0)
            print(f"step {step:>5} {name:<10} " +
                  " ".join(f"{k}={v:.4f}" for k, v in row.items()
                           if k not in ("step", "dataset", "n_prompts")), flush=True)
        if step != 0:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
