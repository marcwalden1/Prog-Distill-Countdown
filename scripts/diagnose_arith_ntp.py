"""Teacher-forced arithmetic diagnosis across base -> SFT -> GRPO checkpoints.

Companion to scripts/diagnose_gt_ntp.py, aimed at *arithmetic evaluation* rather
than at the choice of equation. It exists because the behavioral probe
(scripts/probe_arith_fidelity.py) can only score the arithmetic a model chooses
to show, and base models cannot emit the `search` trace format without few-shot
exemplars — a different conditioning than the SFT/GRPO checkpoints. Teacher
forcing scores base, SFT and every GRPO step on byte-identical continuations.

For each puzzle we render the exact `search`-format response that GT-SFT trains
on (sft_formats.render_response, so the text is byte-identical to the SFT
target) and score it after the byte-exact eval prompt (prompt_utils.render_prompt,
mode="eval"). Char spans are recovered from the rendered text by regex, so the
masks cannot drift from the format definition.

Scored spans, kept separate because they measure different things:

  ce_reject_value  CE on the TRUE value of a rejected attempt
                   ("Try <wrong expr> = <true value>; too high").
                   The clean arithmetic probe: the value is uniquely determined
                   and the expression is fixed by the probe, not model-chosen.
  ce_final_value   CE on the final value, which is always the target. Mostly
                   measures the format prior that a trace ends at the target,
                   NOT arithmetic — reported for contrast, not as the metric.
  ce_verdict       CE on the "too high"/"too low" phrase. Teacher-forced form of
                   the trace-coherence measure (the value is given, so this is
                   purely "can the model compare it to the target").

Plus, at the LAST rejected value slot, the same prefix is scored twice — once
forcing the true value, once forcing the target — giving

  logp_true_slot / logp_target_slot   summed logprob of the whole number string
                                      (numbers are multi-token in OLMo's BPE:
                                      '-13' and '1000' are 2 tokens each)
  p_true_norm = sigmoid(logp_true - logp_target)

which is invariant to global entropy changes, and so survives the sharpening
that makes raw CE curves hard to read across an RL run (see
research_logs/2026-08-06_gt-ntp-distribution-diagnosis-kl3e3-2ep-seed2.md:
entropy 1.31 -> 0.40 nats in the first 150 steps).

Drift controls at the same positions: entropy, top-1 prob, KL(ckpt || base).

Restart-safe: existing (stage, dataset) rows in --out are skipped.

Usage (see scripts/diagnose_arith_ntp.sh):
    python3 scripts/diagnose_arith_ntp.py \
        --base-model $MODEL_DIR/OLMo-2-0425-1B-pt4001B \
        --stage base=$MODEL_DIR/OLMo-2-0425-1B-pt4001B \
        --stage sft=$CHECKPOINT_DIR/sft-checkpoints/<M>/<sft-exp> \
        --ckpt-root $CHECKPOINT_DIR/checkpoints/<M>/<grpo-exp> \
        --steps 100,200,...,1600 \
        --datasets balanced-simple=997 \
        --out analysis_data/arith_ntp_<tag>.csv
"""

import argparse
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_utils import render_prompt  # noqa: E402
from sft_formats import render_response  # noqa: E402

# Same line shape as scripts/probe_arith_fidelity.py, but with spans.
TRY_LINE = re.compile(
    r"Try\s+(.+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*;\s*(too high|too low|that is the target)")
FINAL_VERDICT = "that is the target"
N_ATTEMPTS = 3          # fixed (not the (1,3) SFT range) so every probe row has
                        # the same shape and >=1 rejected attempt


def span_mask(offsets, spans):
    """Token mask for tokens overlapping any (lo, hi) char span."""
    return [any(lo < hi_s and hi > lo_s for lo_s, hi_s in spans)
            for lo, hi in offsets]


def build_examples(data_dir, dataset, n_prompts, tokenizer, seed=0):
    """Fixed, checkpoint-independent probe set."""
    df = pd.read_parquet(os.path.join(data_dir, dataset, "test.parquet"))
    if n_prompts and n_prompts < len(df):
        df = df.sample(n=n_prompts, random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    examples, skipped = [], 0
    for _, row in df.iterrows():
        ei = row["extra_info"]
        nums, target = list(ei["numbers"]), int(row["reward_model"]["ground_truth"])
        prompt_text = render_prompt(tokenizer, list(row["prompt"]), mode="eval")
        cont = render_response("search", prompt_text, nums, target,
                               ei["canonical_pattern"], rng=rng,
                               n_attempts=N_ATTEMPTS)

        matches = list(TRY_LINE.finditer(cont))
        rejected = [m for m in matches if m.group(3) != FINAL_VERDICT]
        final = [m for m in matches if m.group(3) == FINAL_VERDICT]
        if not rejected or not final:
            skipped += 1          # puzzle with too few legal distractors
            continue

        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        enc = tokenizer(cont, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]

        # The A/B slot: last rejected attempt's value. Everything before it is
        # shared by both variants, so the two scores share an identical prefix.
        slot = rejected[-1].span(2)
        pre_text = cont[:slot[0]]
        pre_ids = tokenizer(pre_text, add_special_tokens=False)["input_ids"]
        true_ids = tokenizer(cont[slot[0]:slot[1]], add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(str(target), add_special_tokens=False)["input_ids"]

        examples.append({
            "p_ids": p_ids, "c_ids": enc["input_ids"],
            "m_reject": span_mask(offs, [m.span(2) for m in rejected]),
            "m_final": span_mask(offs, [m.span(2) for m in final]),
            "m_verdict": span_mask(offs, [m.span(3) for m in rejected]),
            "ab_prefix": p_ids + pre_ids, "ab_true": true_ids, "ab_tgt": tgt_ids,
        })
    if skipped:
        print(f"  {dataset}: skipped {skipped} puzzles with <1 usable distractor", flush=True)
    return examples


def _pad_forward(model, seqs, pad_id, device):
    maxlen = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits


@torch.no_grad()
def score_batch(model, base_model, batch, pad_id, device):
    """Full-trace metrics + the A/B slot logprobs for one batch."""
    out = {k: [] for k in ("ce_full", "ce_reject_value", "ce_final_value",
                           "ce_verdict", "entropy", "top1", "kl_base",
                           "logp_true_slot", "logp_target_slot")}

    # --- pass 1: full trace, true values everywhere ---------------------------
    logits = _pad_forward(model, [ex["p_ids"] + ex["c_ids"] for ex in batch], pad_id, device)
    base_logits = _pad_forward(base_model, [ex["p_ids"] + ex["c_ids"] for ex in batch],
                               pad_id, device)
    for i, ex in enumerate(batch):
        lp, lc = len(ex["p_ids"]), len(ex["c_ids"])
        lg = logits[i, lp - 1: lp + lc - 1].float()          # predicts c_ids[0..lc-1]
        tgt = torch.tensor(ex["c_ids"], device=device)
        logprob = F.log_softmax(lg, dim=-1)
        ce = -logprob.gather(1, tgt[:, None]).squeeze(1)
        p = logprob.exp()
        base_lp = F.log_softmax(base_logits[i, lp - 1: lp + lc - 1].float(), dim=-1)
        out["ce_full"].append(ce.mean().item())
        out["entropy"].append(-(p * logprob).sum(-1).mean().item())
        out["top1"].append(p.max(-1).values.mean().item())
        out["kl_base"].append((p * (logprob - base_lp)).sum(-1).mean().item())
        for name, key in (("m_reject", "ce_reject_value"),
                          ("m_final", "ce_final_value"),
                          ("m_verdict", "ce_verdict")):
            m = torch.tensor(ex[name], device=device)
            out[key].append(ce[m].mean().item() if m.any() else float("nan"))

    # --- pass 2: the A/B slot, same prefix, true value vs target -------------
    seqs, meta = [], []
    for ex in batch:
        for key in ("ab_true", "ab_tgt"):
            seqs.append(ex["ab_prefix"] + ex[key])
            meta.append((len(ex["ab_prefix"]), len(ex[key])))
    ab_logits = _pad_forward(model, seqs, pad_id, device)
    sums = []
    for i, (npre, nval) in enumerate(meta):
        lg = ab_logits[i, npre - 1: npre + nval - 1].float()
        tgt = torch.tensor(seqs[i][npre:], device=device)
        sums.append(F.log_softmax(lg, dim=-1).gather(1, tgt[:, None]).sum().item())
    for j in range(len(batch)):
        out["logp_true_slot"].append(sums[2 * j])
        out["logp_target_slot"].append(sums[2 * j + 1])
    return out


def diagnose(model, base_model, examples, pad_id, device, batch_size):
    acc = None
    for lo in range(0, len(examples), batch_size):
        b = score_batch(model, base_model, examples[lo: lo + batch_size], pad_id, device)
        if acc is None:
            acc = {k: [] for k in b}
        for k, v in b.items():
            acc[k].extend(v)
    row = {f"{k}_macro": float(np.nanmean(v)) for k, v in acc.items()
           if not k.startswith("logp_")}
    lt = np.array(acc["logp_true_slot"])
    lg = np.array(acc["logp_target_slot"])
    row["logp_true_slot_macro"] = float(lt.mean())
    row["logp_target_slot_macro"] = float(lg.mean())
    # entropy-invariant: P(true) / (P(true) + P(target)) at an identical prefix
    row["p_true_norm_macro"] = float(np.mean(1.0 / (1.0 + np.exp(-(lt - lg)))))
    row["frac_true_preferred"] = float(np.mean(lt > lg))
    row["ce_reject_p50"] = float(np.nanpercentile(acc["ce_reject_value"], 50))
    row["ce_reject_p90"] = float(np.nanpercentile(acc["ce_reject_value"], 90))
    row["n_prompts"] = len(examples)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="also the KL reference")
    ap.add_argument("--stage", action="append", default=[],
                    help="label=path/to/merged/hf/dir; repeatable, kept in order")
    ap.add_argument("--ckpt-root", help="GRPO run dir holding global_step_*/")
    ap.add_argument("--steps", default="")
    ap.add_argument("--datasets", default="balanced-simple=997",
                    help="name=n_prompts[,...]; n=0 keeps all")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda",
                    help="cpu is usable for a small --datasets dry run")
    args = ap.parse_args()

    device = args.device
    stages = [(s.split("=", 1)[0], None, s.split("=", 1)[1]) for s in args.stage]
    if args.ckpt_root and args.steps:
        for st in [int(x) for x in args.steps.split(",")]:
            stages.append((f"grpo-{st}", st, os.path.join(args.ckpt_root, f"global_step_{st}")))
    datasets = [(kv.split("=")[0], int(kv.split("=")[1])) for kv in args.datasets.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    print("Building probe sets...", flush=True)
    examples = {name: build_examples(args.data_dir, name, n, tokenizer, args.seed)
                for name, n in datasets}
    for name, exs in examples.items():
        print(f"  {name}: {len(exs)} probe rows", flush=True)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype).to(device).eval()

    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        prev = pd.read_csv(args.out)
        done = set(zip(prev["stage"], prev["dataset"]))
        print(f"Resuming: {len(done)} (stage, dataset) rows already in {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for order, (label, step, path) in enumerate(stages):
        todo = [(n, e) for n, e in examples.items() if (label, n) not in done]
        if not todo:
            continue
        if not os.path.exists(os.path.join(path, "model.safetensors")) and \
           not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
            print(f"SKIP {label}: no merged weights at {path}", flush=True)
            continue
        model = (base_model if os.path.abspath(path) == os.path.abspath(args.base_model)
                 else AutoModelForCausalLM.from_pretrained(
                     path, torch_dtype=dtype).to(device).eval())
        for name, exs in todo:
            row = {"stage": label, "step": step, "order": order, "dataset": name}
            row.update(diagnose(model, base_model, exs, pad_id, device, args.batch_size))
            pd.DataFrame([row]).to_csv(
                args.out, mode="a", index=False,
                header=not os.path.exists(args.out) or os.path.getsize(args.out) == 0)
            print(f"{label:<12} {name:<16} " +
                  " ".join(f"{k.replace('_macro','')}={v:.4f}" for k, v in row.items()
                           if isinstance(v, float)), flush=True)
        if model is not base_model:
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
