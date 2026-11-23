#!/usr/bin/env python3
"""Held-out KD loss for Qwen3-4B -> 0.6B distilled students, on PROMPTS no candidate trained on.

The SFT runs draw batches from torch DistributedSampler(shuffle=True, seed=SEED, epoch 0)
over the teacher-rollout parquets: a uniform random permutation `perm` of all rows, and a
run of S steps at global batch B trains on exactly perm[:S*B] (rank r takes perm[r::W];
each step consumes B/W consecutive entries per rank, so the union is a prefix of perm).
A prompt is held out when NONE of its rollouts is in that prefix; with --steps-seen 8000
that is ~549 of 418,136 prompts (x10 rollouts). Shorter runs saw a subset of the prefix,
so the set is held out for every candidate up to --steps-seen.

Per candidate it reports, pooled over response tokens (same masking as training):
  kd   = the training objective: bucketed forward KL(teacher || student) over the
         teacher's top-20 + sampled-token bucket + tail (kd_token_loss, verl patch 8)
  ce   = -log q(teacher's sampled token), comparable with NTP's train/loss
Per-prompt sums are stored, so paired differences get a bootstrap SE over prompts.

    python3 scripts/score_kd_heldout.py --model tag=/path/to/merged/hf [--model ...]
    python3 scripts/score_kd_heldout.py --report-only          # table from the JSON
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = ("/n/netscratch/bingbin_lab/Lab/Prog-Distill-Countdown/distill-data/Qwen3-4B-Base/"
        "balanced-simple-T1.0-total512-n10-top20")
OUT = os.path.join(REPO, "analysis_data", "kd_heldout_scores.json")


def heldout_rows(data, steps_seen, global_batch, seed):
    import pyarrow.parquet as pq
    import torch
    files = sorted(glob.glob(os.path.join(data, "shard_*.parquet")))
    keys = []  # (shard, prompt_idx) per global row, in dataset order
    for si, f in enumerate(files):
        pidx = pq.read_table(f, columns=["prompt_idx"]).column(0).to_numpy()
        keys.append(si * 10**9 + pidx.astype(np.int64))
    keys = np.concatenate(keys)
    n = len(keys)
    g = torch.Generator()
    g.manual_seed(seed)  # DistributedSampler: seed + epoch, epoch 0
    perm = torch.randperm(n, generator=g).numpy()
    seen_rows = perm[: steps_seen * global_batch]
    seen_prompts = np.unique(keys[seen_rows])
    rows = np.where(~np.isin(keys, seen_prompts))[0]
    # shuffle sanity: fraction of rows seen, per decile of dataset order (should be flat)
    seen = np.zeros(n, bool)
    seen[seen_rows] = True
    dec = [round(float(x.mean()), 3) for x in np.array_split(seen, 10)]
    return rows, keys[rows], n, dec


def score(tag, path, rows, rkeys, data, batch_size, max_length):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "verl"))
    from distill_dataset import DistillKDDataset
    from verl.trainer.fsdp_sft_trainer import kd_token_loss

    tok = AutoTokenizer.from_pretrained(path)
    ds = DistillKDDataset(data, tok, OmegaConf.create({"max_length": max_length, "truncation": "right"}))
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).cuda().eval()
    kl_s, ce_s, n_s = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            items = [ds[int(i)] for i in rows[b:b + batch_size]]
            bt = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
            with torch.autocast("cuda", dtype=torch.bfloat16):  # as in training
                logits = model(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"],
                               position_ids=bt["position_ids"], use_cache=False).logits
                kd = [bt[x][:, :-1] for x in ("kd_ids", "kd_logprobs", "kd_mask")]
                kl, ce = kd_token_loss(logits[:, :-1], *kd, bt["input_ids"][:, 1:])
            m = bt["loss_mask"][:, :-1].float()
            sl = slice(b, b + len(items))
            kl_s[sl] = (kl.float() * m).sum(1).cpu().numpy()
            ce_s[sl] = (ce.float() * m).sum(1).cpu().numpy()
            n_s[sl] = m.sum(1).cpu().numpy()
            del logits, kl, ce
    # per-prompt sums (prompt order = sorted unique keys)
    uk, inv = np.unique(rkeys, return_inverse=True)
    agg = lambda v: np.bincount(inv, weights=v, minlength=len(uk))
    res = {"path": path, "kd": float(kl_s.sum() / n_s.sum()), "ce": float(ce_s.sum() / n_s.sum()),
           "tokens": int(n_s.sum()), "rollouts": int(len(rows)), "prompts": int(len(uk)),
           "per_prompt": {"kd_sum": agg(kl_s).tolist(), "ce_sum": agg(ce_s).tolist(), "n": agg(n_s).tolist()}}
    print(f"{tag}: kd {res['kd']:.5f}  ce {res['ce']:.5f}  ({res['tokens']} tokens, "
          f"{res['rollouts']} rollouts, {res['prompts']} prompts)", flush=True)
    return res


def report(db):
    tags = sorted(db["models"], key=lambda t: db["models"][t]["kd"])
    if not tags:
        print("no scores")
        return None
    best = tags[0]
    rng = np.random.default_rng(0)
    P = len(db["models"][best]["per_prompt"]["n"])
    boots = rng.integers(0, P, size=(2000, P))
    pb = db["models"][best]["per_prompt"]
    print(f"\nheld-out set: {db['meta']['prompts']} prompts / {db['meta']['rollouts']} rollouts "
          f"(unseen by runs <= {db['meta']['steps_seen']} steps)")
    print(f"{'candidate':24s} {'KD obj':>9s} {'Δ vs best (±SE)':>20s} {'sampled CE':>11s} {'Δ CE (±SE)':>20s}")
    for t in tags:
        r = db["models"][t]
        pp = r["per_prompt"]
        n = np.array(pp["n"])  # identical token counts across models (same rollouts)
        cells = []
        for m in ("kd", "ce"):
            d = np.array(pp[f"{m}_sum"]) - np.array(pb[f"{m}_sum"])
            se = np.std(d[boots].sum(1) / n[boots].sum(1))
            cells.append((r[m], r[m] - db["models"][best][m], se))
        (kv, kd_d, kd_se), (cv, ce_d, ce_se) = cells
        print(f"{t:24s} {kv:9.5f} {kd_d:+11.5f} ±{kd_se:.5f} {cv:11.5f} {ce_d:+11.5f} ±{ce_se:.5f}")
    print(f"BEST: {best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="tag=path to merged HF dir")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--steps-seen", type=int, default=8000)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--best-file", help="write the best tag here")
    args = ap.parse_args()

    db = json.load(open(args.out)) if os.path.exists(args.out) else {"meta": {}, "models": {}}
    if not args.report_only:
        rows, rkeys, n, dec = heldout_rows(args.data, args.steps_seen, args.global_batch, args.seed)
        meta = {"steps_seen": args.steps_seen, "global_batch": args.global_batch, "seed": args.seed,
                "rows_total": n, "rollouts": int(len(rows)), "prompts": int(len(np.unique(rkeys)))}
        print(f"rows {n}; fraction seen per decile of dataset order: {dec}")
        print(f"held-out: {meta['prompts']} prompts, {meta['rollouts']} rollouts")
        if db["meta"] and any(db["meta"].get(k) != v for k, v in meta.items()):
            sys.exit(f"held-out set differs from the one in {args.out}: {db['meta']} vs {meta}")
        db["meta"] = meta
        for spec in args.model:
            tag, path = spec.split("=", 1)
            db["models"][tag] = score(tag, path, rows, rkeys, args.data, args.batch_size, args.max_length)
            json.dump(db, open(args.out, "w"))
    best = report(db)
    if args.best_file and best:
        open(args.best_file, "w").write(best + "\n")


if __name__ == "__main__":
    main()
