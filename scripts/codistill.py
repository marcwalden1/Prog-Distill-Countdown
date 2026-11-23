"""Co-distillation of two causal LMs on one node (no reward, no RL).

Model A (default Qwen3-0.6B-Base) and model B (default Qwen3-4B-Base) are trained
simultaneously so that they converge to a shared policy, while both are anchored
to a frozen copy of B's initial weights (B0).

Per outer step ("batch"):
  1. Take `--prompts_per_batch` prompts; each model samples `--rollouts_per_model`
     responses per prompt (temperature `--temperature`, response cap
     `--max_response_len`, truncated at the cap). Both models train on ALL of them.
  2. Take `--grad_steps_per_gen` optimizer steps on that fixed batch. Each step is a
     full pass over the batch (micro-batched for memory). At the start of each step
     the bf16 copies of A and B are refreshed from the trainable (FSDP) weights; they
     are the frozen targets for that step (Jacobi update: A learns toward B's
     pre-step policy and vice versa).

Per-token loss for model X with partner Y (both full-vocabulary, exact):
    mutual_weight * JSD_beta(P_Y || P_X)  +  anchor_weight(t) * KL(P_B0 || P_X)
JSD_beta(T || S) = beta * KL(T || M) + (1 - beta) * KL(S || M),  M = beta*T + (1-beta)*S
(beta = weight of the target in the mixture; 0.5 = symmetric). Gradients flow only
through P_X. anchor_weight decays linearly from --anchor_weight_init to
--anchor_weight_final over --anchor_decay_steps (default --total_steps) outer steps.

Token weighting: the loss is averaged over tokens WITHIN each rollout set (A's
rollouts, B's rollouts) and the two sets are combined with --set_weight_a /
(1 - set_weight_a), so the longer-responding model does not dominate.

Layout: torchrun, one process per GPU. Trainable models are FSDP2-sharded (fp32
master weights + AdamW, bf16 compute). Each rank also holds unsharded bf16 copies
of A, B and B0 (generation + targets; no communication).

Checkpoints: <ckpt_root>/checkpoints/<model name>/<exp>/global_step_<N>/ as merged
HF dirs (bf16 safetensors), so a later GRPO stage can use them as MODEL_PATH.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from grader_utils import compute_score  # noqa: E402
from prompt_utils import render_prompt  # noqa: E402


# ----------------------------------------------------------------------------- losses
def jsd_loss(lp_s, lp_t, beta):
    """Per-row generalized JSD. lp_s: student log-probs (grad), lp_t: target (no grad). [N, V] fp32."""
    lp_m = torch.logaddexp(lp_t + math.log(beta), lp_s + math.log1p(-beta))
    kl_t = (lp_t.exp() * (lp_t - lp_m)).sum(-1)
    kl_s = (lp_s.exp() * (lp_s - lp_m)).sum(-1)
    return beta * kl_t + (1.0 - beta) * kl_s


def fwd_kl(lp_t, lp_s):
    """KL(T || S) per row."""
    return (lp_t.exp() * (lp_t - lp_s)).sum(-1)


def entropy(lp):
    return -(lp.exp() * lp).sum(-1)


# ----------------------------------------------------------------------------- dist utils
def rank0():
    return dist.get_rank() == 0


def log0(*a):
    if rank0():
        print(*a, flush=True)


def all_reduce_sum(t):
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def local(t):
    from torch.distributed.tensor import DTensor
    return t._local_tensor if isinstance(t, DTensor) else t


def clip_grad_norm(params, max_norm):
    sq = torch.zeros((), device="cuda", dtype=torch.float32)
    for p in params:
        if p.grad is not None:
            sq += local(p.grad).float().pow(2).sum()
    norm = all_reduce_sum(sq).sqrt()
    coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            local(p.grad).mul_(coef)
    return norm.item()


# ----------------------------------------------------------------------------- models
def load_trainable(path, lr, wd, mesh, attn, grad_ckpt, reshard):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation=attn)
    m.config.use_cache = False
    if grad_ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in m.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(m, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    return m, opt


def load_frozen(path, attn):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    m = m.cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def sync_copy(fsdp_model, rep):
    """Refresh the unsharded bf16 copy from the FSDP model (collective: all ranks)."""
    from torch.distributed.tensor import DTensor
    rep_params = dict(rep.named_parameters(remove_duplicate=False))
    n = 0
    for name, p in fsdp_model.named_parameters():
        full = p.full_tensor() if isinstance(p, DTensor) else p
        rep_params[name].copy_(full.to(torch.bfloat16))
        del full
        n += 1
    return n


# ----------------------------------------------------------------------------- batching
def pack(samples, pad_id, device="cuda"):
    """Left-pad prompt+response so every row ends at the same position.

    Returns input_ids, attention_mask, position_ids [m, L], K (max response len),
    and resp_mask [m, K] marking which of the last K positions are response tokens.
    Logits at kept position j (of K+1 kept) predict the token at L-K+j, so the first
    K kept logits align with the last K tokens.
    """
    seqs = [s["prompt_ids"] + s["response_ids"] for s in samples]
    L = max(len(x) for x in seqs)
    K = max(len(s["response_ids"]) for s in samples)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    rmask = torch.zeros((len(seqs), K), dtype=torch.bool)
    for i, (x, s) in enumerate(zip(seqs, samples)):
        ids[i, L - len(x):] = torch.tensor(x)
        att[i, L - len(x):] = 1
        rmask[i, K - len(s["response_ids"]):] = True
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), att.to(device), pos.to(device), K, rmask.to(device)


def token_logits(model, ids, att, pos, K, rmask):
    """Logits (model dtype, bf16) at the response positions, [Ntok, V]."""
    out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=K + 1)
    return out.logits[:, :K][rmask]


def token_logprobs(model, ids, att, pos, K, rmask):
    return F.log_softmax(token_logits(model, ids, att, pos, K, rmask).float(), dim=-1)


def ls(x):
    return F.log_softmax(x.float(), dim=-1)


@torch.no_grad()
def generate(rep, tok, prompts, n, max_new, temperature, gen_micro):
    """prompts: list of token-id lists. Returns list (len(prompts)*n) of (response_ids, truncated)."""
    eos = tok.eos_token_id
    res = []
    for c in range(0, len(prompts), gen_micro):
        chunk = prompts[c:c + gen_micro]
        P = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), P), eos, dtype=torch.long)
        att = torch.zeros((len(chunk), P), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, P - len(p):] = torch.tensor(p)
            att[i, P - len(p):] = 1
        out = rep.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True,
                           temperature=temperature, top_k=0, top_p=1.0, max_new_tokens=max_new,
                           num_return_sequences=n, pad_token_id=eos, eos_token_id=eos, use_cache=True)
        gen = out[:, P:].tolist()
        for g in gen:
            if eos in g:
                res.append((g[:g.index(eos) + 1], False))
            else:
                res.append((g, True))
    return res


# ----------------------------------------------------------------------------- one pass
METRICS = ["jsd_A", "jsd_B", "anc_A", "anc_B", "kl_AB", "kl_BA", "ent_A", "ent_B"]


def fwd_flops(n_body, d_vocab, ids, K):
    """Approx. forward FLOPs of one micro-batch: body on every (padded) position, lm_head on the K+1 kept."""
    return 2.0 * n_body * ids.numel() + 2.0 * d_vocab * ids.shape[0] * (K + 1)


def run_pass(samples, models, cfg, anchor_w, train, norm, perf):
    """One pass over this rank's samples, sorted into length buckets.

    models: dict with 'A','B' (FSDP, trainable), 'repA','repB','anchor' (bf16 copies).
    norm: per-set token weights {0: w_a / N_a_global, 1: w_b / N_b_global} (train only).
    perf: dict accumulating 'flops', 'tokens' (real), 'padded' for MFU.

    Memory: the three target logit tensors stay in bf16 (as the models emit them);
    every fp32 log-softmax and the loss are computed in chunks of --loss_chunk tokens.
    The trainable model's logits are detached into a leaf, the chunked loss is
    back-propagated into the leaf, then the leaf's gradient into the model, so only
    one chunk's fp32 intermediates are alive at a time.

    Diagnostics are computed from the bf16 copies, which equal the trainable weights
    at the start of the pass (synced right before), i.e. they describe the pre-update
    policies of this gradient step.
    Returns per-set sums of METRICS (tensor [2, len(METRICS)]) and token counts [2].
    """
    world = dist.get_world_size()
    sums = torch.zeros((2, len(METRICS)), device="cuda")
    counts = torch.zeros(2, device="cuda")
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]["prompt_ids"]) + len(samples[i]["response_ids"]))
    mb, C, beta = cfg.micro_batch, cfg.loss_chunk, cfg.jsd_beta
    for c0 in range(0, len(order), mb):
        batch = [samples[i] for i in order[c0:c0 + mb]]
        ids, att, pos, K, rmask = pack(batch, cfg.pad_id)
        src = torch.tensor([s["src"] for s in batch], device="cuda").unsqueeze(1).expand(-1, K)[rmask]
        ntok = src.numel()
        with torch.no_grad():
            LA = token_logits(models["repA"], ids, att, pos, K, rmask)
            LB = token_logits(models["repB"], ids, att, pos, K, rmask)
            LN = token_logits(models["anchor"], ids, att, pos, K, rmask)
            for c in range(0, ntok, C):
                lpa, lpb, lpn = ls(LA[c:c + C]), ls(LB[c:c + C]), ls(LN[c:c + C])
                vals = torch.stack([jsd_loss(lpa, lpb, beta), jsd_loss(lpb, lpa, beta),
                                    fwd_kl(lpn, lpa), fwd_kl(lpn, lpb),
                                    fwd_kl(lpa, lpb), fwd_kl(lpb, lpa), entropy(lpa), entropy(lpb)], 1)
                s = src[c:c + C]
                sums.index_add_(0, s, vals)
                counts.index_add_(0, s, torch.ones_like(s, dtype=counts.dtype))
                del lpa, lpb, lpn, vals
        f = fwd_flops(cfg.n_body_a, cfg.dv_a, ids, K) + 2 * fwd_flops(cfg.n_body_b, cfg.dv_b, ids, K)
        if train:
            w_tok = torch.where(src == 0, norm[0], norm[1]) * world  # FSDP averages grads over ranks
            for name, tgt in (("A", LB), ("B", LA)):
                logits = token_logits(models[name], ids, att, pos, K, rmask)
                leaf = logits.detach().requires_grad_(True)
                for c in range(0, ntok, C):
                    lp = ls(leaf[c:c + C])
                    loss = (cfg.mutual_weight * jsd_loss(lp, ls(tgt[c:c + C]), beta)
                            + anchor_w * fwd_kl(ls(LN[c:c + C]), lp))
                    (loss * w_tok[c:c + C]).sum().backward()
                    del lp, loss
                logits.backward(leaf.grad)
                del logits, leaf
            mult_a = 4 if cfg.gc_a else 3  # fwd + 2x bwd (+ recompute under grad checkpointing)
            mult_b = 4 if cfg.gc_b else 3
            f += mult_a * fwd_flops(cfg.n_body_a, cfg.dv_a, ids, K) + mult_b * fwd_flops(cfg.n_body_b, cfg.dv_b, ids, K)
        perf["flops"] += f
        perf["tokens"] += int(att.sum())
        perf["padded"] += ids.numel()
        del LA, LB, LN
    return METRICS, sums, counts


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True, help="small model (trainable)")
    ap.add_argument("--model_b", required=True, help="large model (trainable)")
    ap.add_argument("--anchor", default=None, help="frozen anchor; default = model_b initial weights")
    ap.add_argument("--train_data", default=os.path.join(REPO, "data/balanced-simple/train.parquet"))
    ap.add_argument("--val_data", default=os.path.join(REPO, "data/balanced-simple/test.parquet"))
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--total_steps", type=int, default=1600, help="outer steps (one generation each)")
    ap.add_argument("--prompts_per_batch", type=int, default=256)
    ap.add_argument("--rollouts_per_model", type=int, default=2)
    ap.add_argument("--grad_steps_per_gen", type=int, default=4)
    ap.add_argument("--lr_a", type=float, default=1e-6)
    ap.add_argument("--lr_b", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--jsd_beta", type=float, default=0.5)
    ap.add_argument("--mutual_weight", type=float, default=1.0)
    ap.add_argument("--anchor_weight_init", type=float, default=1.0)
    ap.add_argument("--anchor_weight_final", type=float, default=0.0)
    ap.add_argument("--anchor_decay_steps", type=int, default=0,
                    help="steps over which the anchor weight decays (0 = total_steps); held at "
                         "--anchor_weight_final afterwards, so a run extended past its original "
                         "total_steps keeps its schedule instead of re-stretching it")
    ap.add_argument("--anchor_decay_start", type=int, default=0,
                    help="outer step after which the decay begins: the weight is --anchor_weight_init "
                         "at step start+1 and reaches --anchor_weight_final at step start+decay_steps "
                         "(0 = from step 1, the original behaviour)")
    ap.add_argument("--set_weight_a", type=float, default=0.5, help="weight of A's rollout set; B's = 1 - this")
    ap.add_argument("--max_prompt_len", type=int, default=128)
    ap.add_argument("--max_response_len", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro_batch", type=int, default=16, help="sequences per GPU per forward/backward")
    ap.add_argument("--loss_chunk", type=int, default=1024, help="tokens per fp32 loss chunk (memory only)")
    ap.add_argument("--gc_a", type=int, default=1, help="gradient checkpointing for model A")
    ap.add_argument("--gc_b", type=int, default=1, help="gradient checkpointing for model B")
    ap.add_argument("--reshard_a", type=int, default=1, help="FSDP reshard_after_forward for model A")
    ap.add_argument("--reshard_b", type=int, default=1, help="FSDP reshard_after_forward for model B")
    ap.add_argument("--resume_freq", type=int, default=25, help="outer steps between full-state saves (latest kept)")
    ap.add_argument("--stop_after", type=int, default=0, help="exit (after saving state) once this many steps ran in this job")
    ap.add_argument("--no_final_save", action="store_true", help="benchmarking: skip all checkpoint saves")
    ap.add_argument("--peak_tflops", type=float, default=989.0, help="per-GPU dense bf16 peak for MFU (H100/H200 SXM)")
    ap.add_argument("--gen_micro", type=int, default=64, help="prompts per generate() call per GPU")
    ap.add_argument("--val_freq", type=int, default=50)
    ap.add_argument("--val_prompts", type=int, default=256)
    ap.add_argument("--save_freq", type=int, default=100)
    ap.add_argument("--log_freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--wandb_tags", default="")
    ap.add_argument("--init_a", default="", help="initial weights for A (HF dir; default --model_a). "
                    "--model_a still names the checkpoints and gives the tokenizer")
    ap.add_argument("--init_b", default="", help="initial weights for B (HF dir; default --model_b)")
    ap.add_argument("--start_step", type=int, default=0,
                    help="outer step the init weights correspond to (e.g. 400 when warm-starting from "
                         "another run's global_step_400); training runs steps start_step+1..total_steps "
                         "with the same prompt order and anchor schedule. Optimizer state starts fresh.")
    ap.add_argument("--lr_warmup_updates", type=int, default=0,
                    help="linear LR warmup over this many optimizer updates after --start_step "
                         "(useful with a fresh optimizer); a function of the step, so resume-safe")
    cfg = ap.parse_args()
    assert 0.0 < cfg.jsd_beta < 1.0
    cfg.anchor = cfg.anchor or cfg.model_b  # the BASE model_b, also when --init_b is set
    cfg.init_a = cfg.init_a or cfg.model_a
    cfg.init_b = cfg.init_b or cfg.model_b

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world,))
    assert cfg.prompts_per_batch % world == 0, "prompts_per_batch must divide by #GPUs"
    torch.manual_seed(cfg.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_a)
    tok_b = AutoTokenizer.from_pretrained(cfg.model_b)
    assert tok.get_vocab() == tok_b.get_vocab(), "models must share a tokenizer"
    cfg.pad_id = tok.eos_token_id
    name_a = os.path.basename(cfg.model_a.rstrip("/"))
    name_b = os.path.basename(cfg.model_b.rstrip("/"))

    t0 = time.time()
    A, optA = load_trainable(cfg.init_a, cfg.lr_a, cfg.weight_decay, mesh, cfg.attn, cfg.gc_a, bool(cfg.reshard_a))
    B, optB = load_trainable(cfg.init_b, cfg.lr_b, cfg.weight_decay, mesh, cfg.attn, cfg.gc_b, bool(cfg.reshard_b))
    models = {"A": A, "B": B, "repA": load_frozen(cfg.init_a, cfg.attn),
              "repB": load_frozen(cfg.init_b, cfg.attn), "anchor": load_frozen(cfg.anchor, cfg.attn)}
    for key, rep in (("a", models["repA"]), ("b", models["repB"])):
        emb = rep.get_input_embeddings().weight
        n_all = sum(p.numel() for p in rep.parameters())  # tied lm_head counted once
        setattr(cfg, f"n_body_{key}", n_all - emb.numel())
        setattr(cfg, f"dv_{key}", emb.numel())  # hidden x vocab (lm_head matmul)
    log0(f"[init] models loaded in {time.time() - t0:.0f}s; "
         f"mem alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
         f"body params A {cfg.n_body_a / 1e9:.2f}B B {cfg.n_body_b / 1e9:.2f}B; "
         f"micro_batch {cfg.micro_batch} gc {cfg.gc_a}/{cfg.gc_b} reshard {cfg.reshard_a}/{cfg.reshard_b}")

    # data
    def load_rows(path):
        df = pd.read_parquet(path, columns=["prompt", "reward_model", "extra_info", "data_source"])
        return df
    train_df, val_df = load_rows(cfg.train_data), load_rows(cfg.val_data)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train_df))
    need = cfg.total_steps * cfg.prompts_per_batch
    if need > len(perm):
        perm = np.concatenate([perm] + [rng.permutation(len(train_df)) for _ in range(need // len(perm))])
    val_idx = np.random.default_rng(12345).permutation(len(val_df))[:cfg.val_prompts]
    val_idx = val_idx[: (len(val_idx) // world) * world]

    def make_prompts(df, idxs):
        out = []
        for i in idxs:
            row = df.iloc[int(i)]
            text = render_prompt(tok, list(row["prompt"]), mode="eval", disable_chat_template=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > cfg.max_prompt_len:
                raise ValueError(f"prompt {i} has {len(ids)} tokens > --max_prompt_len {cfg.max_prompt_len}")
            out.append({"ids": ids, "gt": row["reward_model"]["ground_truth"],
                        "ei": row["extra_info"], "ds": row["data_source"]})
        return out

    def rollouts(prompts, n):
        """Generate n rollouts per prompt from each model; returns samples + per-model stats."""
        samples, stats = [], {}
        for src, rep in ((0, models["repA"]), (1, models["repB"])):
            g = generate(rep, tok, [p["ids"] for p in prompts], n, cfg.max_response_len,
                         cfg.temperature, cfg.gen_micro)
            texts = tok.batch_decode([r for r, _ in g], skip_special_tokens=True)
            sc, ln, tr = [], [], []
            for k, ((resp, trunc), txt) in enumerate(zip(g, texts)):
                p = prompts[k // n]
                samples.append({"prompt_ids": p["ids"], "response_ids": resp, "src": src})
                sc.append(float(compute_score(p["ds"], txt, p["gt"], extra_info=p["ei"], verbose=False)))
                ln.append(len(resp))
                tr.append(float(trunc))
            stats[src] = torch.tensor([np.mean([s == 1.0 for s in sc]), np.mean(sc), np.mean(ln), np.mean(tr)],
                                      device="cuda")
        for s in (0, 1):
            stats[s] = all_reduce_sum(stats[s]) / world
        return samples, stats

    def gather_metrics(metrics, sums, counts, prefix):
        all_reduce_sum(sums)
        all_reduce_sum(counts)
        out = {}
        for s, tag in ((0, "onA"), (1, "onB")):
            for k, m in enumerate(metrics):
                out[f"{prefix}/{m}_{tag}"] = (sums[s, k] / counts[s].clamp(min=1)).item()
        return out

    def stat_dict(stats, prefix):
        out = {}
        for s, nm in ((0, "A"), (1, "B")):
            acc, score, ln, tr = stats[s].tolist()
            out.update({f"{prefix}/acc_{nm}": acc, f"{prefix}/score_{nm}": score,
                        f"{prefix}/resp_len_{nm}": ln, f"{prefix}/trunc_frac_{nm}": tr})
        return out

    def validate(step):
        t = time.time()
        sl = val_idx[rank::world]
        samples, stats = rollouts(make_prompts(val_df, sl), 1)
        metrics, sums, counts = run_pass(samples, models, cfg, 0.0, train=False, norm=None,
                                         perf={"flops": 0.0, "tokens": 0, "padded": 0})
        out = gather_metrics(metrics, sums, counts, "val")
        out.update(stat_dict(stats, "val"))
        out["timing/val_s"] = time.time() - t
        return out

    def save(step):
        for fsdp_m, rep_k, nm in ((A, "repA", name_a), (B, "repB", name_b)):
            sync_copy(fsdp_m, models[rep_k])
            d = os.path.join(cfg.ckpt_root, "checkpoints", nm, cfg.exp_name, f"global_step_{step}")
            if rank0():
                os.makedirs(d, exist_ok=True)
                models[rep_k].save_pretrained(d, safe_serialization=True)
                tok.save_pretrained(d)
                with open(os.path.join(d, "codistill_args.json"), "w") as f:
                    json.dump({k: v for k, v in vars(cfg).items()}, f, indent=1)
                with open(os.path.join(os.path.dirname(d), "latest_checkpointed_iteration.txt"), "w") as f:
                    f.write(str(step))
            dist.barrier()
        log0(f"[save] step {step} -> {cfg.ckpt_root}/checkpoints/{{{name_a},{name_b}}}/{cfg.exp_name}/global_step_{step}")

    # ---- full training state (FSDP shards + AdamW), latest only, for exact resume.
    # Sampling is seeded by (seed, step, rank) and the prompt order by seed, so
    # model + optimizer + step is the whole state.
    import shutil
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    state_dir = os.path.join(cfg.ckpt_root, "checkpoints", "codistill_state", cfg.exp_name)
    latest = os.path.join(state_dir, "latest")

    def full_state():
        msA, osA = get_state_dict(A, optA)
        msB, osB = get_state_dict(B, optB)
        return {"A": {"model": msA, "optim": osA}, "B": {"model": msB, "optim": osB}}

    def save_state(step, wandb_id):
        t = time.time()
        tmp = os.path.join(state_dir, "tmp")
        if rank0():
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(state_dir, exist_ok=True)
        dist.barrier()
        dcp.save(full_state(), checkpoint_id=tmp)
        dist.barrier()
        if rank0():
            with open(os.path.join(tmp, "meta.json"), "w") as f:
                json.dump({"step": step, "wandb_id": wandb_id, "args": vars(cfg)}, f, indent=1)
            old = os.path.join(state_dir, "old")
            shutil.rmtree(old, ignore_errors=True)
            if os.path.exists(latest):
                os.rename(latest, old)
            os.rename(tmp, latest)
            shutil.rmtree(old, ignore_errors=True)
        dist.barrier()
        log0(f"[state] saved step {step} -> {latest} in {time.time() - t:.0f}s")

    start_step, wandb_id = cfg.start_step + 1, None
    if os.path.exists(os.path.join(latest, "meta.json")):
        t = time.time()
        with open(os.path.join(latest, "meta.json")) as f:
            meta = json.load(f)
        sd = full_state()
        dcp.load(sd, checkpoint_id=latest)
        set_state_dict(A, optA, model_state_dict=sd["A"]["model"], optim_state_dict=sd["A"]["optim"])
        set_state_dict(B, optB, model_state_dict=sd["B"]["model"], optim_state_dict=sd["B"]["optim"])
        del sd
        start_step, wandb_id = meta["step"] + 1, meta.get("wandb_id")
        log0(f"[resume] loaded step {meta['step']} from {latest} in {time.time() - t:.0f}s")
    if start_step > cfg.total_steps:
        log0(f"[resume] already at step {start_step - 1} >= total_steps {cfg.total_steps}; nothing to do")
        dist.barrier()
        dist.destroy_process_group()
        return

    wb = None
    if rank0() and cfg.wandb_project:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=f"{name_a}+{name_b}-{cfg.exp_name}",
                        config=vars(cfg), tags=[t for t in cfg.wandb_tags.split(",") if t],
                        id=wandb_id, resume="allow" if wandb_id else None)
        wandb_id = wb.id
    wid = [wandb_id]
    dist.broadcast_object_list(wid, src=0)
    wandb_id = wid[0]

    def emit(d, step):
        if rank0():
            print(f"step:{step} - " + " - ".join(f"{k}:{v:.5g}" for k, v in sorted(d.items())) + " - ", flush=True)
            if wb is not None:
                wb.log(d, step=step)

    if cfg.val_freq > 0 and start_step == cfg.start_step + 1:  # fresh start (not a resume)
        emit(validate(cfg.start_step), cfg.start_step)

    per_rank = cfg.prompts_per_batch // world
    ran = 0
    for step in range(start_step, cfg.total_steps + 1):
        T = time.time()
        decay = cfg.anchor_decay_steps or (cfg.total_steps - cfg.anchor_decay_start)
        frac = min(max(step - 1 - cfg.anchor_decay_start, 0) / max(decay - 1, 1), 1.0)
        anchor_w = cfg.anchor_weight_init + (cfg.anchor_weight_final - cfg.anchor_weight_init) * frac
        lo = (step - 1) * cfg.prompts_per_batch + rank * per_rank
        prompts = make_prompts(train_df, perm[lo:lo + per_rank])
        timing = {}
        log = {"train/anchor_weight": anchor_w}
        for k in range(cfg.grad_steps_per_gen):
            t = time.time()
            sync_copy(A, models["repA"])
            sync_copy(B, models["repB"])
            timing["sync_s"] = timing.get("sync_s", 0) + time.time() - t
            if k == 0:
                t = time.time()
                torch.manual_seed(cfg.seed * 1_000_003 + step * world + rank)
                samples, stats = rollouts(prompts, cfg.rollouts_per_model)
                timing["gen_s"] = time.time() - t
                ntok = torch.zeros(2, device="cuda")
                for s in samples:
                    ntok[s["src"]] += len(s["response_ids"])
                all_reduce_sum(ntok)
                norm = {0: cfg.set_weight_a / ntok[0].item(), 1: (1 - cfg.set_weight_a) / ntok[1].item()}
                log.update(stat_dict(stats, "rollout"))
            torch.cuda.synchronize()
            t = time.time()
            perf = {"flops": 0.0, "tokens": 0, "padded": 0}
            metrics, sums, counts = run_pass(samples, models, cfg, anchor_w, train=True, norm=norm, perf=perf)
            gn_a = clip_grad_norm(list(A.parameters()), cfg.grad_clip)
            gn_b = clip_grad_norm(list(B.parameters()), cfg.grad_clip)
            if cfg.lr_warmup_updates > 0:
                n_upd = (step - cfg.start_step - 1) * cfg.grad_steps_per_gen + k + 1
                f = min(1.0, n_upd / cfg.lr_warmup_updates)
                for opt, lr in ((optA, cfg.lr_a), (optB, cfg.lr_b)):
                    for g in opt.param_groups:
                        g["lr"] = lr * f
                log["train/lr_scale"] = f
            optA.step(); optB.step()
            optA.zero_grad(set_to_none=True); optB.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            timing["update_s"] = timing.get("update_s", 0) + dt
            pf = all_reduce_sum(torch.tensor([perf["flops"], perf["tokens"], perf["padded"]],
                                             device="cuda", dtype=torch.float64))
            log[f"perf/mfu_k{k}"] = pf[0].item() / (dt * world * cfg.peak_tflops * 1e12)
            log[f"perf/pad_frac_k{k}"] = 1.0 - pf[1].item() / pf[2].item()
            m = gather_metrics(metrics, sums, counts, f"train_k{k}")
            if k == 0 or k == cfg.grad_steps_per_gen - 1:
                log.update(m)
            log[f"train_k{k}/grad_norm_A"] = gn_a
            log[f"train_k{k}/grad_norm_B"] = gn_b
        log.update({f"timing/{k}": v for k, v in timing.items()})
        log["timing/step_s"] = time.time() - T
        log["perf/max_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if cfg.val_freq > 0 and (step % cfg.val_freq == 0 or step == cfg.total_steps):
            log.update(validate(step))
        if step % cfg.log_freq == 0 or "val/acc_A" in log or step == cfg.total_steps:
            emit(log, step)
        ran += 1
        stopping = cfg.stop_after > 0 and ran >= cfg.stop_after
        if not cfg.no_final_save:
            if (cfg.save_freq > 0 and step % cfg.save_freq == 0) or step == cfg.total_steps:
                save(step)
            if (cfg.resume_freq > 0 and step % cfg.resume_freq == 0) or step == cfg.total_steps or stopping:
                save_state(step, wandb_id)
        if stopping:
            log0(f"[stop] --stop_after {cfg.stop_after} reached at step {step}")
            break

    if wb is not None:
        wb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
