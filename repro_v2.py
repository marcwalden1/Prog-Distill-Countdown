"""
Diagnostic v2 — find the real bug.

Compare three setups:
  A. load_format=auto                (works in eval.py — control)
  B. load_format=dummy + push HF sd  (mimics verl — broken)
  C. load_format=auto, then push HF sd on top  (does push corrupt?)

For each, also do a per-layer numeric diff vs HF weights to check if push faithful.
"""
import os
os.environ["VLLM_USE_V1"] = "0"
import torch

STUDENT = "/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints/sft-checkpoints/gemma-3-270m/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-5-seed1"

PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers [73, 19, 16], create an equation that equals 76. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""


def get_model(llm):
    eng = llm.llm_engine
    return eng.model_executor.driver_worker.worker.model_runner.model


def gen(llm, label):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=80, n=1)
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    txt = out[0].outputs[0].text
    print(f"\n=== GEN [{label}] ===\n{txt!r}\n")
    return txt


def diff_to_hf(llm, hf_sd, label):
    """For each vLLM param, compare to the corresponding HF tensor (handling pack mappings)."""
    model = get_model(llm)
    bad = []
    print(f"\n=== Compare vLLM params to HF state_dict [{label}] ===")
    for name, p in model.named_parameters():
        # Direct match?
        if name in hf_sd:
            hf_t = hf_sd[name]
            if p.shape != hf_t.shape:
                bad.append(f"  SHAPE MISMATCH {name}: vllm {tuple(p.shape)} vs hf {tuple(hf_t.shape)}")
                continue
            md = (p.float().cpu() - hf_t.float().cpu()).abs().max().item()
            if md > 1e-4:
                bad.append(f"  DIFF {name}: max_abs={md:.4e}")
            continue
        # Packed: qkv_proj or gate_up_proj
        if name.endswith(".qkv_proj.weight"):
            base = name.replace(".qkv_proj.", ".")
            try:
                q = hf_sd[base.replace(".weight", "") + "q_proj.weight".replace("..", ".")]
            except Exception:
                pass
            qn = name.replace("qkv_proj", "q_proj")
            kn = name.replace("qkv_proj", "k_proj")
            vn = name.replace("qkv_proj", "v_proj")
            if qn in hf_sd and kn in hf_sd and vn in hf_sd:
                q, k, v = hf_sd[qn], hf_sd[kn], hf_sd[vn]
                expected = torch.cat([q, k, v], dim=0)
                if p.shape != expected.shape:
                    bad.append(f"  SHAPE MISMATCH {name}: vllm {tuple(p.shape)} vs cat(qkv) {tuple(expected.shape)}")
                    continue
                md = (p.float().cpu() - expected.float().cpu()).abs().max().item()
                if md > 1e-4:
                    bad.append(f"  PACKED DIFF {name}: max_abs={md:.4e}")
                continue
        if name.endswith(".gate_up_proj.weight"):
            gn = name.replace("gate_up_proj", "gate_proj")
            un = name.replace("gate_up_proj", "up_proj")
            if gn in hf_sd and un in hf_sd:
                g, u = hf_sd[gn], hf_sd[un]
                expected = torch.cat([g, u], dim=0)
                if p.shape != expected.shape:
                    bad.append(f"  SHAPE MISMATCH {name}: vllm {tuple(p.shape)} vs cat(gu) {tuple(expected.shape)}")
                    continue
                md = (p.float().cpu() - expected.float().cpu()).abs().max().item()
                if md > 1e-4:
                    bad.append(f"  PACKED DIFF {name}: max_abs={md:.4e}")
                continue
        # Anything else
        bad.append(f"  UNKNOWN MAPPING for {name}")
    print(f"  total mismatches: {len(bad)}")
    for b in bad[:25]:
        print(b)
    if len(bad) > 25:
        print(f"  ... and {len(bad) - 25} more")


def buffers_summary(llm, label):
    model = get_model(llm)
    print(f"\n=== Buffers [{label}] (only ones with non-trivial values) ===")
    for name, b in model.named_buffers():
        try:
            f = b.detach().float().cpu()
            print(f"  {name:60s}  shape={tuple(b.shape)}  mean={f.mean().item():+.4e}  std={f.std().item():+.4e}")
        except Exception:
            pass


def main():
    from vllm import LLM
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16)
    hf_sd = {k: v.detach().clone() for k, v in hf.state_dict().items()}
    del hf

    print("\n" + "#" * 78)
    print("# RUN A: load_format=auto (control — should work)")
    print("#" * 78)
    llmA = LLM(model=STUDENT, tensor_parallel_size=1, dtype="bfloat16", load_format="auto",
               enforce_eager=True, gpu_memory_utilization=0.4, max_model_len=2048,
               trust_remote_code=True, seed=0)
    diff_to_hf(llmA, hf_sd, "auto-load")
    buffers_summary(llmA, "auto-load")
    gen(llmA, "auto-load")
    del llmA
    torch.cuda.empty_cache()

    print("\n" + "#" * 78)
    print("# RUN B: load_format=dummy + push HF state_dict (mimics verl — likely broken)")
    print("#" * 78)
    llmB = LLM(model=STUDENT, tensor_parallel_size=1, dtype="bfloat16", load_format="dummy",
               enforce_eager=True, gpu_memory_utilization=0.4, max_model_len=2048,
               trust_remote_code=True, seed=0)
    modelB = get_model(llmB)
    loaded = modelB.load_weights(((n, t) for n, t in hf_sd.items()))
    print(f"Pushed: load_weights returned {len(loaded) if loaded else 0} names")
    diff_to_hf(llmB, hf_sd, "dummy+push")
    buffers_summary(llmB, "dummy+push")
    gen(llmB, "dummy+push")
    del llmB
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
