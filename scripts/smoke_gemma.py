"""Quick smoke test for the gemma-3-270m vLLM 'persistent=False' patch.

Two vLLM loads of the *same* base gemma-3-270m, different paths:
  RUN A — load_format=auto (control)
  RUN B — load_format=dummy + push HF state_dict (mimics verl's GRPO)

With the vLLM-side patch (gemma3.py:372 has persistent=False), RUN B's
normalizer buffer survives initialize_dummy_weights and the HF push
restores the real parameter weights. Without it, normalizer is clobbered
to ~0 and generation produces multilingual token salad.

Generation uses the base model only — we expect plausible English
continuations, not countdown-task answers. The signal is:
    (1) RUN A and RUN B both report normalizer ≈ sqrt(hidden_size)=sqrt(640)≈25.30
    (2) RUN A and RUN B both produce coherent text (not random multilingual chars)
"""
import os
os.environ["VLLM_USE_V1"] = "0"
import math
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams

MODEL = "/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models/gemma-3-270m"
PROMPT = "The capital of France is"


def get_norm(llm):
    m = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    for mod in m.modules():
        n = getattr(mod, "normalizer", None)
        if isinstance(n, torch.Tensor) and n.numel() == 1:
            return float(n.detach().float().cpu().item())
    return None


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    expected = math.sqrt(cfg.hidden_size)
    print(f"\n{'=' * 60}")
    print(f"MODEL:         {MODEL}")
    print(f"hidden_size:   {cfg.hidden_size}")
    print(f"expected norm: {expected:.4f}")
    print(f"{'=' * 60}")

    print("\n=== RUN A: load_format=auto (control) ===")
    llm = LLM(model=MODEL, load_format="auto", dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=0.4, max_model_len=512,
              tensor_parallel_size=1, seed=0)
    n_a = get_norm(llm)
    out_a = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=20),
                         use_tqdm=False)
    text_a = out_a[0].outputs[0].text
    print(f"  normalizer: {n_a:.4f}")
    print(f"  generated:  {text_a!r}")
    del llm
    torch.cuda.empty_cache()

    print("\n=== RUN B: load_format=dummy + push HF state_dict (verl path) ===")
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    sd = {k: v.detach().clone() for k, v in hf.state_dict().items()}
    del hf
    llm = LLM(model=MODEL, load_format="dummy", dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=0.4, max_model_len=512,
              tensor_parallel_size=1, seed=0)
    m = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    m.load_weights(((n, t) for n, t in sd.items()))
    n_b = get_norm(llm)
    out_b = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=20),
                         use_tqdm=False)
    text_b = out_b[0].outputs[0].text
    print(f"  normalizer: {n_b:.4f}")
    print(f"  generated:  {text_b!r}")

    print(f"\n{'=' * 60}")
    print("VERDICT")
    ok_a = n_a is not None and abs(n_a - expected) < 1.0
    ok_b = n_b is not None and abs(n_b - expected) < 1.0
    print(f"  RUN A normalizer correct (auto):       {'PASS' if ok_a else 'FAIL'}")
    print(f"  RUN B normalizer correct (dummy+push): {'PASS' if ok_b else 'FAIL'}")
    if ok_a and ok_b:
        print("\nPASS: vLLM persistent=False patch is live; gemma-3-270m is GRPO-ready.")
    else:
        print("\nFAIL: check vLLM gemma3.py:372 for `persistent=False`.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
