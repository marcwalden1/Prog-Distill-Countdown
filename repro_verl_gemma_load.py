"""
Reproduce verl's exact vLLM weight-load path for gemma-3-270m and identify
which params silently fail to load.

verl path (per fsdp_vllm.py:update_params):
  1. Initialize vLLM with load_format="dummy" -> random init weights
  2. Take HF model state_dict()
  3. Call vllm_model.load_weights(((name, tensor) for name,tensor in state_dict.items()))
  4. AutoWeightsLoader returns set of loaded names

We diff before/after vllm param hashes to know exactly what survived the push.
Then we generate to confirm the diagnosis.
"""
import os
import hashlib
# Force V0 engine so the model lives in this process and we can introspect it
# (V1 spawns a subprocess; verl gets around this by using a custom worker init).
os.environ["VLLM_USE_V1"] = "0"
import torch

STUDENT = "/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/rl-checkpoints/sft-checkpoints/gemma-3-270m/gemma-270m-balanced-distill-teacher-Qwen1.5B-sftlr1e-5-seed1"

PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers [73, 19, 16], create an equation that equals 76. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer>tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""


def fingerprint(t: torch.Tensor) -> str:
    """Cheap content fingerprint via mean/std/shape — distinguishing dummy-init from real."""
    f = t.detach().to(torch.float32).cpu()
    return f"shape={tuple(t.shape)} mean={f.mean().item():+.6e} std={f.std().item():+.6e} sum={f.sum().item():+.6e}"


def main():
    print("=" * 78)
    print("STAGE 1: Build vLLM with load_format=dummy")
    print("=" * 78)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=STUDENT,
        tensor_parallel_size=1,
        dtype="bfloat16",
        load_format="dummy",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        trust_remote_code=True,
        seed=0,
    )

    # Drill down to the underlying nn.Module — V1 engine has a different path
    eng = llm.llm_engine
    vllm_model = None
    for path_attrs in [
        ("engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("engine_core", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
        ("model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("model_executor", "driver_worker", "model_runner", "model"),
    ]:
        obj = eng
        try:
            for a in path_attrs:
                obj = getattr(obj, a)
            vllm_model = obj
            print(f"  Found model via: llm.llm_engine.{'.'.join(path_attrs)}")
            break
        except AttributeError:
            continue
    if vllm_model is None:
        print(f"  llm.llm_engine attrs: {[a for a in dir(eng) if not a.startswith('_')][:30]}")
        raise RuntimeError("could not locate vLLM model")
    print(f"vLLM model class: {type(vllm_model).__name__}")
    print(f"vLLM total params: {sum(1 for _ in vllm_model.named_parameters())}")

    # Snapshot dummy-init fingerprints
    before = {}
    for name, p in vllm_model.named_parameters():
        before[name] = fingerprint(p)
    print(f"Snapshot taken: {len(before)} params")

    print()
    print("=" * 78)
    print("STAGE 2: Load HF gemma3 student and push state_dict")
    print("=" * 78)
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16)
    sd = hf_model.state_dict()
    print(f"HF state_dict size: {len(sd)}")
    print("First 5 HF keys:", list(sd.keys())[:5])

    # Mimic verl's update_params iterator (no DTensor needed since we're not FSDP-wrapped)
    items = list(sd.items())
    loaded = vllm_model.load_weights(((n, t) for n, t in items))
    print(f"\nload_weights returned {len(loaded) if loaded else 0} loaded names")

    print()
    print("=" * 78)
    print("STAGE 3: Diff which vLLM params changed (loaded) vs unchanged (silently dropped)")
    print("=" * 78)
    after = {}
    for name, p in vllm_model.named_parameters():
        after[name] = fingerprint(p)

    unchanged = []
    changed = []
    for name, fp_before in before.items():
        if after[name] == fp_before:
            unchanged.append(name)
        else:
            changed.append(name)

    print(f"Total vLLM params:       {len(before)}")
    print(f"Changed (loaded ok):     {len(changed)}")
    print(f"Unchanged (NOT loaded):  {len(unchanged)}")

    # Categorize unchanged
    print(f"\n=== UNCHANGED params (still at dummy init — silently dropped) ===")
    cats = {}
    for n in unchanged:
        # last component
        last = n.split(".")[-1]
        suffix = ".".join(n.split(".")[-2:])
        cats.setdefault(suffix, []).append(n)
    for suffix in sorted(cats):
        names = cats[suffix]
        print(f"  {suffix}: {len(names)} params  (sample: {names[0]})")

    print(f"\n=== Sample of CHANGED params (first 5) ===")
    for n in changed[:5]:
        print(f"  {n}")

    # Verify: are the unchanged params present in the HF state_dict at all?
    hf_keys = set(sd.keys())
    print(f"\n=== Cross-check: are unchanged vLLM params present (by name) in HF state_dict? ===")
    in_hf = [n for n in unchanged if n in hf_keys]
    not_in_hf = [n for n in unchanged if n not in hf_keys]
    print(f"  unchanged-and-in-HF (HF has it but loader skipped):      {len(in_hf)}  (sample: {in_hf[:3]})")
    print(f"  unchanged-and-NOT-in-HF (HF doesn't have, expected gap): {len(not_in_hf)}  (sample: {not_in_hf[:3]})")

    print()
    print("=" * 78)
    print("STAGE 4a: Generate WITHOUT process_weights_after_loading (mimics broken verl path)")
    print("=" * 78)
    sp = SamplingParams(temperature=0.0, max_tokens=100, n=1)
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    print(f"GENERATED (broken):\n{out[0].outputs[0].text!r}")

    print()
    print("=" * 78)
    print("STAGE 4b: Apply _process_weights_after_loading (the fix) and generate again")
    print("=" * 78)
    from vllm.model_executor.model_loader.loader import _process_weights_after_loading
    model_config = llm.llm_engine.vllm_config.model_config
    target_device = next(vllm_model.parameters()).device
    _process_weights_after_loading(vllm_model, model_config, target_device)

    after2 = {}
    for name, p in vllm_model.named_parameters():
        after2[name] = fingerprint(p)
    changed_by_fix = [n for n in after if after2[n] != after[n]]
    print(f"Params changed by process_weights_after_loading: {len(changed_by_fix)}")
    cats2 = {}
    for n in changed_by_fix:
        suffix = ".".join(n.split(".")[-2:])
        cats2.setdefault(suffix, []).append(n)
    for suffix in sorted(cats2):
        print(f"  {suffix}: {len(cats2[suffix])} params (sample: {cats2[suffix][0]})")

    out2 = llm.generate([PROMPT], sp, use_tqdm=False)
    print(f"\nGENERATED (after fix):\n{out2[0].outputs[0].text!r}")


if __name__ == "__main__":
    main()
