"""Build a small from-scratch Qwen2-style model with pretrained embeddings.

Creates a randomly initialized Qwen2ForCausalLM whose config is inherited from
a donor checkpoint (Qwen2.5-0.5B) except for depth / FFN width, then copies the
donor's input embedding table verbatim (lm_head follows via weight tying).
Everything else — tokenizer, RoPE, attention layout (14 heads / 2 KV heads,
head_dim 64), vocab — matches the donor, so the result is a stock architecture
that HF transformers, verl, and vLLM all load without custom code.

Presets (~40M non-embedding params at hidden_size 896):
    wide: 5 layers,  intermediate_size 2432  (half of donor's 4864)  -> ~41.9M body
    deep: 8 layers,  intermediate_size 1216  (quarter of donor's)    -> ~40.9M body

Usage:
    python3 scripts/make_scratch_model.py \
        --donor /path/to/models/Qwen2.5-0.5B \
        --preset wide \
        --out /path/to/models/Qwen2.5-40M-wide \
        --seed 1
"""

import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PRESETS = {
    "wide": {"num_hidden_layers": 5, "intermediate_size": 2432},
    "deep": {"num_hidden_layers": 8, "intermediate_size": 1216},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", required=True, help="Path to donor model dir (Qwen2.5-0.5B)")
    parser.add_argument("--out", required=True, help="Output model dir")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--num-layers", type=int, default=None, help="Override num_hidden_layers")
    parser.add_argument("--intermediate-size", type=int, default=None, help="Override intermediate_size")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    overrides = dict(PRESETS[args.preset]) if args.preset else {}
    if args.num_layers is not None:
        overrides["num_hidden_layers"] = args.num_layers
    if args.intermediate_size is not None:
        overrides["intermediate_size"] = args.intermediate_size
    if not overrides:
        parser.error("Provide --preset and/or explicit --num-layers/--intermediate-size")

    config = AutoConfig.from_pretrained(args.donor)
    assert config.model_type == "qwen2", f"Expected a Qwen2 donor, got {config.model_type}"
    assert config.tie_word_embeddings, "Builder assumes tied embeddings (donor has tie_word_embeddings=true)"
    for key, value in overrides.items():
        setattr(config, key, value)
    # Inert while use_sliding_window=false, but keep it consistent with depth.
    config.max_window_layers = config.num_hidden_layers

    print(f"Config overrides: {overrides}")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(config)  # random init, initializer_range from donor

    donor = AutoModelForCausalLM.from_pretrained(args.donor, torch_dtype=torch.float32)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(donor.model.embed_tokens.weight)

    # Verify the copy and that lm_head is tied to it.
    assert torch.equal(model.model.embed_tokens.weight, donor.model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), \
        "lm_head is not tied to embed_tokens"

    n_total = sum(p.numel() for p in model.parameters())  # tied weights counted once
    n_embed = model.model.embed_tokens.weight.numel()
    n_body = n_total - n_embed
    print(f"Params: total={n_total/1e6:.1f}M  embedding={n_embed/1e6:.1f}M  body={n_body/1e6:.1f}M")
    assert 30e6 < n_body < 55e6, f"Body params {n_body/1e6:.1f}M outside the ~40M target range"

    model = model.to(torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.donor).save_pretrained(args.out)
    GenerationConfig.from_pretrained(args.donor).save_pretrained(args.out)

    # Record provenance next to the weights.
    with open(os.path.join(args.out, "scratch_init.json"), "w") as f:
        json.dump(
            {
                "donor": os.path.abspath(args.donor),
                "seed": args.seed,
                "overrides": overrides,
                "embeddings": "copied from donor (trainable, tied lm_head)",
                "body": "random init (initializer_range from donor config)",
                "params_total": n_total,
                "params_embedding": n_embed,
                "params_body": n_body,
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
