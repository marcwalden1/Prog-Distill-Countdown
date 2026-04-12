"""
Generate SFT training data from a teacher model checkpoint.

Runs vLLM inference on the training set, filters to only correct solutions
(score == 1.0), and saves a parquet with 'prompt' and 'response' columns
compatible with verl's SFTDataset.

Usage:
    python3 scripts/generate_sft_data.py \
        --teacher_checkpoint_path /path/to/merged/checkpoint \
        --output_path /path/to/output.parquet \
        [--data_path data/balanced/train.parquet] \
        [--n_responses 16] \
        [--max_length 1024]
"""

import argparse
import os
import sys

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# grader_utils.py is in the repo root; add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader_utils import compute_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint_path", required=True,
                        help="Path to the merged teacher checkpoint (contains model.safetensors)")
    parser.add_argument("--output_path", required=True,
                        help="Where to save the output parquet file")
    parser.add_argument("--data_path", default="data/balanced/train.parquet",
                        help="Training data parquet to run inference on")
    parser.add_argument("--n_responses", type=int, default=16,
                        help="Number of responses to sample per prompt")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max new tokens per response")
    parser.add_argument("--max_prompts", type=int, default=10000,
                        help="Max number of prompts to run inference on (random sample). "
                             "Set to 0 to use all prompts (slow for large datasets).")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"  {len(df)} total prompts")

    if args.max_prompts and args.max_prompts < len(df):
        df = df.sample(n=args.max_prompts, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} prompts (--max_prompts {args.max_prompts})")

    # Extract plain-string prompts.
    # The 'prompt' column is a list of dicts: [{"role": "user", "content": "..."}]
    # SFTDataset expects a plain string — it wraps it in the chat template itself.
    prompt_strings = []
    for row in df["prompt"]:
        # row is a list of message dicts; take the last user message content
        content = row[-1]["content"] if isinstance(row[-1], dict) else row[-1]
        prompt_strings.append(str(content))

    ground_truths = df["reward_model"].apply(lambda x: x["ground_truth"]).tolist()
    extra_infos = df["extra_info"].tolist()

    print(f"Running vLLM inference with n={args.n_responses}, max_tokens={args.max_length}")
    llm = LLM(
        model=args.teacher_checkpoint_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    # Build chat-formatted prompts for vLLM (same format as GRPO training)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_checkpoint_path)

    formatted_prompts = []
    for p in prompt_strings:
        chat = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        # Append the partial assistant turn that training data always starts with
        text += "Let me solve this step by step.\n<think>"
        formatted_prompts.append(text)

    sampling_params = SamplingParams(
        n=args.n_responses,
        temperature=1.0,
        max_tokens=args.max_length,
        stop=["</s>", tokenizer.eos_token] if tokenizer.eos_token else ["</s>"],
    )

    print("Generating responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Filter to correct solutions only
    sft_prompts = []
    sft_responses = []
    n_with_correct = 0
    n_total_correct = 0

    for i, output in enumerate(outputs):
        ground_truth = ground_truths[i]
        extra_info = extra_infos[i]
        prompt_str = prompt_strings[i]

        found_correct = False
        for completion in output.outputs:
            # Reconstruct the full model output (the partial prefix we added + the generated text)
            full_response = "Let me solve this step by step.\n<think>" + completion.text

            score = compute_score(
                data_source=None,
                solution_str=full_response,
                ground_truth=ground_truth,
                extra_info=extra_info,
                verbose=False,
            )
            if score == 1.0:
                sft_prompts.append(prompt_str)
                sft_responses.append(full_response)
                n_total_correct += 1
                found_correct = True

        if found_correct:
            n_with_correct += 1

    print(f"\nCoverage: {n_with_correct}/{len(df)} prompts have ≥1 correct solution "
          f"({100*n_with_correct/len(df):.1f}%)")
    print(f"Total correct (prompt, response) pairs: {n_total_correct}")

    if n_total_correct == 0:
        print("WARNING: No correct solutions found — SFT data will be empty!")

    out_df = pd.DataFrame({"prompt": sft_prompts, "response": sft_responses})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)
    print(f"Saved {len(out_df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
