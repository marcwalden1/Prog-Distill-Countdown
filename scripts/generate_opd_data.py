"""
Generate offline on-policy-distillation (OPD) training data, following
"Lightning OPD" (arXiv:2604.13010): rollouts are sampled ONCE from the SFT'd
student (the reference policy pi_ref), and the teacher is queried ONCE to
precompute per-token log-probabilities over those rollouts. The OPD trainer
then optimizes the student on this frozen dataset with per-token advantages
A_t = log pi_T(a_t|s_t) - log pi_theta(a_t|s_t) (teacher term read from this
parquet, student term computed live).

Teacher consistency: pass the SAME teacher checkpoint that generated the SFT
data of the student (for progdistill, the final-round teacher).

Runs in two stages (separate processes, so each vLLM instance gets the GPU
to itself — the .sh wrapper calls this script twice):

    --stage rollout   sample n responses/prompt from the student; token IDs
                      are stored verbatim (no detokenize/retokenize drift)
    --stage score     teacher prompt_logprobs pass over the frozen rollouts

Output parquet columns:
    prompt            str          raw user prompt (inspection only)
    prompt_ids        list<int>    chat-template prompt token IDs
    response_ids      list<int>    assistant-prefix + completion + eos IDs
    teacher_logprobs  list<float>  teacher logprob per response_ids token
    response          str          decoded response text (inspection only)
    score             float        grader score of the student rollout
    rollout_logprob   float        student's own cumulative logprob at sample time

Usage (via scripts/generate_opd_data.sh):
    python3 scripts/generate_opd_data.py --stage rollout \
        --student_checkpoint_path /path/to/merged/student \
        --output_path /path/to/opd.parquet [--n_responses 4] [--temperature 1.0]
    python3 scripts/generate_opd_data.py --stage score \
        --teacher_checkpoint_path /path/to/merged/teacher \
        --output_path /path/to/opd.parquet
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader_utils import compute_score
from prompt_utils import ASSISTANT_PARTIAL_TURN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["rollout", "score"])
    parser.add_argument("--output_path", required=True,
                        help="Final OPD parquet path. The rollout stage writes "
                             "<output_path>.rollouts.parquet; the score stage "
                             "reads it and writes the final parquet.")
    parser.add_argument("--student_checkpoint_path", default=None,
                        help="[rollout] Merged HF checkpoint of the SFT student (pi_ref)")
    parser.add_argument("--teacher_checkpoint_path", default=None,
                        help="[score] Merged HF checkpoint of the teacher")
    parser.add_argument("--data_path", default="data/balanced/train.parquet",
                        help="[rollout] Prompt source parquet")
    parser.add_argument("--n_responses", type=int, default=4,
                        help="[rollout] Student samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="[rollout] Student sampling temperature (paper used 0.8; "
                             "1.0 matches the true reference policy)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="[rollout] Max new tokens per response")
    parser.add_argument("--max_prompts", type=int, default=10000,
                        help="[rollout] Max prompts (random sample, seed 42; 0 = all)")
    parser.add_argument("--score_chunk_size", type=int, default=2000,
                        help="[score] Rows per vLLM scoring call")
    return parser.parse_args()


def rollouts_path(output_path):
    return output_path + ".rollouts.parquet"


def strip_trailing_eos(token_ids, eos_id):
    ids = list(token_ids)
    while ids and ids[-1] == eos_id:
        ids.pop()
    return ids


def run_rollout(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    assert args.student_checkpoint_path, "--student_checkpoint_path required for --stage rollout"

    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"  {len(df)} total prompts")
    if args.max_prompts and args.max_prompts < len(df):
        df = df.sample(n=args.max_prompts, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} prompts (--max_prompts {args.max_prompts})")

    prompt_strings = []
    for row in df["prompt"]:
        content = row[-1]["content"] if isinstance(row[-1], dict) else row[-1]
        prompt_strings.append(str(content))
    ground_truths = df["reward_model"].apply(lambda x: x["ground_truth"]).tolist()
    extra_infos = df["extra_info"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.student_checkpoint_path)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "OPD data generation currently supports chat-template tokenizers only "
            "(Qwen family). Base-format models would need the template_type='base' "
            "prompt/response split wired through here."
        )
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "Tokenizer must define an eos token"

    # Same prompt/response token split the OPD dataset will train on:
    # prompt = chat-template-rendered user turn; response starts with the
    # standard assistant partial turn (matches SFTDataset / generate_sft_data).
    prefix_ids = tokenizer.encode(ASSISTANT_PARTIAL_TURN, add_special_tokens=False)
    all_prompt_ids = []
    gen_prompts = []
    for p in prompt_strings:
        chat_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        pids = tokenizer.encode(chat_str, add_special_tokens=False)
        all_prompt_ids.append(pids)
        gen_prompts.append(TokensPrompt(prompt_token_ids=pids + prefix_ids))

    print(f"Sampling rollouts from student: n={args.n_responses}, "
          f"temp={args.temperature}, max_tokens={args.max_length}")
    llm = LLM(
        model=args.student_checkpoint_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        n=args.n_responses,
        temperature=args.temperature,
        max_tokens=args.max_length,
        stop=["</s>", tokenizer.eos_token] if tokenizer.eos_token else ["</s>"],
    )
    outputs = llm.generate(gen_prompts, sampling_params)

    rows = {k: [] for k in ["prompt", "prompt_ids", "response_ids", "response",
                            "score", "rollout_logprob"]}
    n_correct = n_partial = n_empty = 0
    for i, output in enumerate(outputs):
        for completion in output.outputs:
            completion_ids = strip_trailing_eos(completion.token_ids, eos_id)
            # Response as the SFT trainer sees it: assistant prefix + sampled
            # completion + eos. eos is appended unconditionally, matching
            # SFTDataset's `response + tokenizer.eos_token`.
            response_ids = prefix_ids + completion_ids + [eos_id]
            full_response = ASSISTANT_PARTIAL_TURN + completion.text
            score = compute_score(
                data_source=None,
                solution_str=full_response,
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
                verbose=False,
            )
            if score == 1.0:
                n_correct += 1
            elif score == 0.1:
                n_partial += 1
            else:
                n_empty += 1
            rows["prompt"].append(prompt_strings[i])
            rows["prompt_ids"].append(all_prompt_ids[i])
            rows["response_ids"].append(response_ids)
            rows["response"].append(full_response)
            rows["score"].append(float(score))
            rows["rollout_logprob"].append(float(completion.cumulative_logprob or 0.0))

    n_total = n_correct + n_partial + n_empty
    print(f"\nStudent rollout breakdown: {n_correct} correct, {n_partial} format-only, "
          f"{n_empty} no-answer-tag (total: {n_total})")

    out_path = rollouts_path(args.output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {n_total} rollouts to {out_path}")


def run_score(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    assert args.teacher_checkpoint_path, "--teacher_checkpoint_path required for --stage score"

    in_path = rollouts_path(args.output_path)
    print(f"Loading student rollouts from {in_path}")
    df = pd.read_parquet(in_path)
    print(f"  {len(df)} rollouts")

    # Qwen's embedding matrix is padded past the tokenizer's vocabulary (151,936
    # rows vs 151,665 defined tokens), so a model can sample ids from the unused
    # tail. vLLM rejects those as input ("Token id N is out of vocabulary"), and
    # the teacher's logprob for them would be meaningless anyway (untrained
    # embedding rows). Well-trained models effectively never emit them;
    # from-scratch students do at a very low rate. Drop the affected rollouts.
    from transformers import AutoTokenizer

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_checkpoint_path)
    max_token_id = max(teacher_tok.get_vocab().values())  # matches vLLM's check
    keep = [
        max(max(p, default=0), max(r, default=0)) <= max_token_id
        for p, r in zip(df["prompt_ids"], df["response_ids"])
    ]
    n_dropped = len(df) - sum(keep)
    if n_dropped:
        pct = 100.0 * n_dropped / len(df)
        print(f"  WARNING: dropping {n_dropped} rollouts ({pct:.3f}%) containing "
              f"token ids > teacher tokenizer max id {max_token_id}")
        df = df[keep].reset_index(drop=True)
        print(f"  {len(df)} rollouts remain")

    # prompt_logprobs is incompatible with prefix caching on the vLLM V1 engine.
    llm = LLM(
        model=args.teacher_checkpoint_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enable_prefix_caching=False,
    )
    # max_tokens=1 is the vLLM minimum; the generated token is discarded. Only
    # the prompt_logprobs of the frozen rollout tokens are kept.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    # Plain Python ints, not numpy scalars: parquet hands back numpy arrays, and
    # vLLM's V1 engine serializes TokensPrompt through msgspec, which rejects
    # numpy integers with `ValidationError: Expected int, got ext`.
    prompt_ids_col = [[int(t) for t in x] for x in df["prompt_ids"]]
    response_ids_col = [[int(t) for t in x] for x in df["response_ids"]]

    teacher_logprobs_col = []
    chunk = max(1, args.score_chunk_size)
    for start in range(0, len(df), chunk):
        end = min(start + chunk, len(df))
        reqs = [
            TokensPrompt(prompt_token_ids=prompt_ids_col[j] + response_ids_col[j])
            for j in range(start, end)
        ]
        outputs = llm.generate(reqs, sampling_params)
        for j, output in zip(range(start, end), outputs):
            full_ids = prompt_ids_col[j] + response_ids_col[j]
            plp = output.prompt_logprobs
            assert plp is not None and len(plp) == len(full_ids), (
                f"row {j}: expected {len(full_ids)} prompt_logprobs, "
                f"got {None if plp is None else len(plp)}"
            )
            n_prompt = len(prompt_ids_col[j])
            lps = []
            for pos in range(n_prompt, len(full_ids)):
                entry = plp[pos]  # dict {token_id: Logprob}; None only at pos 0
                lps.append(float(entry[full_ids[pos]].logprob))
            assert len(lps) == len(response_ids_col[j])
            teacher_logprobs_col.append(lps)
        print(f"  scored {end}/{len(df)}")

    df["teacher_logprobs"] = teacher_logprobs_col

    flat = [lp for lps in teacher_logprobs_col for lp in lps]
    mean_lp = sum(flat) / max(1, len(flat))
    print(f"\nTeacher logprobs: {len(flat)} tokens, mean {mean_lp:.4f}")
    print(f"Mean student rollout logprob/seq at sample time: "
          f"{df['rollout_logprob'].mean():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"Saved {len(df)} rows to {args.output_path}")
    os.remove(in_path)
    print(f"Removed intermediate {in_path}")


def main():
    args = parse_args()
    if args.stage == "rollout":
        run_rollout(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
