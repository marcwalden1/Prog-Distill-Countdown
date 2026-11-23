from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
import json
import torch
from utils import *
from prompt_utils import render_prompt
from sft_formats import FORMATS, DISTRACTOR_STRATEGIES, render_response
from calc_utils import correct_rhs, lhs_expression
import argparse
import random
import re
import pyarrow.parquet as pq

def shot_source_for(eval_dataset):
    """Dataset directory holding the exemplar pool for `eval_dataset`.

    balanced5/balanced6 are test-only (no train.parquet), so their exemplars come
    from the n=3,4 train split of the *same prompt variant* — e.g.
    balanced5-simple -> balanced-simple. Exemplars are therefore never drawn from
    any test set: train/test are disjoint for n=3,4 by construction, and the n=5/6
    test sets are a different puzzle size entirely. A random-subsample suffix
    (balanced5-simple-r1000, from scripts/make_eval_subsample.py) is dropped first.
    """
    return re.sub(r"^balanced[56]", "balanced", re.sub(r"-r\d+$", "", eval_dataset))


def build_fewshot_prefix(dataset_dir, n_shot, fmt, seed, n_attempts, strategy):
    """A fixed, seeded block of `n_shot` solved exemplars, prepended to every prompt.

    Fixed rather than per-prompt: lower variance across prompts, and vLLM's prefix
    cache makes the shared prefix nearly free. Exemplars are rendered with the same
    sft_formats.render_response that builds GT-SFT targets, so `--shot_format X` is
    exactly the format `generate_gt_sft_data.py --format X` would train on.
    """
    path = os.path.join("data", dataset_dir, "train.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No exemplar pool at {path}. Pass --shot_source <dataset dir under data/ "
            f"with a train.parquet>.")
    # pyarrow, not load_dataset: the exemplar pool is the 282 MB train parquet, and
    # only n_shot rows are needed. Reading just the containing row groups also skips
    # the HF-datasets cache, which wedges in uninterruptible I/O on this cluster's
    # NFS home when HF_DATASETS_CACHE is unset (as in scripts/eval_base.sh).
    pf = pq.ParquetFile(path)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(pf.metadata.num_rows), n_shot))

    starts, acc = [], 0
    for g in range(pf.metadata.num_row_groups):
        starts.append(acc)
        acc += pf.metadata.row_group(g).num_rows
    cols = ["prompt", "extra_info", "reward_model"]
    rows = []
    for i in idx:
        g = max(g for g in range(len(starts)) if starts[g] <= i)
        rows.append(pf.read_row_group(g, columns=cols).slice(i - starts[g], 1).to_pylist()[0])

    blocks = []
    for j, row in enumerate(rows):
        prompt = str(row["prompt"][0]["content"])
        ei = row["extra_info"]
        nums = [int(n) for n in ei["numbers"]]
        blocks.append(prompt + render_response(
            fmt, prompt, nums, int(row["reward_model"]["ground_truth"]),
            ei["canonical_pattern"], rng=random.Random(seed * 1_000_003 + j),
            n_attempts=n_attempts, strategy=strategy))
    return "\n\n".join(blocks) + "\n\n"


def main(args):
    dataset_id = f"data/{args.eval_dataset}/test.parquet"

    checkpoint_path = args.checkpoint_path
    if args.result_path is None:
        result_dir = checkpoint_path.replace("checkpoints", "results")
    else:
        result_dir = args.result_path
    # Few-shot results go in their own subdir rather than a filename suffix:
    # plot_results.py globs "{ds}_temp0.6_n32_*.json" under global_step_*/round_*/
    # sft_final, so a suffix-only scheme would silently merge zero-shot and
    # few-shot points into one curve.
    if args.n_shot > 0:
        result_dir = os.path.join(result_dir, f"{args.n_shot}shot-{args.shot_format}")
    # Same reason for a non-default generation cap: the `_*.json` glob would
    # merge max1024 and max4096 files into one curve. 1024 stays in place so
    # every historical eval keeps resolving.
    if args.max_tokens != 1024:
        result_dir = os.path.join(result_dir, f"max{args.max_tokens}")
    if args.calculator:
        result_dir = os.path.join(result_dir, "calc")
    result_path = os.path.join(result_dir, "{}_temp{:.1f}_n{}_max{}.json".format(args.eval_dataset, args.temperature, args.n, args.max_tokens))

    print("Saving results to {}".format(result_path), flush=True)

    os.makedirs(result_dir, exist_ok=True)

    dataset = load_dataset("parquet", data_files=dataset_id, split="train")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Scaffold-free variants (*-simple) are trained on the literal parquet
    # string (train_grpo.sh blanks the chat template); eval must match.
    if args.prompt_format == "auto":
        disable_chat_template = "-simple" in args.eval_dataset
    else:
        disable_chat_template = args.prompt_format == "raw"
    print(f"Prompt format: {'raw' if disable_chat_template else 'chat-template'} "
          f"(--prompt_format={args.prompt_format}, dataset={args.eval_dataset})", flush=True)

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(example):
        return {"prompt": render_prompt(tokenizer, example["prompt"], mode="eval",
                                        disable_chat_template=disable_chat_template)}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x))
    prompts = dataset["prompt"]

    if args.n_shot > 0:
        if not disable_chat_template:
            raise SystemExit(
                "--n_shot requires the raw prompt path: exemplars are plain "
                "prompt+response text and cannot be spliced into a chat template "
                "without changing what the model sees. Re-run with "
                "--prompt_format raw (base models have no chat template anyway).")
        lo, _, hi = args.shot_attempts.partition("-")
        n_attempts = (int(lo), int(hi)) if hi else int(lo)
        shot_source = (shot_source_for(args.eval_dataset)
                       if args.shot_source == "auto" else args.shot_source)
        prefix = build_fewshot_prefix(shot_source, args.n_shot, args.shot_format,
                                      args.shot_seed, n_attempts, args.shot_strategy)
        print(f"Few-shot: {args.n_shot} x {args.shot_format} exemplars from "
              f"data/{shot_source}/train.parquet (seed {args.shot_seed})", flush=True)
        print("---- few-shot prefix ----\n" + prefix + "---- end prefix ----", flush=True)
        prompts = [prefix + p for p in prompts]

    llm = LLM(checkpoint_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16",
              trust_remote_code=True, enforce_eager=args.enforce_eager)
    # Few-shot base models continue the exemplar pattern past their own answer,
    # inventing further puzzles; since the grader reads the LAST <answer> tag,
    # generation must stop at the first one. Default on only for --n_shot > 0, so
    # every historical zero-shot eval keeps its exact sampling behavior.
    stop_after_answer = (args.stop_after_answer if args.stop_after_answer is not None
                         else args.n_shot > 0)
    stop_kwargs = ({"stop": ["</answer>"], "include_stop_str_in_output": True}
                   if stop_after_answer else {})
    if stop_after_answer:
        print("Stopping generation at the first </answer>", flush=True)
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=1, 
        max_tokens=args.max_tokens, 
        n=args.n,
        **stop_kwargs,
    )

    # Generate in chunks and append each finished chunk to a JSONL sidecar, so a
    # walltime kill / preemption / GPU fault loses at most one chunk. A rerun with
    # the same result_path resumes from the sidecar. The 4096-cap n=5/n=6 evals
    # need ~15 h; with one generate() call over all prompts, a timeout saved nothing.
    partial_path = result_path + ".partial.jsonl"
    meta = {"checkpoint_path": os.path.abspath(checkpoint_path), "n_prompts": len(prompts),
            "n": args.n, "temperature": args.temperature, "max_tokens": args.max_tokens,
            "n_shot": args.n_shot, "shot_format": args.shot_format,
            "stop_after_answer": stop_after_answer}
    if args.calculator:
        meta["calculator"] = "stepwise"  # calc_utils.correct_rhs design
    done = load_partial(partial_path, meta)
    todo = [i for i in range(len(prompts)) if i not in done]
    print(f"Partial results: {len(done)}/{len(prompts)} prompts already done "
          f"({partial_path}); {len(todo)} to generate in chunks of {args.save_every}", flush=True)
    # (Re)write the sidecar from what parsed: a torn last line has no newline, and
    # appending after it would corrupt the next record too.
    with open(partial_path + ".tmp", "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for idx, rec in done.items():
            f.write(json.dumps({"idx": idx, **rec}) + "\n")
    os.replace(partial_path + ".tmp", partial_path)

    def save(finished):
        """Append {idx: (texts, calc stats or None)} to the sidecar."""
        with open(partial_path, "a") as f:
            for idx, (texts, calc) in finished.items():
                rec = {"outputs": texts,
                       "target": dataset[idx]["target"],
                       "nums": dataset[idx]["nums"],
                       "canonical_pattern_index": dataset[idx]["canonical_pattern_index"],
                       "canonical_pattern": dataset[idx]["canonical_pattern"]}
                if calc is not None:
                    rec["calc"] = calc
                done[idx] = rec
                f.write(json.dumps({"idx": idx, **rec}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"Saved {len(done)}/{len(prompts)} prompts", flush=True)

    if args.calculator:
        generate_with_calculator(llm, tokenizer, prompts, todo, sampling_params, args, save)
    else:
        for start in range(0, len(todo), args.save_every):
            chunk = todo[start:start + args.save_every]
            gens = llm.generate([prompts[i] for i in chunk], sampling_params=sampling_params,
                                use_tqdm=True)
            save({idx: ([o.text for o in output.outputs], None)
                  for idx, output in zip(chunk, gens)})

    outputs = [done[i] for i in range(len(prompts))]
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(outputs, f, indent=2)
    os.replace(tmp_path, result_path)  # atomic: a half-written JSON never appears
    os.remove(partial_path)


def generate_with_calculator(llm, tokenizer, prompts, todo, base_params, args, save):
    """Sample `args.n` responses per prompt in which every arithmetic step is Python's.

    Each sample is generated in segments that stop at "=". When the text before
    an "=" ends in an expression, the model's right-hand side is checked at the
    next stop (or at EOS / the length cap) with calc_utils.correct_rhs; if any
    number in it is wrong, the segment is cut after the corrected right-hand
    side and generation resumes from there. Segments go straight back into the
    running engine. `--calc_max_inflight` should stay within what the KV cache
    holds: a continuation that waits behind a long queue loses its cached prefix
    and is re-prefilled from scratch, which made a 4096-deep pilot ~4.5x slower
    than plain eval. Token ids are the model's own where possible; only text
    after a correction is re-tokenised. `max_tokens` covers model and inserted
    tokens together.

    Per output, saves [equations checked, equations corrected].
    """
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    prompt_ids = {}
    # Sample state: key "idx:j" -> dict(text, ids, pending, checked, fixed), where
    # pending = (lhs expression, char offset, token offset) of the RHS to check.
    state, pending_keys = {}, [f"{i}:{j}" for i in todo for j in range(args.n)]
    pending_keys.reverse()  # pop() from the end = submit in prompt order
    results = {i: [None] * args.n for i in todo}
    ready, inflight, seg_id = {}, 0, 0
    totals = {"checked": 0, "fixed": 0, "segments": 0}

    def submit(key):
        nonlocal seg_id
        st = state[key]
        budget = args.max_tokens - len(st["ids"])
        if budget <= 0:
            return False
        i = int(key.split(":")[0])
        if i not in prompt_ids:
            prompt_ids[i] = tokenizer.encode(prompts[i])
        params = base_params.clone()
        params.n = 1
        params.max_tokens = budget
        params.stop = ["="]
        params.include_stop_str_in_output = True
        params.output_kind = RequestOutputKind.FINAL_ONLY
        engine.add_request(f"{key}|{seg_id}", {"prompt_token_ids": prompt_ids[i] + st["ids"]},
                           params)
        seg_id += 1
        return True

    def finish(key):
        i, j = map(int, key.split(":"))
        st = state.pop(key)
        results[i][j] = (st["text"], [st["checked"], st["fixed"]])
        if all(r is not None for r in results[i]):
            ready[i] = ([r[0] for r in results[i]], [r[1] for r in results[i]])

    while pending_keys or inflight:
        while pending_keys and inflight < args.calc_max_inflight:
            key = pending_keys.pop()
            state[key] = {"text": "", "ids": [], "pending": None, "checked": 0, "fixed": 0}
            submit(key)
            inflight += 1
        for out in engine.step():
            if not out.finished:
                continue
            totals["segments"] += 1
            key = out.request_id.split("|")[0]
            st, o = state[key], out.outputs[0]
            seg_ids = list(o.token_ids)
            if tokenizer.decode(seg_ids) != o.text:  # stop "=" fell mid-token
                seg_ids = tokenizer.encode(o.text, add_special_tokens=False)
            st["text"] += o.text
            st["ids"] += seg_ids
            at_eq = o.finish_reason == "stop" and o.stop_reason == "="
            corrected = False
            if st["pending"] is not None:
                lhs, c0, t0 = st["pending"]
                st["pending"] = None
                new_seg, corrected = correct_rhs(lhs, st["text"][c0:])
                st["checked"] += 1
                totals["checked"] += 1
                if corrected:
                    st["text"] = st["text"][:c0] + new_seg
                    st["ids"] = st["ids"][:t0] + tokenizer.encode(new_seg, add_special_tokens=False)
                    st["fixed"] += 1
                    totals["fixed"] += 1
            if at_eq and not corrected:
                lhs = lhs_expression(st["text"])
                if lhs is not None:
                    st["pending"] = (lhs, len(st["text"]), len(st["ids"]))
            if (at_eq or corrected) and submit(key):
                continue  # still in flight under a new segment id
            inflight -= 1
            finish(key)
        if len(ready) >= args.save_every or (ready and not pending_keys and not inflight):
            save(ready)
            print(f"  calculator: {totals['checked']} equations checked, "
                  f"{totals['fixed']} corrected, {totals['segments']} segments", flush=True)
            ready = {}


def load_partial(partial_path, meta):
    """{prompt idx: record} from a previous interrupted run, or {} if none.

    The first line holds the run config; resuming under a different config would
    silently mix two evals, so that is a hard error. A torn last line (job killed
    mid-write) is dropped and regenerated.
    """
    done = {}
    if not os.path.exists(partial_path):
        return done
    with open(partial_path) as f:
        for ln, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"Dropping unparsable line {ln} of {partial_path}", flush=True)
                continue
            if "meta" in rec:
                if rec["meta"] != meta:
                    raise SystemExit(
                        f"{partial_path} was written with a different config:\n"
                        f"  file: {rec['meta']}\n  now:  {meta}\n"
                        f"Delete it to start over.")
                continue
            done[rec.pop("idx")] = rec
    return done

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/Qwen2.5-1.5B/balanced-grpo-seed1/global_step_50")
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--eval_dataset", type=str, default="balanced")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=128,
                        help="Prompts per generation chunk; each finished chunk is appended "
                             "to <result>.partial.jsonl so an interrupted eval resumes.")
    parser.add_argument("--calculator", action="store_true",
                        help="Python computes every arithmetic step: after each '<expr> =' "
                             "the model's right-hand side is aligned to the expression and "
                             "wrong numbers are replaced by Python's values (2 decimals), "
                             "then generation resumes (calc_utils.py). Results go to a "
                             "calc/ subdir.")
    parser.add_argument("--calc_max_inflight", type=int, default=512,
                        help="--calculator: max concurrent samples in the engine. Keep "
                             "within the KV cache (see generate_with_calculator).")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable vLLM CUDA graph / torch.compile (avoids cache corruption errors)")
    parser.add_argument("--stop_after_answer", type=lambda v: v.lower() == "true",
                        default=None, choices=[True, False],
                        help="Stop generation at the first </answer> (the tag is kept). "
                             "Default: on for --n_shot > 0, off otherwise, which "
                             "preserves historical zero-shot behavior.")
    parser.add_argument("--n_shot", type=int, default=0,
                        help="Number of solved exemplars prepended to every prompt "
                             "(0 = zero-shot, the historical behavior). Requires "
                             "--prompt_format raw.")
    parser.add_argument("--shot_format", choices=list(FORMATS), default="answer",
                        help="Exemplar response format; see sft_formats.py. Matches "
                             "generate_gt_sft_data.py --format.")
    parser.add_argument("--shot_seed", type=int, default=0,
                        help="Seed selecting which puzzles become exemplars (and the "
                             "distractors inside them).")
    parser.add_argument("--shot_source", default="auto",
                        help="Dataset dir under data/ to draw exemplars from. 'auto' "
                             "maps balanced5/6 to their n=3,4 train split.")
    parser.add_argument("--shot_attempts", default="1-3",
                        help="--shot_format search: rejected attempts per exemplar, "
                             "'K' or an inclusive range 'LO-HI'.")
    parser.add_argument("--shot_strategy", choices=list(DISTRACTOR_STRATEGIES),
                        default="mixed", help="--shot_format search: distractor choice.")
    parser.add_argument("--prompt_format", choices=["auto", "chat", "raw"], default="auto",
                        help="'chat' = wrap with the tokenizer chat template (historical behavior); "
                             "'raw' = literal parquet string, no wrapper; 'auto' (default) = raw for "
                             "*-simple datasets, chat otherwise. Must match how the model was trained.")
    args = parser.parse_args()

    main(args)