## adapted from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import json
import os
import shutil
from datasets import Dataset
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import argparse

DATA_SOURCE = 'countdown'
N2_PATTERNS = 4  # annotated_expressions.json["2"]: A+B, A-B, A*B, A/B -- one per operation

def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'base-simple':
        """This works for any base model"""
        prefix = f"""Solve the Countdown problem.
Using the numbers {numbers}, construct an arithmetic expression that evaluates to {target}.

Rules:
- Use each provided number exactly once.
- Use only the operations +, -, *, /, and parentheses.
- Do not use any other numbers.
- You may use <think>...</think> for scratch work.
- Return the final expression inside <answer>...</answer>. For example, <answer>(1 + 2) / 3</answer>.

Solution:"""
    elif template_type == 'base-simple-gsm':
        """This works for any base model"""
        prefix = f"""Solve the Countdown problem.
Using the numbers {numbers}, construct an arithmetic expression that evaluates to {target}.

Rules:
- Use each provided number exactly once.
- Use only the operations +, -, *, /, and parentheses.
- Do not use any other numbers.
- Use only grade-school math to find solutions. Nothing complicated required.
- You may use <think>...</think> for scratch work.
- Return the final expression inside <answer>...</answer>. For example, <answer>(1 + 2) / 3</answer>.

Solution:"""
    elif template_type == 'base-simple-terse':
        """base-simple + an explicit terse search format (one `expr = value <cmp>` line per try)"""
        prefix = f"""Solve the Countdown problem.
Using the numbers {numbers}, construct an arithmetic expression that evaluates to {target}.

Rules:
- Use each provided number exactly once.
- Use only the operations +, -, *, /, and parentheses.
- Do not use any other numbers.

Output instruction:
- Inside <think>...</think>, try candidate expressions, one per line, written as
  <expression> = <value>, followed by a comparison to the target: "< <target>" if the
  value is lower, "> <target>" if higher, or "= <target>. Found." if it equals the
  target. Stop at the first "Found."
- Do not include other text or comments.
- After </think>, write only the final expression inside <answer>...</answer>, once.
  Do not write anything after it.

For example, for numbers [3, 5, 8], target 19:
<think>
3 + 5 + 8 = 16 < 19
3 * 5 + 8 = 23 > 19
8 * 3 - 5 = 19 = 19. Found.
</think>
<answer>8 * 3 - 5</answer>

Solution for numbers {numbers}, target {target}:"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


def make_map_fn(split, template_type='base'):
    def process_fn(example, idx):
        question = make_prefix(example, template_type=template_type)
        data = {
            "data_source": DATA_SOURCE,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['target']
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "numbers": example['nums'],
                "puzzle_size": example['puzzle_size'],
                "canonical_pattern_index": example['canonical_pattern_index'],
                "canonical_pattern": example['canonical_pattern']
            }
        }
        return data
    return process_fn


def build_balanced234(args):
    """<out_subdir> = the <src_subdir> train set (n=3,4) verbatim + operation-balanced n=2 rows.

    Defaults build data/balanced234 from data/balanced. Point --src_subdir at a prompt
    variant (e.g. balanced-simple, from scripts/make_prompt_variant_data.py) with the
    matching --template_type to extend that variant instead.

    The n=3,4 rows are copied out of <src_subdir>/train.parquet rather than rebuilt from
    the raw per-pattern JSONs, for two reasons: those JSONs are no longer on disk, and a
    round-trip through them could not reproduce the current train set anyway (the test
    dedup and the leakage filter both drop rows that survive in neither parquet). Reading
    the parquet makes the n=3,4 rows byte-identical by construction, so a balanced234 run
    differs from a balanced run by the n=2 rows and nothing else. The test set is copied
    unchanged for the same reason.

    n=2 has exactly one canonical pattern per operation, so taking the same count from each
    pattern file gives exact operation balance. The count is capped by how many unique n=2
    puzzles exist under this file's generation rules (nums 1-99, integer target 1-99, dedup
    on sorted(nums)): + 2450, - 4851, * 241, / 473 -- hence the 241 default.
    """
    src_dir = os.path.join(args.local_dir, args.src_subdir)
    out_dir = os.path.join(args.local_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    base_train = pq.read_table(os.path.join(src_dir, "train.parquet"))
    print(f"n=3,4 train rows carried over from {src_dir}: {base_train.num_rows}")

    # The prompt is baked into the parquet, so --template_type must be the one the
    # source dir was rendered with, or the n=2 rows would carry a different prefix
    # from the n=3,4 rows. Check against a real row rather than trusting the caller.
    probe = base_train.slice(0, 1).to_pylist()[0]
    expected = make_prefix({"nums": probe["nums"], "target": probe["target"]}, args.template_type)
    if expected != probe["prompt"][0]["content"]:
        raise ValueError(
            f"--template_type={args.template_type!r} does not match the prompts already in "
            f"{src_dir}/train.parquet. The n=2 rows would get a different prefix from the "
            f"n=3,4 rows. Pass the template_type that {args.src_subdir} was rendered with."
        )
    # continue extra_info.index past the existing rows
    next_index = pc.max(base_train.column("extra_info").combine_chunks().field("index")).as_py() + 1

    n2_rows = []
    for i in range(N2_PATTERNS):
        path = os.path.join(args.local_dir, f"countdown_2_pattern_{i}.json")
        with open(path) as f:
            data = json.load(f)
        if len(data) < args.n2_per_op:
            raise ValueError(
                f"{path} has only {len(data)} samples, need {args.n2_per_op}. "
                f"Regenerate with a larger --num_data, or lower --n2_per_op."
            )
        n2_rows += data[:args.n2_per_op]

    # same row construction as the JSON path; process_fn returns only the added columns
    process_fn = make_map_fn('train', args.template_type)
    rows = [{**example, **process_fn(example, next_index + idx)} for idx, example in enumerate(n2_rows)]
    # schema=base_train.schema keeps column order, struct field order and dtypes identical
    n2_train = pa.Table.from_pylist(rows, schema=base_train.schema)
    print(f"n=2 train rows added: {n2_train.num_rows} ({args.n2_per_op} per operation)")

    train_table = pa.concat_tables([base_train, n2_train])
    pq.write_table(train_table, os.path.join(out_dir, 'train.parquet'))
    print(f"wrote {out_dir}/train.parquet: {train_table.num_rows} rows")

    shutil.copyfile(os.path.join(src_dir, "test.parquet"), os.path.join(out_dir, "test.parquet"))
    print(f"copied {src_dir}/test.parquet -> {out_dir}/test.parquet (unchanged)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data')
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--mode', choices=['balanced', 'balanced234'], default='balanced',
                        help="'balanced' (default): build balanced/, balanced5/, balanced6/ from the "
                             "raw countdown_{3,4,5,6}_pattern_*.json files. 'balanced234': build "
                             "balanced234/ = the existing balanced/ train set plus n=2 rows.")
    parser.add_argument('--src_subdir', default='balanced',
                        help="--mode balanced234 only: dir under --local_dir holding the n=3,4 "
                             "train/test parquets to extend (e.g. 'balanced-simple')")
    parser.add_argument('--out_subdir', default='balanced234',
                        help="--mode balanced234 only: output dir under --local_dir")
    parser.add_argument('--n2_per_op', type=int, default=241,
                        help="--mode balanced234 only: n=2 samples per operation "
                             "(241 = the most available for '*'; see build_balanced234)")

    args = parser.parse_args()

    if args.mode == 'balanced234':
        build_balanced234(args)
        raise SystemExit(0)

    # Generate train/test data for n = 3, 4
    train_dataset = []
    test_dataset = []
    for i in range(18):
        with open(f"data/countdown_3_pattern_{i}.json") as f:
            data = json.load(f)
        train_dataset += data[10:]
        test_dataset += data[:10]

    for i in range(96):
        with open(f"data/countdown_4_pattern_{i}.json") as f:
            data = json.load(f)
        train_dataset += data[10:]
        test_dataset += data[:10]

    # Deduplicate test by (sorted nums, target) - keep first occurrence
    seen = set()
    deduped = []
    for item in test_dataset:
        key = (tuple(sorted(item['nums'])), item['target'])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    test_dataset = deduped

    # Filter train rows whose (sorted(nums), target) matches any test puzzle.
    # Closes the leakage pathway where the same canonical puzzle appears under
    # different pattern indices in train and test.
    test_keys = set((tuple(sorted(item['nums'])), item['target']) for item in test_dataset)
    before = len(train_dataset)
    train_dataset = [
        item for item in train_dataset
        if (tuple(sorted(item['nums'])), item['target']) not in test_keys
    ]
    print(f"Train leakage filter (n=3,4): {before} -> {len(train_dataset)} rows "
          f"({before - len(train_dataset)} removed)")

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    
    train_dataset = train_dataset.map(function=make_map_fn('train', args.template_type), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', args.template_type), with_indices=True)

    local_dir = os.path.join(args.local_dir, "balanced")

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Generate test data for n = 5
    test_dataset = []
    for i in range(558):
        with open(f"data/countdown_5_pattern_{i}.json") as f:
            data = json.load(f)
        test_dataset += data[:10]

    # Deduplicate by (sorted nums, target) - keep first occurrence
    seen = set()
    deduped = []
    for item in test_dataset:
        key = (tuple(sorted(item['nums'])), item['target'])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    test_dataset = deduped

    test_dataset = Dataset.from_list(test_dataset)
    test_dataset = test_dataset.map(function=make_map_fn('test', args.template_type), with_indices=True)
    local_dir = os.path.join(args.local_dir, "balanced5")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Generate test data for n = 6
    test_dataset = []
    for i in range(4328):
        with open(f"data/countdown_6_pattern_{i}.json") as f:
            data = json.load(f)
        test_dataset += data[:1]

    # Deduplicate by (sorted nums, target) - keep first occurrence
    seen = set()
    deduped = []
    for item in test_dataset:
        key = (tuple(sorted(item['nums'])), item['target'])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    test_dataset = deduped

    test_dataset = Dataset.from_list(test_dataset)
    test_dataset = test_dataset.map(function=make_map_fn('test', args.template_type), with_indices=True)
    local_dir = os.path.join(args.local_dir, "balanced6")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))