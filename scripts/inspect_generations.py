"""Inspect eval generations: format taxonomy, length stats, and sample dumps.

For each (run, temperature, dataset) eval JSON this script reports:
  - token-length stats (mean / median / p90 / fraction at the 1024 cap)
  - a failure taxonomy per sample, mirroring grader_utils.compute_score:
      correct        score 1.0  (equation evaluates to target)
      wrong_value    score 0.1  (valid numbers, evaluates, wrong result)
      bad_numbers    score 0.1  (answer tag, numbers not used exactly once)
      unevaluable    score 0.1  (answer tag + valid numbers, can't evaluate)
      no_answer_tag  score 0.0  (no <answer>...</answer> match)
  - structure markers: fraction of samples containing </think>, <answer>
and dumps readable text samples per file for manual reading.

Usage (repo root; needs transformers, no matplotlib):
    python3 scripts/inspect_generations.py \
        --run base=Qwen2.5-0.5B/base/global_step_0 \
        --run grpo-seed1=Qwen2.5-0.5B/balanced-grpo-seed1/global_step_1600 ...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader_utils import (
    extract_solution,
    validate_equation,
    evaluate_equation,
)

MAX_PROMPTS = None  # None = full prompt set, same as plot_results.py. A head-slice
                    # biases n=5/n=6 (file is ordered by pattern index). Set an int
                    # only for a deliberately fast, approximate stats pass.


def classify(output, target, nums):
    equation = extract_solution(output)
    if equation is None:
        return "no_answer_tag"
    if not validate_equation(equation, nums):
        return "bad_numbers"
    result = evaluate_equation(equation)
    if result is None:
        return "unevaluable"
    try:
        return "correct" if abs(result - target) < 1e-5 else "wrong_value"
    except Exception:
        return "unevaluable"


CLASSES = ["correct", "wrong_value", "bad_numbers", "unevaluable", "no_answer_tag"]


def analyze_file(path, tokenizer, max_len):
    with open(path) as f:
        data = json.load(f)
    data = [d for d in data if d.get("outputs")]
    if MAX_PROMPTS is not None:
        data = data[:MAX_PROMPTS]

    counts = {c: 0 for c in CLASSES}
    lengths = []
    n_think_close = 0
    n_total = 0
    per_item_classes = []
    for item in data:
        item_classes = []
        for out in item["outputs"]:
            cls = classify(out, item["target"], item["nums"])
            counts[cls] += 1
            item_classes.append(cls)
            n_total += 1
            if "</think>" in out:
                n_think_close += 1
        lengths.extend(
            len(ids) for ids in tokenizer(item["outputs"], add_special_tokens=False)["input_ids"]
        )
        per_item_classes.append(item_classes)

    lengths.sort()
    n = len(lengths)
    stats = {
        "n_prompts": len(data),
        "n_samples": n_total,
        "mean_len": sum(lengths) / n,
        "median_len": lengths[n // 2],
        "p90_len": lengths[int(n * 0.9)],
        "frac_truncated": sum(l >= max_len for l in lengths) / n,
        "frac_think_close": n_think_close / n_total,
        **{f"frac_{c}": counts[c] / n_total for c in CLASSES},
    }
    return stats, data, per_item_classes


def dump_samples(out_path, data, per_item_classes, tokenizer, n_prompts, n_outputs, header):
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for i, item in enumerate(data[:n_prompts]):
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"PROMPT {i}: target={item['target']}  nums={item['nums']}  "
                    f"pattern={item.get('canonical_pattern')}\n")
            for j, out in enumerate(item["outputs"][:n_outputs]):
                ntok = len(tokenizer(out, add_special_tokens=False)["input_ids"])
                f.write("-" * 78 + "\n")
                f.write(f"[output {j}] class={per_item_classes[i][j]}  tokens={ntok}\n")
                f.write(out + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=SUBDIR",
                        help="<label>=<subdir under result root, e.g. Qwen2.5-0.5B/base/global_step_0>")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.6, 1.0])
    parser.add_argument("--datasets", nargs="+", default=["balanced", "balanced5"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tokenizer",
                        default="/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/Qwen2.5-0.5B")
    parser.add_argument("--n-sample-prompts", type=int, default=8)
    parser.add_argument("--n-sample-outputs", type=int, default=4)
    parser.add_argument("--out-dir", default="results/generation_samples")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for label, subdir in [r.split("=", 1) for r in args.run]:
        for temp in args.temps:
            for ds in args.datasets:
                path = os.path.join(
                    args.result_root, subdir,
                    f"{ds}_temp{temp:.1f}_n{args.n}_max{args.max_tokens}.json")
                if not os.path.exists(path):
                    print(f"[missing] {path}")
                    continue
                print(f"[analyzing] {path}", flush=True)
                stats, data, classes = analyze_file(path, tokenizer, args.max_tokens)
                stats.update({"run": label, "temp": temp, "dataset": ds})
                rows.append(stats)
                sample_path = os.path.join(args.out_dir, f"{label}_{ds}_temp{temp:.1f}.txt")
                dump_samples(sample_path, data, classes, tokenizer,
                             args.n_sample_prompts, args.n_sample_outputs,
                             header=f"run={label}  dataset={ds}  temp={temp}  source={path}")
                print(f"  samples -> {sample_path}")

    cols = ["run", "temp", "dataset", "mean_len", "median_len", "p90_len",
            "frac_truncated", "frac_think_close",
            "frac_correct", "frac_wrong_value", "frac_bad_numbers",
            "frac_unevaluable", "frac_no_answer_tag"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.out_dir, "summary.tsv"), "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {os.path.join(args.out_dir, 'summary.tsv')}")


if __name__ == "__main__":
    main()
