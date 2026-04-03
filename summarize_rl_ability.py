import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grader_utils import compute_score


def score_result_file(path):
    with open(path) as f:
        data = json.load(f)

    correct = 0
    total = 0
    for item in data:
        for output in item.get("outputs", []):
            score = compute_score(None, output, item["target"], {"numbers": item["nums"]}, verbose=False)
            correct += score == 1.0
            total += 1

    if total == 0:
        raise ValueError(f"No outputs found in {path}")

    return correct / total


def load_or_init_summary(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": []}


def replace_run(summary, run_record):
    kept = [
        run for run in summary["runs"]
        if not (
            run.get("model_name") == run_record["model_name"]
            and run.get("exp_name") == run_record["exp_name"]
        )
    ]
    kept.append(run_record)
    summary["runs"] = sorted(kept, key=lambda run: (run["model_name"], run["exp_name"]))
    return summary


def format_markdown(summary):
    lines = ["# RL Ability Runs", ""]
    for run in summary["runs"]:
        lines.append(f"## {run['model_name']} / {run['exp_name']}")
        lines.append("")
        lines.append(f"- Baseline result: `{run['baseline_result_path']}`")
        lines.append(f"- Train job id: `{run.get('train_job_id', 'pending')}`")
        lines.append(f"- Eval job id: `{run.get('eval_job_id', 'pending')}`")
        lines.append(f"- Analyze job id: `{run.get('analyze_job_id', 'pending')}`")
        lines.append(f"- Baseline accuracy: `{run['baseline_accuracy']:.4f}`")
        lines.append(f"- Final checkpoint: `step {run['final_step']}`")
        lines.append(f"- Final accuracy: `{run['final_accuracy']:.4f}`")
        lines.append(f"- Final gain vs baseline: `{run['final_gain']:+.4f}`")
        lines.append(f"- Best checkpoint: `step {run['best_step']}`")
        lines.append(f"- Best accuracy: `{run['best_accuracy']:.4f}`")
        lines.append(f"- Best gain vs baseline: `{run['best_gain']:+.4f}`")
        lines.append(f"- Checkpoint root: `{run['checkpoint_root']}`")
        lines.append(f"- Result root: `{run['result_root']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--baseline_result_path", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--summary_json", default="results/rl_ability_runs.json")
    parser.add_argument("--summary_md", default="results/rl_ability_runs.md")
    parser.add_argument("--train_job_id")
    parser.add_argument("--eval_job_id")
    parser.add_argument("--analyze_job_id")
    args = parser.parse_args()

    baseline_accuracy = score_result_file(args.baseline_result_path)

    pattern = os.path.join(args.result_root, "global_step_*", "*.json")
    candidates = []
    for path in glob(pattern):
        if path.endswith("_annotated.json"):
            continue
        if os.path.basename(path).startswith("."):
            continue
        step_name = os.path.basename(os.path.dirname(path))
        if not step_name.startswith("global_step_"):
            continue
        step = int(step_name.split("_")[-1])
        candidates.append((step, path))

    if not candidates:
        raise FileNotFoundError(f"No evaluated checkpoint results found under {args.result_root}")

    scored = []
    for step, path in sorted(candidates):
        scored.append((step, path, score_result_file(path)))

    final_step, final_path, final_accuracy = scored[-1]
    best_step, best_path, best_accuracy = max(scored, key=lambda item: item[2])

    run_record = {
        "model_name": args.model_name,
        "exp_name": args.exp_name,
        "train_job_id": args.train_job_id,
        "eval_job_id": args.eval_job_id,
        "analyze_job_id": args.analyze_job_id,
        "baseline_result_path": args.baseline_result_path,
        "checkpoint_root": args.checkpoint_root,
        "result_root": args.result_root,
        "baseline_accuracy": baseline_accuracy,
        "final_step": final_step,
        "final_result_path": final_path,
        "final_accuracy": final_accuracy,
        "final_gain": final_accuracy - baseline_accuracy,
        "best_step": best_step,
        "best_result_path": best_path,
        "best_accuracy": best_accuracy,
        "best_gain": best_accuracy - baseline_accuracy,
    }

    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    summary = load_or_init_summary(args.summary_json)
    summary = replace_run(summary, run_record)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(args.summary_md, "w") as f:
        f.write(format_markdown(summary))


if __name__ == "__main__":
    main()
