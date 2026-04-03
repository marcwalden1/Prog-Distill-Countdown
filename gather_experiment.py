"""
gather_experiment.py

Reads training logs, eval results, and annotated outputs for a given experiment,
generates plots, and appends a structured entry to experiments.md.

Usage:
    python gather_experiment.py --model_name Qwen2.5-1.5B --exp_name balanced-grpo-seed1
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--eval_dataset", default="balanced")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# A. Training log parsing
# ---------------------------------------------------------------------------

def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_training_log(log_dir, model_name, exp_name):
    """Find the training log for this experiment. Prefers config-header match,
    falls back to model-name string search. Returns (path, job_id)."""
    candidates = sorted(glob.glob(os.path.join(log_dir, "train_grpo.sh-*.out")))
    # Try header match first (logs with config block)
    for path in candidates:
        try:
            with open(path) as f:
                content = f.read(4096)
            clean = strip_ansi(content)
            if f"Model:            {model_name}" in clean and \
               f"Exp name:         {exp_name}" in clean:
                job_id = re.search(r"train_grpo\.sh-(\d+)-", os.path.basename(path))
                return path, job_id.group(1) if job_id else None
        except Exception:
            continue
    # Fall back: scan for model name + exp name anywhere in the file
    for path in candidates:
        try:
            with open(path) as f:
                content = f.read()
            if model_name in content and exp_name in content:
                job_id = re.search(r"train_grpo\.sh-(\d+)-", os.path.basename(path))
                return path, job_id.group(1) if job_id else None
        except Exception:
            continue
    return None, None


def parse_config_header(content):
    """Extract fields from the EXPERIMENT CONFIG header block.
    Only parses within the ==== EXPERIMENT CONFIG ==== block to avoid
    matching text from model-generated outputs in the log."""
    content = strip_ansi(content)
    config = {}

    # Extract just the config block
    block_match = re.search(
        r"={20,}\s*\nEXPERIMENT CONFIG\s*\n(.+?)\n={20,}",
        content, re.DOTALL
    )
    if not block_match:
        return config  # old log without header — return empty, caller handles fallback

    block = block_match.group(1)
    patterns = {
        "date":            r"^Date:\s+(.+)",
        "job_id":          r"^SLURM Job ID:\s+(\S+)",
        "array_id":        r"^SLURM Array ID:\s+(\S+)",
        "node":            r"^Node:\s+(\S+)",
        "n_gpus":          r"^GPUs:\s+(\S+)",
        "model":           r"^Model:\s+(\S+)",
        "exp_name":        r"^Exp name:\s+(\S+)",
        "data_source":     r"^Data source:\s+(\S+)",
        "max_length":      r"^Max length:\s+(\S+)",
        "checkpoint_path": r"^Checkpoint path:\s+(.+)",
        "project_dir":     r"^Project dir:\s+(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, block, re.MULTILINE)
        if m:
            config[key] = m.group(1).strip()
    return config


def get_job_duration(job_id):
    """Use sacct to get wall-clock start/end for the job."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=Start,End", "-X", "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None, None, None
        parts = lines[0].split("|")
        if len(parts) < 2:
            return None, None, None
        start_str, end_str = parts[0], parts[1]
        fmt = "%Y-%m-%dT%H:%M:%S"
        start = datetime.strptime(start_str, fmt)
        end = datetime.strptime(end_str, fmt)
        delta = end - start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes = remainder // 60
        duration_str = f"{hours}h {minutes:02d}m"
        return start_str, end_str, duration_str
    except Exception:
        return None, None, None


def parse_step_metrics(content):
    """Parse per-step training metrics from the log. Returns dict of step -> metrics.

    Each step is logged on a single line (possibly very long) with format:
      (TaskRunner pid=N) step:S - key:value - key:value ...
    Lines also contain ANSI escape codes which are stripped first.
    """
    steps = {}
    ansi_free = strip_ansi(content)

    for line in ansi_free.splitlines():
        m = re.search(r"\bstep:(\d+) - ", line)
        if not m:
            continue
        step_num = int(m.group(1))
        entry = {"step": step_num}

        def extract(pat):
            hit = re.search(pat, line)
            if hit:
                return float(hit.group(1))
            return None

        v = extract(r"val-core/countdown/reward/mean@4:(?:np\.float64\()?([\d.eE+\-]+)\)?")
        if v is not None:
            entry["val_reward"] = v

        v = extract(r"actor/entropy:([\d.eE+\-]+)")
        if v is not None:
            entry["entropy"] = v

        v = extract(r"actor/kl_loss:(?:np\.float64\()?([\d.eE+\-]+)\)?")
        if v is not None:
            entry["kl_loss"] = v

        v = extract(r"actor/grad_norm:(?:np\.float64\()?([\d.eE+\-]+)\)?")
        if v is not None:
            entry["grad_norm"] = v

        v = extract(r"response_length/mean:([\d.eE+\-]+)")
        if v is not None:
            entry["response_length_mean"] = v

        v = extract(r"response_length/clip_ratio:([\d.eE+\-]+)")
        if v is not None:
            entry["clip_ratio"] = v

        if len(entry) > 1:
            steps[step_num] = entry

    return steps


def parse_hyperparams(train_script="scripts/train_grpo.sh"):
    """Read key hyperparameters directly from the training script."""
    params = {}
    try:
        with open(train_script) as f:
            content = f.read()
        mappings = {
            "lr":            r"actor_rollout_ref\.actor\.optim\.lr=([\S]+)",
            "train_batch":   r"data\.train_batch_size=(\d+)",
            "rollout_n":     r"actor_rollout_ref\.rollout\.n=(\d+)",
            "val_n":         r"actor_rollout_ref\.rollout\.val_kwargs\.n=(\d+)",
            "kl_coef":       r"actor_rollout_ref\.actor\.kl_loss_coef=([\S]+)",
            "kl_type":       r"actor_rollout_ref\.actor\.kl_loss_type=(\S+)",
            "entropy_coeff": r"actor_rollout_ref\.actor\.entropy_coeff=(\S+)",
            "loss_agg":      r"actor_rollout_ref\.actor\.loss_agg_mode=(\S+)",
            "save_freq":     r"trainer\.save_freq=(\d+)",
            "total_epochs":  r"trainer\.total_epochs=(\d+)",
            "gpu_mem_util":  r"actor_rollout_ref\.rollout\.gpu_memory_utilization=([\S]+)",
            "norm_adv":      r"algorithm\.norm_adv_by_std_in_grpo=(\S+)",
            "use_kl_reward": r"algorithm\.use_kl_in_reward=(\S+)",
            "prompt_max_len":r"data\.max_prompt_length=(\d+)",
        }
        for key, pat in mappings.items():
            m = re.search(pat, content)
            if m:
                params[key] = m.group(1).strip().rstrip("\\")
    except Exception:
        pass
    return params


# ---------------------------------------------------------------------------
# B. Eval accuracy curve
# ---------------------------------------------------------------------------

def compute_eval_accuracy(result_dir, model_name, exp_name, eval_dataset):
    """Score each checkpoint's eval results. Returns sorted list of (step, accuracy)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grader_utils import compute_score

    pattern = os.path.join(result_dir, model_name, exp_name,
                           "global_step_*", f"{eval_dataset}_temp*.json")
    files = glob.glob(pattern)
    results = []

    for path in files:
        step_match = re.search(r"global_step_(\d+)", path)
        if not step_match:
            continue
        step = int(step_match.group(1))

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        correct = total = 0
        # Suppress grader_utils stdout
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            for item in data:
                for output in item.get("outputs", []):
                    score = compute_score(None, output, item["target"],
                                         {"numbers": item["nums"]})
                    correct += score == 1.0
                    total += 1
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        if total > 0:
            results.append((step, correct / total))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# C. Plots
# ---------------------------------------------------------------------------

def make_plots(model_name, exp_name, step_metrics, eval_curve, figures_dir,
               annotated_path=None):
    os.makedirs(figures_dir, exist_ok=True)

    title_prefix = f"{model_name} / {exp_name}"
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    # 1. Training curve (val reward vs step)
    steps_with_reward = [(s, d["val_reward"]) for s, d in sorted(step_metrics.items())
                         if "val_reward" in d]
    if steps_with_reward:
        xs, ys = zip(*steps_with_reward)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, linewidth=1.5, color="#2563eb")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Val reward mean@4")
        ax.set_title(f"Training curve — {title_prefix}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(figures_dir, "training_curve.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")

    # 2. Eval accuracy curve
    if eval_curve:
        xs, ys = zip(*eval_curve)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, linewidth=1.5, color="#16a34a", marker="o", markersize=3)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Accuracy (pass@1)")
        ax.set_title(f"Eval accuracy — {title_prefix}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(figures_dir, "eval_accuracy.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")

    # 3. Pattern breakdown (if annotated results exist)
    if annotated_path and os.path.exists(annotated_path):
        try:
            with open(annotated_path) as f:
                annotated = json.load(f)
            pattern_correct = {}
            pattern_total = {}
            for item in annotated:
                pat = item.get("canonical_pattern", "unknown")
                scores = item.get("scores", [])
                pattern_correct[pat] = pattern_correct.get(pat, 0) + sum(scores)
                pattern_total[pat] = pattern_total.get(pat, 0) + len(scores)

            # top 20 by frequency
            top = sorted(pattern_total.items(), key=lambda x: -x[1])[:20]
            labels = [p for p, _ in top]
            accs = [pattern_correct.get(p, 0) / pattern_total[p] for p, _ in top]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(labels)), accs, color="#7c3aed")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Pattern breakdown (top 20) — {title_prefix}")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out = os.path.join(figures_dir, "pattern_breakdown.png")
            fig.savefig(out)
            plt.close(fig)
            print(f"  Saved: {out}")
            return True  # annotated plot was made
        except Exception as e:
            print(f"  Warning: could not generate pattern breakdown: {e}")

    return False


# ---------------------------------------------------------------------------
# D. Write experiments.md entry
# ---------------------------------------------------------------------------

def format_entry(config, hyperparams, step_metrics, eval_curve, duration_str,
                 model_name, exp_name, eval_dataset, has_pattern_plot, figures_dir):
    # Final step metrics
    final_step = max(step_metrics.keys()) if step_metrics else "?"
    final = step_metrics.get(final_step, {}) if step_metrics else {}

    def fmt(val, spec):
        try:
            return format(float(val), spec)
        except Exception:
            return str(val)

    # Final eval accuracy
    final_eval_step, final_acc = (eval_curve[-1] if eval_curve else ("?", "?"))

    fig_base = f"figures/{model_name}/{exp_name}"
    repl_checkpoint_dir = "/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/rl-checkpoints"

    pattern_section = ""
    if has_pattern_plot:
        pattern_section = f"""
### Pattern breakdown (final checkpoint)
![Pattern breakdown]({fig_base}/pattern_breakdown.png)
"""

    entry = f"""
---

## {model_name} / {exp_name}

**Date:** {config.get('date', 'unknown')}
**Duration:** {duration_str or 'unknown'}
**SLURM Job ID:** {config.get('job_id', 'unknown')}

### Setup
| Field | Value |
|---|---|
| Base model | {model_name} |
| Data source | `data/{config.get('data_source', hyperparams.get('data_source', '?'))}/` |
| Algorithm | GRPO |
| Train batch size | {hyperparams.get('train_batch', 256)} |
| Rollout n | {hyperparams.get('rollout_n', 4)} |
| Val n | {hyperparams.get('val_n', 4)} |
| Learning rate | {hyperparams.get('lr', '1e-6')} |
| Max response length | {config.get('max_length', hyperparams.get('max_length', 1024))} |
| KL loss coef | {hyperparams.get('kl_coef', '0.001')} |
| KL loss type | {hyperparams.get('kl_type', 'low_var_kl')} |
| Entropy coeff | {hyperparams.get('entropy_coeff', '0')} |
| Loss aggregation | {hyperparams.get('loss_agg', 'token-mean')} |
| Advantage normalization | {hyperparams.get('norm_adv', 'False')} |
| KL in reward | {hyperparams.get('use_kl_reward', 'False')} |
| Save/test freq | every {hyperparams.get('save_freq', 50)} steps |
| Total epochs | {hyperparams.get('total_epochs', 1)} |
| vLLM GPU mem util | {hyperparams.get('gpu_mem_util', '0.6')} |
| GPUs | {config.get('n_gpus', '4')} × nvidia_h100_80gb_hbm3 |
| Node | {config.get('node', 'unknown')} |
| Total steps | {final_step} |

### Training metrics (step {final_step})
| Metric | Value |
|---|---|
| Val reward mean@4 | {fmt(final.get('val_reward', '?'), '.4f')} |
| Actor entropy | {fmt(final.get('entropy', '?'), '.4f')} |
| KL loss | {fmt(final.get('kl_loss', '?'), '.6f')} |
| Grad norm | {fmt(final.get('grad_norm', '?'), '.4f')} |
| Response length (mean) | {fmt(final.get('response_length_mean', '?'), '.1f')} |
| Response clip ratio | {fmt(final.get('clip_ratio', '?'), '.4f')} |

### Eval metrics (step {final_eval_step}, n=32, temp=0.6, dataset={eval_dataset})
| Dataset | Accuracy |
|---|---|
| {eval_dataset} (n=3,4) | {fmt(final_acc, '.4f') if final_acc != '?' else '?'} |

### Training curve
![Training curve]({fig_base}/training_curve.png)

### Eval accuracy curve
![Eval accuracy]({fig_base}/eval_accuracy.png)
{pattern_section}
### Replication
```bash
MODEL_NAME={model_name} EXP_NAME={exp_name} DATA_SOURCE={config.get('data_source', '?')} sbatch scripts/train_grpo.sh
MODEL_NAME={model_name} EXP_NAME={exp_name} CHECKPOINT_DIR={repl_checkpoint_dir} sbatch --array=1-32 scripts/eval.sh
```
"""
    return entry


def append_to_experiments_md(entry, output_path="experiments.md"):
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write("# Experiments\n\nThis file tracks completed training runs.\n")
    with open(output_path, "a") as f:
        f.write(entry)
    print(f"  Appended entry to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_name = args.model_name
    exp_name = args.exp_name

    print(f"\nGathering experiment: {model_name} / {exp_name}")
    print("=" * 60)

    # --- A. Training log ---
    print("\n[1] Parsing training log...")
    log_path, log_job_id = find_training_log(args.log_dir, model_name, exp_name)
    if log_path:
        print(f"  Found log: {log_path}")
        with open(log_path) as f:
            log_content = f.read()
        config = parse_config_header(log_content)
        step_metrics = parse_step_metrics(log_content)
        print(f"  Config fields found: {list(config.keys())}")
        print(f"  Steps parsed: {len(step_metrics)}")
    else:
        print("  WARNING: No training log found. Config will be empty.")
        config = {}
        step_metrics = {}
        log_content = ""
        log_job_id = None

    # Fill in fields that old logs (no header) won't have
    if "job_id" not in config and log_job_id:
        config["job_id"] = log_job_id
    if "data_source" not in config:
        # Infer from exp_name convention: {data_source}-grpo-seed{N}
        ds_match = re.match(r"^(.+?)-(?:grpo|ppo)-", exp_name)
        config["data_source"] = ds_match.group(1) if ds_match else "balanced"
    if "n_gpus" not in config:
        config["n_gpus"] = "4"  # default

    # Duration via sacct
    job_id = config.get("job_id")
    duration_str = None
    if job_id:
        _, _, duration_str = get_job_duration(job_id)
        if duration_str:
            print(f"  Training duration: {duration_str}")
        else:
            # Fall back: estimate from log timestamps
            ts_matches = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                                    strip_ansi(log_content))
            if len(ts_matches) >= 2:
                fmt = "%Y-%m-%d %H:%M:%S"
                try:
                    t0 = datetime.strptime(ts_matches[0], fmt)
                    t1 = datetime.strptime(ts_matches[-1], fmt)
                    delta = t1 - t0
                    h, rem = divmod(int(delta.total_seconds()), 3600)
                    duration_str = f"{h}h {rem//60:02d}m (estimated)"
                    print(f"  Training duration (estimated): {duration_str}")
                except Exception:
                    pass

    # Hyperparams from training script
    hyperparams = parse_hyperparams("scripts/train_grpo.sh")
    print(f"  Hyperparams parsed: {list(hyperparams.keys())}")

    # --- B. Eval accuracy ---
    print("\n[2] Computing eval accuracy curve...")
    eval_curve = compute_eval_accuracy(args.result_dir, model_name, exp_name, args.eval_dataset)
    print(f"  Checkpoints scored: {len(eval_curve)}")
    if eval_curve:
        print(f"  Final: step {eval_curve[-1][0]}, accuracy {eval_curve[-1][1]:.4f}")

    # --- C. Plots ---
    print("\n[3] Generating plots...")
    figures_dir = os.path.join("figures", model_name, exp_name)

    # Check for annotated results at final eval step
    annotated_path = None
    if eval_curve:
        final_eval_step = eval_curve[-1][0]
        candidates = glob.glob(os.path.join(
            args.result_dir, model_name, exp_name,
            f"global_step_{final_eval_step}", f"{args.eval_dataset}_temp*_annotated.json"
        ))
        if candidates:
            annotated_path = candidates[0]

    has_pattern_plot = make_plots(
        model_name, exp_name, step_metrics, eval_curve, figures_dir, annotated_path
    )

    # --- D. Write experiments.md ---
    print("\n[4] Writing experiments.md entry...")
    entry = format_entry(
        config, hyperparams, step_metrics, eval_curve, duration_str,
        model_name, exp_name, args.eval_dataset, has_pattern_plot, figures_dir
    )
    append_to_experiments_md(entry)

    print("\nDone.")
    if step_metrics:
        final_step = max(step_metrics.keys())
        final = step_metrics[final_step]
        print(f"  Final step:        {final_step}")
        print(f"  Val reward mean@4: {final.get('val_reward', '?'):.4f}" if 'val_reward' in final else "  Val reward: ?")
    if eval_curve:
        print(f"  Eval accuracy:     {eval_curve[-1][1]:.4f} (step {eval_curve[-1][0]})")


if __name__ == "__main__":
    main()
