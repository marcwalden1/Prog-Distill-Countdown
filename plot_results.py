"""
plot_results.py

Produces 4 plots for a given model/experiment run:
  1. val_reward_vs_step.png  — val reward mean@4 vs. training step (from log)
  2. eval_n34.png            — mean@1 + mean@32 vs. checkpoint step (n=3,4)
  3. eval_n5.png             — mean@1 + mean@32 vs. checkpoint step (n=5)
  4. eval_n6.png             — mean@1 + mean@32 vs. checkpoint step (n=6)

Plots 2-4 are skipped gracefully if the eval results don't exist.

Usage:
    python3 plot_results.py --model_name Qwen2.5-1.5B --exp_name balanced-grpo-seed1
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gather_experiment import find_training_logs, parse_step_metrics
from grader_utils import compute_score

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STYLE = {"font.size": 11, "figure.dpi": 150}
COLOR_VAL   = "#2563eb"   # blue  — val reward
COLOR_MEAN1 = "#16a34a"   # green — mean@1
COLOR_MEAN32 = "#dc2626"  # red   — mean@32
LINEWIDTH  = 1.5
MARKERSIZE = 3
FIGSIZE    = (7, 4)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   required=True)
    parser.add_argument("--exp_name",     required=True)
    parser.add_argument("--result_dir",   default="results")
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--figures_dir",  default="figures")
    parser.add_argument("--model_base_dir",
                        default="/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models")
    parser.add_argument("--condition", default=None,
                        help="Condition tag for WandB (e.g. rl-only, distill, progdistill)")
    parser.add_argument("--dataset_suffix", default="",
                        help="suffix on the eval dataset dir names, e.g. '-simple' for "
                             "prompt-variant runs evaluated on balanced-simple/balanced5-simple/...")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="eval generation cap to plot. eval.py writes caps other than 1024 "
                             "into a global_step_*/max<M>/ subdir; plot_results.sh derives this "
                             "from --max_tokens in EVAL_EXTRA_ARGS. Non-1024 figures get a _max<M> tag.")
    parser.add_argument("--ood_suffix", default="",
                        help="extra suffix on the n=5/n=6 dataset names only, e.g. '-r1000' for "
                             "the random subsets from scripts/make_eval_subsample.py "
                             "(run_pipeline.sh OOD_SUBSAMPLE=1000). Figures and wandb keys get the "
                             "tag too, so they never overwrite full-set ones.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_reward_curve(log_dir, model_name, exp_name):
    """Return sorted list of (step, val_reward) from the training log(s).

    A resumed run (walltime kill -> resubmit, verl resume_mode=auto) has one log
    per SLURM job covering a disjoint step range, so all of them are merged.
    Later jobs win on any overlapping step: after a resume from global_step_N
    the new job's val_before_train re-validates at N with the same weights, and
    that fresher measurement is the one to keep.
    """
    logs = find_training_logs(log_dir, model_name, exp_name)
    if not logs:
        print(f"  WARNING: No training log found for {model_name}/{exp_name}")
        return []
    if len(logs) > 1:
        print(f"  Merging {len(logs)} training logs (resumed run): "
              + ", ".join(job_id or "?" for _, job_id in logs))
    step_metrics = {}
    for log_path, _ in logs:
        with open(log_path) as f:
            step_metrics.update(parse_step_metrics(f.read()))
    return [
        (s, d["val_reward"])
        for s, d in sorted(step_metrics.items())
        if "val_reward" in d
    ]


def _score_item(args):
    """Score one prompt's outputs. Top-level for multiprocessing pickling."""
    outputs, target, nums = args
    if not outputs:
        return None
    extra = {"numbers": nums}
    s1 = compute_score(None, outputs[0], target, extra, verbose=False)
    mean1 = float(s1 == 1.0)
    scores = [compute_score(None, out, target, extra, verbose=False) for out in outputs]
    mean32 = sum(scores) / len(scores)
    return mean1, mean32


MAX_PROMPTS = None  # None = score the full prompt set (n=5/n=6 have >1000 prompts,
                    # and the file is ordered by pattern index so a head-slice is a
                    # biased/easier subsample — see git history). Set to an int only
                    # for a deliberately fast, approximate plot.


def _extract_eval_step(path):
    """Extract the training step from a result path."""
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return int(m.group(1))

    m = re.search(r"round_(\d+)", path)
    if m:
        return int(m.group(1))

    if "/sft_final/" in path:
        return 1600

    return None


EVAL_TEMP = "0.6"   # canonical eval sampling temperature (used for globbing + filenames)
EVAL_N = "32"       # canonical samples-per-prompt (n=32)
EVAL_MAX_TOKENS = 1024  # set from --max_tokens in main(); mirrors eval.py's result layout


def _cap_subdir():
    # eval.py: caps other than 1024 live in <step>/max<M>/, 1024 stays in <step>/.
    return "" if EVAL_MAX_TOKENS == 1024 else f"max{EVAL_MAX_TOKENS}"


def _result_files(result_dir, model_name, exp_name, eval_dataset):
    # Only the canonical eval config (temp 0.6, n=32) is plotted — one point per
    # step. Matching `temp*` here would also pick up stray temp-sweep runs
    # (e.g. temp1.0/temp2.0) and n64 files, producing multiple points per step.
    glob_name = f"{eval_dataset}_temp{EVAL_TEMP}_n{EVAL_N}_*.json"
    cap = _cap_subdir()
    patterns = [
        os.path.join(result_dir, model_name, exp_name, "global_step_*", cap, glob_name),
        os.path.join(result_dir, model_name, exp_name, "round_*", cap, glob_name),
        os.path.join(result_dir, model_name, exp_name, "sft_final", cap, glob_name),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files), key=lambda path: (_extract_eval_step(path) is None, _extract_eval_step(path), path))


def _score_checkpoint(path):
    """Return (step, mean1, mean32) for one result file, using .scores cache if fresh."""
    cache_path = path + ".scores"
    try:
        if os.path.getmtime(cache_path) >= os.path.getmtime(path):
            with open(cache_path) as f:
                cached = tuple(json.load(f))
            print(f"    [cache hit]", flush=True)
            return cached
    except Exception:
        pass

    step = _extract_eval_step(path)
    if step is None:
        return None

    t_load = time.time()
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    print(f"    json load: {time.time()-t_load:.1f}s  ({len(data)} prompts in file)", flush=True)

    if MAX_PROMPTS is not None and len(data) > MAX_PROMPTS:
        data = data[:MAX_PROMPTS]
        print(f"    truncated to {MAX_PROMPTS} prompts", flush=True)

    items = [
        (item["outputs"], item["target"], item["nums"])
        for item in data if item.get("outputs")
    ]
    if not items:
        return None

    t_score = time.time()
    from multiprocessing import Pool
    with Pool(8) as pool:
        scored = pool.map(_score_item, items)
    print(f"    scoring {len(items)} prompts x32: {time.time()-t_score:.1f}s", flush=True)

    scored = [s for s in scored if s is not None]
    if not scored:
        return None

    n_prompts = len(scored)
    sum1  = sum(s[0] for s in scored)
    sum32 = sum(s[1] for s in scored)
    result = (step, sum1 / n_prompts, sum32 / n_prompts)
    try:
        with open(cache_path, "w") as f:
            json.dump(list(result), f)
    except Exception:
        pass
    return result


def _length_checkpoint(path, tokenizer):
    """Return (step, mean_response_length_tokens) using .lengths cache."""
    cache_path = path + ".lengths"
    try:
        if os.path.getmtime(cache_path) >= os.path.getmtime(path):
            with open(cache_path) as f:
                return tuple(json.load(f))
    except Exception:
        pass

    step = _extract_eval_step(path)
    if step is None:
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    all_outputs = [out for item in data for out in item.get("outputs", [])]
    if not all_outputs:
        return None

    encoded = tokenizer(all_outputs, add_special_tokens=False)["input_ids"]
    mean_len = sum(len(ids) for ids in encoded) / len(encoded)

    result = (step, mean_len)
    try:
        with open(cache_path, "w") as f:
            json.dump(list(result), f)
    except Exception:
        pass
    return result


def compute_mean_lengths(result_dir, model_name, exp_name, eval_dataset, tokenizer):
    """Return sorted list of (step, mean_tokens) for each checkpoint."""
    files = _result_files(result_dir, model_name, exp_name, eval_dataset)
    if not files:
        return []

    results = []
    for path in sorted(files):
        result = _length_checkpoint(path, tokenizer)
        if result is not None:
            results.append(result)
    results.sort(key=lambda x: x[0])
    return results


def compute_mean_metrics(result_dir, model_name, exp_name, eval_dataset):
    """
    For each checkpoint step, compute mean@1 and mean@32.

    mean@1  — score outputs[0] as binary (1.0 correct else 0.0), average across prompts
    mean@32 — average raw compute_score() across all 32 outputs per prompt, then across prompts

    Returns sorted list of (step, mean1, mean32), or [] if no results found.
    Uses .scores cache files to avoid re-scoring unchanged result files.
    """
    files = _result_files(result_dir, model_name, exp_name, eval_dataset)
    if not files:
        return []

    results = []
    for path in sorted(files):
        step = _extract_eval_step(path)
        label = f"step_{step}" if step is not None else path
        print(f"  Scoring {label}...", end=" ", flush=True)
        t0 = time.time()
        result = _score_checkpoint(path)
        if result is not None:
            results.append(result)
        print(f"{time.time() - t0:.1f}s", flush=True)

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_val_reward(val_curve, model_name, exp_name, figures_dir):
    if not val_curve:
        print("  Skipping val_reward plot (no data)")
        return
    xs, ys = zip(*val_curve)
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, linewidth=LINEWIDTH, color=COLOR_VAL)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Val reward mean@4")
    ax.set_title(f"Val reward — {model_name} / {exp_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_dir, "val_reward_vs_step.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_response_length(lengths_by_dataset, figures_dir, title_prefix, file_tag=""):
    """Plot mean response length vs. checkpoint step for all datasets on one figure.

    Labels may carry a subsample tag ('n=5/r1000'); colors key on the part before '/'.
    `file_tag` ('_r1000') keeps a subsample plot from overwriting the full-set one.
    """
    dataset_colors = {"n=3,4": COLOR_MEAN1, "n=5": COLOR_MEAN32, "n=6": COLOR_VAL}
    has_data = any(v for v in lengths_by_dataset.values())
    if not has_data:
        print("  Skipping response length plot (no data)", flush=True)
        return
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n_label, lengths in lengths_by_dataset.items():
        if not lengths:
            continue
        xs, ys = zip(*lengths)
        ax.plot(xs, ys, linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE,
                color=dataset_colors[n_label.split("/")[0]], label=n_label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", va="bottom", fontsize=6, color=dataset_colors[n_label.split("/")[0]])
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Mean response length (tokens)")
    ax.set_title(f"Response length (temp {EVAL_TEMP}) — {title_prefix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"response_length{file_tag}_temp{EVAL_TEMP}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)


def plot_eval(metrics, n_label, out_path, title_prefix):
    """Plot mean@1 and mean@32 on the same axes vs. checkpoint step."""
    xs      = [r[0] for r in metrics]
    mean1s  = [r[1] for r in metrics]
    mean32s = [r[2] for r in metrics]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, mean1s,  linewidth=LINEWIDTH, color=COLOR_MEAN1,
            marker="o", markersize=MARKERSIZE, label="mean@1")
    ax.plot(xs, mean32s, linewidth=LINEWIDTH, color=COLOR_MEAN32,
            marker="o", markersize=MARKERSIZE, label="mean@32")
    # Annotate each marker with its value (mean@1 above, mean@32 below).
    for x, y in zip(xs, mean1s):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", va="bottom", fontsize=6, color=COLOR_MEAN1)
    for x, y in zip(xs, mean32s):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -8),
                    ha="center", va="top", fontsize=6, color=COLOR_MEAN32)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Eval [{n_label}] (temp {EVAL_TEMP}) — {title_prefix}")
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_name = args.model_name
    exp_name   = args.exp_name

    figures_dir = os.path.join(args.figures_dir, model_name, exp_name)
    os.makedirs(figures_dir, exist_ok=True)

    title_prefix = f"{model_name} / {exp_name}"
    print(f"\nPlotting: {title_prefix}", flush=True)
    print("=" * 60, flush=True)

    # --- Plot 1: val reward vs. training step ---
    print("\n[1] Val reward vs. training step...", flush=True)
    val_curve = load_val_reward_curve(args.log_dir, model_name, exp_name)
    plot_val_reward(val_curve, model_name, exp_name, figures_dir)

    # --- Plots 2-4: mean@1 + mean@32 per eval dataset ---
    global EVAL_MAX_TOKENS
    EVAL_MAX_TOKENS = args.max_tokens
    cap_file = "" if EVAL_MAX_TOKENS == 1024 else f"_max{EVAL_MAX_TOKENS}"
    sfx = args.dataset_suffix
    # '-r1000' -> label 'n=5/r1000' (wandb key n5_r1000), figure eval_n5_r1000_*.png
    ood = args.ood_suffix
    ood_tag = ood.lstrip("-")
    ood_label = f"/{ood_tag}" if ood_tag else ""
    ood_file = f"_{ood_tag}" if ood_tag else ""
    dataset_configs = [
        (f"balanced{sfx}",  "n=3,4", f"eval_n34{cap_file}_temp{EVAL_TEMP}.png"),
        (f"balanced5{sfx}{ood}", f"n=5{ood_label}", f"eval_n5{ood_file}{cap_file}_temp{EVAL_TEMP}.png"),
        (f"balanced6{sfx}{ood}", f"n=6{ood_label}", f"eval_n6{ood_file}{cap_file}_temp{EVAL_TEMP}.png"),
    ]

    for dataset, n_label, fname in dataset_configs:
        print(f"\n[eval] Dataset: {dataset} ({n_label})", flush=True)
        metrics = compute_mean_metrics(args.result_dir, model_name, exp_name, dataset)
        if not metrics:
            print(f"  Skipping: no eval results found for {dataset}", flush=True)
            continue
        print(f"  Checkpoints found: {len(metrics)}", flush=True)
        out = os.path.join(figures_dir, fname)
        plot_eval(metrics, n_label, out, title_prefix)

    # --- Plot 4: response length vs. checkpoint step ---
    print(f"\n[length] Response length...", flush=True)
    from transformers import AutoTokenizer
    tokenizer_path = os.path.join(args.model_base_dir, model_name)
    print(f"  Loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths_by_dataset = {
        n_label: compute_mean_lengths(args.result_dir, model_name, exp_name, dataset, tokenizer)
        for dataset, n_label, _ in dataset_configs
    }
    plot_response_length(lengths_by_dataset, figures_dir, title_prefix, file_tag=ood_file + cap_file)

    # --- WandB: log eval metrics and val reward curve ---
    print("\n[wandb] Logging eval metrics...", flush=True)
    log_to_wandb(
        model_name=model_name,
        exp_name=exp_name,
        val_curve=val_curve,
        metrics_by_dataset={
            n_label: compute_mean_metrics(args.result_dir, model_name, exp_name, dataset)
            for dataset, n_label, _ in dataset_configs
        },
        lengths_by_dataset=lengths_by_dataset,
        condition=args.condition,
    )

    print("\nDone.", flush=True)


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------

_FAMILY_SIZE_RE = re.compile(r"^[0-9.]+[BbMm]$")     # 0.5B, 7B, 270m, 40M
_FAMILY_DATE_RE = re.compile(r"^[0-9]{4}$")          # OLMo-2-0425-1B release date


def model_family(name, max_segments=2):
    """Model-family tag for a model name (or a run name that starts with one).

    Keeps the leading hyphen-separated segments up to the first size-like or
    4-digit release-date segment: Qwen2.5-0.5B -> "Qwen2.5",
    Qwen3-0.6B-Base -> "Qwen3", OLMo-2-1124-7B-pt135B -> "OLMo-2",
    gemma-3-270m -> "gemma-3". Mirrors _model_family() in
    scripts/train_{grpo,sft,opd}.sh -- keep the two in sync.

    Returns "" when no size/date boundary is found, or when more than
    max_segments precede it, so hand-named runs don't get a junk family tag.
    """
    out = []
    for seg in name.split("-"):
        if _FAMILY_SIZE_RE.match(seg) or _FAMILY_DATE_RE.match(seg):
            return "-".join(out) if 0 < len(out) <= max_segments else ""
        out.append(seg)
    return ""


def wandb_tags_for_run(model_name, exp_name, condition=None, stage="eval"):
    tags = [stage]

    inferred_condition = condition
    if inferred_condition is None:
        if "progdistill" in exp_name:
            inferred_condition = "progdistill"
        elif "distill" in exp_name:
            inferred_condition = "distill"
        else:
            inferred_condition = "rl-only"

    tags.append(inferred_condition)
    if "grpo" in exp_name:
        tags.append("grpo")
    elif stage == "eval":
        tags.append("sft")
    # From-scratch 40M models are named Qwen2.5-40M-<shape>; keep the size in
    # the tag (e.g. "40M-wide", "40M-deep") rather than just the shape suffix.
    # Qwen3 base models are named Qwen3-<size>-Base; drop the -Base suffix so
    # the tag is the size ("0.6B") rather than the literal "Base".
    size_tag = model_name.removesuffix("-Base").split("-")[-1]
    if "-40M-" in model_name:
        size_tag = "40M-" + size_tag
    tags.append(size_tag)
    # Family tag ("Qwen2.5"/"Qwen3"/"OLMo-2"/...) -- size_tag drops it, and the
    # family is otherwise only a prefix of the run name, which isn't filterable
    # as a tag in the UI.
    family_tag = model_family(model_name)
    if family_tag:
        tags.append(family_tag)

    for tag in os.environ.get("WANDB_TAGS", "").split(","):
        tag = tag.strip()
        if tag:
            tags.append(tag)

    return list(dict.fromkeys(tags)), inferred_condition


def log_to_wandb(model_name, exp_name, val_curve, metrics_by_dataset, lengths_by_dataset,
                 condition=None):
    # Namespace by user so two people running the same exp_name don't collide
    # on the same deterministic run id (which silently resumes the other user's
    # run and rejects every log as "step less than current").
    user = os.environ.get("USER", "unknown")
    run_id = hashlib.md5(f"{user}-{model_name}-{exp_name}-eval".encode()).hexdigest()[:8]

    tags, condition = wandb_tags_for_run(model_name, exp_name, condition, stage="eval")
    tags.append(f"user:{user}")   # keep per-user filtering tag (helper omits it)

    # Pass x_service_wait directly via Settings — relying on WANDB__SERVICE_WAIT
    # env var has been unreliable on FAS-RC compute nodes (job 13718245 still
    # timed out at 30s despite the env var being exported).
    init_kwargs = dict(
        project="prog_distill",
        entity=os.environ.get("WANDB_ENTITY", "bingbin"),
        name=f"{model_name}-{exp_name}-eval",
        id=run_id,
        resume="allow",
        tags=tags,
        config={"model_name": model_name, "exp_name": exp_name,
                "condition": condition, "user": user},
        settings=wandb.Settings(x_service_wait=180.0),
    )
    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        # wandb-core service can fail to start on flaky nodes. Don't bring
        # down the plot job over it — fall back to offline and continue.
        print(f"  WandB online init failed ({e}); retrying offline...", flush=True)
        try:
            wandb.teardown()
        except Exception:
            pass
        init_kwargs["settings"] = wandb.Settings(
            x_service_wait=180.0, mode="offline"
        )
        try:
            run = wandb.init(**init_kwargs)
        except Exception as e2:
            print(f"  WandB offline init also failed ({e2}); skipping wandb logging.",
                  flush=True)
            return

    # Build per-step dicts for val reward and eval metrics
    val_dict = dict(val_curve) if val_curve else {}

    # Collect all steps that appear in any dataset
    all_steps = sorted(set(
        [s for s, _ in val_curve]
        + [s for metrics in metrics_by_dataset.values() for s, _, _ in metrics]
    ))

    eval_by_step = {}
    for n_label, metrics in metrics_by_dataset.items():
        key = n_label.replace("=", "").replace(",", "_").replace("/", "_")  # n3_4, n5, n6
        for step, m1, m32 in metrics:
            eval_by_step.setdefault(step, {})[f"eval/{key}/mean_at_1"]  = m1
            eval_by_step.setdefault(step, {})[f"eval/{key}/mean_at_32"] = m32

    length_by_step = {}
    for n_label, lengths in lengths_by_dataset.items():
        key = n_label.replace("=", "").replace(",", "_").replace("/", "_")
        for step, mean_len in lengths:
            length_by_step.setdefault(step, {})[f"response_length/{key}"] = mean_len

    for step in all_steps:
        log_dict = {}
        if step in val_dict:
            log_dict["val/reward_mean_at_4"] = val_dict[step]
        log_dict.update(eval_by_step.get(step, {}))
        log_dict.update(length_by_step.get(step, {}))
        if log_dict:
            wandb.log(log_dict, step=step)

    wandb.finish()
    print(f"  WandB run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
