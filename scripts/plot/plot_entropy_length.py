"""Plot sample entropy and generation length vs. training step on the ID val set.

For each run (one curve per run), reads the per-checkpoint eval JSONs
(balanced_temp0.6_n32_max1024.json, n=3,4 val set) and computes:

- sample entropy: per prompt, the Shannon entropy (bits) of the empirical
  distribution over the 32 sampled final answers (last <answer>...</answer>
  span, whitespace-normalized; missing answer is its own bucket), averaged
  over prompts. Max is log2(32) = 5 bits (all 32 answers distinct).
  NOTE: this is sampled-solution diversity, not token-level policy entropy —
  the eval JSONs store no logprobs.
- generation length: mean response length in tokens, same definition as
  plot_results.py (.lengths caches are reused when fresh, written when possible).

Usage: python3 scripts/plot/plot_entropy_length.py  (paths/runs configured below)
"""

import json
import math
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

RUNS = [
    ("Qwen2.5-1.5B GRPO",
     "results/Qwen2.5-1.5B/balanced-grpo-lr3e-6-kl3e-3-seed1"),
    ("Qwen2.5-0.5B GRPO",
     "results/Qwen2.5-0.5B/balanced-grpo-seed1"),
    ("Qwen2.5-0.5B KD + GRPO",
     "/n/holylabs/LABS/kempner_bingbin_lab/Lab/sdholakia/results/Qwen2.5-0.5B/"
     "balanced-distill-sftlr3e-5-grpo-lr1e-6-kl3e-4-seed1"),
    ("Qwen2.5-0.5B PD mixture + GRPO",
     "results/Qwen2.5-0.5B/"
     "balanced-progdistill-mixture-t1p5b-sftlr3e-5-steps1600-grpo-lr1e-6-kl3e-3-seed1"),
]

EVAL_FILE = "balanced_temp0.6_n32_max1024.json"
TOKENIZER_PATH = "/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models/Qwen2.5-0.5B"
FIGURES_DIR = "figures/entropy_length"

# dataviz categorical slots 1-4 (validated); markers double as CVD-safe encoding
# COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
COLORS = {
    "Qwen2.5-1.5B GRPO": mcolors.CSS4_COLORS["lightcoral"],
    "Qwen2.5-0.5B GRPO": mcolors.CSS4_COLORS["dodgerblue"],
    "Qwen2.5-0.5B KD + GRPO": mcolors.CSS4_COLORS["forestgreen"],
    "Qwen2.5-0.5B PD + GRPO": mcolors.CSS4_COLORS["darkorange"],
    "Qwen2.5-0.5B PD mixture + GRPO": mcolors.CSS4_COLORS["mediumorchid"],
}
MARKERS = ["o", "s", "^", "D"]

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")
STEP_RE = re.compile(r"global_step_(\d+)")


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return " ".join(matches[-1].split())


def sample_entropy_bits(outputs):
    counts = Counter(extract_answer(o) for o in outputs)
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def load_cache(cache_path, json_path):
    try:
        if os.path.getmtime(cache_path) >= os.path.getmtime(json_path):
            with open(cache_path) as f:
                return tuple(json.load(f))
    except Exception:
        pass
    return None


def save_cache(cache_path, value):
    try:
        with open(cache_path, "w") as f:
            json.dump(list(value), f)
    except Exception:
        pass  # other users' trees may be read-only


def collect_run(run_dir, tokenizer):
    """Return sorted lists [(step, entropy_bits)], [(step, mean_len_tokens)]."""
    entropies, lengths = [], []
    for name in sorted(os.listdir(run_dir)):
        m = STEP_RE.fullmatch(name)
        json_path = os.path.join(run_dir, name, EVAL_FILE)
        if not m or not os.path.isfile(json_path):
            continue
        step = int(m.group(1))

        ent = load_cache(json_path + ".entropy32", json_path)
        length = load_cache(json_path + ".lengths", json_path)

        if ent is None or length is None:
            with open(json_path) as f:
                data = json.load(f)
            if ent is None:
                mean_ent = sum(sample_entropy_bits(it["outputs"]) for it in data) / len(data)
                ent = (step, mean_ent)
                save_cache(json_path + ".entropy32", ent)
            if length is None:
                all_outputs = [out for it in data for out in it["outputs"]]
                ids = tokenizer(all_outputs, add_special_tokens=False)["input_ids"]
                length = (step, sum(len(x) for x in ids) / len(ids))
                save_cache(json_path + ".lengths", length)

        entropies.append(ent)
        lengths.append(length)
    return sorted(entropies), sorted(lengths)


def plot_curves(series_by_run, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, series) in enumerate(series_by_run.items()):
        if not series:
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, color=COLORS[label], marker=MARKERS[i], markersize=5,
                linewidth=2, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ent_by_run, len_by_run = {}, {}
    for label, run_dir in RUNS:
        ent, lens = collect_run(run_dir, tokenizer)
        ent_by_run[label], len_by_run[label] = ent, lens
        print(f"\n{label}  ({run_dir})")
        for (s, e), (_, l) in zip(ent, lens):
            print(f"  step {s:>5}  entropy {e:.3f} bits  mean length {l:7.1f} tok")

    plot_curves(ent_by_run, "Sample entropy (bits, 32 samples/prompt)",
                "Answer-distribution entropy on ID val set (n=3,4), temp 0.6",
                os.path.join(FIGURES_DIR, "entropy_vs_step_balanced.png"))
    plot_curves(len_by_run, "Mean response length (tokens)",
                "Generation length on ID val set (n=3,4), temp 0.6",
                os.path.join(FIGURES_DIR, "length_vs_step_balanced.png"))


if __name__ == "__main__":
    main()
