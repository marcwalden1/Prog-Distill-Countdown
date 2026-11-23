#!/usr/bin/env python3
"""Draft ID-vs-OOD plots for the Gemma-270M GRPO groups in results/grpo_ckpts.md.

The four groups under the `# Gemma-270M` heading of results/grpo_ckpts.md
("Mine (bingbin)", "base GRPO", "KD (distill) + GRPO", "PD (progdistill) + GRPO")
each get one figure. Every run in a group is drawn as an ID-vs-OOD accuracy
trajectory: in-distribution (balanced, n=3,4) on x vs OOD (n=5 or n=6) on y, one
marker per evaluated checkpoint, connected in step order (open circle = first
step, star = last).

Each figure is a 2x2 grid: rows = metric (mean@1, mean@32), cols = OOD set
(n=5, n=6). Reads only the tiny `.scores` caches ([step, mean@1, mean@32]) next
to each eval JSON, so it is safe to run locally (like scripts/plot/plot_pareto_2d.py
and scripts/plot/plot_id_vs_ood.py).

Usage: python3 scripts/plot/plot_id_vs_ood_gemma_groups.py
"""
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
# Score directly from the full eval JSON via _score_item. We deliberately do NOT
# use plot_results._score_checkpoint / the adjacent `.scores` caches: the caches
# in results_shlok/ were written by an old plot_results.py that capped scoring at
# the first 1000 prompts, which overestimates n=5/n=6 accuracy (the JSON is
# ordered by pattern index, so the head slice is an easier subsample).
from plot_results import _score_item
CKPTS_MD = os.path.join(REPO, "results", "grpo_ckpts.md")
OUT_DIR = os.path.join(REPO, "figures", "gemma-3-270m", "id_vs_ood_draft")

TEMP = "0.6"
ID_DATASET = "balanced"                       # n=3,4 in-distribution
OOD = [("balanced5", "n=5"), ("balanced6", "n=6")]
METRICS = [("mean@1", 1), ("mean@32", 2)]     # (name, index into .scores record)

# (heading marker in the md, slug for the output filename)
GROUPS = [
    ("## Mine (bingbin)", "mine"),
    ("### base GRPO", "base_grpo"),
    ("### KD (distill) + GRPO", "kd_distill_grpo"),
    ("### PD (progdistill) + GRPO", "pd_progdistill_grpo"),
]

plt.rcParams.update({"font.size": 12})


def parse_groups():
    """{slug: [run_dir, ...]} parsed from the Gemma-270M section of grpo_ckpts.md."""
    with open(CKPTS_MD) as f:
        lines = f.readlines()
    # restrict to the Gemma-270M section
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "# Gemma-270M")
    section = lines[start:]

    markers = {m: slug for m, slug in GROUPS}
    out = {slug: [] for _, slug in GROUPS}
    cur = None
    for ln in section:
        stripped = ln.strip()
        # a heading switches the active group (match longest marker first)
        matched = None
        for m in sorted(markers, key=len, reverse=True):
            if stripped.startswith(m):
                matched = markers[m]
                break
        if matched is not None:
            cur = matched
            continue
        if stripped.startswith("Notes"):     # trailing prose bullets are not runs
            cur = None
            continue
        if cur is None or not stripped.startswith("- "):
            continue
        paths = re.findall(r"`([^`]+)`", ln)
        if paths:
            out[cur].append(paths[-1])   # last backtick token on the line = the path
    return out


def resolve_score_dir(path):
    """Return a dir that contains eval JSONs, trying `path` then results/gemma-3-270m/<base>."""
    candidates = [path,
                  os.path.join(REPO, "results", "gemma-3-270m", os.path.basename(path))]
    for c in candidates:
        if glob.glob(os.path.join(c, "global_step_*", f"*_temp{TEMP}_*.json")):
            return c
    return None


# Local memoization keyed by (json path, mtime). `full` in the name marks that
# these are full-set scores, so a stale cache from the buggy 1k-capped path can't
# be reused. Needed because the read-only results_shlok/ tree can't cache next to
# its JSONs.
CACHE_FILE = os.path.join(OUT_DIR, ".score_cache_full.json")
_CACHE = {}   # abspath(json) -> [json_mtime, step, mean@1, mean@32]


def _load_cache():
    global _CACHE
    try:
        with open(CACHE_FILE) as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = {}


def _save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_CACHE, f)
    except Exception:
        pass


def _score_full_json(jp):
    """(mean@1, mean@32) over ALL prompts in one eval JSON, or None."""
    try:
        with open(jp) as f:
            data = json.load(f)
    except Exception:
        return None
    items = [(it["outputs"], it["target"], it["nums"])
             for it in data if it.get("outputs")]
    if not items:
        return None
    with Pool(8) as pool:
        scored = [s for s in pool.map(_score_item, items) if s is not None]
    if not scored:
        return None
    n = len(scored)
    return (sum(s[0] for s in scored) / n, sum(s[1] for s in scored) / n)


def load_scores(run_dir, dataset):
    """{step: (mean@1, mean@32)} for one (run, dataset), scored over the FULL
    prompt set (not the first 1k). Memoized in a local cache keyed by json mtime.
    """
    out = {}
    pattern = os.path.join(run_dir, "global_step_*",
                           f"{dataset}_temp{TEMP}_*.json")
    for jp in glob.glob(pattern):
        m = re.search(r"global_step_(\d+)", jp)
        if not m:
            continue
        step = int(m.group(1))
        key = os.path.abspath(jp)
        try:
            mt = os.path.getmtime(jp)
        except OSError:
            continue
        rec = _CACHE.get(key)
        if rec and rec[0] >= mt:
            out[step] = (float(rec[2]), float(rec[3]))
            continue
        res = _score_full_json(jp)           # (mean@1, mean@32) or None
        if res is not None:
            _CACHE[key] = [mt, step, float(res[0]), float(res[1])]
            out[step] = (float(res[0]), float(res[1]))
    return out


def run_label(path):
    """Short legend label from the run dir basename (sftlr / lr / kl / early flag)."""
    b = os.path.basename(path)
    parts = []
    for tag, pat in (("sftlr", r"sftlr([0-9]+e-?[0-9]+)"),
                     ("lr", r"grpo-lr([0-9]+e-?[0-9]+)"),
                     ("kl", r"kl([0-9]+e-?[0-9]+)")):
        m = re.search(pat, b)
        if m:
            parts.append(f"{tag}{m.group(1)}")
    if "early400" in b:
        parts.append("early400")
    return " ".join(parts) if parts else b


def plot_group(slug, run_dirs):
    runs = []
    for p in run_dirs:
        d = resolve_score_dir(p)
        if d is None:
            print(f"  [skip] no eval JSONs under {p}")
            continue
        runs.append((run_label(p), d))
    if not runs:
        print(f"[{slug}] no usable runs")
        return

    # per-run data: {label: {dataset: {step: (mean@1, mean@32)}}}
    wanted = [ID_DATASET] + [ds for ds, _ in OOD]
    data = {lbl: {ds: load_scores(d, ds) for ds in wanted} for lbl, d in runs}

    cmap = plt.get_cmap("tab20")
    colors = {lbl: cmap(i % 20) for i, (lbl, _) in enumerate(runs)}

    fig, axes = plt.subplots(len(METRICS), len(OOD), figsize=(14, 12))
    for r, (metric_name, mi) in enumerate(METRICS):
        idx = mi - 1
        for c, (ood_ds, ood_label) in enumerate(OOD):
            ax = axes[r][c]
            for lbl, _d in runs:
                id_by, ood_by = data[lbl][ID_DATASET], data[lbl][ood_ds]
                steps = sorted(set(id_by) & set(ood_by))
                if not steps:
                    continue
                color = colors[lbl]
                xs = [id_by[s][idx] for s in steps]
                ys = [ood_by[s][idx] for s in steps]
                ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1.6, zorder=1)
                ax.scatter(xs, ys, s=28, color=color, label=lbl, zorder=2)
                ax.scatter([xs[0]], [ys[0]], s=90, facecolors="none",
                           edgecolors=color, linewidths=1.6, zorder=3)
                ax.scatter([xs[-1]], [ys[-1]], s=150, marker="*", color=color,
                           edgecolors="#0b0b0b", linewidths=0.5, zorder=4)
            if slug != "base_grpo":   # base GRPO spans low ID accuracy; don't clip it
                ax.set_xlim(0.6, 1.0)
            ax.set_xlabel(f"ID {metric_name} (n=3,4)")
            ax.set_ylabel(f"OOD {metric_name} ({ood_label})")
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False,
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               title="○ first step   ★ last step", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out = os.path.join(OUT_DIR, f"{slug}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(runs)} runs)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _load_cache()
    groups = parse_groups()
    for _, slug in GROUPS:
        print(f"[{slug}] {len(groups[slug])} run(s) listed")
        plot_group(slug, groups[slug])
    _save_cache()


if __name__ == "__main__":
    main()
