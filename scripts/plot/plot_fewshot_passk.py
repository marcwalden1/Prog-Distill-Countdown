#!/usr/bin/env python3
"""pass@k curves for the few-shot base-model evals (eval.py --n_shot).

One sub-plot per response format, one colour per model, line style by shot count:

    dotted = 0-shot,  dashed = 2-shot,  solid = 8-shot

A 0-shot arm has no exemplars and therefore no format, so its JSON lives at the
root of results/baselines/<model>/ (eval.py only adds the <K>shot-<format>
subdir for --n_shot > 0). The same 0-shot curve is drawn in both panels as the
shared no-exemplar baseline the two formats are measured against.

Which models are drawn comes from MODEL_PRESETS[--tag] (add a preset, or pass
--models "dir:label:#rrggbb,..."), so one script covers the OLMo-2 ladder and
the Qwen3 base pair.

x is the number of samples drawn per prompt (1..32), y is pass@k -- the fraction
of prompts solved by at least one of k samples. Each eval JSON holds 32 samples
per prompt, so a whole curve comes out of data already on disk: with N samples of
which c are correct,

    pass@k = 1 - C(N-c, k) / C(N, k)                (Chen et al. 2021)

the unbiased estimate over all size-k subsets, shared with
scripts/plot/plot_passk_curve.py rather than re-derived. pass@1 equals mean@1 and
pass@32 equals the pass@32 column of scripts/score_fewshot_evals.py.

Samples are truncated at the first </answer> before scoring, exactly as
score_fewshot_evals.py does: a few-shot base model answers the real puzzle and
then invents more, and the grader reads the LAST <answer> tag. See
research_logs/2026-09-11_sft-formats-and-fewshot-eval.md.

    python3 scripts/plot/plot_fewshot_passk.py
    python3 scripts/plot/plot_fewshot_passk.py --dataset balanced-simple --tag olmo2
    python3 scripts/plot/plot_fewshot_passk.py --tag qwen3_base --shots 0,2,8
    python3 scripts/plot/plot_fewshot_passk.py --tag qwen25_base --shots 0,2,8 \
        --ks 1,2,4,8,16,32,64,128,256

Writes figures/fewshot/fewshot_passk_<tag>.png (+ .pdf) and the same numbers as
analysis_data/fewshot_passk_<tag>.csv.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/plot
SCRIPTS = os.path.dirname(HERE)                            # scripts
ROOT = os.path.dirname(SCRIPTS)                            # repo root
for _p in (ROOT, SCRIPTS, HERE):
    sys.path.insert(0, _p)
from grader_utils import compute_score  # noqa: E402
from plot_passk_curve import pass_at_k  # noqa: E402  (unbiased estimator)
from score_fewshot_evals import truncate_at_answer  # noqa: E402

# Categorical slots of the validated default palette (1 blue, 2 orange, 3 aqua,
# 7 violet), assigned smallest-model-first so every panel of a family reads the
# same way. Within a preset a model keeps its hue in every figure, so a filtered
# rerun never repaints the survivors. Slots restart at 1 per family because the
# families live in separate figures; a CROSS-FAMILY figure must therefore pass
# --models explicitly to avoid two models sharing blue. Each preset's hue set is
# validated as a categorical palette (the qwen 2-slot pair: worst adjacent CVD
# dE 24.7 protan, normal-vision 33.6, contrast >= 3:1 on a light surface).
MODEL_PRESETS = {
    # tag -> [(model dir, short label, colour)]
    "olmo2": [
        ("OLMo-2-0425-1B-pt1007B", "1B / 1007B", "#2a78d6"),
        ("OLMo-2-0425-1B-pt4001B", "1B / 4001B", "#eb6834"),
        ("OLMo-2-1124-7B-pt135B",  "7B / 135B",  "#1baf7a"),
        ("OLMo-2-1124-7B-pt3896B", "7B / 3896B", "#4a3aa7"),
    ],
    "qwen3_base": [
        ("Qwen3-0.6B-Base", "Qwen3 0.6B", "#2a78d6"),
        ("Qwen3-1.7B-Base", "Qwen3 1.7B", "#eb6834"),
        ("Qwen3-4B-Base",   "Qwen3 4B",   "#1baf7a"),
    ],
    # Qwen2.5 base models carry no "-Base" suffix -- the plain name IS the base
    # model in that family, and only the instruct variants are suffixed.
    "qwen25_base": [
        ("Qwen2.5-0.5B", "Qwen2.5 0.5B", "#2a78d6"),
        ("Qwen2.5-1.5B", "Qwen2.5 1.5B", "#eb6834"),
    ],
}
# Sparser line = fewer exemplars, so the legend reads in the same order as the
# ablation. 4-shot is here only so an in-between arm does not crash the lookup.
SHOT_STYLE = {0: (":", "0-shot"), 2: ("--", "2-shot"),
              4: ("-.", "4-shot"), 8: ("-", "8-shot")}
FORMATS = ["answer", "search"]
FORMAT_TITLE = {"answer": "answer-only exemplars", "search": "search-trace exemplars"}


def _count_correct(rec):
    outs = [truncate_at_answer(o) for o in rec["outputs"]]
    extra = {"numbers": rec["nums"]}
    c = sum(1 for o in outs if compute_score(None, o, rec["target"], extra,
                                             verbose=False) == 1.0)
    return len(outs), c


_N_RE = re.compile(r"_n(\d+)_max")


def pick_eval_json(paths, want_n=None):
    """(path, N) of the eval JSON to read when several sample budgets coexist.

    eval.py names files <ds>_temp<T>_n<N>_max<M>.json, so a 32-sample and a
    256-sample eval of the same arm sit side by side. A plain sorted()[0] would
    take n256 over n32 only because '2' sorts before '3' -- decide on N itself.
    Default is the largest budget on disk, since pass@k needs N >= k.
    """
    cand = []
    for path in paths:
        m = _N_RE.search(os.path.basename(path))
        if m:
            cand.append((int(m.group(1)), path))
    if not cand:
        return None, None
    if want_n is not None:
        hit = [c for c in cand if c[0] == want_n]
        return (hit[0][1], want_n) if hit else (None, None)
    n, path = max(cand)
    return path, n


def curve_for(path, ks, workers=16):
    """[(k, pass@k)] for one eval JSON, cached next to it as .passkcurve."""
    cache = path + ".passkcurve"
    try:
        if os.path.getmtime(cache) >= os.path.getmtime(path):
            with open(cache) as f:
                rec = json.load(f)
            if rec["ks"] == ks:
                return list(zip(rec["ks"], rec["vals"])), rec["n_prompts"]
    except Exception:
        pass
    with open(path) as f:
        data = json.load(f)
    data = [r for r in data if r.get("outputs")]
    with Pool(workers) as pool:
        counts = pool.map(_count_correct, data)
    vals = [sum(pass_at_k(n, c, k) for n, c in counts) / len(counts) for k in ks]
    try:
        with open(cache, "w") as f:
            json.dump({"ks": ks, "vals": vals, "n_prompts": len(counts)}, f)
    except Exception:
        pass
    return list(zip(ks, vals)), len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--dataset", default="balanced-simple")
    ap.add_argument("--shots", default="2,8")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--figures-dir", default=os.path.join("figures", "fewshot"))
    ap.add_argument("--csv-dir", default="analysis_data")
    ap.add_argument("--tag", default="olmo2",
                    help="Names the output files and, unless --models is given, "
                         "selects the model set from MODEL_PRESETS.")
    ap.add_argument("--models", default=None,
                    help='Override the preset: "dir:label:#rrggbb,dir:label:#rrggbb".')
    ap.add_argument("--samples", type=int, default=None,
                    help="Read the eval JSON with exactly this many samples per "
                         "prompt (the _n<N>_ field). Default: the largest budget "
                         "on disk for each arm.")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.models:
        model_style = [tuple(spec.split(":", 2)) for spec in args.models.split(",")]
    elif args.tag in MODEL_PRESETS:
        model_style = MODEL_PRESETS[args.tag]
    else:
        raise SystemExit(f"ERROR: --tag {args.tag} is not in MODEL_PRESETS "
                         f"({', '.join(MODEL_PRESETS)}); pass --models explicitly.")

    ks = [int(x) for x in args.ks.split(",")]
    shots = [int(x) for x in args.shots.split(",")]
    unknown = [s for s in shots if s not in SHOT_STYLE]
    if unknown:
        raise SystemExit(f"ERROR: no line style for shot count(s) {unknown}; "
                         f"add them to SHOT_STYLE.")

    rows, panels = [], {f: [] for f in FORMATS}
    for fmt in FORMATS:
        for model, label, color in model_style:
            for k_shot in shots:
                # 0-shot has no exemplars and so no format subdir; the same JSON
                # is the baseline curve in both panels.
                subdir = () if k_shot == 0 else (f"{k_shot}shot-{fmt}",)
                pat = os.path.join(args.result_dir, "baselines", model, *subdir,
                                   f"{args.dataset}_temp*_n*_max*.json")
                hits = [p for p in glob.glob(pat) if p.endswith(".json")]
                path, n_samples = pick_eval_json(hits, args.samples)
                if path is None:
                    want = "" if args.samples is None else f" at n={args.samples}"
                    print(f"  SKIP {model} {k_shot}-shot {fmt}: no eval JSON{want}")
                    continue
                # pass@k is undefined for k > N (one cannot draw 64 samples from
                # 32), so an arm evaluated at a smaller budget contributes a
                # shorter curve rather than crashing the sum in curve_for.
                arm_ks = [k for k in ks if k <= n_samples]
                if len(arm_ks) < len(ks):
                    print(f"  NOTE {model} {k_shot}-shot {fmt}: N={n_samples}, "
                          f"dropping k>{n_samples}")
                curve, n_prompts = curve_for(path, arm_ks, args.workers)
                panels[fmt].append((label, color, k_shot, curve))
                print(f"  {label:11s} {k_shot}-shot {fmt:7s} "
                      f"prompts={n_prompts} N={n_samples}  " +
                      "  ".join(f"pass@{k}={v:.4f}" for k, v in curve))
                for k, v in curve:
                    rows.append(dict(model=model, label=label, format=fmt,
                                     shots=k_shot, k=k, pass_at_k=round(v, 6),
                                     n_prompts=n_prompts, n_samples=n_samples,
                                     dataset=args.dataset))
    if not any(panels.values()):
        raise SystemExit("ERROR: no few-shot eval JSONs found — nothing to plot.")

    ymax = max(v for _, _, _, c in sum(panels.values(), []) for _, v in c)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, fmt in zip(axes, FORMATS):
        ends = []
        for label, color, k_shot, curve in panels[fmt]:
            ls, _ = SHOT_STYLE[k_shot]
            xs = [k for k, _ in curve]
            ys = [v for _, v in curve]
            ax.plot(xs, ys, ls, color=color, linewidth=2, marker="o",
                    markersize=4.5, zorder=3)
            # Direct label on the solid (8-shot) curve: identity is never carried
            # by colour alone, and it is the relief the contrast WARN requires.
            if k_shot == max(shots):
                ends.append([ys[-1], ys[-1], label, color, xs[-1]])

        # Nudge labels apart where two curves finish close together (the 1B pair
        # in the search panel lands ~0.01 apart, which overlaps at 8pt).
        gap = ymax * 0.055
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            if ends[i][1] - ends[i - 1][1] < gap:
                ends[i][1] = ends[i - 1][1] + gap
        for y_data, y_text, label, color, x in ends:
            ax.annotate(label, (x, y_data), xytext=(6, 0), va="center",
                        textcoords="offset points", color=color, fontsize=8,
                        zorder=4,
                        xycoords="data" if y_text == y_data else None,
                        annotation_clip=False)
            if y_text != y_data:   # redraw at the nudged height with a leader
                ax.texts[-1].remove()
                ax.annotate(label, xy=(x, y_data), xytext=(x * 1.10, y_text),
                            va="center", color=color, fontsize=8, zorder=4,
                            annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.7, alpha=0.6,
                                            shrinkA=0, shrinkB=2))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("samples per prompt (k)")
        ax.set_title(FORMAT_TITLE[fmt], fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(min(ks) * 0.9, max(ks) * 1.45)
        ax.set_ylim(0, ymax * 1.12)
    axes[0].set_ylabel("pass@k")

    handles = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in model_style]
    handles += [Line2D([], [], color="0.35", lw=2, ls=SHOT_STYLE[s][0],
                       label=SHOT_STYLE[s][1]) for s in shots]
    axes[0].legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
                   ncol=2, columnspacing=1.2)
    kind = "Base-model" if 0 in shots else "Few-shot base-model"
    budgets = sorted({r["n_samples"] for r in rows})
    budget = (f"{budgets[0]} samples/prompt" if len(budgets) == 1
              else f"{min(budgets)}-{max(budgets)} samples/prompt")
    fig.suptitle(f"{kind} pass@k — {args.dataset} ({budget}, temp 0.6)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(args.figures_dir, exist_ok=True)
    base = os.path.join(args.figures_dir, f"fewshot_passk_{args.tag}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nWrote {base}.png / .pdf")

    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, f"fewshot_passk_{args.tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
