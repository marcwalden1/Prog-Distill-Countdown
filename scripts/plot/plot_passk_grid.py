#!/usr/bin/env python3
"""pass@k vs training step, one row per k and one column per dataset.

Same runs and CLI shape as `plot_olmo2_ladder_pareto.py`, but the metric axis is
expanded: rows are k = 1, 2, 4, 8, 16, 32 instead of just pass@1 / pass@32, so
the sharpening-vs-coverage trade-off can be read at every budget rather than at
its two endpoints.

Reads the `.passk_multi` caches (`scripts/compute_passk_multi.py`) — the
correct-count histogram per eval file — and turns them into pass@k with the
unbiased estimator

    pass@k = 1 - C(N-c, k) / C(N, k)                (Chen et al. 2021)

averaged over prompts. NOTE this makes the k=1 row differ slightly from the
`.passk` pass@1 used by the pareto figures: that one is "the FIRST sample was
correct" (a single draw), this one is c/N (its expectation). Same quantity, less
noise; they agree to ~0.005 in practice.

One figure per temperature. The temperature is part of the eval filename, so
`--temp` selects the eval files and also names the output.

    python3 scripts/plot/plot_passk_grid.py \
        --run "1B 4001B=OLMo-2-0425-1B-pt4001B/balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1" \
        --origin "1B 4001B=balanced-simple-gtsft200-lr3e-5-seed1" \
        --temp 0.6 --tag 7b-vs-1b --set 7b-vs-1b-sftgrpo-lr3e-6kl3e-3
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_passk_curve import pass_at_k  # noqa: E402  (same directory)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _step(path):
    m = re.search(r"global_step_(\d+)", path)
    return int(m.group(1)) if m else None


def _passk_from_hist(rec, ks):
    """{k: pass@k} for one eval file's histogram, or None if k > N for all k."""
    N, hist, n = rec["n_samples"], rec["hist"], rec["n_prompts"]
    if not n:
        return None
    out = {}
    for k in ks:
        if k > N:
            continue
        out[k] = sum(cnt * pass_at_k(N, c, k) for c, cnt in enumerate(hist)) / n
    return out or None


def load_curve(result_dir, model, exp, dataset, temp, ks):
    """{step: {k: pass@k}} across checkpoints of one (model, exp, dataset)."""
    out = {}
    pat = os.path.join(result_dir, model, exp, "global_step_*",
                       f"{dataset}_temp{temp}_n*_max*.json.passk_multi")
    for p in glob.glob(pat):
        s = _step(p)
        if s is None:
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        v = _passk_from_hist(rec, ks)
        if v:
            out[s] = v
    return dict(sorted(out.items()))


def load_origin(result_dir, model, sft_exp, dataset, temp, ks):
    """{k: pass@k} for the pre-GRPO SFT checkpoint, or None."""
    if not sft_exp:
        return None
    pat = os.path.join(result_dir, model, sft_exp, "sft_final",
                       f"{dataset}_temp{temp}_n*_max*.json.passk_multi")
    for p in sorted(glob.glob(pat)):
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        v = _passk_from_hist(rec, ks)
        if v:
            return v
    return None


# Linestyle encodes temperature when more than one is plotted; colour always
# encodes the run. Solid is the first --temp given.
TEMP_LS = ["-", "--", "-.", ":"]


def plot_grid(runs, datasets, ks, temps, out_path, title):
    multi = len(temps) > 1
    fig, axes = plt.subplots(len(ks), len(datasets),
                             figsize=(5.2 * len(datasets), 2.9 * len(ks)),
                             sharex=True, squeeze=False)
    for ri, k in enumerate(ks):
        for ci, (ds, ds_label) in enumerate(datasets):
            ax = axes[ri][ci]
            drawn = False
            for r in runs:
                for ti, t in enumerate(temps):
                    c = r["color"]
                    ls = TEMP_LS[ti % len(TEMP_LS)] if multi else r.get("ls", "-")
                    curve = r["curves"][t][ds]
                    steps = [s for s in sorted(curve) if k in curve[s]]
                    ys = [curve[s][k] for s in steps]
                    if steps:
                        drawn = True
                        ax.plot(steps, ys, marker="o", ls=ls, color=c, ms=3.0,
                                lw=1.5, alpha=0.85, label=r["label"], zorder=2)
                        ax.plot(steps[-1], ys[-1], marker="*", ms=14, color=c,
                                mec="black", mew=0.7, ls="none", zorder=5,
                                label="_nolegend_")
                    o = (r["origins"][t].get(ds) or {}).get(k)
                    if o is not None:
                        drawn = True
                        # Hollow origin marker for the non-solid temperatures, so
                        # two SFT dots of the same colour stay distinguishable.
                        ax.plot(0, o, marker="o", ms=7, color=c, mec="black",
                                mew=0.7, ls="none", zorder=5, label="_nolegend_",
                                mfc=(c if ti == 0 else "#fcfcfb"))
                        if steps:
                            ax.plot([0, steps[0]], [o, ys[0]], ls=ls, color=c,
                                    lw=1.0, alpha=0.4, zorder=1, label="_nolegend_")
            if not drawn:
                ax.text(0.5, 0.5, f"no data for k={k}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#b00")
            if ri == 0:
                ax.set_title(ds_label, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"pass@{k}", fontsize=10, fontweight="bold")
            if ri == len(ks) - 1:
                ax.set_xlabel("GRPO step   (0 = SFT student, pre-GRPO)")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
    # Legend once, outside the panels: 6x3 axes leave no safe in-panel corner.
    shapes = [Line2D([], [], ls="none", marker="*", ms=14, mfc="0.55", mec="black",
                     mew=0.7, label="last available RL checkpoint")]
    if any(any(r["origins"][t] for t in temps) for r in runs):
        shapes.append(Line2D([], [], ls="none", marker="o", ms=7, mfc="0.55",
                             mec="black", mew=0.7,
                             label="SFT (pre-GRPO), drawn at step 0"))
    handles = [Line2D([], [], color=r["color"],
                      ls=("-" if multi else r.get("ls", "-")), lw=2,
                      marker="o", ms=4, label=r["label"]) for r in runs]
    if multi:
        handles += [Line2D([], [], color="0.35", ls=TEMP_LS[i % len(TEMP_LS)],
                           lw=2, label=f"temp {t}") for i, t in enumerate(temps)]
    fig.legend(handles=handles + shapes, loc="upper center", ncol=min(7, len(handles) + 2),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.945))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=MODEL/EXP")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--origin", action="append", default=None, metavar="LABEL=SFT_EXP")
    ap.add_argument("--id-dataset", default="balanced-simple")
    ap.add_argument("--ood", action="append", default=None, metavar="DATASET=LABEL")
    ap.add_argument("--run-color", action="append", default=None, metavar="LABEL=#HEX")
    ap.add_argument("--run-linestyle", action="append", default=None, metavar="LABEL=LS")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--temp", default="0.6",
                    help="eval temperature, or a comma-separated list. With several, "
                         "linestyle encodes temperature (first = solid) and colour "
                         "still encodes the run.")
    ap.add_argument("--set", dest="set_dir", default=None, metavar="FOLDER")
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    temps = [t.strip() for t in args.temp.split(",") if t.strip()]
    if len(temps) > 1 and args.run_linestyle:
        print("NOTE: --run-linestyle ignored — with several --temp values the "
              "linestyle encodes temperature, not run.")
    ood_list = ([tuple(o.split("=", 1)) for o in args.ood] if args.ood
                else [("balanced5-simple", "n=5"), ("balanced6-simple", "n=6")])
    datasets = [(args.id_dataset, "n=3,4 (in-dist)")] + [(d, f"{l} (OOD)") for d, l in ood_list]
    all_ds = [d for d, _ in datasets]

    origin_of = dict(o.split("=", 1) for o in (args.origin or []))
    color_of = dict(o.split("=", 1) for o in (args.run_color or []))
    ls_of = dict(o.split("=", 1) for o in (args.run_linestyle or []))
    labels = [spec.split("=", 1)[0] for spec in args.run]
    for lbl in origin_of:
        if lbl not in labels:
            raise SystemExit(f"ERROR: --origin names run '{lbl}', which is not among "
                             f"--run labels {labels}")

    runs = []
    for ci, spec in enumerate(args.run):
        label, path = spec.split("=", 1)
        model, exp = path.split("/", 1)
        sft_exp = origin_of.get(label, "")
        curves, origins = {}, {}
        print(f"{label}: {model}/{exp}")
        for t in temps:
            curves[t] = {ds: load_curve(args.result_dir, model, exp, ds, t, ks)
                         for ds in all_ds}
            origins[t] = {ds: o for ds in all_ds
                          if (o := load_origin(args.result_dir, model, sft_exp, ds, t, ks))}
            counts = {ds: len(c) for ds, c in curves[t].items()}
            print(f"    temp {t}: checkpoints per dataset: {counts}   "
                  f"SFT origin: {sorted(origins[t])}")
        if not any(any(curves[t].values()) or origins[t] for t in temps):
            raise SystemExit(
                f"ERROR: no .passk_multi caches for {model}/{exp} at temp(s) "
                f"{', '.join(temps)}.\n"
                f"       Build them:  python3 scripts/compute_passk_multi.py "
                f"--model {model} --exp {exp} --temp {temps[0]}")
        runs.append({"label": label, "model": model, "exp": exp, "curves": curves,
                     "origins": origins,
                     "color": color_of.get(label, COLORS[ci % len(COLORS)]),
                     "ls": ls_of.get(label, "-")})

    fig_dir = args.figures_dir or os.path.join("figures", "olmo2-ladder",
                                               args.set_dir or args.tag or "untagged")
    suffix = f"_{args.tag}" if args.tag else ""
    title = args.title or "pass@k vs GRPO step"
    temp_tag = "-".join(temps)
    temp_desc = (f"sampling temperature {temps[0]}" if len(temps) == 1 else
                 "sampling temperature: " +
                 ", ".join(f"{t} ({TEMP_LS[i % len(TEMP_LS)]})"
                           for i, t in enumerate(temps)))
    plot_grid(runs, datasets, ks, temps,
              os.path.join(fig_dir, f"passk_grid_temp{temp_tag}{suffix}.png"),
              f"{title}\n{temp_desc}   ·   "
              f"pass@k = fraction of prompts solved by >=1 of k samples "
              f"(unbiased, from 32 draws)")

    # Final-checkpoint table, so the figure's endpoints are readable as numbers.
    w = max([len(r["label"]) for r in runs] + [3])
    for ds, ds_label in datasets:
        for t in temps:
            print(f"\nFinal checkpoint, {ds_label} (temp {t}):")
            hdr = (f"  {'run':<{w}} {'step':>5} "
                   + " ".join(f"{'pass@' + str(k):>9}" for k in ks))
            print(hdr); print("  " + "-" * (len(hdr) - 2))
            for r in runs:
                c = r["curves"][t][ds]
                if not c:
                    continue
                last = max(c)
                print(f"  {r['label']:<{w}} {last:>5} "
                      + " ".join(f"{c[last].get(k, float('nan')):>9.4f}" for k in ks))


if __name__ == "__main__":
    main()
