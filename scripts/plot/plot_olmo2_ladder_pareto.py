#!/usr/bin/env python3
"""ID-vs-OOD and pass@1-vs-pass@32 plots comparing several (model, exp) runs.

Reads ONLY the `.passk` caches ([step, pass@1, pass@32, n_prompts]) written by
scripts/compute_passk.py, so plotting is seconds rather than hours.

  pass@1  — fraction of prompts whose FIRST sample is exactly correct.
            (Numerically identical to plot_results.py's mean@1.)
  pass@32 — fraction of prompts with AT LEAST ONE exactly-correct sample of 32.
            A coverage metric.

Deliberately does NOT plot mean@32. With scores in {0.0, 0.1, 1.0} and format
compliance near 100%, mean@32 reduces to `0.1 + 0.9 * mean@1` (verified: max
residual 0.012 over 102 points), i.e. a monotone rescaling of mean@1 whose gap is
pure format credit. On n=6 that credit is ~88% of the number. It carries no
information beyond mean@1, so it is omitted to avoid overstating performance.

Unlike scripts/plot/plot_pareto_2d.py (one model, two exps) this compares runs
that differ in *model*, which is what the OLMo-2 pretraining ladder needs.

Usage:
  python3 scripts/plot/plot_olmo2_ladder_pareto.py \
      --run 1007B=OLMo-2-0425-1B-pt1007B/balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1 \
      --run 4001B=OLMo-2-0425-1B-pt4001B/balanced-simple-gtsft200-grpo-lr3e-6-kl3e-3-seed1 \
      --origin "1007B SFT+GRPO=balanced-simple-gtsft200-lr3e-5-seed1" \
      --tag ladder
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

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
PAD = 0.04          # fraction of each axis' own data range added as margin


def _step(path):
    m = re.search(r"global_step_(\d+)", path)
    return int(m.group(1)) if m else None


def load_curve(result_dir, model, exp, dataset):
    """{step: (pass@1, pass@32)} across checkpoints of one (model, exp, dataset)."""
    out = {}
    pat = os.path.join(result_dir, model, exp, "global_step_*", f"{dataset}_temp*.json.passk")
    for p in glob.glob(pat):
        s = _step(p)
        if s is None:
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        if isinstance(rec, list) and len(rec) >= 3:
            out[s] = (rec[1], rec[2])
    return dict(sorted(out.items()))


def load_origin(result_dir, model, sft_exp, dataset):
    """(pass@1, pass@32) for the pre-GRPO SFT checkpoint, or None."""
    if not sft_exp:
        return None
    pat = os.path.join(result_dir, model, sft_exp, "sft_final", f"{dataset}_temp*.json.passk")
    for p in sorted(glob.glob(pat)):
        try:
            with open(p) as f:
                rec = json.load(f)
            if isinstance(rec, list) and len(rec) >= 3:
                return (rec[1], rec[2])
        except Exception:
            pass
    return None


def _annotate_ends(ax, xs, ys, steps, color):
    """Label first and last checkpoint with their step number."""
    if not xs:
        return
    for i, tag in ((0, f"s{steps[0]}"), (len(xs) - 1, f"s{steps[-1]}")):
        ax.annotate(tag, (xs[i], ys[i]), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=color)


def _shape_legend_handles(with_origin):
    """Marker semantics, drawn once per panel in neutral grey (run identity is colour)."""
    h = [
        Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec="0.25", mew=1.6,
               label="first RL checkpoint"),
        Line2D([], [], ls="none", marker="*", ms=15, mfc="0.55", mec="black", mew=0.7,
               label="last available RL checkpoint"),
    ]
    if with_origin:
        h.append(Line2D([], [], ls="none", marker="o", ms=7, mfc="0.55", mec="black",
                        mew=0.7, label="SFT (pre-GRPO)"))
    return h


def _fit_axes(ax, xs, ys, diagonal=False):
    """Scale each axis to its OWN data range. x and y ranges are independent, so a
    y-heavy panel does not stretch x and leave dead space on the right."""
    if not xs or not ys:
        return
    def lim(v):
        lo, hi = min(v), max(v)
        pad = (hi - lo) * PAD if hi > lo else (abs(hi) * PAD or 0.01)
        return lo - pad, hi + pad
    xlo, xhi = lim(xs)
    ylo, yhi = lim(ys)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    if diagonal:
        # Draw y=x only across the visible overlap; it may be a corner segment or
        # absent entirely once the axes are scaled independently.
        t0, t1 = max(xlo, ylo), min(xhi, yhi)
        if t1 > t0:
            ax.plot([t0, t1], [t0, t1], ls=":", c="grey", lw=1, label="y = x", zorder=0)


def _draw_run(ax, r, xds, yds, xi, yi):
    """One run's trajectory + endpoint markers + SFT origin.

    Markers: first RL checkpoint = circled dot, final RL checkpoint = star,
    SFT (pre-GRPO) = plain dot. Returns (xs, ys) of everything drawn, for autoscale.
    """
    xc, yc = r["curves"][xds], r["curves"][yds]
    steps = [s for s in xc if s in yc]
    xs = [xc[s][xi] for s in steps]
    ys = [yc[s][yi] for s in steps]
    c = r["color"]
    if xs:
        ax.plot(xs, ys, marker="o", ls=r.get("ls", "-"), color=c, ms=3.5, lw=1.6,
                alpha=0.85, label=r["label"], zorder=2)
        # first RL checkpoint: open ring around a filled centre dot
        ax.plot(xs[0], ys[0], marker="o", ms=11, mfc="none", mec=c, mew=1.8,
                ls="none", zorder=4, label="_nolegend_")
        ax.plot(xs[0], ys[0], marker="o", ms=3.5, color=c, ls="none", zorder=5,
                label="_nolegend_")
        # final RL checkpoint
        ax.plot(xs[-1], ys[-1], marker="*", ms=16, color=c, mec="black", mew=0.7,
                ls="none", zorder=5, label="_nolegend_")
        _annotate_ends(ax, xs, ys, steps, c)
    ox, oy = r["origins"].get(xds), r["origins"].get(yds)
    if ox and oy:
        ax.plot(ox[xi], oy[yi], marker="o", ms=7.5, color=c, mec="black", mew=0.7,
                ls="none", zorder=5, label="_nolegend_")
        xs, ys = xs + [ox[xi]], ys + [oy[yi]]
    return xs, ys


def _coverage_note(ax, runs, dss):
    """Footnote naming runs absent from this panel, or plotted from only part of
    their checkpoints — a 2-of-16 trajectory must not read like a full one.

    Fewer checkpoints is not by itself partial coverage: a run saved every 200
    steps has half the points of one saved every 100 but reaches the same final
    step. Those are called out as a coarser cadence, and only a run that stops
    short of the furthest final step is flagged as truncated."""
    steps = {r["label"]: [s for s in r["curves"][dss[0]] if s in r["curves"][dss[-1]]]
             for r in runs}
    counts = {l: len(s) for l, s in steps.items()}
    full = max(counts.values()) if counts else 0
    furthest = max((max(s) for s in steps.values() if s), default=0)
    absent = [l for l, n in counts.items() if n == 0]
    truncated, coarser = [], []
    for l, n in counts.items():
        if n == 0 or n >= full:
            continue
        if max(steps[l]) < furthest:
            truncated.append(f"{l} (to s{max(steps[l])} of s{furthest})")
        else:
            coarser.append(f"{l} ({n} ckpts)")
    msgs = []
    if absent:
        msgs.append("no data: " + ", ".join(absent))
    if truncated:
        msgs.append("partial: " + ", ".join(truncated))
    if coarser:
        msgs.append(f"coarser cadence, full range: " + ", ".join(coarser))
    if msgs:
        ax.text(0.5, -0.13, "   ".join(msgs), transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="#b00")


def plot_id_vs_ood(runs, id_ds, ood_list, mi, metric, out_path, title):
    fig, axes = plt.subplots(1, len(ood_list), figsize=(6.0 * len(ood_list), 5.2), squeeze=False)
    for ax, (ood_ds, ood_label) in zip(axes[0], ood_list):
        ax_x, ax_y = [], []
        for r in runs:
            xs, ys = _draw_run(ax, r, id_ds, ood_ds, mi, mi)
            ax_x += xs; ax_y += ys
        _fit_axes(ax, ax_x, ax_y)
        _coverage_note(ax, runs, [id_ds, ood_ds])
        ax.set_xlabel(f"in-distribution {metric}  (n=3,4)")
        ax.set_ylabel(f"OOD {metric}  ({ood_label})")
        ax.set_title(f"ID vs {ood_label}")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + _shape_legend_handles(any(r["origins"] for r in runs)),
                  fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pass1_vs_pass32(runs, datasets, out_path, title):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 5.2), squeeze=False)
    for ax, (ds, label) in zip(axes[0], datasets):
        ax_x, ax_y = [], []
        for r in runs:
            xs, ys = _draw_run(ax, r, ds, ds, 0, 1)
            ax_x += xs; ax_y += ys
        _fit_axes(ax, ax_x, ax_y, diagonal=True)
        _coverage_note(ax, runs, [ds])
        ax.set_xlabel("pass@1   (first sample exactly correct)")
        ax.set_ylabel("pass@32  (>=1 of 32 samples exactly correct)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + _shape_legend_handles(any(r["origins"] for r in runs)),
                  fontsize=8, loc="best")
    fig.suptitle(title + "\npass@32 = coverage: fraction of prompts solved by at least one of 32 samples",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_vs_step(runs, datasets, out_path, title):
    """Both metrics against GRPO step: rows = pass@1 / pass@32, cols = dataset.

    The pareto panels show where a run ends up but hide *when* it got there; this
    is the same data read along the training axis. The SFT origin is drawn at
    step 0 and joined to the first RL checkpoint by a dashed segment, so the
    warmup-to-RL jump is visible but not mistaken for a measured step.

    Each row shares a y-axis across datasets only if the scales are comparable —
    they are not (n=6 pass@1 is ~1/30 of n=3,4), so every panel autoscales.
    """
    metrics = [(0, "pass@1"), (1, "pass@32")]
    fig, axes = plt.subplots(len(metrics), len(datasets),
                             figsize=(5.4 * len(datasets), 4.4 * len(metrics)),
                             sharex=True, squeeze=False)
    for ri, (mi, m_label) in enumerate(metrics):
        for ci, (ds, ds_label) in enumerate(datasets):
            ax = axes[ri][ci]
            for r in runs:
                c = r["color"]
                curve = r["curves"][ds]
                steps = sorted(curve)
                ys = [curve[s][mi] for s in steps]
                if steps:
                    ax.plot(steps, ys, marker="o", ls=r.get("ls", "-"), color=c,
                            ms=3.5, lw=1.6, alpha=0.85, label=r["label"], zorder=2)
                    ax.plot(steps[-1], ys[-1], marker="*", ms=16, color=c,
                            mec="black", mew=0.7, ls="none", zorder=5,
                            label="_nolegend_")
                o = r["origins"].get(ds)
                if o:
                    ax.plot(0, o[mi], marker="o", ms=7.5, color=c, mec="black",
                            mew=0.7, ls="none", zorder=5, label="_nolegend_")
                    if steps:
                        ax.plot([0, steps[0]], [o[mi], ys[0]], ls="--", color=c,
                                lw=1.0, alpha=0.5, zorder=1, label="_nolegend_")
            if ri == len(metrics) - 1:
                _coverage_note(ax, runs, [ds])
            ax.set_title(f"{ds_label} — {m_label}")
            ax.set_ylabel(m_label)
            if ri == len(metrics) - 1:
                ax.set_xlabel("GRPO step   (0 = SFT student, pre-GRPO)")
            ax.grid(alpha=0.3)
            if ri == 0 and ci == 0:
                # Only the markers this figure actually draws: no "first RL
                # checkpoint" ring here, since step is on the x-axis already.
                shapes = [Line2D([], [], ls="none", marker="*", ms=15, mfc="0.55",
                                 mec="black", mew=0.7,
                                 label="last available RL checkpoint")]
                if any(r["origins"] for r in runs):
                    shapes.append(Line2D([], [], ls="none", marker="o", ms=7,
                                         mfc="0.55", mec="black", mew=0.7,
                                         label="SFT (pre-GRPO), drawn at step 0"))
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles=handles + shapes, fontsize=8, loc="best")
    fig.suptitle(title + "\nboth metrics along the training axis", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=MODEL/EXP")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--origin", action="append", default=None, metavar="LABEL=SFT_EXP",
                    help="repeatable: draw run LABEL's trajectory origin from "
                         "<its model>/<SFT_EXP>/sft_final. Only runs named here get an "
                         "origin dot, so a no-SFT run of the same model is not given one "
                         "by mistake.")
    ap.add_argument("--id-dataset", default="balanced-simple")
    ap.add_argument("--ood", action="append", default=None, metavar="DATASET=LABEL",
                    help="default: balanced5-simple=n=5, balanced6-simple=n=6")
    ap.add_argument("--run-color", action="append", default=None, metavar="LABEL=#HEX",
                    help="repeatable: override run LABEL's colour. Use to encode a "
                         "dimension in hue/shade (e.g. one hue per model, shaded by "
                         "pretraining budget) instead of the default per-run cycle.")
    ap.add_argument("--run-linestyle", action="append", default=None, metavar="LABEL=LS",
                    help="repeatable: matplotlib linestyle for run LABEL (e.g. '--'). "
                         "Use to encode a second dimension (e.g. SFT format) so colour "
                         "does not have to carry both.")
    ap.add_argument("--set", dest="set_dir", default=None, metavar="FOLDER",
                    help="sub-directory of figures/olmo2-ladder/ for this comparison "
                         "set (default: --tag). Set folders are named descriptively "
                         "and do NOT have to match the tag — pass --set to write into "
                         "an existing one instead of creating a new folder.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default=None,
                    help="suptitle override; the default names the 1B ladder, which is "
                         "wrong once runs of another size are on the same axes.")
    args = ap.parse_args()

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
        curves = {ds: load_curve(args.result_dir, model, exp, ds) for ds in all_ds}
        sft_exp = origin_of.get(label, "")
        origins = {ds: o for ds in all_ds
                   if (o := load_origin(args.result_dir, model, sft_exp, ds))}
        print(f"{label}: {model}/{exp}")
        counts = {ds: len(c) for ds, c in curves.items()}
        print(f"    checkpoints per dataset: {counts}   SFT origin: {sorted(origins)}")
        missing = [ds for ds, n in counts.items() if n == 0]
        if missing:
            print(f"    WARNING: no pass@k data for {missing} — this run will be absent "
                  f"from those panels")
        if not any(curves.values()):
            raise SystemExit(
                f"ERROR: no .passk caches for {model}/{exp} under {args.result_dir}.\n"
                f"       Build them:  python3 scripts/compute_passk.py "
                f"--model {model} --exp {exp}")
        runs.append({"label": label, "model": model, "exp": exp, "curves": curves,
                     "origins": origins,
                     "color": color_of.get(label, COLORS[ci % len(COLORS)]),
                     "ls": ls_of.get(label, "-")})

    # One sub-directory per comparison set, named by --tag, so the ladder folder
    # holds folders rather than dozens of loose PNGs. Filenames keep the tag
    # suffix, so a figure is still self-identifying once moved or pasted
    # elsewhere. An explicit --figures-dir is used verbatim.
    fig_dir = args.figures_dir or os.path.join("figures", "olmo2-ladder",
                                               args.set_dir or args.tag or "untagged")
    suffix = f"_{args.tag}" if args.tag else ""
    title = args.title or "OLMo-2-1B pretraining ladder: GT-SFT warmup -> GRPO (lr 3e-6, kl 3e-3)"

    for mi, name in ((0, "pass@1"), (1, "pass@32")):
        plot_id_vs_ood(runs, args.id_dataset, ood_list, mi, name,
                       os.path.join(fig_dir, f"id_vs_ood_{name.replace('@','')}{suffix}.png"),
                       f"{title}\nmetric: {name}")
    plot_pass1_vs_pass32(runs, datasets,
                         os.path.join(fig_dir, f"pass1_vs_pass32{suffix}.png"), title)
    plot_vs_step(runs, datasets,
                 os.path.join(fig_dir, f"vs_step{suffix}.png"), title)

    w = max([len(r["label"]) for r in runs] + [3])
    print("\nFinal checkpoint per run (pass@1 / pass@32):")
    hdr = f"  {'run':<{w}} {'step':>5} " + " ".join(f"{l:>22}" for _, l in datasets)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in runs:
        idc = r["curves"][args.id_dataset]
        if not idc:
            continue
        last = max(idc)
        cells = []
        for ds, _ in datasets:
            c = r["curves"][ds]
            if not c:
                cells.append("  --  ")
                continue
            k = max(c)
            v = c[k]
            cells.append(f"{v[0]:.4f} / {v[1]:.4f}" + ("" if k == last else f" @s{k}"))
        print(f"  {r['label']:<{w}} {last:>5} " + " ".join(f"{c:>22}" for c in cells))
    if any(r["origins"] for r in runs):
        print("\nSFT origin (pre-GRPO):")
        for r in runs:
            cells = []
            for ds, _ in datasets:
                o = r["origins"].get(ds)
                cells.append(f"{o[0]:.4f} / {o[1]:.4f}" if o else "  --  ")
            print(f"  {r['label']:<{w}} {'sft':>5} " + " ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
