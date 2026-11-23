"""Behavioral arithmetic-fidelity probe for `search`-format rollouts.

`sft_formats.py --format search` installs trace lines of a fixed shape

    Try <expr> = <value>; too high|too low.
    Try <expr> = <target>; that is the target.

so the arithmetic-claim positions are *specified by the format*, not guessed
from free-form CoT. For every such line in a stored eval JSON we recompute
eval(expr) with grader_utils.evaluate_equation and compare it against the
value the model claimed, splitting two things that behave very differently:

  rejected lines ("too high"/"too low")  -> pure arithmetic evaluation
  final line ("that is the target")      -> contaminated by the format prior
                                            that the last line ends at the target

and, independently of arithmetic, whether the too-high/too-low verdict agrees
with the model's *own* claimed value relative to the target ("verdict error" —
surface coherence of the trace, which can be right while every number is wrong).

Reads eval JSONs only; no GPU. Validate with --self-test, which runs the parser
over ground-truth traces rendered by sft_formats and asserts 0 miscomputations.

Usage:
    python3 scripts/probe_arith_fidelity.py --self-test
    python3 scripts/probe_arith_fidelity.py \
        --stage "base-8shot=results/baselines/<M>/8shot-search/<F>.json" \
        --stage "SFT-lr3e-5=results/<M>/<sft-exp>/sft_final/<F>.json" \
        --grpo-root results/<M>/<grpo-exp> --steps 100,200,...,1600 \
        --dataset-file balanced-simple_temp0.6_n32_max1024.json \
        --out analysis_data/arith_fidelity_<tag>.csv
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader_utils import evaluate_equation  # noqa: E402

# The literal line shape sft_formats._think_body emits for fmt="search".
TRY_LINE = re.compile(
    r"Try\s+(.+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*;\s*(too high|too low|that is the target)",
    re.IGNORECASE)
FINAL_VERDICT = "that is the target"


def approx_eq(true_v, claimed):
    """Compare at the claimed value's printed precision: sft_formats._fmt_value
    renders non-integers as '%.2f', so 49/12 -> '4.08' is a correct claim."""
    s = str(claimed)
    dec = len(s.split(".")[1].rstrip("0")) if "." in s else 0
    tol = 10 ** (-dec) / 2 + 1e-9 if dec else 1e-6
    return abs(true_v - claimed) <= tol


def probe_text(text, target):
    """Per-output counters for one generation."""
    c = dict(rej_n=0, rej_wrong=0, fin_n=0, fin_wrong=0, fin_claims_target=0,
             verdict_n=0, verdict_wrong=0, unparseable=0, lines=0,
             rej_wrong_toward_target=0)
    for expr, claimed_s, verdict in TRY_LINE.findall(text):
        c["lines"] += 1
        claimed = float(claimed_s)
        verdict = verdict.lower()
        is_final = verdict == FINAL_VERDICT

        # Verdict coherence is judged against the model's OWN claimed value,
        # so it is independent of whether the arithmetic is right.
        c["verdict_n"] += 1
        should = ("that is the target" if abs(claimed - target) < 1e-6
                  else "too high" if claimed > target else "too low")
        if should != verdict:
            c["verdict_wrong"] += 1

        true_v = evaluate_equation(expr.strip())
        if true_v is None:
            c["unparseable"] += 1
            continue
        try:
            true_v = float(true_v)
        except (TypeError, ValueError):
            c["unparseable"] += 1
            continue
        wrong = not approx_eq(true_v, claimed)
        if is_final:
            c["fin_n"] += 1
            c["fin_wrong"] += wrong
            c["fin_claims_target"] += abs(claimed - target) < 1e-6
        else:
            c["rej_n"] += 1
            c["rej_wrong"] += wrong
            if wrong and abs(claimed - target) < abs(true_v - target) - 1e-9:
                c["rej_wrong_toward_target"] += 1
    return c


def probe_file(path, max_samples):
    data = json.load(open(path))
    tot = dict(outs=0, outs_with_line=0)
    keys = ("rej_n", "rej_wrong", "fin_n", "fin_wrong", "fin_claims_target",
            "verdict_n", "verdict_wrong", "unparseable", "lines",
            "rej_wrong_toward_target")
    for k in keys:
        tot[k] = 0
    for row in data:
        target = float(row["target"])
        for text in row["outputs"][:max_samples]:
            tot["outs"] += 1
            c = probe_text(text, target)
            tot["outs_with_line"] += c["lines"] > 0
            for k in keys:
                tot[k] += c[k]
    r = lambda a, b: tot[a] / tot[b] if tot[b] else float("nan")  # noqa: E731
    return dict(
        n_prompts=len(data), n_outputs=tot["outs"],
        fmt_adherence=r("outs_with_line", "outs"),
        try_lines_per_output=tot["lines"] / max(tot["outs"], 1),
        unparseable_frac=r("unparseable", "lines"),
        n_rejected=tot["rej_n"], rej_miscompute=r("rej_wrong", "rej_n"),
        rej_wrong_toward_target=(tot["rej_wrong_toward_target"] / tot["rej_wrong"]
                                 if tot["rej_wrong"] else float("nan")),
        n_final=tot["fin_n"], final_miscompute=r("fin_wrong", "fin_n"),
        final_claims_target=r("fin_claims_target", "fin_n"),
        verdict_error=r("verdict_wrong", "verdict_n"),
    )


def self_test(n=300, dataset="balanced", data_dir="data"):
    """The parser must find 0 miscomputations in ground-truth search traces."""
    import pandas as pd
    from sft_formats import render_response
    df = pd.read_parquet(os.path.join(data_dir, dataset, "test.parquet")).head(n)
    rng = random.Random(0)
    lines = wrong = verdict_wrong = 0
    for _, row in df.iterrows():
        ei, tgt = row["extra_info"], int(row["reward_model"]["ground_truth"])
        text = render_response("search", "Solution:", list(ei["numbers"]), tgt,
                               ei["canonical_pattern"], rng=rng)
        c = probe_text(text, float(tgt))
        lines += c["lines"]
        wrong += c["rej_wrong"] + c["fin_wrong"]
        verdict_wrong += c["verdict_wrong"]
    print(f"self-test: {lines} Try-lines from {n} GT traces -> "
          f"{wrong} miscomputations, {verdict_wrong} verdict errors")
    assert lines > 0 and wrong == 0 and verdict_wrong == 0, "parser does not match sft_formats"
    print("PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", default=[],
                    help="label=path/to/eval.json; repeatable, kept in order")
    ap.add_argument("--grpo-root", help="run dir holding global_step_*/")
    ap.add_argument("--steps", default="", help="comma-separated GRPO steps")
    ap.add_argument("--dataset-file", default="balanced-simple_temp0.6_n32_max1024.json")
    ap.add_argument("--max-samples", type=int, default=8,
                    help="generations per prompt to read (eval JSONs hold 32)")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    stages = []
    for s in args.stage:
        label, path = s.split("=", 1)
        stages.append((label, None, path))
    if args.grpo_root and args.steps:
        for st in [int(x) for x in args.steps.split(",")]:
            stages.append((f"grpo-{st}", st,
                           os.path.join(args.grpo_root, f"global_step_{st}", args.dataset_file)))

    rows = []
    hdr = (f"{'stage':<22}{'outs':>7}{'fmt':>6}{'ln/out':>8}{'n_rej':>8}{'REJ_MIS':>9}"
           f"{'n_fin':>7}{'FIN_MIS':>9}{'fin=tgt':>9}{'VERD_ERR':>10}")
    print(hdr)
    print("-" * len(hdr))
    for order, (label, step, path) in enumerate(stages):
        if not os.path.exists(path):
            print(f"{label:<22} MISSING {path}")
            continue
        m = probe_file(path, args.max_samples)
        m.update(stage=label, step=step, order=order, source=path)
        rows.append(m)
        print(f"{label:<22}{m['n_outputs']:>7}{m['fmt_adherence']:>6.3f}"
              f"{m['try_lines_per_output']:>8.2f}{m['n_rejected']:>8}"
              f"{m['rej_miscompute']:>9.3f}{m['n_final']:>7}{m['final_miscompute']:>9.3f}"
              f"{m['final_claims_target']:>9.3f}{m['verdict_error']:>10.3f}")

    if args.out and rows:
        import pandas as pd
        cols = ["stage", "step", "order", "n_prompts", "n_outputs", "fmt_adherence",
                "try_lines_per_output", "unparseable_frac", "n_rejected",
                "rej_miscompute", "rej_wrong_toward_target", "n_final",
                "final_miscompute", "final_claims_target", "verdict_error", "source"]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows)[cols].to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
