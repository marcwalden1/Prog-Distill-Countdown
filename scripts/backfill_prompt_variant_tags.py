#!/usr/bin/env python3
"""Retro-tag W&B runs with the prompt variant (data source) they used.

The prompt text is baked into the parquet at preprocessing time, so the only
record of which prompt format a run used is its data directory --
`balanced` (the original scaffolded prompt), `balanced-simple`,
`balanced-simple-gsm`, ... See preprocess_balanced.make_prefix.

scripts/train_grpo.sh and scripts/train_sft.sh now append that directory to
WANDB_TAGS, and the run_olmo2_*.sh launchers include it in the WANDB_TAGS they
preset. This script covers everything launched before that. It was used on
2026-09-22 to tag all 110 runs then in bingbin/prog_distill.

Variant resolution, per run:
  1. config.data.train_files -- verl logs the whole Hydra config, so every
     train/SFT run carries it. Handles both `../data/<variant>/train.parquet`
     and the GT-SFT `sft-data/gt/<variant>/gt-*.parquet` layout.
  2. Longest match on the run name against the data/ listing, for runs with no
     data config (the `*-eval` runs written by plot_results.py log only
     model_name/exp_name/condition/user).
Longest-match matters: `balanced` is a prefix of `balanced-simple`, which is a
prefix of `balanced-simple-gsm`. When both sources resolve and disagree, the
run is reported and skipped rather than guessed at.

    python3 scripts/backfill_prompt_variant_tags.py            # dry run
    python3 scripts/backfill_prompt_variant_tags.py --apply

Tags are added, never replaced. A 403 means the key is valid but lacks write
access to that run (see CLAUDE.md) -- reported and skipped, the pass continues.

Distinct from the two neighbouring scripts: tag_wandb_runs.py adds tags you
name to runs you select by name substring (it has no exclusion, so it cannot
separate balanced-simple from balanced-simple-gsm, and it never reads config);
backfill_wandb_tags.py infers a fixed stage/size vocabulary from run names.
"""
import argparse
import collections
import os
import re

import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_variants(data_dir):
    """data/ subdirs that can be a *training* set, longest name first.

    balanced5*/balanced6* are test-only, and their names appear inside eval-run
    names, so they must not be candidates for a run's training variant.
    """
    dirs = [d for d in os.listdir(data_dir)
            if d.startswith("balanced") and os.path.isdir(os.path.join(data_dir, d))]
    return sorted((d for d in dirs if not re.match(r"^balanced[56]", d)),
                  key=len, reverse=True)


def from_path(train_files, variants):
    if not train_files:
        return None
    return next((seg for seg in str(train_files).split("/") if seg in variants), None)


def from_name(name, variants):
    return next((v for v in variants if v in name), None)   # variants are longest-first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="bingbin/prog_distill")
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                    help="source of the variant vocabulary (default: <repo>/data)")
    ap.add_argument("--apply", action="store_true", help="default is a dry run")
    args = ap.parse_args()

    variants = train_variants(args.data_dir)
    print(f"train variants (longest-match order): {variants}\n")

    api = wandb.Api()
    runs = list(api.runs(args.project, per_page=200))
    print(f"{len(runs)} runs in {args.project}\n")

    plan, problems = [], []
    for r in runs:
        data_cfg = r.config.get("data")
        train_files = data_cfg.get("train_files") if isinstance(data_cfg, dict) else None
        v_cfg = from_path(train_files, variants)
        v_name = from_name(r.name, variants)
        if v_cfg and v_name and v_cfg != v_name:
            problems.append((r, f"config={v_cfg} but name={v_name}", train_files))
        elif v_cfg or v_name:
            plan.append((r, v_cfg or v_name, "config" if v_cfg else "name"))
        else:
            problems.append((r, "no variant inferable", train_files))

    for (variant, src), n in sorted(collections.Counter(
            (v, src) for _, v, src in plan).items()):
        print(f"  {n:4d}  {variant:22s} (from {src})")

    todo = [(r, v, s) for r, v, s in plan if v not in r.tags]
    print(f"\n{len(plan) - len(todo)} already tagged, {len(todo)} to tag, "
          f"{len(problems)} unresolved\n")

    for r, why, train_files in problems:
        print(f"  [SKIP] {r.name}\n         {why}  tags={r.tags}  train_files={train_files}")

    changed = failed = 0
    for r, variant, src in sorted(todo, key=lambda x: x[0].name):
        if not args.apply:
            print(f"  [dry]  {r.name}  + {variant}  ({src})")
            changed += 1
            continue
        try:
            r.tags = list(r.tags) + [variant]
            r.update()
            print(f"  [tag]  {r.name}  -> {r.tags}")
            changed += 1
        except Exception as e:                      # 403 etc: report, keep going
            print(f"  [FAIL] {r.name}: {type(e).__name__}: {e}")
            failed += 1

    verb = "tagged" if args.apply else "would tag"
    print(f"\n{verb}: {changed}, failed: {failed}, unresolved: {len(problems)}")


if __name__ == "__main__":
    main()
