#!/usr/bin/env python3
"""Add W&B tags to existing runs selected by name substring.

Distinct from scripts/backfill_wandb_tags.py, which *infers* a fixed tag
vocabulary from run names. This one adds tags you name, to runs you select --
for retro-tagging a dimension that was not in the launcher's WANDB_TAGS when the
runs were submitted.

    python3 scripts/tag_wandb_runs.py --name-contains gtsft200-search- \
        --add-tag format=search [--dry-run]

Tags are added, never replaced; runs already carrying the tag are left alone. A
403 means the key is valid but lacks write access to that run (see CLAUDE.md) --
the run is reported and skipped rather than aborting the pass.
"""
import argparse

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="bingbin/prog_distill")
    ap.add_argument("--name-contains", action="append", required=True,
                    help="repeatable; a run matches if ALL substrings are present")
    ap.add_argument("--add-tag", action="append", required=True, help="repeatable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = wandb.Api()
    runs = [r for r in api.runs(args.project, per_page=200)
            if all(s in r.name for s in args.name_contains)]
    print(f"{len(runs)} run(s) match {args.name_contains}")

    changed = skipped = failed = 0
    for r in sorted(runs, key=lambda r: r.name):
        missing = [t for t in args.add_tag if t not in r.tags]
        if not missing:
            print(f"  [has]  {r.name}  {r.tags}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [dry]  {r.name}  + {missing}")
            changed += 1
            continue
        try:
            r.tags = list(r.tags) + missing
            r.update()
            print(f"  [tag]  {r.name}  -> {r.tags}")
            changed += 1
        except Exception as e:                      # 403 etc: report, keep going
            print(f"  [FAIL] {r.name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{changed} tagged, {skipped} already had it, {failed} failed")


if __name__ == "__main__":
    main()
