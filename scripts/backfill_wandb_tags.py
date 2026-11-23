#!/usr/bin/env python3
"""Backfill inferred W&B tags for prog_distill runs."""

import argparse
import os
import re
import sys

import wandb
from wandb.errors import CommError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_results import model_family  # noqa: E402  (shared family-tag rule)


def infer_tags(name):
    lower = name.lower()
    tags = []

    if lower.endswith("-eval") or "-eval" in lower or "evals" in lower:
        tags.append("eval")
    if "sft-round" in lower or re.search(r"(^|-)sft($|-)", lower):
        tags.append("sft")

    if "progdistill" in lower:
        tags.append("progdistill")
    elif "distill" in lower:
        tags.append("distill")

    if "grpo" in lower:
        tags.append("grpo")

    if "270m" in lower:
        tags.append("270m")
    elif "qwen2.5-0.5b" in lower or "s0p5b" in lower:
        tags.append("0.5b")
    elif lower.startswith("qwen2.5-1.5b"):
        tags.append("1.5b")

    # Model family ("Qwen2.5", "Qwen3", "OLMo-2", "gemma-3"). Run names are
    # "{MODEL_NAME}-{EXP_NAME}", so the same rule the training scripts apply to
    # MODEL_NAME works on the name; it returns "" for hand-named runs that don't
    # start with a model name (e.g. "shlok-best-1point5-model").
    family = model_family(name)
    if family:
        tags.append(family)

    return dedupe_tags(tags)


def dedupe_tags(tags):
    seen = set()
    out = []
    for tag in tags:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "bingbin"))
    parser.add_argument("--project", default="prog_distill")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    changed = 0
    failed = 0
    scanned = 0
    for run in api.runs(f"{args.entity}/{args.project}"):
        scanned += 1
        inferred = infer_tags(run.name or "")
        if not inferred:
            continue

        current = list(run.tags or [])
        merged = dedupe_tags(current + inferred)
        if merged == current:
            continue

        changed += 1
        if not args.quiet:
            print(f"{run.id}\t{run.name}\t{current} -> {merged}")
        if not args.dry_run:
            try:
                run.tags = merged
                run.update()
            except CommError as exc:
                failed += 1
                print(f"failed\t{run.id}\t{run.name}\t{exc}")

    print(f"scanned={scanned} changed={changed} failed={failed} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
