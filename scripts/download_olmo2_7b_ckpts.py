#!/usr/bin/env python3
"""Download OLMo-2-1124-7B stage1 pretraining checkpoints into a local HF cache
and expose each as $MODEL_DIR/OLMo-2-1124-7B-pt<tokens>B.

The 1B ladder reads from the shared read-only dam_lab cache; that cache holds no
7B repo, so these are downloaded into the user's own cache. ~29.2 GB per revision
(fp32, 6 shards), 4 revisions = ~117 GB.

Each revision is verified after download: every file present, sizes matching the
Hub, and every LFS blob's sha256 matching (in an HF cache the blob filename *is*
the content sha256, so this is a free integrity check).

Usage:
    python3 scripts/download_olmo2_7b_ckpts.py            # download + link
    python3 scripts/download_olmo2_7b_ckpts.py --check    # verify only, no download
"""
import argparse
import os
import pathlib
import sys

REPO = "allenai/OLMo-2-1124-7B"
# tag -> revision.  Requested token budgets; all four exist upstream.
REVISIONS = {
    # Original ladder.
    "pt101B":  "stage1-step24000-tokens101B",
    "pt252B":  "stage1-step60000-tokens252B",
    "pt500B":  "stage1-step119000-tokens500B",
    "pt1007B": "stage1-step240000-tokens1007B",
    # Added 2026-08-31 to bracket the 1B-pt1007B post-SFT match. The 1B target is
    # pass@1 0.2758 / pass@32 0.7793 (n=3,4, its best LR 1e-4) and 7B-pt101B already
    # exceeds it (0.2839 / 0.7854), so the match sits BELOW 101B tokens — hence the
    # sub-101B points. 151B is included for curve shape.
    "pt42B":   "stage1-step10000-tokens42B",
    "pt63B":   "stage1-step15000-tokens63B",
    "pt84B":   "stage1-step20000-tokens84B",
    "pt151B":  "stage1-step36000-tokens151B",
    # Target corrected to the 1B pt4001B model (pass@1 0.2979 / pass@32 0.7984 at its
    # best LR 3e-5). That sits between 7B pt101B (0.2839/0.7854) and pt252B
    # (0.3089/0.8275), so 126/151/202B pin down the crossing.
    "pt126B":  "stage1-step30000-tokens126B",
    "pt202B":  "stage1-step48000-tokens202B",
    # 135B refines the 126-202B crossing region, where 126B and 202B came out
    # tied on joint distance to the 1B pt4001B target (0.0112 vs 0.0114).
    "pt135B":  "stage1-step32000-tokens135B",
    # Final stage1 checkpoint (top rung of the ladder), added 2026-09-06 to
    # complete the pretraining-budget curve at 7B; mirrors the 1B ladder's pt4001B.
    "pt3896B": "stage1-step928646-tokens3896B",
}
CACHE = pathlib.Path(os.environ.get(
    "OLMO7B_CACHE", "/n/netscratch/kempner_bingbin_lab/Lab/bingbin/hf-cache/hub"))
MODELS = pathlib.Path(os.environ.get(
    "MODEL_DIR", "/n/holylabs/LABS/kempner_bingbin_lab/Lab/bingbin/models"))


def verify(api, snap_dir, rev):
    """Return list of problems for a downloaded snapshot ([] == clean)."""
    problems = []
    want = api.repo_info(REPO, revision=rev, files_metadata=True).siblings
    for sib in want:
        if sib.rfilename == ".gitattributes":
            continue
        f = pathlib.Path(snap_dir) / sib.rfilename
        if not f.is_file():
            problems.append(f"missing {sib.rfilename}")
            continue
        if sib.size and f.stat().st_size != sib.size:
            problems.append(f"size mismatch {sib.rfilename}: "
                            f"{f.stat().st_size} != {sib.size}")
        if sib.lfs and f.resolve().name != sib.lfs.sha256:
            problems.append(f"sha256 mismatch {sib.rfilename}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. pt101B,pt252B")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_CACHE", str(CACHE))
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()

    tags = [t.strip() for t in args.only.split(",") if t.strip()] or list(REVISIONS)
    CACHE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    failures = []
    for tag in tags:
        rev = REVISIONS[tag]
        dest = MODELS / f"OLMo-2-1124-7B-{tag}"
        print(f"\n=== {tag}  ({rev})", flush=True)

        if args.check:
            if not dest.exists():
                print("   NOT PRESENT"); failures.append(tag); continue
            snap = dest.resolve()
        else:
            snap = pathlib.Path(snapshot_download(REPO, revision=rev, cache_dir=str(CACHE),
                                                 max_workers=8))
            print(f"   downloaded -> {snap}", flush=True)

        problems = verify(api, snap, rev)
        if problems:
            print("   VERIFY FAILED:", flush=True)
            for p in problems[:8]:
                print(f"     {p}", flush=True)
            failures.append(tag)
            continue
        print("   verified: all files present, sizes and sha256 match the Hub", flush=True)

        if not args.check:
            if dest.is_symlink() or dest.exists():
                print(f"   link exists: {dest}", flush=True)
            else:
                dest.symlink_to(snap)
                print(f"   linked {dest} -> {rev}", flush=True)

    print("\n" + "=" * 60)
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)
    print("All requested revisions present and verified.")
    for tag in tags:
        d = MODELS / f"OLMo-2-1124-7B-{tag}"
        print(f"  {d}")


if __name__ == "__main__":
    main()
