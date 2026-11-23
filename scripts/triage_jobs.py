#!/usr/bin/env python3
"""Classify failed tasks of one or more SLURM jobs/arrays, and optionally hold the rest.

Implements the "batch failure policy" in CLAUDE.md. For every task that ended
badly it reads the tail of its log and assigns one class:

    node     hardware / node fault (CUDA launch failure, ECC, Xid, NODE_FAIL, ...)
             -> exclude the node and release the held tasks
    oom      GPU out of memory (CUDA OOM / torch OutOfMemoryError)
             -> lower the micro-batch knobs and resubmit the held tasks
    hostmem  host-RAM OOM (sacct OUT_OF_MEMORY / oom-kill) -> treat as "other"
    timeout  hit the walltime -> "other": budget is wrong, discuss first
    other    anything else -> pause and ask the user

It also prints an ETA for RUNNING eval tasks from eval.py's "Saved k/N prompts"
lines, flagging any whose projected finish exceeds its walltime. That is the
check that would have caught the 2026-10-11 long-cap evals (87 TIMEOUTs x 8 h)
in the first hour instead of after the fact.

Usage:
    python3 scripts/triage_jobs.py 52292727 52292735 ...      # report only
    python3 scripts/triage_jobs.py --hold <jobids>            # + hold PENDING tasks
    python3 scripts/triage_jobs.py --release <jobids>         # release held tasks
    python3 scripts/triage_jobs.py --exclude NODE --release <jobids>
          # add NODE to every pending task's exclude list, then release

Exit code: 0 if no failures, 1 if any failure (so it can gate a watcher loop).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAIL_BYTES = 400_000  # eval logs are MBs of tqdm; the error is always at the end

# Order matters: the first class whose pattern matches wins.
PATTERNS = [
    ("oom", r"CUDA out of memory|torch\.(cuda\.)?OutOfMemoryError|OutOfMemoryError: CUDA"),
    ("node", r"unspecified launch failure|uncorrectable ECC|ECC error|Xid|"
             r"GPU is lost|fallen off the bus|No CUDA GPUs are available|"
             r"CUDA driver initialization failed|cudaErrorDevicesUnavailable|"
             r"illegal memory access was encountered|NCCL.*(unhandled system error|"
             r"remote process exited)|Stale file handle|Input/output error|"
             r"Transport endpoint is not connected"),
    ("hostmem", r"oom[-_]kill|Killed process|MemoryError"),
]
BAD_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def tasks(jobids):
    out = sh(["sacct", "-j", ",".join(jobids), "-X", "-n", "-P", "-o",
              "JobID,JobName,State,ExitCode,NodeList,Elapsed,Timelimit"])
    rows = []
    for line in out.splitlines():
        jid, name, state, rc, node, elapsed, limit = line.split("|")
        if "[" in jid:  # still-pending array range, e.g. 123_[4-16]
            rows.append(dict(jid=jid, name=name, state="PENDING", rc=rc, node="",
                             elapsed=elapsed, limit=limit))
            continue
        rows.append(dict(jid=jid, name=name, state=state.split()[0], rc=rc, node=node,
                         elapsed=elapsed, limit=limit))
    return rows


def log_path(t):
    arr, _, task = t["jid"].partition("_")
    pats = [f"logs/{t['name']}-{arr}-{task}.out"] if task else [f"logs/*-{arr}-*.out", f"logs/*-{arr}.out"]
    for p in pats:
        hits = glob.glob(os.path.join(REPO, p))
        if hits:
            return hits[0]
    return None


def tail(path):
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - TAIL_BYTES))
        return f.read().decode("utf-8", "replace")


def classify(t):
    text = tail(t["log"]) if t["log"] else ""
    for cls, pat in PATTERNS:
        m = re.search(pat, text)
        if m:
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]
            return cls, line.strip()[-160:]
    if t["state"] == "NODE_FAIL":
        return "node", "sacct state NODE_FAIL"
    if t["state"] == "OUT_OF_MEMORY":
        return "hostmem", "sacct state OUT_OF_MEMORY (host RAM, not GPU)"
    if t["state"] == "TIMEOUT":
        return "timeout", f"hit walltime {t['limit']}"
    last = [l for l in text.splitlines() if l.strip() and "Error" in l]
    return "other", (last[-1].strip()[-160:] if last else f"exit {t['rc']}, no error line in log")


def secs(s):
    if not s or s in ("UNLIMITED", "INVALID"):
        return None
    d, _, hms = s.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def eta(t):
    """(fraction done, projected total seconds) from eval.py progress lines."""
    if not t["log"]:
        return None
    hits = re.findall(r"Saved (\d+)/(\d+) prompts", tail(t["log"]))
    first = re.search(r"Partial results: (\d+)/", tail(t["log"]))
    el = secs(t["elapsed"])
    if not hits or not el:
        return None
    k, n = map(int, hits[-1])
    k0 = int(first.group(1)) if first else 0
    if k <= k0:
        return None
    return k / n, el * (n - k0) / (k - k0)


def scontrol_each(jobids, action, extra=()):
    pend = sh(["squeue", "-h", "-t", "PENDING", "-o", "%i", "-j", ",".join(jobids)]).split()
    for j in pend:
        subprocess.run(["scontrol", action, j, *extra] if action != "update"
                       else ["scontrol", "update", f"JobId={j}", *extra])
    return pend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobids", nargs="+")
    ap.add_argument("--hold", action="store_true", help="scontrol hold every PENDING task")
    ap.add_argument("--exclude", help="node(s) to add to pending tasks' ExcludeNodeList")
    ap.add_argument("--release", action="store_true", help="scontrol release held tasks")
    args = ap.parse_args()

    rows = tasks(args.jobids)
    for t in rows:
        t["log"] = log_path(t) if t["state"] not in ("PENDING",) else None

    bad = [t for t in rows if t["state"] in BAD_STATES]
    counts = {}
    for t in rows:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    print("states:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    by_cls = {}
    for t in bad:
        cls, ev = classify(t)
        by_cls.setdefault(cls, []).append(t)
        print(f"  {t['jid']:16s} {t['state']:10s} {t['node']:16s} {cls:8s} {ev}")

    # Silent failures: an eval task that is COMPLETED but whose result JSON does
    # not exist. (eval.sh now propagates eval.py's exit code, so this guards
    # tasks submitted before that fix.)
    for t in rows:
        if t["state"] == "COMPLETED" and t["log"]:
            with open(t["log"], "rb") as f:
                m = re.search(rb"Saving results to (\S+\.json)", f.read(200_000))
            if m and not os.path.exists(m.group(1).decode()):
                cls, ev = classify(t)
                by_cls.setdefault(cls, []).append(t)
                print(f"  {t['jid']:16s} COMPLETED* {t['node']:16s} {cls:8s} "
                      f"no result JSON; {ev}")

    for t in rows:
        if t["state"] == "RUNNING":
            e = eta(t)
            lim = secs(t["limit"])
            if e and lim:
                frac, total = e
                flag = "  <-- WILL TIME OUT" if total > lim else ""
                print(f"  {t['jid']:16s} RUNNING    {frac:5.1%} done, projected "
                      f"{total/3600:.1f} h of {lim/3600:.1f} h{flag}")
                if total > lim:
                    by_cls.setdefault("will-timeout", []).append(t)

    if by_cls:
        print("\nsummary:", ", ".join(f"{k}={len(v)}" for k, v in by_cls.items()))
        nodes = sorted({t["node"] for t in by_cls.get("node", []) if t["node"]})
        if nodes:
            print(f"bad nodes: {','.join(nodes)}  -> add to #SBATCH --exclude in the job script, "
                  f"then: python3 scripts/triage_jobs.py --exclude {','.join(nodes)} --release <jobids>")

    if args.hold:
        held = scontrol_each(args.jobids, "hold")
        print(f"held {len(held)} pending task(s)")
    if args.exclude:
        # NB this REPLACES the pending task's ExcludeNodeList; merge with the script's.
        prev = sh(["scontrol", "show", "job", args.jobids[0]])
        m = re.search(r"ExcNodeList=(\S+)", prev)
        merged = ",".join(sorted(set(filter(None, [*(m.group(1).split(",") if m and m.group(1) != "(null)" else []),
                                                    *args.exclude.split(",")]))))
        upd = scontrol_each(args.jobids, "update", [f"ExcNodeList={merged}"])
        print(f"set ExcNodeList={merged} on {len(upd)} pending task(s)")
    if args.release:
        rel = scontrol_each(args.jobids, "release")
        print(f"released {len(rel)} task(s)")

    sys.exit(1 if by_cls else 0)


if __name__ == "__main__":
    main()
