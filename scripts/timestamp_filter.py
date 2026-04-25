"""Forwards stdin to stdout. When a line contains `step:N` for a step N
that is a positive multiple of every_n_steps (default 100), prints a
`[YYYY-MM-DD HH:MM:SS | elapsed HH:MM:SS] === step N reached ===` marker
line right before forwarding the original line. Each milestone is only
marked once per run, so duplicate logs of the same step (e.g. train then
val on the same step) don't double-print.
"""

import argparse
import datetime
import re
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=100)
    args = parser.parse_args()

    pat = re.compile(r"step:(\d+)\b")
    seen: set[int] = set()
    start = time.time()

    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_ts}] === run started ===", flush=True)

    for line in sys.stdin:
        m = pat.search(line)
        if m:
            n = int(m.group(1))
            if n > 0 and n % args.every == 0 and n not in seen:
                seen.add(n)
                elapsed = int(time.time() - start)
                h, rem = divmod(elapsed, 3600)
                mm, ss = divmod(rem, 60)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts} | elapsed {h:02d}:{mm:02d}:{ss:02d}] "
                    f"=== step {n} reached ===",
                    flush=True,
                )
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
