"""Write per-shard memory-mappable sidecars of the teacher top-k data for KD training.

The top-k columns of the distill-data parquets (scripts/generate_distill_data.py)
are ~90 GB in total, too big to hold in RAM per training rank, and parquet has
no cheap random row access. For every <data_dir>/shard_XXXX.parquet this writes

    <data_dir>/kd/shard_XXXX.topk_ids.npy   int32   [T, k]  (T = response tokens in shard)
    <data_dir>/kd/shard_XXXX.topk_lp.npy    float16 [T, k]  teacher logprobs, sorted by rank
    <data_dir>/kd/shard_XXXX.samp_lp.npy    float32 [T]     teacher logprob of the sampled token

with tokens in parquet row order, rows concatenated. A row's token offset is the
cumsum of len(response_ids) (DistillKDDataset recomputes it). float16 logprobs:
resolution <= 0.008 nats for logprobs > -16, far below the ~0.1-nat bf16 noise
between vLLM and HF; anything below -65504 cannot occur (vLLM floors at -inf ->
clipped to -1e4 here).

Skips shards whose sidecars exist; writes atomically. Resumable / requeue-safe.

    python3 scripts/convert_distill_topk.py <data_dir> [--topk 20]
"""
import argparse, glob, os, sys, time

import numpy as np
import pyarrow.parquet as pq


def flat(col, dtype):
    arr = col.combine_chunks()
    off = np.asarray(arr.offsets)
    vals = np.asarray(arr.values)
    return vals[off[0]:off[-1]].astype(dtype, copy=False), off - off[0]


def save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--only", type=str, default="", help="comma list of shard indices (testing)")
    a = ap.parse_args()
    k = a.topk
    out = os.path.join(a.data_dir, "kd")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data_dir, "shard_*.parquet")))
    if a.only:
        keep = {f"shard_{int(s):04d}.parquet" for s in a.only.split(",")}
        files = [f for f in files if os.path.basename(f) in keep]
    n_done = 0
    for f in files:
        stem = os.path.basename(f)[: -len(".parquet")]
        paths = {x: os.path.join(out, f"{stem}.{x}.npy") for x in ("topk_ids", "topk_lp", "samp_lp")}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        t0 = time.time()
        t = pq.read_table(f, columns=["response_ids", "sampled_logprobs", "topk_ids", "topk_logprobs"])
        r_ids, r_off = flat(t.column("response_ids"), np.int32)
        samp, s_off = flat(t.column("sampled_logprobs"), np.float32)
        tids, t_off = flat(t.column("topk_ids"), np.int32)
        tlps, l_off = flat(t.column("topk_logprobs"), np.float32)
        T = len(r_ids)
        # every per-row list must line up with the row's response length
        assert np.array_equal(s_off, r_off), stem
        assert np.array_equal(t_off, r_off * k) and np.array_equal(l_off, r_off * k), stem
        assert len(samp) == T and len(tids) == T * k and len(tlps) == T * k, stem
        tids = tids.reshape(T, k)
        tlps = tlps.reshape(T, k)
        # sanity: top-k sorted descending; sampled logprob equals its top-k entry when present
        assert (np.diff(tlps, axis=1) <= 1e-6).all(), f"{stem}: top-k not sorted"
        hit = tids == r_ids[:, None]
        in_top = hit.any(1)
        if in_top.any():
            j = hit.argmax(1)[in_top]
            d = np.abs(tlps[in_top, j] - samp[in_top])
            assert d.max() < 1e-4, f"{stem}: sampled lp != top-k lp (max diff {d.max()})"
        save(paths["topk_ids"], tids)
        save(paths["topk_lp"], np.clip(tlps, -1e4, 0).astype(np.float16))
        save(paths["samp_lp"], np.clip(samp, -1e4, 0).astype(np.float32))
        n_done += 1
        print(f"{stem}: rows {t.num_rows} tokens {T} in_top{k} {in_top.mean():.4f} "
              f"{time.time() - t0:.1f}s", flush=True)
    print(f"converted {n_done} shards; {len(files)} parquet files in {a.data_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
