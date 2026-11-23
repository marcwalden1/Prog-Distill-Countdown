"""Token-id SFT dataset for offline distillation data (scripts/generate_distill_data.py).

Loaded via verl's `data.custom_cls` hook:

    data.custom_cls.path=$PROJECT_DIR/distill_dataset.py \
    data.custom_cls.name=DistillNTPDataset

Next-token prediction on the teacher's EXACT sampled tokens: no retokenization and,
unlike SFTDataset, no EOS appended. A rollout that stopped on EOS already carries
the EOS id as its last response token (so it is trained on); a rollout truncated
at max_tokens has no EOS and the student is not taught to stop there. Empty
rollouts (response = [EOS]) are kept: they teach the teacher's immediate-stop rate.

Only `prompt_ids` / `response_ids` are read (the top-k logprob columns, ~100 GB,
are skipped). Token ids are held as one flat int32 array + offsets per column, so
4.2M rollouts cost ~5 GB per rank rather than millions of small numpy arrays.
Batch contract matches SFTDataset/OPDDataset (input_ids, attention_mask,
position_ids, loss_mask).
"""

import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset

from verl.utils.model import compute_position_id_with_mask


def _expand(files):
    if not isinstance(files, list | ListConfig):
        files = [files]
    out = []
    for f in files:
        f = os.path.expanduser(str(f))
        out += sorted(glob.glob(os.path.join(f, "shard_*.parquet"))) if os.path.isdir(f) else [f]
    if not out:
        raise FileNotFoundError(f"no parquet files in {files}")
    return out


def _flat(chunked):
    arr = chunked.combine_chunks() if isinstance(chunked, pa.ChunkedArray) else chunked
    offsets = np.asarray(arr.offsets, dtype=np.int64)
    values = np.asarray(arr.values, dtype=np.int32)
    base = offsets[0]
    return values[base:offsets[-1]].copy(), offsets - base


class DistillNTPDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        self.max_length = config.get("max_length", 2048)
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ["error", "right"]

        files = _expand(parquet_files)
        tables = [pq.read_table(f, columns=["prompt_ids", "response_ids"]) for f in files]
        table = pa.concat_tables(tables)
        self.p_vals, self.p_off = _flat(table.column("prompt_ids"))
        self.r_vals, self.r_off = _flat(table.column("response_ids"))
        del tables, table
        print(f"DistillNTPDataset: {len(files)} files, {len(self)} rollouts, "
              f"{len(self.r_vals)} response tokens", flush=True)

    def __len__(self):
        return len(self.p_off) - 1

    def __getitem__(self, item):
        prompt_ids = torch.from_numpy(self.p_vals[self.p_off[item]:self.p_off[item + 1]].astype(np.int64))
        response_ids = torch.from_numpy(self.r_vals[self.r_off[item]:self.r_off[item + 1]].astype(np.int64))
        prompt_length, response_length = prompt_ids.shape[0], response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            input_ids = torch.cat((input_ids, torch.full((pad,), self.pad_token_id, dtype=input_ids.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad, dtype=attention_mask.dtype)))
        elif seq_len > self.max_length:
            if self.truncation != "right":
                raise NotImplementedError(f"{seq_len=} is larger than {self.max_length=}")
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        position_ids = compute_position_id_with_mask(attention_mask)
        # Shifted frame: position i predicts input_ids[i+1]. Keep exactly the
        # predictions of response tokens (same semantics as SFTDataset/OPDDataset).
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "position_ids": position_ids, "loss_mask": loss_mask}


class DistillKDDataset(DistillNTPDataset):
    """DistillNTPDataset + teacher top-k targets for logit distillation (data.kd=True).

    Needs the sidecars from scripts/convert_distill_topk.py in <data_dir>/kd/.
    Adds, aligned to input positions in the shifted frame (position i predicts
    input_ids[i+1], i.e. response token r sits at position P-1+r):

        kd_ids       int64 [L, k+1]  teacher top-k ids, then the sampled token
        kd_logprobs  f32   [L, k+1]  teacher logprobs of those ids
        kd_mask      bool  [L, k+1]  which slots are buckets: all top-k slots
                                     (duplicate ids masked), the sampled-token
                                     slot only when it is NOT in the top-k

    Everything else in the vocab is the tail bucket (built in the trainer).
    Non-response positions are all-False; the trainer's loss_mask zeroes them.
    """

    def __init__(self, parquet_files, tokenizer, config):
        super().__init__(parquet_files, tokenizer, config)
        files = _expand(parquet_files)
        self.kd_paths = []
        row_shard, row_start = [], []
        r_len = np.diff(self.r_off)
        row = 0
        for si, f in enumerate(files):
            d, stem = os.path.dirname(f), os.path.basename(f)[: -len(".parquet")]
            paths = {x: os.path.join(d, "kd", f"{stem}.{x}.npy") for x in ("topk_ids", "topk_lp", "samp_lp")}
            for p in paths.values():
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{p} missing: run scripts/convert_distill_topk.py {d}")
            n_rows = pq.ParquetFile(f).metadata.num_rows
            lens = r_len[row:row + n_rows]
            T = int(np.load(paths["samp_lp"], mmap_mode="r").shape[0])
            assert T == int(lens.sum()), f"{stem}: sidecar has {T} tokens, parquet {int(lens.sum())}"
            row_shard.append(np.full(n_rows, si, np.int32))
            row_start.append(np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))
            self.kd_paths.append(paths)
            row += n_rows
        assert row == len(self), (row, len(self))
        self.row_shard = np.concatenate(row_shard)
        self.row_start = np.concatenate(row_start)
        self._mm = {}  # opened lazily (per dataloader worker)
        self.k = int(np.load(self.kd_paths[0]["topk_ids"], mmap_mode="r").shape[1])

    def _arrays(self, si):
        if si not in self._mm:
            p = self.kd_paths[si]
            self._mm[si] = tuple(np.load(p[x], mmap_mode="r") for x in ("topk_ids", "topk_lp", "samp_lp"))
        return self._mm[si]

    def __getitem__(self, item):
        out = super().__getitem__(item)
        k, L = self.k, self.max_length
        P = int(self.p_off[item + 1] - self.p_off[item])
        r_ids = self.r_vals[self.r_off[item]:self.r_off[item + 1]]
        R = len(r_ids)
        ids, lps, samp = self._arrays(int(self.row_shard[item]))
        s0 = int(self.row_start[item])
        tid = np.asarray(ids[s0:s0 + R], dtype=np.int64)           # [R, k]
        tlp = np.asarray(lps[s0:s0 + R], dtype=np.float32)          # [R, k]
        slp = np.asarray(samp[s0:s0 + R], dtype=np.float32)         # [R]
        y = r_ids.astype(np.int64)
        # duplicate ids (vLLM returned < k entries, padded by repeating the last) -> mask repeats
        dup = ((tid[:, :, None] == tid[:, None, :]) & np.tri(k, k, -1, dtype=bool)[None]).any(-1)
        in_top = (tid == y[:, None]).any(1)
        kd_ids = np.zeros((L, k + 1), np.int64)
        kd_lp = np.zeros((L, k + 1), np.float32)
        kd_mask = np.zeros((L, k + 1), bool)
        # response token r is predicted at position P-1+r; keep positions < L (shifted frame)
        n = max(0, min(R, L - P + 1))
        pos = slice(P - 1, P - 1 + n)
        kd_ids[pos, :k], kd_lp[pos, :k], kd_mask[pos, :k] = tid[:n], tlp[:n], ~dup[:n]
        kd_ids[pos, k], kd_lp[pos, k], kd_mask[pos, k] = y[:n], slp[:n], ~in_top[:n]
        out["kd_ids"] = torch.from_numpy(kd_ids)
        out["kd_logprobs"] = torch.from_numpy(kd_lp)
        out["kd_mask"] = torch.from_numpy(kd_mask)
        return out
