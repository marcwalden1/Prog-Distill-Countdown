"""Offline on-policy-distillation (OPD) dataset for verl's FSDP SFT trainer.

Loaded via verl's `data.custom_cls` hook (no verl-side file needed):

    data.custom_cls.path=$PROJECT_DIR/opd_dataset.py \
    data.custom_cls.name=OPDDataset

Reads parquets produced by scripts/generate_opd_data.py. Unlike SFTDataset it
does NOT retokenize text: `prompt_ids` / `response_ids` are the exact token IDs
of the student rollout, and `teacher_logprobs[i]` is the teacher's logprob of
`response_ids[i]` given everything before it. Returned batches match
SFTDataset's contract (input_ids / attention_mask / position_ids / loss_mask,
identical loss-mask semantics) plus a `teacher_logprobs` tensor aligned to
input positions (0.0 at prompt/pad positions, which the loss mask excludes).
The OPD loss branch in fsdp_sft_trainer.py (local patch 6) consumes it.
"""

import numpy as np
import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class OPDDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config):
        # tokenizer is unused (token IDs are stored) except for pad_token_id.
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        self.max_length = config.get("max_length", 2048)
        truncation = config.get("truncation", "error")
        assert truncation in ["error", "right"], (
            f"OPDDataset supports truncation error/right, got {truncation}"
        )
        self.truncation = truncation

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        local_files = [copy_to_local(f, verbose=True) for f in parquet_files]
        dataframe = pd.concat([pd.read_parquet(f) for f in local_files])

        # np.array (not asarray) so tensors never alias parquet's read-only buffers
        self.prompt_ids = [np.array(x, dtype=np.int64) for x in dataframe["prompt_ids"]]
        self.response_ids = [np.array(x, dtype=np.int64) for x in dataframe["response_ids"]]
        self.teacher_logprobs = [
            np.array(x, dtype=np.float32) for x in dataframe["teacher_logprobs"]
        ]
        for i, (r, t) in enumerate(zip(self.response_ids, self.teacher_logprobs)):
            assert len(r) == len(t), (
                f"row {i}: response_ids ({len(r)}) and teacher_logprobs ({len(t)}) disagree"
            )

    def __len__(self):
        return len(self.prompt_ids)

    def __getitem__(self, item):
        prompt_ids = torch.from_numpy(self.prompt_ids[item])
        response_ids = torch.from_numpy(self.response_ids[item])
        teacher_lp = torch.from_numpy(self.teacher_logprobs[item])

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.ones_like(input_ids)
        # teacher_logprobs[i] corresponds to input_ids[i] for response positions.
        teacher_logprobs = torch.cat(
            (torch.zeros(prompt_length, dtype=torch.float32), teacher_lp), dim=-1
        )

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            input_ids = torch.cat(
                (input_ids, torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype))
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype))
            )
            teacher_logprobs = torch.cat(
                (teacher_logprobs, torch.zeros(pad_len, dtype=torch.float32))
            )
        elif sequence_length > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                teacher_logprobs = teacher_logprobs[: self.max_length]
            else:
                raise NotImplementedError(
                    f"{sequence_length=} is larger than {self.max_length=}"
                )

        position_ids = compute_position_id_with_mask(attention_mask)

        # Same semantics as SFTDataset: in the shifted-loss frame, position i
        # predicts input_ids[i+1], so masking [:prompt_length-1] keeps exactly
        # the predictions of response tokens (incl. eos).
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "teacher_logprobs": teacher_logprobs,
        }
