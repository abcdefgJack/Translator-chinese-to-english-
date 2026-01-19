from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    """
    Loads tokenized examples saved by preprocess.py:
      dict: {"examples": [{"src_ids": [...], "tgt_ids": [...]}, ...], "meta": {...}}
    """
    def __init__(self, pt_path: str):
        obj = torch.load(pt_path, map_location="cpu")
        self.examples = obj["examples"]
        self.meta = obj["meta"]

        self.pad_id = int(self.meta["pad_id"])
        self.bos_id = int(self.meta["bos_id"])
        self.eos_id = int(self.meta["eos_id"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "src_ids": torch.tensor(ex["src_ids"], dtype=torch.long),
            "tgt_ids": torch.tensor(ex["tgt_ids"], dtype=torch.long),
        }


def make_padding_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    x: (B, T)
    return: (B, 1, 1, T) mask with True for tokens to attend (non-pad), False for pad
    This shape works nicely for multi-head attention broadcasting.
    """
    return (x != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """
    Causal (look-ahead) mask for decoder self-attn.
    return shape: (1, 1, T, T) with True allowed, False blocked (future positions)
    """
    # lower-triangular True
    mask = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class Batch:
    src: torch.Tensor           # (B, S)
    tgt_in: torch.Tensor        # (B, T-1)
    tgt_out: torch.Tensor       # (B, T-1)
    src_pad_mask: torch.Tensor  # (B, 1, 1, S)
    tgt_pad_mask: torch.Tensor  # (B, 1, 1, T-1)
    tgt_causal_mask: torch.Tensor  # (1, 1, T-1, T-1)


def collate_fn_builder(pad_id: int):
    """
    Returns a collate_fn that pads src/tgt and creates masks.
    """

    def collate(batch: List[Dict[str, Any]]) -> Batch:
        # batch: list of {"src_ids": (S,), "tgt_ids": (T,)}
        src_list = [b["src_ids"] for b in batch]
        tgt_list = [b["tgt_ids"] for b in batch]

        # Pad to max length in batch
        src = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)

        # Teacher forcing:
        # decoder input: tgt_in = tgt[:, :-1]
        # target labels: tgt_out = tgt[:, 1:]
        tgt_in = tgt[:, :-1].contiguous()
        tgt_out = tgt[:, 1:].contiguous()

        device = src.device  # usually cpu here; train.py will move to cuda

        src_pad_mask = make_padding_mask(src, pad_id)          # (B,1,1,S)
        tgt_pad_mask = make_padding_mask(tgt_in, pad_id)       # (B,1,1,T-1)
        tgt_causal_mask = make_causal_mask(tgt_in.size(1), device=device)  # (1,1,T-1,T-1)

        return Batch(
            src=src,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
            tgt_causal_mask=tgt_causal_mask,
        )

    return collate
