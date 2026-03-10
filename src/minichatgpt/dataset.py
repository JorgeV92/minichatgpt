from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import BytePairTokenizer


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window next-token dataset for language model pretraining."""

    def __init__(self, token_ids: list[int], block_size: int, stride: int | None = None) -> None:
        self.tokens = token_ids
        self.block_size = block_size
        self.stride = stride or max(1, block_size // 2)
        self.starts = list(range(0, max(0, len(token_ids) - block_size - 1), self.stride))
        if not self.starts and len(token_ids) > block_size + 1:
            self.starts = [0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        chunk = self.tokens[start : start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class ChatSFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Supervised fine-tuning dataset for chat data.

    Expected JSONL format per line:
    {"user": "...", "assistant": "..."}
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: BytePairTokenizer, block_size: int) -> None:
        self.block_size = block_size
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        path = Path(jsonl_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            prompt = self.format_prompt(row["user"])
            full = prompt + row["assistant"] + "\n"
            prompt_ids = tokenizer.encode(prompt)
            full_ids = tokenizer.encode(full)
            if len(full_ids) < 2:
                continue
            full_ids = full_ids[: block_size + 1]
            x = torch.tensor(full_ids[:-1], dtype=torch.long)
            y = torch.tensor(full_ids[1:], dtype=torch.long)
            assistant_start = max(0, min(len(prompt_ids) - 1, len(y)))
            y[:assistant_start] = -100
            self.examples.append((x, y))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]

    @staticmethod
    def format_prompt(user_message: str) -> str:
        return f"### User:\n{user_message}\n\n### Assistant:\n"


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x_pad, y_pad
