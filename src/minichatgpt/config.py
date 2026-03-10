from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 128
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate : float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 20
    device: str = "cpu"
    num_workers: int = 0