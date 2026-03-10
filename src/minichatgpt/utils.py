from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch


def configure_torch_runtime() -> None:
    num_threads = int(os.environ.get("MINICHATGPT_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, num_threads))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch may disallow resetting this after parallel work has started.
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    configure_torch_runtime()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    configure_torch_runtime()
    return torch.load(path, map_location=device)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")