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

def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    configure_torch_runtime()
    return torch.load(path, map_location=device)