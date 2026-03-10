from __future__ import annotations

from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import BytePairTokenizer
from .utils import load_checkpoint

DEFAULT_SYSTEM_PROMPT = (
    "You are MiniChatGPT, a helpful assistant built for truth. "
    "Be clear, concise, and honest about uncertainty."
)

def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> tuple[GPT, BytePairTokenizer]:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    gpt_config = GPTConfig(**checkpoint["config"]["gpt_config"])
    model = GPT(gpt_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    return model, tokenizer


def render_history(history: list[tuple[str, str]], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    parts = [f"### System:\n{system_prompt}\n"]
    for user_text, assistant_text in history:
        parts.append(f"\n### User:\n{user_text}\n\n### Assistant:\n{assistant_text}\n")
    return "".join(parts)