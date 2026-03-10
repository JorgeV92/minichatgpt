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


@torch.no_grad()
def answer(
    model: GPT,
    tokenizer: BytePairTokenizer,
    history: list[tuple[str, str]],
    user_message: str,
    device: str = "cpu",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    prompt = render_history(history, system_prompt=system_prompt)
    prompt += f"\n### User:\n{user_message}\n\n### Assistant:\n"
    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tokenizer.decode(y[0].tolist())
    marker = "### Assistant:\n"
    answer_text = text.rsplit(marker, 1)[-1]
    if "### User:" in answer_text:
        answer_text = answer_text.split("### User:", 1)[0]
    return answer_text.strip()