import torch

from minichatgpt.config import GPTConfig
from minichatgpt.model import GPT

def test_model_forward_shape() -> None:
    config = GPTConfig(vocab_size=300, block_size=16, n_layers=2, n_heads=2, n_embd=32)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (4, 16))
    y = torch.randint(0, config.vocab_size, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0