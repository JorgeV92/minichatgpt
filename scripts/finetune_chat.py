#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minichatgpt.config import GPTConfig, TrainConfig
from minichatgpt.dataset import ChatSFTDataset, pad_collate
from minichatgpt.model import GPT
from minichatgpt.tokenizer import BytePairTokenizer
from minichatgpt.trainer import Trainer
from minichatgpt.utils import load_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tune MiniChatGPT on chat JSONL data.")
    parser.add_argument("--data", type=Path, required=True, help="Path to chat JSONL file")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for finetuned model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base pretrained checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    tokenizer = BytePairTokenizer.load(args.tokenizer)
    gpt_config = GPTConfig(**checkpoint["config"]["gpt_config"])
    model = GPT(gpt_config)
    model.load_state_dict(checkpoint["model_state"])

    dataset = ChatSFTDataset(args.data, tokenizer, block_size=gpt_config.block_size)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    trainer = Trainer(model, dataset, train_config, out_dir=args.out_dir, collate_fn=pad_collate)
    best_path = trainer.fit()
    print(f"finetuned checkpoint: {best_path}")


if __name__ == "__main__":
    main()
