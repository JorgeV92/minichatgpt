#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minichatgpt.config import GPTConfig, TrainConfig
from minichatgpt.dataset import NextTokenDataset
from minichatgpt.model import GPT
from minichatgpt.tokenizer import BytePairTokenizer
from minichatgpt.trainer import Trainer
from minichatgpt.utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser(description="Pretrain a tiny GPT model on raw text.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a training text file")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for checkpoints")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    text = args.input.read_text(encoding="utf-8")
    tokenizer = BytePairTokenizer.load(args.tokenizer)
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(token_ids, block_size=args.block_size)

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    model = GPT(gpt_config)
    trainer = Trainer(model, dataset, train_config, out_dir=args.out_dir)
    best_path = trainer.fit()
    save_json(
        args.out_dir / "run_summary.json",
        {
            "num_parameters": model.num_parameters(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "checkpoint": str(best_path),
        },
    )
    print(f"best checkpoint: {best_path}")
    print(f"parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    main()
