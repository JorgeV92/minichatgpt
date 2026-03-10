#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minichatgpt.tokenizer import BytePairTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a training text file")
    parser.add_argument("--output", type=Path, required=True, help="Where to save tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=384, help="Final vocabulary size")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    tokenizer = BytePairTokenizer()
    tokenizer.train(text, vocab_size=args.vocab_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print(f"saved tokenizer to {args.output}")
    print(f"vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
