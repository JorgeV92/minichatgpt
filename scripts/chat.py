#!/usr/bin/env python3
from __future__ import annotations

import argparse

from minichatgpt.chat import answer, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Terminal chat loop for MiniChatGPT.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device=args.device)
    history: list[tuple[str, str]] = []

    print("MiniChatGPT terminal chat")
    print("Type 'exit' to quit.\n")
    while True:
        user_message = input("you> ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("bye")
            break
        reply = answer(
            model,
            tokenizer,
            history,
            user_message,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        history.append((user_message, reply))
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
