from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class BytePairTokenizer:
    """
        TODO: add documentation 
    """

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int = 512, verbose: bool = True) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        tokens = list(text.encode("utf-8"))
        while len(self.vocab) < vocab_size:
            stats = self._get_pair_stats(tokens)
            if not stats:
                break
            pair, count = stats.most_common(1)[0]
            if count < 2:
                break
            new_id = len(self.vocab)
            self.merges.append(pair)
            self.merge_ranks[pair] = len(self.merges) - 1
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            tokens = self._merge_pair(tokens, pair, new_id)
            if verbose and new_id % 25 == 0:
                print(f"[tokenizer] vocab={len(self.vocab)} last_merge={pair} count={count}")

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            best_index = -1
            best_pair: tuple[int, int] | None = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_index = i
            if best_pair is None:
                break
            merged_id = 256 + self.merge_ranks[best_pair]
            tokens = tokens[:best_index] + [merged_id] + tokens[best_index + 2 :]
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        raw = b"".join(self.vocab[token_id] for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "version": 1,
            "merges": self.merges,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer.merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(tokenizer.merges, start=256):
            tokenizer.vocab[i] = tokenizer.vocab[a] + tokenizer.vocab[b]
        return tokenizer

    @staticmethod
    def _get_pair_stats(tokens: list[int]) -> Counter[tuple[int, int]]:
        stats: Counter[tuple[int, int]] = Counter()
        for i in range(len(tokens) - 1):
            stats[(tokens[i], tokens[i + 1])] += 1
        return stats

    @staticmethod
    def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
