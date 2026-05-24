from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]")


def basic_tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.strip()]


class SimpleBPETokenizer:
    def __init__(self, special_tokens: Sequence[str] | None = None) -> None:
        self.special_tokens = list(special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
        self.merges: List[Tuple[str, str]] = []
        self.vocab: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _word_to_symbols(word: str) -> Tuple[str, ...]:
        return tuple(list(word) + ["</w>"])

    @staticmethod
    def _merge_once(symbols: Sequence[str], pair: Tuple[str, str]) -> Tuple[str, ...]:
        merged: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return tuple(merged)

    def train_from_token_sequences(
        self,
        token_sequences: Sequence[Sequence[str]],
        vocab_size: int = 2000,
        min_frequency: int = 2,
    ) -> None:
        word_freq = Counter()
        for sequence in token_sequences:
            for token in sequence:
                normalized = token.lower().strip()
                if normalized:
                    word_freq[normalized] += 1

        if not word_freq:
            raise ValueError("Cannot train tokenizer on an empty corpus.")

        vocab = {word: self._word_to_symbols(word) for word in word_freq}
        merges: List[Tuple[str, str]] = []

        def pair_statistics(current_vocab: Dict[str, Tuple[str, ...]]) -> Counter:
            stats: Counter = Counter()
            for word, symbols in current_vocab.items():
                frequency = word_freq[word]
                for pair in zip(symbols, symbols[1:]):
                    stats[pair] += frequency
            return stats

        target_vocab_size = max(vocab_size, len(self.special_tokens) + 32)
        while True:
            symbols = set()
            for symbol_seq in vocab.values():
                symbols.update(symbol_seq)
            if len(symbols) + len(self.special_tokens) >= target_vocab_size:
                break

            stats = pair_statistics(vocab)
            if not stats:
                break
            best_pair, best_count = stats.most_common(1)[0]
            if best_count < min_frequency:
                break

            merges.append(best_pair)
            vocab = {word: self._merge_once(symbols_seq, best_pair) for word, symbols_seq in vocab.items()}

        vocab_tokens = set(self.special_tokens)
        for symbols in vocab.values():
            for symbol in symbols:
                cleaned = symbol.replace("</w>", "")
                if cleaned:
                    vocab_tokens.add(cleaned)

        self.merges = merges
        self.merge_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.vocab = self.special_tokens + sorted(token for token in vocab_tokens if token not in self.special_tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def _apply_bpe(self, word: str) -> List[str]:
        symbols = list(word.lower()) + ["</w>"]
        if len(symbols) == 1:
            return symbols

        while True:
            pairs = list(zip(symbols, symbols[1:]))
            ranked_pairs = [pair for pair in pairs if pair in self.merge_ranks]
            if not ranked_pairs:
                break
            best_pair = min(ranked_pairs, key=lambda pair: self.merge_ranks[pair])
            symbols = list(self._merge_once(symbols, best_pair))
        return [symbol.replace("</w>", "") for symbol in symbols if symbol.replace("</w>", "")]

    def encode_tokens(self, tokens: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
        subwords: List[str] = []
        alignment: List[List[str]] = []
        for token in tokens:
            pieces = self._apply_bpe(token)
            if not pieces:
                pieces = ["[UNK]"]
            alignment.append(pieces)
            subwords.extend(pieces)
        return subwords, alignment

    def encode(self, text: str) -> List[str]:
        return self.encode_tokens(basic_tokenize(text))[0]

    def token_to_id_fn(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id.get("[UNK]", 1))

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id_fn(token) for token in tokens]

    def to_dict(self) -> Dict[str, object]:
        return {
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimpleBPETokenizer":
        tokenizer = cls(payload.get("special_tokens", ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
        tokenizer.merges = [tuple(pair) for pair in payload.get("merges", [])]
        tokenizer.merge_ranks = {pair: index for index, pair in enumerate(tokenizer.merges)}
        tokenizer.vocab = list(payload.get("vocab", []))
        tokenizer.token_to_id = {token: index for index, token in enumerate(tokenizer.vocab)}
        tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
        return tokenizer

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleBPETokenizer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)