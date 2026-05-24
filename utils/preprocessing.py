from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import SimpleBPETokenizer, basic_tokenize


INTENTS = [
    "search_movie",
    "check_showtime",
    "book_ticket",
    "cancel_ticket",
    "select_seat",
    "check_booking_status",
    "greeting",
    "goodbye",
]

ENTITY_LABELS = [
    "MOVIE_NAME",
    "THEATER_NAME",
    "CITY",
    "DATE",
    "TIME",
    "NUM_TICKETS",
    "SEAT_TYPE",
    "SEAT_NUMBER",
    "LANGUAGE",
    "SCREEN_TYPE",
]

ENTITY_BANK: Dict[str, List[str]] = {
    "MOVIE_NAME": [
        "Dune Part Two",
        "Oppenheimer",
        "Barbie",
        "Interstellar",
        "Jawan",
        "Leo",
        "Kalki 2898 AD",
        "Tiger 3",
        "Spider Man No Way Home",
        "The Batman",
        "Avatar The Way of Water",
        "Top Gun Maverick",
    ],
    "THEATER_NAME": [
        "PVR Nexus Mall",
        "INOX City Center",
        "Cinepolis Central",
        "Luxe IMAX",
        "Miraj Cinemas",
        "Carnival City Mall",
        "Wave Cinemas",
        "Raj Mandir",
    ],
    "CITY": [
        "Mumbai",
        "Delhi",
        "Bengaluru",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Kolkata",
        "Ahmedabad",
        "Jaipur",
        "Kochi",
    ],
    "DATE": [
        "today",
        "tomorrow",
        "this evening",
        "tonight",
        "this weekend",
        "next Friday",
        "next Saturday",
        "25 May",
        "26 May",
        "27 May",
    ],
    "TIME": [
        "10 am",
        "11:30 am",
        "1 pm",
        "3:15 pm",
        "5 pm",
        "6:45 pm",
        "8 pm",
        "9:30 pm",
        "11 pm",
    ],
    "NUM_TICKETS": ["1", "2", "3", "4", "5", "6"],
    "SEAT_TYPE": ["regular", "vip", "premium", "recliner", "balcony"],
    "SEAT_NUMBER": ["A10", "A11", "B7", "B12", "C3", "C4", "D8", "E15"],
    "LANGUAGE": ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Kannada", "French"],
    "SCREEN_TYPE": ["2D", "3D", "IMAX", "Dolby", "4DX"],
}

INTENT_TEMPLATES: Dict[str, List[List[object]]] = {
    "search_movie": [
        ["find", ("MOVIE_NAME",), "in", ("CITY",)],
        ["what movies like", ("MOVIE_NAME",), "are running in", ("CITY",)],
        ["show me", ("MOVIE_NAME",), "sessions in", ("CITY",)],
        ["is", ("MOVIE_NAME",), "playing near", ("CITY",)],
        ["any", ("LANGUAGE",), "movies in", ("CITY",)],
    ],
    "check_showtime": [
        ["when is", ("MOVIE_NAME",), "showing at", ("THEATER_NAME",)],
        ["show me showtimes for", ("MOVIE_NAME",), "on", ("DATE",)],
        ["what time does", ("MOVIE_NAME",), "start in", ("CITY",)],
        ["tell me the timings for", ("MOVIE_NAME",), "at", ("THEATER_NAME",)],
        ["is there a", ("SCREEN_TYPE",), "show of", ("MOVIE_NAME",), "today"],
    ],
    "book_ticket": [
        ["book", ("NUM_TICKETS",), ("SEAT_TYPE",), "tickets for", ("MOVIE_NAME",), "at", ("THEATER_NAME",)],
        ["i need", ("NUM_TICKETS",), "tickets for", ("MOVIE_NAME",), "this", ("DATE",)],
        ["reserve", ("NUM_TICKETS",), ("SEAT_TYPE",), "seats in", ("CITY",)],
        ["please book me", ("NUM_TICKETS",), "for", ("MOVIE_NAME",), "at", ("TIME",)],
        ["can u book", ("MOVIE_NAME",), "for", ("NUM_TICKETS",), "people"],
    ],
    "cancel_ticket": [
        ["cancel my", ("MOVIE_NAME",), "booking at", ("THEATER_NAME",)],
        ["i want to cancel tickets for", ("MOVIE_NAME",), "today"],
        ["cancel the", ("NUM_TICKETS",), "seats in", ("CITY",)],
        ["please refund my booking for", ("DATE",)],
        ["drop my reservation at", ("THEATER_NAME",)],
    ],
    "select_seat": [
        ["select seat", ("SEAT_NUMBER",)],
        ["pick", ("SEAT_TYPE",), "seats near", ("SEAT_NUMBER",)],
        ["i want", ("SEAT_NUMBER",), "for the show"],
        ["choose", ("SEAT_TYPE",), "seat", ("SEAT_NUMBER",)],
        ["give me", ("SEAT_TYPE",), "around", ("SEAT_NUMBER",)],
    ],
    "check_booking_status": [
        ["check my booking status for", ("MOVIE_NAME",)],
        ["where is my ticket for", ("DATE",)],
        ["show my booking details at", ("THEATER_NAME",)],
        ["is my reservation confirmed for", ("TIME",)],
        ["look up my movie booking in", ("CITY",)],
    ],
    "greeting": [
        ["hey there"],
        ["hello"],
        ["hi, need help"],
        ["good morning"],
        ["yo"],
    ],
    "goodbye": [
        ["thanks bye"],
        ["see you later"],
        ["goodbye"],
        ["that's all thanks"],
        ["bye and thanks"],
    ],
}

COMMON_TYPOS = {
    "movie": "movi",
    "movies": "muvies",
    "theater": "thetre",
    "theatre": "thtr",
    "tickets": "tix",
    "ticket": "tkt",
    "book": "buk",
    "showtime": "shoetime",
    "please": "plz",
    "reserve": "resrv",
    "cancel": "cncl",
    "status": "statuz",
    "select": "slect",
    "hello": "helo",
    "thanks": "thx",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean_join(tokens: Sequence[str]) -> str:
    return " ".join(tokens).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")


def _apply_noise(tokens: Sequence[str], rng: random.Random) -> List[str]:
    noisy_tokens: List[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in COMMON_TYPOS and rng.random() < 0.35:
            noisy_tokens.append(COMMON_TYPOS[lowered])
            continue
        if len(token) > 4 and rng.random() < 0.12:
            index = rng.randint(1, len(token) - 2)
            noisy_tokens.append(token[:index] + token[index + 1 :])
            continue
        if len(token) > 4 and rng.random() < 0.08:
            index = rng.randint(1, len(token) - 2)
            replacement = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
            noisy_tokens.append(token[:index] + replacement + token[index + 1 :])
            continue
        noisy_tokens.append(token)
    if rng.random() < 0.15:
        noisy_tokens.insert(0, rng.choice(["pls", "hey", "yo", "um", "quick"]))
    if rng.random() < 0.12:
        noisy_tokens.append(rng.choice(["pls", "thx", "now", "please", "?"]))
    return noisy_tokens


def _render_template(template: Sequence[object], rng: random.Random) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    tags: List[str] = []
    for segment in template:
        if isinstance(segment, tuple):
            label = segment[0]
            value = rng.choice(ENTITY_BANK[label])
            value_tokens = basic_tokenize(value)
            for index, token in enumerate(value_tokens):
                tags.append(f"{'B' if index == 0 else 'I'}-{label}")
                tokens.append(token)
        else:
            literal_tokens = basic_tokenize(str(segment))
            tokens.extend(literal_tokens)
            tags.extend(["O"] * len(literal_tokens))
    return tokens, tags


def build_example(intent: str, rng: random.Random) -> Dict[str, object]:
    template = rng.choice(INTENT_TEMPLATES[intent])
    tokens, tags = _render_template(template, rng)
    if intent not in {"greeting", "goodbye"} and rng.random() < 0.5:
        tokens = [rng.choice(["hey", "pls", "um", "yo"])]+tokens
        tags = ["O"] + tags
    if rng.random() < 0.45:
        tokens = _apply_noise(tokens, rng)
    sentence = _clean_join(tokens)
    return {
        "sentence": sentence,
        "intent": intent,
        "tokens": tokens,
        "BIO_tags": tags,
    }


def generate_synthetic_dataset(num_examples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    examples: List[Dict[str, object]] = []
    per_intent = num_examples // len(INTENTS)
    remainder = num_examples % len(INTENTS)
    for intent_index, intent in enumerate(INTENTS):
        count = per_intent + (1 if intent_index < remainder else 0)
        for _ in range(count):
            examples.append(build_example(intent, rng))
    rng.shuffle(examples)
    dataframe = pd.DataFrame(examples)
    dataframe["tokens"] = dataframe["tokens"].apply(json.dumps)
    dataframe["BIO_tags"] = dataframe["BIO_tags"].apply(json.dumps)
    return dataframe


def split_dataset(dataframe: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for _, group in dataframe.groupby("intent"):
        indices = group.index.to_list()
        rng.shuffle(indices)
        total = len(indices)
        train_end = max(1, int(round(total * 0.70)))
        val_end = max(train_end + 1, int(round(total * 0.85)))
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])
    train_df = dataframe.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = dataframe.loc[val_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = dataframe.loc[test_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def build_label_maps(dataframe: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    intents = sorted(dataframe["intent"].unique().tolist())
    intent_to_id = {intent: index for index, intent in enumerate(intents)}
    all_tags = {"O"}
    for entity_label in ENTITY_LABELS:
        all_tags.add(f"B-{entity_label}")
        all_tags.add(f"I-{entity_label}")
    for tags in dataframe["BIO_tags"]:
        all_tags.update(json.loads(tags) if isinstance(tags, str) else tags)
    ordered_tags = ["O"] + sorted(tag for tag in all_tags if tag != "O")
    tag_to_id = {tag: index for index, tag in enumerate(ordered_tags)}
    id_to_intent = {index: intent for intent, index in intent_to_id.items()}
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}
    return intent_to_id, tag_to_id, id_to_intent, id_to_tag


def serialize_dataframe(dataframe: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def deserialize_tokens(column_value: str) -> List[str]:
    return list(json.loads(column_value))


def deserialize_tags(column_value: str) -> List[str]:
    return list(json.loads(column_value))


def align_tags_to_subwords(tags: Sequence[str], alignment: Sequence[Sequence[str]]) -> List[str]:
    subword_tags: List[str] = []
    for tag, pieces in zip(tags, alignment):
        if tag == "O":
            subword_tags.extend(["O"] * len(pieces))
            continue
        prefix, label = tag.split("-", 1)
        if not pieces:
            continue
        subword_tags.append(f"B-{label}" if prefix == "B" else f"I-{label}")
        subword_tags.extend([f"I-{label}"] * (len(pieces) - 1))
    return subword_tags


def encode_dataframe(
    dataframe: pd.DataFrame,
    tokenizer: SimpleBPETokenizer,
    intent_to_id: Dict[str, int],
    tag_to_id: Dict[str, int],
    max_length: int = 48,
) -> List[Dict[str, object]]:
    encoded_samples: List[Dict[str, object]] = []
    pad_id = tokenizer.token_to_id.get("[PAD]", 0)
    cls_id = tokenizer.token_to_id.get("[CLS]", 2)
    sep_id = tokenizer.token_to_id.get("[SEP]", 3)

    for _, row in dataframe.iterrows():
        tokens = deserialize_tokens(row["tokens"])
        tags = deserialize_tags(row["BIO_tags"])
        subwords, alignment = tokenizer.encode_tokens(tokens)
        subword_tags = align_tags_to_subwords(tags, alignment)
        trimmed_subwords = subwords[: max_length - 2]
        trimmed_tags = subword_tags[: max_length - 2]
        if len(trimmed_tags) < len(trimmed_subwords):
            trimmed_tags.extend(["O"] * (len(trimmed_subwords) - len(trimmed_tags)))
        elif len(trimmed_tags) > len(trimmed_subwords):
            trimmed_tags = trimmed_tags[: len(trimmed_subwords)]
        input_ids = [cls_id] + tokenizer.convert_tokens_to_ids(trimmed_subwords) + [sep_id]
        attention_mask = [1] * len(input_ids)
        ner_labels = [-100] + [tag_to_id[tag] for tag in trimmed_tags] + [-100]
        if len(input_ids) < max_length:
            padding = max_length - len(input_ids)
            input_ids.extend([pad_id] * padding)
            attention_mask.extend([0] * padding)
            ner_labels.extend([-100] * padding)
        encoded_samples.append(
            {
                "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
                "intent_id": torch.tensor(intent_to_id[row["intent"]], dtype=torch.long),
                "ner_labels": torch.tensor(ner_labels[:max_length], dtype=torch.long),
                "sentence": row["sentence"],
                "intent": row["intent"],
                "tokens": tokens,
                "tags": tags,
                "subwords": ["[CLS]"] + trimmed_subwords + ["[SEP]"],
                "subword_tags": ["O"] + trimmed_tags + ["O"],
            }
        )
    return encoded_samples


class EncodedDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, object]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["input_ids", "attention_mask", "intent_id", "ner_labels"]
    collated: Dict[str, object] = {}
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    for key in batch[0].keys():
        if key not in tensor_keys:
            collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(samples: Sequence[Dict[str, object]], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(EncodedDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def compute_oov_rate(dataframe: pd.DataFrame, token_set: Sequence[str]) -> float:
    total = 0
    missing = 0
    vocabulary = set(token_set)
    for _, row in dataframe.iterrows():
        tokens = deserialize_tokens(row["tokens"])
        for token in tokens:
            total += 1
            if token.lower() not in vocabulary:
                missing += 1
    return missing / total if total else 0.0


def compute_basic_vocab(train_df: pd.DataFrame) -> List[str]:
    vocab = set()
    for tokens in train_df["tokens"]:
        vocab.update(token.lower() for token in json.loads(tokens))
    return sorted(vocab)


def entity_distribution(dataframe: pd.DataFrame) -> Dict[str, int]:
    counts = Counter()
    for tags in dataframe["BIO_tags"]:
        tag_list = json.loads(tags)
        for tag in tag_list:
            if tag != "O":
                counts[tag.split("-", 1)[1]] += 1
    return dict(counts)


def dataset_statistics(dataframe: pd.DataFrame) -> Dict[str, object]:
    lengths = [len(json.loads(tokens)) for tokens in dataframe["tokens"]]
    intent_counts = dataframe["intent"].value_counts().to_dict()
    entity_counts = entity_distribution(dataframe)
    return {
        "num_examples": len(dataframe),
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_length": int(np.max(lengths)) if lengths else 0,
        "intent_counts": intent_counts,
        "entity_counts": entity_counts,
    }


def save_json(path: str, payload: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)