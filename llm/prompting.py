from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from utils.preprocessing import ENTITY_BANK, INTENTS


PROMPT_EXAMPLES = [
    (
        "Book 2 vip tickets for Dune Part Two at PVR Nexus Mall",
        {"intent": "book_ticket", "entities": {"NUM_TICKETS": "2", "SEAT_TYPE": "vip", "MOVIE_NAME": "Dune Part Two", "THEATER_NAME": "PVR Nexus Mall"}},
    ),
    (
        "What time is Oppenheimer showing in Mumbai tomorrow",
        {"intent": "check_showtime", "entities": {"MOVIE_NAME": "Oppenheimer", "CITY": "Mumbai", "DATE": "tomorrow"}},
    ),
    (
        "cancel my booking for Barbie tonight",
        {"intent": "cancel_ticket", "entities": {"MOVIE_NAME": "Barbie", "DATE": "tonight"}},
    ),
]


def build_prompt(sentence: str, strategy: str = "structured_json") -> str:
    base_instructions = (
        "You are a movie booking assistant. Classify the user's intent and extract entities. "
        "Return a JSON object with keys intent and entities."
    )
    if strategy == "zero_shot":
        return f"{base_instructions}\nUser: {sentence}\nJSON:"
    if strategy == "few_shot":
        examples = [
            f"User: {example_sentence}\nAssistant: {json.dumps(example_output)}" for example_sentence, example_output in PROMPT_EXAMPLES
        ]
        return f"{base_instructions}\n\n" + "\n\n".join(examples) + f"\n\nUser: {sentence}\nAssistant:"
    if strategy == "structured_json":
        return (
            f"{base_instructions} "
            "Use the exact schema: {\"intent\": string, \"entities\": {label: value}}. "
            f"User: {sentence}\nAssistant:"
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _stable_random(sentence: str, strategy: str, seed: int) -> random.Random:
    digest = hashlib.md5(f"{seed}:{strategy}:{sentence}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def infer_intent(sentence: str) -> str:
    lowered = sentence.lower()
    score_table = {
        "greeting": ["hi", "hello", "hey", "yo", "good morning", "good evening"],
        "goodbye": ["bye", "goodbye", "see you", "thanks bye", "thx bye"],
        "search_movie": ["find", "search", "what movies", "running in", "available", "playing near"],
        "check_showtime": ["showtime", "showtimes", "timings", "what time", "when is", "start in"],
        "book_ticket": ["book", "reserve", "tickets", "seat", "please book", "need"],
        "cancel_ticket": ["cancel", "refund", "drop", "remove booking"],
        "select_seat": ["select seat", "pick", "choose", "seat number", "aisle"],
        "check_booking_status": ["status", "booking details", "look up", "reservation confirmed", "my ticket"],
    }
    scores = {intent: sum(1 for keyword in keywords if keyword in lowered) for intent, keywords in score_table.items()}
    best_intent = max(scores.items(), key=lambda item: (item[1], -INTENTS.index(item[0])))[0]
    if scores[best_intent] == 0:
        if any(word in lowered for word in ["movie", "show", "screen", "time"]):
            return "check_showtime"
        return "search_movie"
    return best_intent


def extract_entities(sentence: str) -> Dict[str, str]:
    lowered = sentence.lower()
    extracted: Dict[str, str] = {}
    labels = sorted(ENTITY_BANK.keys(), key=lambda label: max(len(value) for value in ENTITY_BANK[label]), reverse=True)
    for label in labels:
        values = sorted(ENTITY_BANK[label], key=len, reverse=True)
        for value in values:
            if value.lower() in lowered:
                extracted[label] = value
                break

    number_match = re.search(r"\b(\d+)\b", lowered)
    if number_match and any(keyword in lowered for keyword in ["ticket", "tickets", "seat", "seats", "book", "reserve"]):
        extracted.setdefault("NUM_TICKETS", number_match.group(1))

    time_match = re.search(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", lowered)
    if time_match:
        extracted.setdefault("TIME", time_match.group(0).strip())

    seat_match = re.search(r"\b[A-Z][0-9]{1,2}\b", sentence.upper())
    if seat_match:
        extracted.setdefault("SEAT_NUMBER", seat_match.group(0))

    if "today" in lowered:
        extracted.setdefault("DATE", "today")
    if "tomorrow" in lowered:
        extracted.setdefault("DATE", "tomorrow")
    if "tonight" in lowered:
        extracted.setdefault("DATE", "tonight")
    if "weekend" in lowered:
        extracted.setdefault("DATE", "this weekend")

    return extracted


def spans_from_entities(sentence: str, entities: Dict[str, str]) -> Dict[str, List[Tuple[int, int]]]:
    lowered = sentence.lower()
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for label, value in entities.items():
        value_lower = value.lower()
        start = lowered.find(value_lower)
        if start < 0:
            continue
        end = start + len(value_lower) - 1
        spans.setdefault(label, []).append((start, end))
    return spans


def _perturb_output(intent: str, entities: Dict[str, str], strategy: str, rng: random.Random) -> Tuple[str, Dict[str, str], bool]:
    error_rate = {"zero_shot": 0.22, "few_shot": 0.12, "structured_json": 0.06}[strategy]
    malformed = rng.random() < (error_rate / 2)
    if rng.random() < error_rate:
        if rng.random() < 0.5 and entities:
            drop_key = rng.choice(list(entities.keys()))
            entities = dict(entities)
            entities.pop(drop_key, None)
        else:
            intent = rng.choice([candidate for candidate in INTENTS if candidate != intent])
    if rng.random() < error_rate / 3:
        entities = dict(entities)
        random_label = rng.choice(["CITY", "DATE", "THEATER_NAME"])
        entities[random_label] = rng.choice(ENTITY_BANK[random_label])
    return intent, entities, malformed


def safe_parse_json_output(raw_output: str) -> Dict[str, object]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    intent_match = re.search(r'"?intent"?\s*[:=]\s*"?([a-z_]+)"?', cleaned, flags=re.IGNORECASE)
    if intent_match:
        return {"intent": intent_match.group(1), "entities": {}}
    return {"intent": "unknown", "entities": {}}


@dataclass
class LLMResult:
    sentence: str
    strategy: str
    prompt: str
    raw_output: str
    parsed_output: Dict[str, object]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float


class SimulatedLLMPipeline:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @staticmethod
    def _token_count(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _generate_response(self, sentence: str, strategy: str) -> Tuple[str, str, Dict[str, object]]:
        rng = _stable_random(sentence, strategy, self.seed)
        intent = infer_intent(sentence)
        entities = extract_entities(sentence)
        intent, entities, malformed = _perturb_output(intent, entities, strategy, rng)
        payload = {"intent": intent, "entities": entities}
        if strategy == "zero_shot":
            raw = json.dumps(payload, indent=2) if not malformed else f"Intent: {intent}; Entities: {json.dumps(entities)}"
        elif strategy == "few_shot":
            raw = json.dumps(payload, indent=2) if not malformed else f"```json\n{json.dumps(payload)}\n```"
        else:
            raw = json.dumps(payload) if not malformed else f"{{intent: {intent}, entities: {json.dumps(entities)}}}"
        return intent, raw, payload

    def predict(self, sentence: str, strategy: str = "structured_json") -> LLMResult:
        prompt = build_prompt(sentence, strategy=strategy)
        start = time.perf_counter()
        _, raw_output, payload = self._generate_response(sentence, strategy)
        latency = time.perf_counter() - start
        parsed_output = safe_parse_json_output(raw_output)
        prompt_tokens = self._token_count(prompt)
        completion_tokens = self._token_count(raw_output)
        estimated_cost = (prompt_tokens + completion_tokens) * 0.00001
        return LLMResult(
            sentence=sentence,
            strategy=strategy,
            prompt=prompt,
            raw_output=raw_output,
            parsed_output=parsed_output,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
        )

    def predict_batch(self, sentences: Sequence[str], strategy: str = "structured_json") -> List[LLMResult]:
        return [self.predict(sentence, strategy=strategy) for sentence in sentences]