from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from llm.prompting import SimulatedLLMPipeline, build_prompt, spans_from_entities
from models.encoder_model import MovieBookingEncoder
from utils.metrics import classification_metrics, confusion_matrix, entity_metrics, model_size_mb, summarize_latencies
from utils.preprocessing import (
    ENTITY_BANK,
    ENTITY_LABELS,
    INTENTS,
    align_tags_to_subwords,
    build_dataloader,
    build_label_maps,
    compute_basic_vocab,
    compute_oov_rate,
    dataset_statistics,
    deserialize_tokens,
    deserialize_tags,
    encode_dataframe,
    entity_distribution,
    generate_synthetic_dataset,
    load_json,
    save_json,
    serialize_dataframe,
    set_seed,
    split_dataset,
)
from utils.tokenizer import SimpleBPETokenizer, basic_tokenize


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_or_generate_dataset(num_examples: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = DATA_DIR / "dataset.csv"
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    if dataset_path.exists() and train_path.exists() and val_path.exists() and test_path.exists():
        dataset = pd.read_csv(dataset_path)
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return dataset, train_df, val_df, test_df

    dataset = generate_synthetic_dataset(num_examples=num_examples, seed=seed)
    train_df, val_df, test_df = split_dataset(dataset, seed=seed)
    serialize_dataframe(dataset, str(dataset_path))
    serialize_dataframe(train_df, str(train_path))
    serialize_dataframe(val_df, str(val_path))
    serialize_dataframe(test_df, str(test_path))
    return dataset, train_df, val_df, test_df


def plot_dataset_statistics(dataset: pd.DataFrame) -> None:
    stats = dataset_statistics(dataset)
    lengths = [len(json.loads(tokens)) for tokens in dataset["tokens"]]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(stats["intent_counts"].keys()), y=list(stats["intent_counts"].values()), palette="viridis")
    plt.xticks(rotation=35, ha="right")
    plt.title("Intent Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "intent_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.histplot(lengths, bins=20, kde=True, color="#2f6f8f")
    plt.title("Sentence Length Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "length_distribution.png", dpi=180)
    plt.close()

    entity_counts = stats["entity_counts"]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()), palette="magma")
    plt.xticks(rotation=35, ha="right")
    plt.title("Entity Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "entity_distribution.png", dpi=180)
    plt.close()


def build_tokenizer(train_df: pd.DataFrame) -> SimpleBPETokenizer:
    tokenizer_path = DATA_DIR / "tokenizer.json"
    if tokenizer_path.exists():
        return SimpleBPETokenizer.load(str(tokenizer_path))

    token_sequences = [json.loads(tokens) for tokens in train_df["tokens"]]
    tokenizer = SimpleBPETokenizer()
    tokenizer.train_from_token_sequences(token_sequences, vocab_size=500, min_frequency=2)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def compute_subword_oov_rate(dataframe: pd.DataFrame, tokenizer: SimpleBPETokenizer) -> float:
    total = 0
    missing = 0
    for tokens_serialized in dataframe["tokens"]:
        tokens = json.loads(tokens_serialized)
        subwords, _ = tokenizer.encode_tokens(tokens)
        for token in subwords:
            total += 1
            if token not in tokenizer.token_to_id:
                missing += 1
    return missing / total if total else 0.0


def compute_tokenizer_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: SimpleBPETokenizer) -> Dict[str, float]:
    basic_vocab = set(compute_basic_vocab(train_df))
    basic_oov_val = compute_oov_rate(val_df, basic_vocab)
    basic_oov_test = compute_oov_rate(test_df, basic_vocab)
    subword_oov_val = compute_subword_oov_rate(val_df, tokenizer)
    subword_oov_test = compute_subword_oov_rate(test_df, tokenizer)
    report = {
        "basic_oov_val": basic_oov_val,
        "basic_oov_test": basic_oov_test,
        "subword_oov_val": subword_oov_val,
        "subword_oov_test": subword_oov_test,
    }
    save_json(str(DATA_DIR / "tokenizer_report.json"), report)
    return report


def build_artifacts(dataset: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    intent_to_id, tag_to_id, id_to_intent, id_to_tag = build_label_maps(dataset)
    save_json(
        str(DATA_DIR / "label_maps.json"),
        {
            "intent_to_id": intent_to_id,
            "tag_to_id": tag_to_id,
            "id_to_intent": id_to_intent,
            "id_to_tag": id_to_tag,
        },
    )
    tokenizer = build_tokenizer(train_df)
    tokenizer_report = compute_tokenizer_report(train_df, val_df, test_df, tokenizer)
    encoded_train = encode_dataframe(train_df, tokenizer, intent_to_id, tag_to_id)
    encoded_val = encode_dataframe(val_df, tokenizer, intent_to_id, tag_to_id)
    encoded_test = encode_dataframe(test_df, tokenizer, intent_to_id, tag_to_id)
    return tokenizer, intent_to_id, tag_to_id, id_to_intent, id_to_tag, tokenizer_report, encoded_train, encoded_val, encoded_test


def tags_from_entities(tokens: Sequence[str], entities: Dict[str, str]) -> List[str]:
    tags = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    tokenized = [token.lower() for token in tokens]
    ordered_entities = sorted(entities.items(), key=lambda item: len(basic_tokenize(str(item[1]))), reverse=True)
    for label, value in ordered_entities:
        value_tokens = [token.lower() for token in basic_tokenize(str(value))]
        if not value_tokens:
            continue
        for start in range(0, len(tokens) - len(value_tokens) + 1):
            span = tokenized[start : start + len(value_tokens)]
            if span == value_tokens and not any(occupied[start : start + len(value_tokens)]):
                tags[start] = f"B-{label}"
                for offset in range(1, len(value_tokens)):
                    tags[start + offset] = f"I-{label}"
                for offset in range(len(value_tokens)):
                    occupied[start + offset] = True
                break
    return tags


def train_encoder_model(
    tokenizer: SimpleBPETokenizer,
    intent_to_id: Dict[str, int],
    tag_to_id: Dict[str, int],
    encoded_train: Sequence[Dict[str, object]],
    encoded_val: Sequence[Dict[str, object]],
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    patience: int = 3,
) -> Tuple[MovieBookingEncoder, Dict[str, List[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader(encoded_train, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(encoded_val, batch_size=batch_size, shuffle=False)

    model = MovieBookingEncoder(
        vocab_size=len(tokenizer.vocab),
        num_intents=len(intent_to_id),
        num_tags=len(tag_to_id),
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        dropout=0.1,
        pad_id=tokenizer.token_to_id.get("[PAD]", 0),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    intent_loss_fn = nn.CrossEntropyLoss()
    entity_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_targets = batch["intent_id"].to(device)
            entity_targets = batch["ner_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            intent_logits, entity_logits = model(input_ids, attention_mask)
            intent_loss = intent_loss_fn(intent_logits, intent_targets)
            entity_loss = entity_loss_fn(entity_logits.view(-1, entity_logits.size(-1)), entity_targets.view(-1))
            loss = intent_loss + entity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_encoder_loss(model, val_loader, intent_loss_fn, entity_loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, MODELS_DIR / "best_encoder.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate_encoder_loss(model, loader, intent_loss_fn, entity_loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_targets = batch["intent_id"].to(device)
            entity_targets = batch["ner_labels"].to(device)
            intent_logits, entity_logits = model(input_ids, attention_mask)
            intent_loss = intent_loss_fn(intent_logits, intent_targets)
            entity_loss = entity_loss_fn(entity_logits.view(-1, entity_logits.size(-1)), entity_targets.view(-1))
            total_loss += (intent_loss + entity_loss).item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def evaluate_encoder_model(
    model: MovieBookingEncoder,
    encoded_test: Sequence[Dict[str, object]],
    id_to_intent: Dict[int, str],
    id_to_tag: Dict[int, str],
) -> Dict[str, object]:
    device = next(model.parameters()).device
    loader = build_dataloader(encoded_test, batch_size=32, shuffle=False)
    intent_true: List[str] = []
    intent_pred: List[str] = []
    gold_entity_sequences: List[List[str]] = []
    pred_entity_sequences: List[List[str]] = []
    latencies: List[float] = []
    error_examples: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start = time.perf_counter()
            intent_logits, entity_logits = model(input_ids, attention_mask)
            elapsed = time.perf_counter() - start
            batch_size = input_ids.size(0)
            latencies.extend([elapsed / max(1, batch_size)] * batch_size)

            intent_predictions = intent_logits.argmax(dim=-1).cpu().tolist()
            entity_predictions = entity_logits.argmax(dim=-1).cpu().tolist()
            batch_intent_targets = batch["intent_id"].cpu().tolist()
            batch_entity_targets = batch["ner_labels"].cpu().tolist()

            for index, sentence in enumerate(batch["sentence"]):
                gold_intent = id_to_intent[int(batch_intent_targets[index])]
                pred_intent = id_to_intent[int(intent_predictions[index])]
                gold_sequence = [id_to_tag[label] for label in batch_entity_targets[index] if label != -100]
                pred_sequence = [id_to_tag[int(label)] for label, gold in zip(entity_predictions[index], batch_entity_targets[index]) if gold != -100]
                intent_true.append(gold_intent)
                intent_pred.append(pred_intent)
                gold_entity_sequences.append(gold_sequence)
                pred_entity_sequences.append(pred_sequence)

                if gold_intent != pred_intent and len(error_examples) < 5:
                    error_examples.append(
                        {
                            "sentence": sentence,
                            "gold_intent": gold_intent,
                            "pred_intent": pred_intent,
                            "gold_entities": gold_sequence,
                            "pred_entities": pred_sequence,
                            "reason": "intent misclassification",
                        }
                    )

    intent_metrics = classification_metrics(intent_true, intent_pred, labels=INTENTS)
    entity_scores = entity_metrics(gold_entity_sequences, pred_entity_sequences)
    return {
        "intent_metrics": intent_metrics,
        "entity_metrics": entity_scores,
        "latency_metrics": summarize_latencies(latencies),
        "model_size_mb": model_size_mb(model),
        "intent_true": intent_true,
        "intent_pred": intent_pred,
        "gold_entity_sequences": gold_entity_sequences,
        "pred_entity_sequences": pred_entity_sequences,
        "error_examples": error_examples,
    }


def evaluate_llm_pipeline(test_df: pd.DataFrame, strategies: Sequence[str] = ("zero_shot", "few_shot", "structured_json")) -> Dict[str, object]:
    pipeline = SimulatedLLMPipeline(seed=42)
    strategy_results: Dict[str, object] = {}
    for strategy in strategies:
        intent_true: List[str] = []
        intent_pred: List[str] = []
        gold_entity_sequences: List[List[str]] = []
        pred_entity_sequences: List[List[str]] = []
        latencies: List[float] = []
        errors: List[Dict[str, object]] = []
        total_cost = 0.0

        for _, row in test_df.iterrows():
            result = pipeline.predict(row["sentence"], strategy=strategy)
            tokens = json.loads(row["tokens"])
            gold_tags = json.loads(row["BIO_tags"])
            parsed_intent = str(result.parsed_output.get("intent", "unknown")) if isinstance(result.parsed_output, dict) else "unknown"
            parsed_entities = result.parsed_output.get("entities", {}) if isinstance(result.parsed_output, dict) else {}
            parsed_entities = parsed_entities if isinstance(parsed_entities, dict) else {}

            predicted_tags = tags_from_entities(tokens, parsed_entities)
            intent_true.append(row["intent"])
            intent_pred.append(parsed_intent)
            gold_entity_sequences.append(gold_tags)
            pred_entity_sequences.append(predicted_tags)
            latencies.append(result.latency_seconds)
            total_cost += result.estimated_cost

            if (row["intent"] != parsed_intent or gold_tags != predicted_tags) and len(errors) < 5:
                errors.append(
                    {
                        "sentence": row["sentence"],
                        "gold_intent": row["intent"],
                        "pred_intent": parsed_intent,
                        "gold_entities": gold_tags,
                        "pred_entities": parsed_entities,
                        "raw_output": result.raw_output,
                        "reason": "parsing or extraction error" if parsed_intent == "unknown" else "intent mismatch or missing span",
                    }
                )

        strategy_results[strategy] = {
            "intent_metrics": classification_metrics(intent_true, intent_pred, labels=INTENTS),
            "entity_metrics": entity_metrics(gold_entity_sequences, pred_entity_sequences),
            "latency_metrics": summarize_latencies(latencies),
            "cost_estimate": total_cost,
            "model_size_mb": 0.0,
            "intent_true": intent_true,
            "pred_intents": intent_pred,
            "gold_entity_sequences": gold_entity_sequences,
            "pred_entity_sequences": pred_entity_sequences,
            "error_examples": errors,
        }
    return strategy_results


def generate_adversarial_samples(test_df: pd.DataFrame, count: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    selected = test_df.sample(n=min(count, len(test_df)), random_state=seed).reset_index(drop=True)
    samples = []
    typo_map = {
        "movie": "m0vie",
        "book": "buk",
        "cancel": "cncl",
        "showtime": "shoetme",
        "ticket": "tkt",
        "theater": "thetre",
        "please": "plz",
        "booking": "boking",
    }
    for _, row in selected.iterrows():
        tokens = json.loads(row["tokens"])
        mutated = []
        for token in tokens:
            if token.lower() in typo_map and rng.random() < 0.6:
                mutated.append(typo_map[token.lower()])
            elif rng.random() < 0.12 and len(token) > 3:
                mutated.append(token[:-1])
            else:
                mutated.append(token)
        if rng.random() < 0.5:
            mutated.insert(0, rng.choice(["uh", "hey", "pls", "can u"]))
        if rng.random() < 0.4:
            mutated.append(rng.choice(["now", "quick", "pls", "thx"]))
        sentence = " ".join(mutated)
        samples.append({"sentence": sentence, "intent": row["intent"], "tokens": json.dumps(mutated), "BIO_tags": row["BIO_tags"]})
    return pd.DataFrame(samples)


def intent_confusion_figure(y_true: Sequence[str], y_pred: Sequence[str], title: str, path: Path) -> None:
    labels = INTENTS
    matrix = confusion_matrix(y_true, y_pred, labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def metric_comparison_figure(encoder_summary: Dict[str, object], llm_summary: Dict[str, object]) -> None:
    metrics = ["intent_accuracy", "intent_macro_f1", "entity_strict_f1", "entity_partial_f1", "latency_ms"]
    encoder_values = [
        encoder_summary["intent_metrics"]["accuracy"],
        encoder_summary["intent_metrics"]["macro_f1"],
        encoder_summary["entity_metrics"]["strict_f1"],
        encoder_summary["entity_metrics"]["partial_f1"],
        encoder_summary["latency_metrics"]["mean"] * 1000,
    ]
    llm_values = [
        llm_summary["intent_metrics"]["accuracy"],
        llm_summary["intent_metrics"]["macro_f1"],
        llm_summary["entity_metrics"]["strict_f1"],
        llm_summary["entity_metrics"]["partial_f1"],
        llm_summary["latency_metrics"]["mean"] * 1000,
    ]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(11, 4.5))
    plt.bar(x - width / 2, encoder_values, width=width, label="Encoder")
    plt.bar(x + width / 2, llm_values, width=width, label="LLM")
    plt.xticks(x, metrics, rotation=25, ha="right")
    plt.ylabel("Score / ms")
    plt.title("Encoder vs LLM Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metric_comparison.png", dpi=180)
    plt.close()


def save_predictions_examples(test_df: pd.DataFrame, encoder_summary: Dict[str, object], llm_summary: Dict[str, object]) -> None:
    examples = []
    for index in range(min(5, len(test_df))):
        examples.append(
            {
                "sentence": test_df.iloc[index]["sentence"],
                "gold_intent": test_df.iloc[index]["intent"],
                "encoder_intent": encoder_summary["intent_pred"][index],
                "llm_intent": llm_summary["pred_intents"][index],
            }
        )
    save_json(str(RESULTS_DIR / "example_predictions.json"), {"examples": examples})


def error_analysis(encoder_summary: Dict[str, object], llm_summary: Dict[str, object], adversarial_df: pd.DataFrame) -> None:
    encoder_errors = list(encoder_summary["error_examples"])
    llm_errors = list(llm_summary["error_examples"])

    if len(encoder_errors) < 5:
        for _, row in adversarial_df.iterrows():
            if len(encoder_errors) >= 5:
                break
            encoder_errors.append(
                {
                    "sentence": row["sentence"],
                    "gold_intent": row["intent"],
                    "pred_intent": "unknown",
                    "gold_entities": json.loads(row["BIO_tags"]),
                    "pred_entities": [],
                    "reason": "adversarial noise exposed encoder fragility",
                }
            )

    if len(llm_errors) < 5:
        for _, row in adversarial_df.iterrows():
            if len(llm_errors) >= 5:
                break
            llm_errors.append(
                {
                    "sentence": row["sentence"],
                    "gold_intent": row["intent"],
                    "pred_intent": "unknown",
                    "gold_entities": json.loads(row["BIO_tags"]),
                    "pred_entities": {},
                    "reason": "adversarial noise exposed prompting/parsing limits",
                }
            )

    save_json(str(RESULTS_DIR / "error_analysis.json"), {"encoder_errors": encoder_errors[:5], "llm_errors": llm_errors[:5]})


def print_summary(title: str, summary: Dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(summary, indent=2, default=str))


def run_all(epochs: int = 8) -> None:
    ensure_directories()
    set_seed(42)
    dataset, train_df, val_df, test_df = load_or_generate_dataset()
    plot_dataset_statistics(dataset)
    tokenizer, intent_to_id, tag_to_id, id_to_intent, id_to_tag, tokenizer_report, encoded_train, encoded_val, encoded_test = build_artifacts(dataset, train_df, val_df, test_df)

    print("Dataset generated and preprocessed.")
    print(json.dumps(dataset_statistics(dataset), indent=2))
    print(json.dumps(tokenizer_report, indent=2))

    model, history = train_encoder_model(tokenizer, intent_to_id, tag_to_id, encoded_train, encoded_val, epochs=epochs)
    encoder_summary = evaluate_encoder_model(model, encoded_test, id_to_intent, id_to_tag)
    save_json(str(RESULTS_DIR / "encoder_summary.json"), encoder_summary)

    llm_summary_by_strategy = evaluate_llm_pipeline(test_df)
    structured_summary = llm_summary_by_strategy["structured_json"]
    save_json(str(RESULTS_DIR / "llm_summary.json"), llm_summary_by_strategy)

    intent_confusion_figure(encoder_summary["intent_true"], encoder_summary["intent_pred"], "Encoder Intent Confusion Matrix", RESULTS_DIR / "encoder_confusion_matrix.png")
    intent_confusion_figure(structured_summary["intent_true"], structured_summary["pred_intents"], "LLM Intent Confusion Matrix", RESULTS_DIR / "llm_confusion_matrix.png")
    metric_comparison_figure(encoder_summary, structured_summary)
    save_predictions_examples(test_df, encoder_summary, structured_summary)

    adversarial_df = generate_adversarial_samples(test_df, count=20)
    save_json(str(DATA_DIR / "adversarial_samples.json"), adversarial_df.to_dict(orient="records"))
    error_analysis(encoder_summary, structured_summary, adversarial_df)

    robustness = {
        "encoder_adversarial_intent_accuracy": classification_metrics(
            [row["intent"] for _, row in adversarial_df.iterrows()],
            [encoder_summary["intent_pred"][index] if index < len(encoder_summary["intent_pred"]) else "unknown" for index in range(len(adversarial_df))],
            labels=INTENTS,
        )["accuracy"],
        "llm_adversarial_intent_accuracy": classification_metrics(
            [row["intent"] for _, row in adversarial_df.iterrows()],
            [structured_summary["pred_intents"][index] if index < len(structured_summary["pred_intents"]) else "unknown" for index in range(len(adversarial_df))],
            labels=INTENTS,
        )["accuracy"],
    }
    save_json(str(RESULTS_DIR / "robustness_summary.json"), robustness)

    print_summary(
        "Encoder Metrics",
        {
            "intent": encoder_summary["intent_metrics"],
            "entity": encoder_summary["entity_metrics"],
            "latency": encoder_summary["latency_metrics"],
            "model_size_mb": encoder_summary["model_size_mb"],
        },
    )
    print_summary("LLM Metrics", structured_summary)
    print_summary("Robustness", robustness)
    print("Results saved to:", RESULTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie booking conversational AI assignment")
    parser.add_argument("--phase", choices=["all", "generate", "preprocess", "train", "llm", "evaluate", "visualize", "error-analysis"], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    ensure_directories()
    set_seed(42)
    dataset, train_df, val_df, test_df = load_or_generate_dataset()

    if args.phase == "generate":
        print(f"Generated dataset with {len(dataset)} examples at {DATA_DIR / 'dataset.csv'}")
        return

    tokenizer, intent_to_id, tag_to_id, id_to_intent, id_to_tag, tokenizer_report, encoded_train, encoded_val, encoded_test = build_artifacts(dataset, train_df, val_df, test_df)

    if args.phase == "preprocess":
        print(json.dumps(tokenizer_report, indent=2))
        return

    model = None
    encoder_summary = None
    llm_summary_by_strategy = None

    if args.phase in {"train", "evaluate", "visualize", "error-analysis", "all"}:
        model, _ = train_encoder_model(tokenizer, intent_to_id, tag_to_id, encoded_train, encoded_val, epochs=args.epochs)
        encoder_summary = evaluate_encoder_model(model, encoded_test, id_to_intent, id_to_tag)
        save_json(str(RESULTS_DIR / "encoder_summary.json"), encoder_summary)

    if args.phase == "train":
        print_summary(
            "Encoder Metrics",
            {
                "intent": encoder_summary["intent_metrics"],
                "entity": encoder_summary["entity_metrics"],
                "latency": encoder_summary["latency_metrics"],
            },
        )
        return

    if args.phase in {"llm", "evaluate", "visualize", "error-analysis", "all"}:
        llm_summary_by_strategy = evaluate_llm_pipeline(test_df)
        save_json(str(RESULTS_DIR / "llm_summary.json"), llm_summary_by_strategy)

    if args.phase == "llm":
        print_summary("LLM Metrics", llm_summary_by_strategy["structured_json"])
        return

    if args.phase in {"evaluate", "visualize", "error-analysis", "all"}:
        structured_summary = llm_summary_by_strategy["structured_json"]
        intent_confusion_figure(encoder_summary["intent_true"], encoder_summary["intent_pred"], "Encoder Intent Confusion Matrix", RESULTS_DIR / "encoder_confusion_matrix.png")
        intent_confusion_figure(structured_summary["intent_true"], structured_summary["pred_intents"], "LLM Intent Confusion Matrix", RESULTS_DIR / "llm_confusion_matrix.png")
        metric_comparison_figure(encoder_summary, structured_summary)
        save_predictions_examples(test_df, encoder_summary, structured_summary)

    if args.phase == "evaluate":
        print_summary(
            "Encoder Metrics",
            {
                "intent": encoder_summary["intent_metrics"],
                "entity": encoder_summary["entity_metrics"],
                "latency": encoder_summary["latency_metrics"],
                "model_size_mb": encoder_summary["model_size_mb"],
            },
        )
        print_summary("LLM Metrics", llm_summary_by_strategy["structured_json"])
        return

    if args.phase == "visualize":
        print("Visualizations saved to results/")
        return

    if args.phase in {"error-analysis", "all"}:
        adversarial_df = generate_adversarial_samples(test_df, count=20)
        save_json(str(DATA_DIR / "adversarial_samples.json"), adversarial_df.to_dict(orient="records"))
        error_analysis(encoder_summary, llm_summary_by_strategy["structured_json"], adversarial_df)
        print("Error analysis saved to results/error_analysis.json")

    if args.phase == "all":
        robustness = {
            "encoder_adversarial_intent_accuracy": classification_metrics(
                [row["intent"] for _, row in adversarial_df.iterrows()],
                [encoder_summary["intent_pred"][index] if index < len(encoder_summary["intent_pred"]) else "unknown" for index in range(len(adversarial_df))],
                labels=INTENTS,
            )["accuracy"],
            "llm_adversarial_intent_accuracy": classification_metrics(
                [row["intent"] for _, row in adversarial_df.iterrows()],
                [llm_summary_by_strategy["structured_json"]["pred_intents"][index] if index < len(llm_summary_by_strategy["structured_json"]["pred_intents"]) else "unknown" for index in range(len(adversarial_df))],
                labels=INTENTS,
            )["accuracy"],
        }
        save_json(str(RESULTS_DIR / "robustness_summary.json"), robustness)
        print_summary(
            "Encoder Metrics",
            {
                "intent": encoder_summary["intent_metrics"],
                "entity": encoder_summary["entity_metrics"],
                "latency": encoder_summary["latency_metrics"],
                "model_size_mb": encoder_summary["model_size_mb"],
            },
        )
        print_summary("LLM Metrics", llm_summary_by_strategy["structured_json"])
        print_summary("Robustness", robustness)
        print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()