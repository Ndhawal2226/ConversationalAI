# Movie Booking Conversational AI — Repository Flow

This document explains the repository structure and how data moves through the project.

## Repository Map

```mermaid
flowchart TD
    Root[ConversationalAI Repository] --> Main[main.py]
    Root --> Notebook[main.ipynb]
    Root --> Data[data/]
    Root --> Models[models/]
    Root --> Utils[utils/]
    Root --> LLM[llm/]
    Root --> Results[results/]
    Root --> Artifacts[artifacts/]
    Root --> Pdf[Assignment_PS1_NLU.pdf]

    Data --> Dataset[dataset.csv]
    Data --> Splits[train.csv / val.csv / test.csv]
    Data --> Tokenizer[tokenizer.json]
    Data --> Labels[label_maps.json]
    Data --> TokenizerReport[tokenizer_report.json]

    Models --> Encoder[best_encoder.pt]
    Results --> Eval[encoder_summary.json]
    Results --> LLMEval[llm_summary.json]
    Results --> Charts[confusion matrices / comparison plots]
    Results --> Analysis[error_analysis.json / example_predictions.json]
```

## Project Flow

```mermaid
flowchart LR
    A[Load or generate synthetic movie-booking data] --> B[Split into train / val / test]
    B --> C[Train custom BPE tokenizer]
    C --> D[Build label maps for intents and BIO tags]
    D --> E[Encode examples with subword alignment]
    E --> F[Train custom Transformer encoder]
    F --> G[Evaluate encoder on test set]
    E --> H[Simulated LLM prompting pipeline]
    H --> I[Evaluate LLM predictions]
    G --> J[Save metrics, plots, and analysis]
    I --> J
    J --> K[Store artifacts in data/, models/, and results/]
```

## Runtime Roles

```mermaid
flowchart TD
    MainPy[main.py] --> Prep[Dataset generation + preprocessing]
    MainPy --> Train[Encoder training]
    MainPy --> Eval[Encoder / LLM evaluation]
    MainPy --> Viz[Visualization + error analysis]
    MainPy --> Save[Artifact export]

    Notebook[main.ipynb] --> Prep
    Notebook --> Train
    Notebook --> Eval
    Notebook --> Viz
    Notebook --> Save
```

## What Each Folder Does

- `data/`: raw/generated dataset, splits, tokenizer, and label metadata.
- `models/`: saved encoder checkpoint.
- `utils/`: preprocessing, tokenization, and metrics helpers.
- `models/`: custom Transformer encoder implementation.
- `llm/`: prompt-building and simulated LLM evaluation.
- `results/`: metrics, charts, error analysis, and comparison outputs.
- `artifacts/`: extra exported assets and scratch outputs.

## End-to-End Summary

1. Generate or load the movie-booking dataset.
2. Split data and train the tokenizer.
3. Convert intents and BIO tags into model-ready IDs.
4. Train the custom encoder on the labeled data.
5. Run the simulated LLM prompting baseline.
6. Compare both systems with metrics and plots.
7. Save outputs for inspection in `data/`, `models/`, and `results/`.
