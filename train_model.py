"""Train a unified BioBERT medical NER model on d4data/biomedical-ner-all.

The goal of this script is to fine-tune BioBERT for a broader clinical entity
extraction task that can learn multiple medical categories at once, including
Diseases, Symptoms, Medications, and Anatomy.

Key design goals:
- use the public Hugging Face dataset `d4data/biomedical-ner-all`,
- use `dmis-lab/biobert-v1.1` as the backbone transformer,
- align labels safely from words to subword pieces,
- ignore punctuation and special tokens during training so indexing stays safe,
- report clean validation Precision / Recall / F1 / Accuracy via `seqeval`,
- and save the final model to `./smartmed_biobert`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "dmis-lab/biobert-v1.1"
DATASET_NAME = "d4data/biomedical-ner-all"
OUTPUT_DIR = "./smartmed_biobert"


# Matches tokens that are only punctuation or underscores. These are masked out
# during label alignment so they do not become training targets.
PUNCTUATION_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


def is_punctuation(token: str) -> bool:
    """Return True if a token is empty or punctuation-like."""

    if token is None:
        return True

    cleaned_token = str(token).strip()
    return cleaned_token == "" or bool(PUNCTUATION_PATTERN.match(cleaned_token))


def load_and_prepare_dataset() -> Tuple[DatasetDict, List[str]]:
    """Load the biomedical NER dataset and ensure a validation split exists."""

    dataset = load_dataset(DATASET_NAME)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(train=split["train"], validation=split["test"])

    label_feature = dataset["train"].features["ner_tags"].feature
    if not isinstance(label_feature, ClassLabel):
        raise TypeError(
            "Expected dataset['train'].features['ner_tags'].feature to be a ClassLabel."
        )

    return dataset, list(label_feature.names)


def build_label_maps(label_list: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Build stable label lookup maps for the classifier head and metrics."""

    id2label = {index: label for index, label in enumerate(label_list)}
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize word sequences and align word labels to subword tokens safely.

    Safety rules:
    - special tokens receive -100 so they are ignored by the loss,
    - punctuation tokens are masked with -100,
    - only the first subword of each word receives the original label,
    - repeated subwords get -100 so the loss ignores them,
    - all indexing is bounds-checked to avoid out-of-range errors.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    aligned_labels = []

    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []
        tokens = examples["tokens"][batch_index]

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                continue

            if word_id < 0 or word_id >= len(word_labels) or word_id >= len(tokens):
                label_ids.append(-100)
                continue

            token_text = str(tokens[word_id])
            if is_punctuation(token_text):
                label_ids.append(-100)
                previous_word_id = word_id
                continue

            if word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def build_compute_metrics(id2label: Dict[int, str]) -> Callable:
    """Create a seqeval metric function for the Hugging Face Trainer."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction_row, label_row in zip(predictions, labels):
            row_predictions = []
            row_labels = []

            for prediction_id, label_id in zip(prediction_row, label_row):
                if label_id == -100:
                    continue

                row_predictions.append(id2label.get(int(prediction_id), "O"))
                row_labels.append(id2label.get(int(label_id), "O"))

            true_predictions.append(row_predictions)
            true_labels.append(row_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def print_validation_metrics(metrics: Dict[str, float]) -> None:
    """Print a concise validation summary after training finishes."""

    print("\nValidation Metrics")
    print("-" * 60)
    print(f"Precision: {metrics.get('eval_precision', 0.0):.4f}")
    print(f"Recall:    {metrics.get('eval_recall', 0.0):.4f}")
    print(f"F1-score:  {metrics.get('eval_f1', 0.0):.4f}")
    print(f"Accuracy:  {metrics.get('eval_accuracy', 0.0):.4f}")
    print("-" * 60)


def main() -> None:
    """Run the full BioBERT fine-tuning workflow."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset, label_list = load_and_prepare_dataset()
    id2label, label2id = build_label_maps(label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and aligning labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
        seed=42,
        optim="adamw_torch",
        auto_find_batch_size=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )

    trainer.train()

    validation_metrics = trainer.evaluate()
    print_validation_metrics(validation_metrics)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete. Model and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()