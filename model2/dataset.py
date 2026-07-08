"""
dataset.py
----------
Load prescription NER annotations, build BIO label maps, and provide
PyTorch Dataset / DataLoader helpers for training and evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer, PreTrainedTokenizerBase

from config import Config
from utils import get_logger, load_json, save_json, tokenize_words

logger = get_logger("dataset")


@dataclass
class NERSplit:
    examples: List[Dict]
    label_list: List[str]
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def build_label_maps(label_list: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_list = label_list or Config.build_label_list()
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def save_label_maps(label_list: List[str], path: str = Config.LABEL_MAP_PATH) -> None:
    save_json({"label_list": label_list}, path)


def load_label_maps(path: str = Config.LABEL_MAP_PATH) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if not os.path.exists(path):
        label_list, label2id, id2label = build_label_maps()
        save_label_maps(label_list, path)
        return label_list, label2id, id2label

    payload = load_json(path)
    label_list = payload["label_list"]
    return build_label_maps(label_list)


def _char_entities_to_bio(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    words = tokenize_words(text)
    if not words:
        return [], []

    # Reconstruct character offsets for each word token.
    search_start = 0
    word_spans: List[Tuple[int, int]] = []
    for word in words:
        idx = text.find(word, search_start)
        if idx < 0:
            idx = search_start
        word_spans.append((idx, idx + len(word)))
        search_start = idx + len(word)

    labels = ["O"] * len(words)
    sorted_entities = sorted(entities, key=lambda item: (item["start"], item["end"]))

    for entity in sorted_entities:
        entity_label = entity["label"].upper()
        if entity_label not in Config.ENTITY_TYPES:
            continue

        covered = [
            i
            for i, (start, end) in enumerate(word_spans)
            if start < entity["end"] and end > entity["start"]
        ]
        if not covered:
            continue

        labels[covered[0]] = f"B-{entity_label}"
        for idx in covered[1:]:
            labels[idx] = f"I-{entity_label}"

    return words, labels


def normalize_example(raw: Dict) -> Dict:
    """Accept either token-level or character-span annotations."""
    if "tokens" in raw and "labels" in raw:
        tokens = [str(token) for token in raw["tokens"]]
        labels = [str(label) for label in raw["labels"]]
        if len(tokens) != len(labels):
            raise ValueError("tokens and labels must have the same length")
        return {"tokens": tokens, "labels": labels, "text": raw.get("text", " ".join(tokens))}

    if "text" in raw and "entities" in raw:
        words, labels = _char_entities_to_bio(raw["text"], raw["entities"])
        return {"tokens": words, "labels": labels, "text": raw["text"]}

    raise ValueError("Each example must provide either (tokens, labels) or (text, entities)")


def load_ner_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NER dataset not found: {path}")

    raw_examples = load_json(path)
    if not isinstance(raw_examples, list):
        raise ValueError(f"Expected a JSON list in {path}")

    examples = [normalize_example(item) for item in raw_examples]
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def load_ner_data(
    train_json: str = Config.TRAIN_JSON,
    val_json: str = Config.VAL_JSON,
    test_json: str = Config.TEST_JSON,
) -> Tuple[NERSplit, NERSplit, NERSplit]:
    label_list, label2id, id2label = load_label_maps()

    train = NERSplit(load_ner_json(train_json), label_list, label2id, id2label)
    val = NERSplit(load_ner_json(val_json), label_list, label2id, id2label)
    test = NERSplit(load_ner_json(test_json), label_list, label2id, id2label)
    return train, val, test


class PrescriptionNERDataset(Dataset):
    """Token-classification dataset aligned to word boundaries."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_length: int = Config.MAX_SEQ_LENGTH,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.pad_label_id = label2id["O"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        words = example["tokens"]
        labels = example["labels"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels: List[int] = []
        previous_word_id: Optional[int] = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                tag = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(self.label2id.get(tag, self.label2id["O"]))
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        item["example_index"] = torch.tensor(index, dtype=torch.long)
        return item


def build_dataloader(
    split: NERSplit,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PrescriptionNERDataset(
        examples=split.examples,
        tokenizer=tokenizer,
        label2id=split.label2id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )


def get_tokenizer(model_name: str = Config.PRETRAINED_MODEL_NAME) -> PreTrainedTokenizerBase:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ValueError:
        logger.warning(
            "Fast tokenizer unavailable for %s; retrying with use_fast=False.",
            model_name,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except ValueError:
            logger.warning(
                "AutoTokenizer failed for %s; using BertTokenizer (transformers v5 fallback).",
                model_name,
            )
            tokenizer = BertTokenizer.from_pretrained(model_name)
    os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(Config.TOKENIZER_DIR)
    return tokenizer
