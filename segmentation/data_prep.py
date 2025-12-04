"""Dataset preparation utilities for the boundary segmenter."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .chunking import chunk_text, chunks_to_texts
from .config import DEFAULT_CHUNKING, DEFAULT_TRAINING, ChunkingConfig
from .soft_break import SoftBreakDetector


def load_texts(dataset_path: Path | str) -> List[str]:
    path = Path(dataset_path)
    raw = json.loads(path.read_text())
    return [
        item["text"]
        for item in raw
        if item.get("text") and item.get("type") != "mixed"
    ]


def build_synthetic_sequences(
    texts: Sequence[str],
    *,
    num_samples: int = DEFAULT_TRAINING.num_synthetic_sequences,
    min_articles: int = DEFAULT_TRAINING.min_articles,
    max_articles: int = DEFAULT_TRAINING.max_articles,
    chunk_cfg: ChunkingConfig = DEFAULT_CHUNKING,
    seed: int = DEFAULT_TRAINING.seed,
    max_chunks_per_sequence: int | None = DEFAULT_TRAINING.max_chunks_per_sequence,
    soft_break_detector: Optional[SoftBreakDetector] = None,
) -> List[Dict[str, List[str]]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, List[str]]] = []

    if not texts:
        return reservoir

    total_texts = len(texts)

    attempts = 0
    max_attempts = num_samples * 10
    while len(reservoir) < num_samples and attempts < max_attempts:
        attempts += 1
        k = rng.randint(min_articles, max_articles)
        indices = rng.sample(range(total_texts), k=k)
        chunks: List[str] = []
        labels: List[int] = []

        for pos, idx in enumerate(indices):
            chunk_objs = chunk_text(texts[idx], chunk_cfg=chunk_cfg, soft_break_detector=soft_break_detector)
            if not chunk_objs:
                continue
            chunk_texts = chunks_to_texts(chunk_objs)
            if not chunk_texts:
                continue
            chunks.extend(chunk_texts)
            is_last_article = pos == len(indices) - 1
            local_labels = [0] * (len(chunk_texts) - 1)
            last_label = 0 if is_last_article else 1
            local_labels.append(last_label)
            labels.extend(local_labels)

        if len(chunks) < 2 or len(chunks) != len(labels) or sum(labels) == 0:
            continue

        sequences = _split_sequences(chunks, labels, max_chunks_per_sequence)
        for sequence in sequences:
            reservoir.append(sequence)
            if len(reservoir) >= num_samples:
                break

    if len(reservoir) < num_samples:
        raise RuntimeError(
            f"Only built {len(reservoir)} samples out of requested {num_samples}. "
            "Reduce the number of synthetic samples or provide more texts."
        )

    return reservoir


def _split_sequences(
    chunks: List[str],
    labels: List[int],
    max_chunks: int | None,
) -> List[Dict[str, List[str]]]:
    if not max_chunks or len(chunks) <= max_chunks:
        return [{"chunks": chunks, "labels": labels}]

    slices: List[Dict[str, List[str]]] = []
    start = 0
    while start < len(chunks):
        end = min(start + max_chunks, len(chunks))
        new_chunks = list(chunks[start:end])
        new_labels = list(labels[start:end])
        if end < len(chunks):
            new_labels[-1] = 0  # boundary cannot cross artificial split
        slices.append({"chunks": new_chunks, "labels": new_labels})
        start = end
    return slices


def split_dataset(
    examples: Sequence[Dict[str, List[str]]],
    val_split: float = DEFAULT_TRAINING.val_split,
    test_split: float = DEFAULT_TRAINING.test_split,
    seed: int = DEFAULT_TRAINING.seed,
) -> Tuple[List[Dict[str, List[str]]], List[Dict[str, List[str]]], List[Dict[str, List[str]]]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    total = len(examples)
    val_size = int(total * val_split)
    test_size = int(total * test_split)

    val_idx = indices[:val_size]
    test_idx = indices[val_size : val_size + test_size]
    train_idx = indices[val_size + test_size :]

    def select(idxs: Sequence[int]) -> List[Dict[str, List[str]]]:
        return [examples[i] for i in idxs]

    return select(train_idx), select(val_idx), select(test_idx)
