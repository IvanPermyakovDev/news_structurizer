"""Configuration helpers for the boundary segmenter pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from .soft_break import SoftBreakConfig


def _default_soft_breaks() -> Tuple[str, ...]:
    return (
        "далее",
        "итак",
        "эфир",
        "сейчас",
        "таким",
        "таким образом",
        "кстати",
        "между тем",
        "вообще",
        "короче",
        "впрочем",
        "значит",
        "посмотрим",
    )


@dataclass(frozen=True)
class ChunkingConfig:
    min_words: int = 25
    max_words: int = 40
    soft_breaks: Sequence[str] = field(default_factory=_default_soft_breaks)
    use_adaptive_boundaries: bool = True
    embedding_dim: int = 64
    break_threshold: float = 0.45
    divergence_weight: float = 0.6
    variance_weight: float = 0.2
    soft_break_weight: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    pretrained_model_name: str = "cointegrated/rubert-tiny2"
    max_seq_len: int = 256
    context_layers: int = 2
    context_heads: int = 6
    context_dropout: float = 0.1
    chunk_batch_size: int = 32
    pooling: str = "cls"  # or "mean"
    use_positional: bool = True
    positional_max_positions: int = 512
    classifier_dropout: float = 0.1
    use_crf: bool = False
    use_contrastive: bool = False
    contrastive_weight: float = 0.1
    triplet_margin: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1
    lr: float = 2e-5
    epochs: int = 5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_clip: float = 1.0
    num_synthetic_sequences: int = 1500
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 13
    max_chunks_per_sequence: int = 128
    min_articles: int = 1
    max_articles: int = 10


DEFAULT_CHUNKING = ChunkingConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_TRAINING = TrainingConfig()
DEFAULT_SOFT_BREAK = SoftBreakConfig()
