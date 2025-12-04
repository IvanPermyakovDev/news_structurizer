"""Utilities for splitting ASR-style news streams into adaptive chunks."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_CHUNKING, ChunkingConfig

if TYPE_CHECKING:  # pragma: no cover
    from .soft_break import SoftBreakDetector

_WORD_SPLIT = re.compile(r"\s+")


@dataclass
class Chunk:
    text: str
    start_token: int
    end_token: int


def _prepare_words(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [w for w in _WORD_SPLIT.split(normalized) if w]


def chunk_text(
    text: str,
    chunk_cfg: ChunkingConfig = DEFAULT_CHUNKING,
    soft_break_detector: Optional["SoftBreakDetector"] = None,
) -> List[Chunk]:
    """Split text into pseudo-sentences of adaptive word length."""
    words = _prepare_words(text)
    if not words:
        return []

    chunks: List[Chunk] = []
    idx = 0
    total = len(words)

    embed_helpers = None
    prev_embed: Optional[np.ndarray] = None
    if chunk_cfg.use_adaptive_boundaries:
        embed_helpers = _EmbeddingHelper(words, chunk_cfg.embedding_dim)

    while idx < total:
        remaining = total - idx
        if remaining <= chunk_cfg.max_words:
            cut = total
        else:
            cut = min(total, idx + chunk_cfg.max_words)
            adaptive_cut = None
            min_cut = min(total, idx + chunk_cfg.min_words)
            if chunk_cfg.use_adaptive_boundaries and embed_helpers is not None:
                for pos in range(min_cut, cut):
                    mean_vec, variance = embed_helpers.span_stats(idx, pos)
                    divergence = _divergence(prev_embed, mean_vec)
                    soft_prob = 0.0
                    if soft_break_detector is not None:
                        window_size = soft_break_detector.config.window_size
                        start = max(idx, pos - window_size)
                        window = words[start:pos]
                        if len(window) == window_size:
                            soft_prob = soft_break_detector.score(window)
                    score = (
                        chunk_cfg.divergence_weight * divergence
                        + chunk_cfg.variance_weight * variance
                        + chunk_cfg.soft_break_weight * soft_prob
                    )
                    if score >= chunk_cfg.break_threshold:
                        adaptive_cut = pos
                        break
                if adaptive_cut is not None and adaptive_cut > idx:
                    cut = adaptive_cut
            else:
                cut = min(total, idx + chunk_cfg.max_words)
        chunk_words = words[idx:cut]
        chunks.append(Chunk(" ".join(chunk_words), idx, cut))
        if chunk_cfg.use_adaptive_boundaries and embed_helpers is not None:
            prev_embed, _ = embed_helpers.span_stats(idx, cut)
        idx = cut

    return chunks


def chunks_to_texts(chunks: Sequence[Chunk]) -> List[str]:
    return [chunk.text for chunk in chunks]


def tokenize_words(text: str) -> List[str]:
    return _prepare_words(text)


class _EmbeddingHelper:
    def __init__(self, words: Sequence[str], dim: int) -> None:
        self.dim = dim
        vectors = np.stack([_token_vector(word, dim) for word in words])
        self.cumsum = np.cumsum(vectors, axis=0)
        self.cumsum_sq = np.cumsum(vectors ** 2, axis=0)

    def span_stats(self, start: int, end: int) -> Tuple[np.ndarray, float]:
        if end <= start:
            raise ValueError("Invalid span")
        length = end - start
        sum_vec = self.cumsum[end - 1] - (self.cumsum[start - 1] if start > 0 else 0)
        mean_vec = sum_vec / length
        sum_sq = self.cumsum_sq[end - 1] - (self.cumsum_sq[start - 1] if start > 0 else 0)
        variance_vec = np.maximum(sum_sq / length - mean_vec ** 2, 0)
        variance = float(np.mean(variance_vec))
        return mean_vec, variance


@lru_cache(maxsize=200000)
def _token_vector(token: str, dim: int) -> np.ndarray:
    digest = hashlib.sha1(f"{token}:{dim}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim, dtype=np.float32)


def _divergence(prev: Optional[np.ndarray], current: np.ndarray) -> float:
    if prev is None:
        return 0.5
    prev_norm = float(np.linalg.norm(prev))
    curr_norm = float(np.linalg.norm(current))
    if prev_norm == 0 or curr_norm == 0:
        return 0.5
    cosine = float(np.dot(prev, current) / (prev_norm * curr_norm))
    cosine = max(min(cosine, 1.0), -1.0)
    return 0.5 * (1 - cosine)
