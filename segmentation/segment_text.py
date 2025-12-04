"""Inference helpers for running the trained segmenter."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from .chunking import Chunk, chunk_text, tokenize_words
from .config import DEFAULT_CHUNKING, DEFAULT_MODEL, ChunkingConfig, ModelConfig
from .soft_break import SoftBreakDetector
from .model import BoundarySegmenter


def load_segmenter(
    checkpoint_path: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[BoundarySegmenter, AutoTokenizer, ChunkingConfig, ModelConfig, Optional[SoftBreakDetector]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model_cfg_dict = checkpoint.get("model_config", DEFAULT_MODEL.__dict__)
    chunk_cfg_dict = checkpoint.get("chunk_config", DEFAULT_CHUNKING.__dict__)
    model_cfg = ModelConfig(**model_cfg_dict)
    chunk_cfg = ChunkingConfig(**chunk_cfg_dict)
    soft_break_state = checkpoint.get("soft_break_state")
    soft_break_detector = None
    if soft_break_state:
        soft_break_detector = SoftBreakDetector.from_state_dict(soft_break_state)

    tokenizer_name = checkpoint.get("tokenizer_name", model_cfg.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = BoundarySegmenter(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, chunk_cfg, model_cfg, soft_break_detector


def _prepare_batch(
    chunks: Sequence[Chunk],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    texts = [chunk.text for chunk in chunks]
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].unsqueeze(0)
    attention_mask = encoding["attention_mask"].unsqueeze(0)
    chunk_mask = torch.ones(1, len(chunks), dtype=torch.bool)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "chunk_mask": chunk_mask.to(device),
    }


def _pick_boundaries(
    scores: Sequence[float],
    *,
    k_known: Optional[int] = None,
    percentile: float = 0.9,
    min_gap: int = 2,
) -> List[int]:
    if not scores:
        return []
    if k_known is not None:
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(k_known - 1, 0)]
        return sorted(top)

    tensor_scores = torch.tensor(scores)
    if len(scores) == 1:
        threshold = tensor_scores.item()
    else:
        quantile = min(max(percentile, 0.0), 0.999)
        threshold = torch.quantile(tensor_scores, quantile).item()

    candidate = [idx for idx, score in enumerate(scores) if score >= threshold]
    candidate.sort()

    selected: List[int] = []
    for idx in candidate:
        if selected and idx - selected[-1] < min_gap:
            if scores[idx] > scores[selected[-1]]:
                selected[-1] = idx
            continue
        selected.append(idx)
    return selected


def segment_text(
    text: str,
    model: BoundarySegmenter,
    tokenizer: AutoTokenizer,
    *,
    model_cfg: ModelConfig = DEFAULT_MODEL,
    chunk_cfg: ChunkingConfig = DEFAULT_CHUNKING,
    device: str | torch.device = "cpu",
    k_known: Optional[int] = None,
    percentile: float = 0.9,
    min_gap_chunks: int = 2,
    soft_break_detector: Optional[SoftBreakDetector] = None,
) -> Dict[str, object]:
    chunks = chunk_text(text, chunk_cfg=chunk_cfg, soft_break_detector=soft_break_detector)
    if not chunks:
        return {
            "token_boundaries": [],
            "chunk_boundaries": [],
            "scores": [],
            "segments": [text],
        }

    batch = _prepare_batch(chunks, tokenizer, model_cfg.max_seq_len, torch.device(device))
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits[0][: len(chunks)]
    scores = torch.sigmoid(logits).cpu().tolist()

    boundary_idxs = _pick_boundaries(scores, k_known=k_known, percentile=percentile, min_gap=min_gap_chunks)
    word_tokens = tokenize_words(text)
    token_boundaries = [chunks[idx].end_token for idx in boundary_idxs if chunks[idx].end_token < len(word_tokens)]

    segments: List[str] = []
    prev = 0
    for boundary in token_boundaries + [len(word_tokens)]:
        segment_words = word_tokens[prev:boundary]
        segments.append(" ".join(segment_words))
        prev = boundary

    return {
        "token_boundaries": token_boundaries,
        "chunk_boundaries": boundary_idxs,
        "scores": scores,
        "segments": segments,
    }
