from __future__ import annotations

import inspect
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils import normalize_text


class NewsSegmenter:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self._accepts_token_type_ids = self._check_token_type_ids()

    def _check_token_type_ids(self) -> bool:
        try:
            return "token_type_ids" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _predict_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        batch_size = 32
        all_probs: list[float] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            lefts = [normalize_text(p[0]) for p in batch]
            rights = [normalize_text(p[1]) for p in batch]

            inputs: Any = self.tokenizer(
                lefts,
                rights,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            if not self._accepts_token_type_ids and "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())

        return all_probs

    def segment(self, text: str) -> list[str]:
        words = text.split()

        min_len = 10
        confirm_thr = 0.8

        scan_indices = list(range(min_len, len(words) - min_len))
        if not scan_indices:
            return [text]

        candidates: list[tuple[str, str]] = []
        for i in scan_indices:
            ctx_left = " ".join(words[max(0, i - 50) : i])
            ctx_right = " ".join(words[i : min(len(words), i + 50)])
            candidates.append((ctx_left, ctx_right))

        probs = self._predict_batch(candidates)

        split_indices = [0]
        i = 0
        while i < len(probs):
            prob = probs[i]
            idx = scan_indices[i]

            is_peak = True
            if i > 0 and probs[i - 1] >= prob:
                is_peak = False
            if i < len(probs) - 1 and probs[i + 1] > prob:
                is_peak = False

            if is_peak and prob > confirm_thr:
                if idx - split_indices[-1] >= min_len:
                    split_indices.append(idx)
                    while i < len(scan_indices) and scan_indices[i] < idx + min_len:
                        i += 1
                    continue
            i += 1

        split_indices.append(len(words))

        segments: list[str] = []
        for k in range(len(split_indices) - 1):
            segments.append(" ".join(words[split_indices[k] : split_indices[k + 1]]))

        return segments

