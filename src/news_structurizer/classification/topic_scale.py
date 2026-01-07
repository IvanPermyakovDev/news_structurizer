from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NewsClassifier:
    def __init__(
        self,
        topic_model_path: str,
        scale_model_path: str,
        device: str | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device

        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_path)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_path)
        self.topic_model.to(self.device)
        self.topic_model.eval()

        self.scale_tokenizer = AutoTokenizer.from_pretrained(scale_model_path)
        self.scale_model = AutoModelForSequenceClassification.from_pretrained(scale_model_path)
        self.scale_model.to(self.device)
        self.scale_model.eval()

    def _predict(
        self,
        text: str,
        tokenizer: Any,
        model: Any,
        max_len: int = 256,
    ) -> dict[str, Any]:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())

        label = model.config.id2label[pred_idx]
        return {"label": label, "confidence": float(probs[pred_idx])}

    def classify(self, text: str) -> dict[str, Any]:
        topic_result = self._predict(text, self.topic_tokenizer, self.topic_model)
        scale_result = self._predict(text, self.scale_tokenizer, self.scale_model)
        return {
            "topic": topic_result["label"],
            "topic_confidence": topic_result["confidence"],
            "scale": scale_result["label"],
            "scale_confidence": scale_result["confidence"],
        }

