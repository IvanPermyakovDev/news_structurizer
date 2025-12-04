"""Adaptive soft break detector for chunking."""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _tokenize(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    tokens = [w for w in normalized.split(" ") if w]
    return tokens


@dataclass
class SoftBreakConfig:
    window_size: int = 3
    epochs: int = 3
    lr: float = 0.2
    threshold: float = 0.6
    max_positive: int = 20000
    max_negative: int = 40000
    seed: int = 17


class SoftBreakDetector:
    def __init__(self, config: SoftBreakConfig | None = None):
        self.config = config or SoftBreakConfig()
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0

    def _generate_samples(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
    ) -> List[Tuple[List[str], int]]:
        rng = random.Random(self.config.seed)
        phrase_tokens = [tuple(p.lower().split()) for p in phrases]
        positives: List[Tuple[List[str], int]] = []
        negatives: List[Tuple[List[str], int]] = []

        for text in texts:
            tokens = _tokenize(text.lower())
            if not tokens:
                continue
            t_len = len(tokens)
            for idx in range(t_len):
                window = tokens[idx : idx + self.config.window_size]
                if len(window) < self.config.window_size:
                    break
                label = 0
                for phrase in phrase_tokens:
                    phrase_len = len(phrase)
                    if idx + phrase_len <= t_len and tuple(tokens[idx : idx + phrase_len]) == phrase:
                        label = 1
                        break
                if label:
                    positives.append((window, 1))
                else:
                    negatives.append((window, 0))
                if len(positives) >= self.config.max_positive and len(negatives) >= self.config.max_negative:
                    break
            if len(positives) >= self.config.max_positive and len(negatives) >= self.config.max_negative:
                break

        if len(negatives) > self.config.max_negative:
            negatives = rng.sample(negatives, self.config.max_negative)
        if len(positives) > self.config.max_positive:
            positives = rng.sample(positives, self.config.max_positive)

        samples = positives + negatives
        rng.shuffle(samples)
        return samples

    def train(self, texts: Sequence[str], phrases: Sequence[str]) -> None:
        samples = self._generate_samples(texts, phrases)
        if not samples:
            return
        rng = random.Random(self.config.seed)
        lr = self.config.lr

        for _ in range(self.config.epochs):
            rng.shuffle(samples)
            for tokens, label in samples:
                activation = self.bias
                for token in tokens:
                    activation += self.weights.get(token, 0.0)
                pred = 1 / (1 + math.exp(-activation))
                grad = pred - label
                for token in tokens:
                    self.weights[token] = self.weights.get(token, 0.0) - lr * grad
                self.bias -= lr * grad

    def score(self, tokens: Sequence[str]) -> float:
        activation = self.bias
        for token in tokens:
            activation += self.weights.get(token.lower(), 0.0)
        return 1 / (1 + math.exp(-activation))

    def state_dict(self) -> Dict[str, object]:
        return {
            "config": self.config.__dict__,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "SoftBreakDetector":
        config = SoftBreakConfig(**state.get("config", {}))
        detector = cls(config)
        detector.weights = {str(k): float(v) for k, v in state.get("weights", {}).items()}
        detector.bias = float(state.get("bias", 0.0))
        return detector

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SoftBreakDetector":
        data = json.loads(path.read_text())
        return cls.from_state_dict(data)
