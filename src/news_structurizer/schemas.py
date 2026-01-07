from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    segmenter_model_path: str
    topic_model_path: str
    scale_model_path: str
    extractor_model_path: str
    asr_model: str | None = None
    asr_language: str = "kk"
    device: str | None = None


@dataclass(frozen=True)
class NewsItem:
    id: int
    text: str
    title: str
    key_events: str
    location: str
    key_names: str
    topic: str
    topic_confidence: float
    scale: str
    scale_confidence: float


@dataclass(frozen=True)
class Report:
    created_at: str
    text: str
    news: list[NewsItem] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
