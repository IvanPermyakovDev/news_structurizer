from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .asr.whisper import WhisperConfig, WhisperTranscriber
    from .classification.topic_scale import NewsClassifier
    from .extraction.attributes import AttributeExtractor, GenerationConfig
    from .pipeline import Pipeline
    from .schemas import NewsItem, PipelineConfig, Report
    from .segmentation.topicsegmenter import NewsSegmenter

__all__ = [
    "AttributeExtractor",
    "GenerationConfig",
    "NewsClassifier",
    "NewsItem",
    "NewsSegmenter",
    "Pipeline",
    "PipelineConfig",
    "Report",
    "WhisperConfig",
    "WhisperTranscriber",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AttributeExtractor": ("news_structurizer.extraction.attributes", "AttributeExtractor"),
    "GenerationConfig": ("news_structurizer.extraction.attributes", "GenerationConfig"),
    "NewsClassifier": ("news_structurizer.classification.topic_scale", "NewsClassifier"),
    "NewsItem": ("news_structurizer.schemas", "NewsItem"),
    "NewsSegmenter": ("news_structurizer.segmentation.topicsegmenter", "NewsSegmenter"),
    "Pipeline": ("news_structurizer.pipeline", "Pipeline"),
    "PipelineConfig": ("news_structurizer.schemas", "PipelineConfig"),
    "Report": ("news_structurizer.schemas", "Report"),
    "WhisperConfig": ("news_structurizer.asr.whisper", "WhisperConfig"),
    "WhisperTranscriber": ("news_structurizer.asr.whisper", "WhisperTranscriber"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))
