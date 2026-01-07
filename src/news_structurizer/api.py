from __future__ import annotations

from .classification import NewsClassifier
from .extraction import AttributeExtractor, GenerationConfig
from .segmentation import NewsSegmenter


def segment_text(*, text: str, segmenter_model_path: str, device: str | None = None) -> list[str]:
    return NewsSegmenter(model_path=segmenter_model_path, device=device).segment(text)


def classify_text(
    *,
    text: str,
    topic_model_path: str,
    scale_model_path: str,
    device: str | None = None,
) -> dict:
    return NewsClassifier(
        topic_model_path=topic_model_path,
        scale_model_path=scale_model_path,
        device=device,
    ).classify(text)


def extract_attributes(
    *,
    text: str,
    extractor_model_path: str,
    device: str | None = None,
    config: GenerationConfig | None = None,
) -> dict[str, str]:
    extractor = AttributeExtractor(model_path=extractor_model_path, device=device)
    return extractor.extract(text, config=config)
