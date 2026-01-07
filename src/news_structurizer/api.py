from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extraction.attributes import GenerationConfig

def segment_text(*, text: str, segmenter_model_path: str, device: str | None = None) -> list[str]:
    from .segmentation.topicsegmenter import NewsSegmenter

    return NewsSegmenter(model_path=segmenter_model_path, device=device).segment(text)


def classify_text(
    *,
    text: str,
    topic_model_path: str,
    scale_model_path: str,
    device: str | None = None,
) -> dict:
    from .classification.topic_scale import NewsClassifier

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
    from .extraction.attributes import AttributeExtractor

    extractor = AttributeExtractor(model_path=extractor_model_path, device=device)
    return extractor.extract(text, config=config)
