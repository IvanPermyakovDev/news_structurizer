from __future__ import annotations

from .classification import NewsClassifier
from .extraction import AttributeExtractor
from .schemas import NewsItem, PipelineConfig, Report
from .segmentation import NewsSegmenter


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        self.segmenter = NewsSegmenter(
            model_path=config.segmenter_model_path,
            device=config.device,
        )
        self.classifier = NewsClassifier(
            topic_model_path=config.topic_model_path,
            scale_model_path=config.scale_model_path,
            device=config.device,
        )
        self.extractor = AttributeExtractor(
            model_path=config.extractor_model_path,
            device=config.device,
        )

    def process_text(self, text: str) -> Report:
        segments = self.segmenter.segment(text)
        news: list[NewsItem] = []

        for idx, segment in enumerate(segments, start=1):
            cls = self.classifier.classify(segment)
            attrs = self.extractor.extract(segment)

            news.append(
                NewsItem(
                    id=idx,
                    text=segment,
                    title=attrs["title"],
                    key_events=attrs["key_events"],
                    location=attrs["location"],
                    key_names=attrs["key_names"],
                    topic=cls["topic"],
                    topic_confidence=cls["topic_confidence"],
                    scale=cls["scale"],
                    scale_confidence=cls["scale_confidence"],
                )
            )

        return Report(
            created_at=Report.now_iso(),
            text=text,
            news=news,
            meta={},
        )

