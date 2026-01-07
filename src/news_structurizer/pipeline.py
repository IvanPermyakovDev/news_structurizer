from __future__ import annotations

import os

from .asr import WhisperConfig, WhisperTranscriber
from .classification import NewsClassifier
from .extraction import AttributeExtractor
from .schemas import NewsItem, PipelineConfig, Report
from .segmentation import NewsSegmenter


class Pipeline:
    DEFAULT_ASR_MODEL = "abilmansplus/whisper-turbo-ksc2"

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
        self._transcriber: WhisperTranscriber | None = None

    def _build_report(self, text: str, *, meta: dict | None = None) -> Report:
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
            meta=meta or {},
        )

    def process_text(self, text: str) -> Report:
        return self._build_report(text)

    def _get_transcriber(self) -> WhisperTranscriber:
        if self._transcriber is not None:
            return self._transcriber

        model = (
            self.config.asr_model
            or os.environ.get("NS_ASR_MODEL")
            or self.DEFAULT_ASR_MODEL
        )
        language = self.config.asr_language or (os.environ.get("NS_ASR_LANGUAGE") or "kk")
        asr_cfg = WhisperConfig(model=model, language=language)
        self._transcriber = WhisperTranscriber(asr_cfg, device=self.config.device)
        return self._transcriber

    def process_audio(self, audio_path: str) -> Report:
        transcriber = self._get_transcriber()
        asr = transcriber.transcribe(audio_path)
        text = asr.get("text", "").strip()
        if not text:
            raise RuntimeError("ASR returned empty text.")

        meta = {
            "audio_path": audio_path,
            "asr": {
                "model": self.config.asr_model,
                "language": self.config.asr_language,
            },
        }
        return self._build_report(text, meta=meta)
