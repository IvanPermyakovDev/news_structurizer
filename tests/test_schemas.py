from __future__ import annotations

from news_structurizer.schemas import NewsItem, PipelineConfig, Report


def test_report_to_dict_contains_expected_keys() -> None:
    report = Report(
        created_at="2026-01-01T00:00:00+00:00",
        text="hello",
        news=[
            NewsItem(
                id=1,
                text="segment",
                title="title",
                key_events="events",
                location="location",
                key_names="names",
                topic="topic",
                topic_confidence=0.9,
                scale="scale",
                scale_confidence=0.8,
            )
        ],
        meta={"source": "test"},
    )

    data = report.to_dict()
    assert set(data.keys()) == {"created_at", "text", "news", "meta"}
    assert data["text"] == "hello"
    assert data["meta"]["source"] == "test"
    assert data["news"][0]["id"] == 1
    assert data["news"][0]["topic_confidence"] == 0.9


def test_report_now_iso_is_utc() -> None:
    ts = Report.now_iso()
    assert "T" in ts
    assert ts.endswith("+00:00")


def test_pipeline_config_defaults() -> None:
    cfg = PipelineConfig(
        segmenter_model_path="/models/segmenter",
        topic_model_path="/models/topic",
        scale_model_path="/models/scale",
        extractor_model_path="/models/extractor",
    )
    assert cfg.asr_language == "kk"
