from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils import import_module_from_path


def _load_worker_settings_module():
    root = Path(__file__).resolve().parents[1]
    return import_module_from_path(
        "news_structurizer_tests_worker_settings",
        root / "apps" / "worker" / "app" / "settings.py",
    )


def test_worker_settings_missing_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    settings_mod = _load_worker_settings_module()

    for key in [
        "NS_SEGMENTER_MODEL",
        "NS_TOPIC_MODEL",
        "NS_SCALE_MODEL",
        "NS_EXTRACTOR_MODEL",
        "NS_ASR_MODEL",
        "NS_ASR_LANGUAGE",
        "NS_DEVICE",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        settings_mod.load_settings()
    msg = str(excinfo.value)
    assert "NS_SEGMENTER_MODEL" in msg
    assert "NS_TOPIC_MODEL" in msg
    assert "NS_SCALE_MODEL" in msg
    assert "NS_EXTRACTOR_MODEL" in msg


def test_worker_settings_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings_mod = _load_worker_settings_module()

    monkeypatch.setenv("NS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NS_SEGMENTER_MODEL", "/models/segmenter")
    monkeypatch.setenv("NS_TOPIC_MODEL", "/models/topic")
    monkeypatch.setenv("NS_SCALE_MODEL", "/models/scale")
    monkeypatch.setenv("NS_EXTRACTOR_MODEL", "/models/extractor")

    monkeypatch.setenv("RABBITMQ_HOST", "example")
    monkeypatch.setenv("RABBITMQ_PORT", "5673")
    monkeypatch.setenv("RABBITMQ_QUEUE", "recordings")

    settings = settings_mod.load_settings()
    assert settings.rabbitmq_host == "example"
    assert settings.rabbitmq_port == 5673
    assert settings.queue_name == "recordings"
    assert settings.data_dir == tmp_path
