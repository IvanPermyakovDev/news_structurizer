from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSettings:
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_username: str
    rabbitmq_password: str
    queue_name: str

    data_dir: Path

    segmenter_model_path: str
    topic_model_path: str
    scale_model_path: str
    extractor_model_path: str
    asr_model: str
    asr_language: str
    device: str | None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def load_settings() -> WorkerSettings:
    data_dir = Path(_env("NS_DATA_DIR", "/data") or "/data")

    rabbitmq_port_s = _env("RABBITMQ_PORT", "5672") or "5672"
    rabbitmq_port = int(rabbitmq_port_s)

    # Model paths must be provided by env in container deployments.
    # For local runs, you can point them to `research/ml/...`.
    segmenter_model = _env("NS_SEGMENTER_MODEL")
    topic_model = _env("NS_TOPIC_MODEL")
    scale_model = _env("NS_SCALE_MODEL")
    extractor_model = _env("NS_EXTRACTOR_MODEL")
    default_asr_model = "abilmansplus/whisper-turbo-ksc2"
    asr_model = _env("NS_ASR_MODEL", default_asr_model) or default_asr_model

    missing = [k for k, v in {
        "NS_SEGMENTER_MODEL": segmenter_model,
        "NS_TOPIC_MODEL": topic_model,
        "NS_SCALE_MODEL": scale_model,
        "NS_EXTRACTOR_MODEL": extractor_model,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return WorkerSettings(
        rabbitmq_host=_env("RABBITMQ_HOST", "rabbitmq") or "rabbitmq",
        rabbitmq_port=rabbitmq_port,
        rabbitmq_username=_env("RABBITMQ_USERNAME", "guest") or "guest",
        rabbitmq_password=_env("RABBITMQ_PASSWORD", "guest") or "guest",
        queue_name=_env("RABBITMQ_QUEUE", "recordings") or "recordings",
        data_dir=data_dir,
        segmenter_model_path=segmenter_model,
        topic_model_path=topic_model,
        scale_model_path=scale_model,
        extractor_model_path=extractor_model,
        asr_model=asr_model,
        asr_language=_env("NS_ASR_LANGUAGE", "kk") or "kk",
        device=_env("NS_DEVICE"),
    )
