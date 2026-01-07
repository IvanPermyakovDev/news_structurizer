from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordingMessage:
    job_id: str
    audio_path: str
    station_name: str | None = None
    source_url: str | None = None


def parse_message(body: bytes) -> RecordingMessage:
    raw = body.decode("utf-8", errors="replace").strip()
    if not raw:
        raise ValueError("Empty message body")

    # Backward compatible: recorder used to publish plain `audio_path` string.
    if not raw.startswith("{"):
        return RecordingMessage(job_id=str(uuid.uuid4()), audio_path=raw)

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        return RecordingMessage(job_id=str(uuid.uuid4()), audio_path=raw)

    if not isinstance(payload, dict):
        return RecordingMessage(job_id=str(uuid.uuid4()), audio_path=raw)

    job_id = str(payload.get("job_id") or uuid.uuid4())
    audio_path = payload.get("audio_path")
    if not audio_path:
        raise ValueError("Missing `audio_path` in message")

    return RecordingMessage(
        job_id=job_id,
        audio_path=str(audio_path),
        station_name=payload.get("station_name"),
        source_url=payload.get("source_url"),
    )

