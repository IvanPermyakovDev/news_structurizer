from __future__ import annotations

from pydantic import BaseModel


class RecordNowDTO(BaseModel):
    station_name: str
    stream_url: str
    duration_sec: int
    is_hls: bool

