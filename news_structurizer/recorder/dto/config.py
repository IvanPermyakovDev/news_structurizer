from pydantic import BaseModel
from typing import Union, List, Literal

class ScheduleRule(BaseModel):
    type: Literal["fixed_hours", "hourly_range", "custom_durations"]
    hours: List[int] = None
    start_hour: int = None
    end_hour: int = None
    durations: List[tuple] = None

class CreateRecordingConfigDTO(BaseModel):
    station_name: str
    stream_url: str
    duration_sec: int
    is_hls: bool
    schedule: ScheduleRule

class DeleteRecordingConfigDTO(BaseModel):
    config_id: str
