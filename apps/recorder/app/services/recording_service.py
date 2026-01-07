import uuid
import schedule
import threading
import time
from typing import Union, List

from ..dto.config import ScheduleRule
from ..dto.jobs import RecordNowDTO
from ..coordination.recording_coordinator import RecordingCoordinator
from .job_storage_service import JobStorageService


class RecordingService:
    def __init__(self):
        self.configs = {}
        self.coordinator = RecordingCoordinator()
        self.job_storage = JobStorageService()
        self._start_scheduler()

    def _start_scheduler(self):
        def run():
            while True:
                schedule.run_pending()
                time.sleep(1)

        threading.Thread(target=run, daemon=True).start()

    def _parse_schedule(self, rule: ScheduleRule) -> Union[List[int], List[tuple], str]:
        if rule.type == "fixed_hours":
            if not rule.hours:
                raise ValueError("Missing 'hours' for fixed_hours schedule")
            return rule.hours
        elif rule.type == "hourly_range":
            if rule.start_hour is None or rule.end_hour is None:
                raise ValueError("Missing 'start_hour' or 'end_hour'")
            return "hourly_from_0700_to_2400"
        elif rule.type == "custom_durations":
            if not rule.durations:
                raise ValueError("Missing 'durations' for custom_durations")
            return rule.durations
        else:
            raise ValueError(f"Unknown schedule type: {rule.type}")

    def _add_job_to_schedule(self, job_id: str, job_func, schedule_rule):
        if isinstance(schedule_rule, list):
            if schedule_rule and isinstance(schedule_rule[0], tuple):
                for hour, duration in schedule_rule:
                    schedule.every().day.at(f"{int(hour):02d}:00").do(
                        job_func, duration_override=duration
                    )
            else:
                for hour in schedule_rule:
                    schedule.every().day.at(f"{int(hour):02d}:00").do(job_func)
        elif schedule_rule == "hourly_from_0700_to_2400":
            for hour in range(7, 24):
                schedule.every().day.at(f"{hour:02d}:00").do(job_func)
        else:
            raise ValueError("Invalid schedule rule")

    def create_config(self, dto) -> str:
        config_id = str(uuid.uuid4())
        url = dto.stream_url.strip()
        schedule_rule = self._parse_schedule(dto.schedule)

        job_func = self.coordinator.get_recording_job(
            dto.station_name, url, dto.duration_sec, dto.is_hls
        )

        self._add_job_to_schedule(config_id, job_func, schedule_rule)

        self.configs[config_id] = {
            "station_name": dto.station_name,
            "schedule_rule": schedule_rule,
        }

        return config_id

    def delete_config(self, config_id: str) -> bool:
        if config_id not in self.configs:
            return False
        del self.configs[config_id]
        return True

    def record_now(self, dto: RecordNowDTO) -> str:
        job_id = str(uuid.uuid4())
        url = dto.stream_url.strip()

        self.job_storage.upsert_job(
            job_id,
            {
                "status": "queued",
                "station_name": dto.station_name,
                "source_url": url,
                "duration_sec": dto.duration_sec,
                "is_hls": dto.is_hls,
            },
        )

        self.coordinator.start_recording(
            job_id=job_id,
            station_name=dto.station_name,
            stream_url=url,
            duration_sec=dto.duration_sec,
            is_hls=dto.is_hls,
        )
        return job_id
