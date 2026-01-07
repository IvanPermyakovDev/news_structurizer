#!/bin/bash

# Создаём структуру
mkdir -p app/dto app/controllers app/services app/coordination

# requirements.txt
cat > requirements.txt << 'REQ'
fastapi==0.115.0
uvicorn==0.32.0
schedule==1.2.2
requests==2.31.0
pika==1.3.2
python-multipart==0.0.9
REQ

# Dockerfile
cat > Dockerfile << 'DOCKER'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKER

# docker-compose.yml
cat > docker-compose.yml << 'COMPOSE'
version: '3.8'

services:
  audio-recorder:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - recordings_volume:/data/recordings
    depends_on:
      - rabbitmq
    restart: unless-stopped

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    restart: unless-stopped

volumes:
  recordings_volume:
COMPOSE

# app/__init__.py
touch app/__init__.py

# app/main.py
cat > app/main.py << 'MAIN'
from fastapi import FastAPI
from .controllers.recording_controller import router as recording_router

app = FastAPI(title="Audio Recorder Service")

app.include_router(recording_router, prefix="/api/v1/recording")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
MAIN

# app/dto/config.py
cat > app/dto/config.py << 'DTO'
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
DTO

# app/controllers/recording_controller.py
cat > app/controllers/recording_controller.py << 'CTRL'
from fastapi import APIRouter, HTTPException
from ..dto.config import CreateRecordingConfigDTO, DeleteRecordingConfigDTO
from ..services.recording_service import RecordingService

router = APIRouter()
recording_service = RecordingService()


@router.post("/configs")
async def create_recording_config(dto: CreateRecordingConfigDTO):
    try:
        config_id = recording_service.create_config(dto)
        return {"config_id": config_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/configs")
async def delete_recording_config(dto: DeleteRecordingConfigDTO):
    success = recording_service.delete_config(dto.config_id)
    if not success:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"message": "Config deleted"}
CTRL

# app/services/file_storage_service.py
cat > app/services/file_storage_service.py << 'FILE'
import os
from pathlib import Path
from datetime import datetime


class FileStorageService:
    def __init__(self, base_output_dir: str = "/data/recordings"):
        self.base_output_dir = Path(base_output_dir)

    def generate_filepath(self, station_name: str, extension: str = ".mp3") -> str:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        station_dir = self.base_output_dir / date_str / station_name
        station_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{station_name}_{now.strftime('%Y%m%d_%H%M')}{extension}"
        return str(station_dir / filename)
FILE

# app/services/rabbitmq_producer.py
cat > app/services/rabbitmq_producer.py << 'RABBIT'
import pika
from typing import Optional


class RabbitMQProducer:
    def __init__(
        self,
        host: str = "rabbitmq",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        queue_name: str = "recordings",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.queue_name = queue_name
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.BlockingChannel] = None
        self._connect()

    def _connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
        )
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self.queue_name, durable=True)

    def publish(self, message: str):
        if self._connection.is_closed:
            self._connect()
        self._channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2),
        )

    def close(self):
        if self._connection and self._connection.is_open:
            self._connection.close()
RABBIT

# app/coordination/recording_coordinator.py
cat > app/coordination/recording_coordinator.py << 'COORD'
import threading
import subprocess
import requests
import time
from typing import Callable, Optional

from ..services.file_storage_service import FileStorageService
from ..services.rabbitmq_producer import RabbitMQProducer


class RecordingCoordinator:
    def __init__(self, ffmpeg_path: str = "/usr/bin/ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.file_service = FileStorageService()
        self.mq_producer = RabbitMQProducer()

    def record_http_stream(
        self,
        station_name: str,
        stream_url: str,
        duration_sec: int,
        duration_override: Optional[int] = None,
    ):
        duration = duration_override or duration_sec
        filepath = self.file_service.generate_filepath(station_name, ".mp3")

        try:
            with requests.get(
                stream_url, stream=True, timeout=duration + 60, verify=False
            ) as resp:
                resp.raise_for_status()
                start_time = time.time()
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if time.time() - start_time > duration:
                            break
                        if chunk:
                            f.write(chunk)
            self.mq_producer.publish(filepath)
        except Exception as e:
            print(f"[HTTP RECORD ERROR] {station_name}: {e}")

    def record_hls_stream(
        self,
        station_name: str,
        stream_url: str,
        duration_sec: int,
        duration_override: Optional[int] = None,
    ):
        duration = duration_override or duration_sec
        filepath = self.file_service.generate_filepath(station_name, ".mp3")

        try:
            cmd = [
                self.ffmpeg_path,
                "-i",
                stream_url,
                "-t",
                str(duration),
                "-vn",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                "-y",
                filepath,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                self.mq_producer.publish(filepath)
            else:
                print(f"[FFMPEG ERROR] Non-zero exit for {station_name}")
        except Exception as e:
            print(f"[HLS RECORD ERROR] {station_name}: {e}")

    def get_recording_job(
        self, station_name: str, stream_url: str, duration_sec: int, is_hls: bool
    ) -> Callable:
        record_func = (
            self.record_hls_stream if is_hls else self.record_http_stream
        )

        def job(duration_override: Optional[int] = None):
            thread = threading.Thread(
                target=record_func,
                args=(station_name, stream_url, duration_sec),
                kwargs={"duration_override": duration_override},
                daemon=True,
            )
            thread.start()

        return job
COORD

# app/services/recording_service.py
cat > app/services/recording_service.py << 'SERVICE'
import uuid
import schedule
import threading
import time
from typing import Union, List

from ..dto.config import ScheduleRule
from ..coordination.recording_coordinator import RecordingCoordinator


class RecordingService:
    def __init__(self):
        self.configs = {}
        self.coordinator = RecordingCoordinator()
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
SERVICE

echo "✅ Проект успешно создан!"
echo "Запустите: docker-compose up --build"
