import threading
import subprocess
import requests
import time
import json
import uuid
from typing import Callable, Optional

from ..services.file_storage_service import FileStorageService
from ..services.job_storage_service import JobStorageService
from ..services.rabbitmq_producer import RabbitMQProducer


class RecordingCoordinator:
    def __init__(self, ffmpeg_path: str = "/usr/bin/ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.file_service = FileStorageService()
        self.job_storage = JobStorageService()
        self.mq_producer = RabbitMQProducer()

    def record_http_stream(
        self,
        job_id: str,
        station_name: str,
        stream_url: str,
        duration_sec: int,
        duration_override: Optional[int] = None,
    ):
        duration = duration_override or duration_sec
        filepath = self.file_service.generate_job_filepath(job_id, station_name, ".mp3")

        try:
            self.job_storage.upsert_job(
                job_id,
                {
                    "status": "recording",
                    "station_name": station_name,
                    "source_url": stream_url,
                    "duration_sec": duration,
                    "is_hls": False,
                    "audio_path": filepath,
                },
            )
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
            self.job_storage.upsert_job(job_id, {"status": "recorded", "audio_path": filepath})
            self.mq_producer.publish(
                json.dumps(
                    {
                        "job_id": job_id,
                        "audio_path": filepath,
                        "station_name": station_name,
                        "source_url": stream_url,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            self.job_storage.upsert_job(job_id, {"status": "failed", "error": str(e)})
            print(f"[HTTP RECORD ERROR] {station_name}: {e}")

    def record_hls_stream(
        self,
        job_id: str,
        station_name: str,
        stream_url: str,
        duration_sec: int,
        duration_override: Optional[int] = None,
    ):
        duration = duration_override or duration_sec
        filepath = self.file_service.generate_job_filepath(job_id, station_name, ".mp3")

        try:
            self.job_storage.upsert_job(
                job_id,
                {
                    "status": "recording",
                    "station_name": station_name,
                    "source_url": stream_url,
                    "duration_sec": duration,
                    "is_hls": True,
                    "audio_path": filepath,
                },
            )
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
                self.job_storage.upsert_job(job_id, {"status": "recorded", "audio_path": filepath})
                self.mq_producer.publish(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "audio_path": filepath,
                            "station_name": station_name,
                            "source_url": stream_url,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                self.job_storage.upsert_job(
                    job_id,
                    {"status": "failed", "error": "ffmpeg returned non-zero exit code"},
                )
                print(f"[FFMPEG ERROR] Non-zero exit for {station_name}")
        except Exception as e:
            self.job_storage.upsert_job(job_id, {"status": "failed", "error": str(e)})
            print(f"[HLS RECORD ERROR] {station_name}: {e}")

    def get_recording_job(
        self, station_name: str, stream_url: str, duration_sec: int, is_hls: bool
    ) -> Callable:
        record_func = (
            self.record_hls_stream if is_hls else self.record_http_stream
        )

        def job(duration_override: Optional[int] = None):
            job_id = str(uuid.uuid4())
            self.job_storage.upsert_job(
                job_id,
                {
                    "status": "queued",
                    "station_name": station_name,
                    "source_url": stream_url,
                    "duration_sec": duration_override or duration_sec,
                    "is_hls": is_hls,
                },
            )
            thread = threading.Thread(
                target=record_func,
                args=(job_id, station_name, stream_url, duration_sec),
                kwargs={"duration_override": duration_override},
                daemon=True,
            )
            thread.start()

        return job

    def start_recording(
        self,
        *,
        job_id: str,
        station_name: str,
        stream_url: str,
        duration_sec: int,
        is_hls: bool,
    ) -> None:
        record_func = self.record_hls_stream if is_hls else self.record_http_stream
        thread = threading.Thread(
            target=record_func,
            args=(job_id, station_name, stream_url, duration_sec),
            daemon=True,
        )
        thread.start()
