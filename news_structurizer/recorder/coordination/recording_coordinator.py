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
