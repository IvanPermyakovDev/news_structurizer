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
