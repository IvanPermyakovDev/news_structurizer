from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class JobStorageService:
    def __init__(self, base_dir: str = "/data") -> None:
        self.base_dir = Path(base_dir)

    def job_dir(self, job_id: str) -> Path:
        return self.base_dir / "jobs" / job_id

    def job_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def transcript_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "transcript.txt"

    def report_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "report.json"

    def read_job(self, job_id: str) -> dict[str, Any]:
        path = self.job_json_path(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def upsert_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        current: dict[str, Any] = {}
        path = self.job_json_path(job_id)
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        merged = dict(current)
        merged.update(dict(updates))
        merged.setdefault("job_id", job_id)
        merged.setdefault("created_at", current.get("created_at") or _now_iso())
        merged["updated_at"] = _now_iso()

        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        return merged

