from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class JobPaths:
    job_dir: Path
    job_json: Path
    transcript_txt: Path
    report_json: Path


class JobStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def paths(self, job_id: str) -> JobPaths:
        job_dir = self.base_dir / "jobs" / job_id
        return JobPaths(
            job_dir=job_dir,
            job_json=job_dir / "job.json",
            transcript_txt=job_dir / "transcript.txt",
            report_json=job_dir / "report.json",
        )

    def write_job_state(self, job_id: str, state: dict[str, Any]) -> None:
        p = self.paths(job_id)
        p.job_dir.mkdir(parents=True, exist_ok=True)
        incoming = dict(state)

        current: dict[str, Any] = {}
        if p.job_json.exists():
            try:
                current = json.loads(p.job_json.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        merged = dict(current)
        merged.update(incoming)
        merged.setdefault("job_id", job_id)
        merged.setdefault("created_at", current.get("created_at") or _now_iso())
        merged["updated_at"] = _now_iso()

        p.job_json.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_transcript(self, job_id: str, text: str) -> None:
        p = self.paths(job_id)
        p.job_dir.mkdir(parents=True, exist_ok=True)
        p.transcript_txt.write_text(text, encoding="utf-8")

    def write_report(self, job_id: str, report: dict[str, Any]) -> None:
        p = self.paths(job_id)
        p.job_dir.mkdir(parents=True, exist_ok=True)
        p.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
