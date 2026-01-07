from __future__ import annotations

import json
from pathlib import Path

from tests.utils import import_module_from_path


def _load_worker_storage_module():
    root = Path(__file__).resolve().parents[1]
    return import_module_from_path(
        "news_structurizer_tests_worker_storage",
        root / "apps" / "worker" / "app" / "storage.py",
    )


def test_job_storage_writes_and_merges_state(tmp_path: Path) -> None:
    storage_mod = _load_worker_storage_module()

    storage = storage_mod.JobStorage(tmp_path)
    storage.write_job_state("job-1", {"status": "queued"})

    p = storage.paths("job-1")
    assert p.job_json.exists()

    first = json.loads(p.job_json.read_text(encoding="utf-8"))
    assert first["job_id"] == "job-1"
    assert first["status"] == "queued"
    assert "created_at" in first
    assert "updated_at" in first

    storage.write_job_state("job-1", {"status": "processing", "station_name": "radio"})
    second = json.loads(p.job_json.read_text(encoding="utf-8"))
    assert second["status"] == "processing"
    assert second["station_name"] == "radio"
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] != first["updated_at"]


def test_job_storage_writes_report_and_transcript(tmp_path: Path) -> None:
    storage_mod = _load_worker_storage_module()

    storage = storage_mod.JobStorage(tmp_path)
    storage.write_transcript("job-2", "hello")
    storage.write_report("job-2", {"text": "hello", "news": []})

    p = storage.paths("job-2")
    assert p.transcript_txt.read_text(encoding="utf-8") == "hello"
    assert json.loads(p.report_json.read_text(encoding="utf-8"))["news"] == []
