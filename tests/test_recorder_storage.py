from __future__ import annotations

from pathlib import Path

from tests.utils import import_module_from_path


def _load_job_storage_service_module():
    root = Path(__file__).resolve().parents[1]
    return import_module_from_path(
        "news_structurizer_tests_recorder_job_storage_service",
        root / "apps" / "recorder" / "app" / "services" / "job_storage_service.py",
    )


def _load_file_storage_service_module():
    root = Path(__file__).resolve().parents[1]
    return import_module_from_path(
        "news_structurizer_tests_recorder_file_storage_service",
        root / "apps" / "recorder" / "app" / "services" / "file_storage_service.py",
    )


def test_recorder_job_storage_service_upsert_and_read(tmp_path: Path) -> None:
    mod = _load_job_storage_service_module()

    storage = mod.JobStorageService(base_dir=str(tmp_path))
    job_id = "job-123"
    stored = storage.upsert_job(job_id, {"status": "queued"})
    assert stored["job_id"] == job_id
    assert stored["status"] == "queued"

    loaded = storage.read_job(job_id)
    assert loaded["job_id"] == job_id
    assert loaded["status"] == "queued"

    storage.upsert_job(job_id, {"status": "done"})
    loaded2 = storage.read_job(job_id)
    assert loaded2["status"] == "done"
    assert loaded2["created_at"] == loaded["created_at"]


def test_recorder_job_storage_service_paths(tmp_path: Path) -> None:
    mod = _load_job_storage_service_module()
    storage = mod.JobStorageService(base_dir=str(tmp_path))
    assert storage.report_path("x").name == "report.json"
    assert storage.transcript_path("x").name == "transcript.txt"
    assert storage.job_json_path("x").name == "job.json"


def test_file_storage_service_generates_job_filepath(tmp_path: Path) -> None:
    mod = _load_file_storage_service_module()
    storage = mod.FileStorageService(base_output_dir=str(tmp_path))

    path = Path(storage.generate_job_filepath(job_id="job-1", station_name="station"))
    assert path.name == "job-1.mp3"
    assert path.parent.exists()
