from __future__ import annotations

import importlib
import uuid

import pytest
from fastapi.testclient import TestClient

import json
from pathlib import Path
from apps.recorder.app.services.job_storage_service import JobStorageService


class DummyMQProducer:
    def __init__(self, *args, **kwargs):
        pass

    def publish(self, message: str):
        return None

    def close(self):
        return None


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    mq_mod = importlib.import_module("apps.recorder.app.services.rabbitmq_producer")
    monkeypatch.setattr(mq_mod, "RabbitMQProducer", DummyMQProducer)

    main_mod = importlib.import_module("apps.recorder.app.main")
    return TestClient(main_mod.app)


def test_get_job_not_found(client: TestClient):
    job_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 404


def test_get_job_report_not_found(client: TestClient):
    job_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/jobs/{job_id}/report")
    assert resp.status_code == 404


def test_get_job_transcript_not_found(client: TestClient):
    job_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/jobs/{job_id}/transcript")
    assert resp.status_code == 404


def test_get_existing_job_returns_data(client: TestClient, tmp_path: Path):
    job_id = "test-job-123"

    # JobStorageService expects: <base_dir>/jobs/<job_id>/job.json
    jobs_dir = tmp_path / "jobs" / job_id
    jobs_dir.mkdir(parents=True)

    job_data = {
        "job_id": job_id,
        "status": "recorded",
        "station_name": "test_station",
    }

    (jobs_dir / "job.json").write_text(json.dumps(job_data), encoding="utf-8")

    # Replace the controller's storage with one pointing to tmp_path
    jobs_ctrl = importlib.import_module("apps.recorder.app.controllers.jobs_controller")
    jobs_ctrl.job_storage = JobStorageService(base_dir=str(tmp_path))

    resp = client.get(f"/api/v1/jobs/{job_id}")

    assert resp.status_code == 200
    assert resp.json() == job_data


def test_record_now_returns_job_id(client: TestClient, monkeypatch):
    fixed_job_id = "job-fixed-001"

    jobs_ctrl = importlib.import_module("apps.recorder.app.controllers.jobs_controller")

    # Replace the real method to avoid starting threads / ffmpeg / network
    monkeypatch.setattr(
        jobs_ctrl.recording_service,
        "record_now",
        lambda dto: fixed_job_id,
    )

    payload = {
        "station_name": "test_station",
        "stream_url": "http://example.com/stream",
        "duration_sec": 5,
        "is_hls": False,
    }

    resp = client.post("/api/v1/jobs/record-now", json=payload)

    assert resp.status_code == 200
    assert resp.json() == {"job_id": fixed_job_id}
