from __future__ import annotations

import importlib
import uuid

import pytest
from fastapi.testclient import TestClient


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


def test_delete_recording_config_not_found(client: TestClient):
    config_id = str(uuid.uuid4())

    resp = client.request(
        "DELETE",
        "/api/v1/recording/configs",
        json={"config_id": config_id},
    )

    assert resp.status_code == 404
