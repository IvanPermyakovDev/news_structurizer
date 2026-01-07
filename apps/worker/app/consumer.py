from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Callable

import pika

from .message import RecordingMessage, parse_message


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RabbitConsumer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        queue_name: str,
        on_message: Callable[[RecordingMessage], None],
        prefetch_count: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.queue_name = queue_name
        self.on_message = on_message
        self.prefetch_count = prefetch_count

    def run_forever(self) -> None:
        while True:
            try:
                self._run_once()
            except Exception as exc:
                print(
                    json.dumps(
                        {"ts": _now_iso(), "event": "consumer_error", "error": str(exc)},
                        ensure_ascii=False,
                    )
                )
                time.sleep(5)

    def _run_once(self) -> None:
        credentials = pika.PlainCredentials(self.username, self.password)
        conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
        )
        channel = conn.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)
        channel.basic_qos(prefetch_count=self.prefetch_count)

        def callback(ch, method, properties, body: bytes) -> None:  # noqa: ANN001
            try:
                msg = parse_message(body)
                self.on_message(msg)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as exc:
                # Persisted failure is handled by on_message; still ack to avoid poison-loop.
                print(
                    json.dumps(
                        {
                            "ts": _now_iso(),
                            "event": "message_failed",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=self.queue_name, on_message_callback=callback)
        print(json.dumps({"ts": _now_iso(), "event": "consumer_started", "queue": self.queue_name}))
        try:
            channel.start_consuming()
        finally:
            try:
                conn.close()
            except Exception:
                pass
