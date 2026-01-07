from __future__ import annotations

import json

from news_structurizer import Pipeline, PipelineConfig

from .settings import load_settings
from .storage import JobStorage
from .consumer import RabbitConsumer
from .message import RecordingMessage


def main() -> None:
    settings = load_settings()
    storage = JobStorage(settings.data_dir)

    pipeline = Pipeline(
        PipelineConfig(
            segmenter_model_path=settings.segmenter_model_path,
            topic_model_path=settings.topic_model_path,
            scale_model_path=settings.scale_model_path,
            extractor_model_path=settings.extractor_model_path,
            asr_model=settings.asr_model,
            asr_language=settings.asr_language,
            device=settings.device,
        )
    )

    def handle(msg: RecordingMessage) -> None:
        storage.write_job_state(
            msg.job_id,
            {
                "status": "processing",
                "audio_path": msg.audio_path,
                "station_name": msg.station_name,
                "source_url": msg.source_url,
            },
        )

        try:
            report = pipeline.process_audio(msg.audio_path)
            storage.write_transcript(msg.job_id, report.text)
            storage.write_report(msg.job_id, report.to_dict())
            storage.write_job_state(
                msg.job_id,
                {
                    "status": "done",
                    "audio_path": msg.audio_path,
                    "station_name": msg.station_name,
                    "source_url": msg.source_url,
                },
            )
        except Exception as exc:
            storage.write_job_state(
                msg.job_id,
                {
                    "status": "failed",
                    "audio_path": msg.audio_path,
                    "station_name": msg.station_name,
                    "source_url": msg.source_url,
                    "error": str(exc),
                },
            )
            raise

    consumer = RabbitConsumer(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        username=settings.rabbitmq_username,
        password=settings.rabbitmq_password,
        queue_name=settings.queue_name,
        on_message=handle,
    )
    print(json.dumps({"event": "worker_ready", "queue": settings.queue_name}, ensure_ascii=False))
    consumer.run_forever()


if __name__ == "__main__":
    main()
