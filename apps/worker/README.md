# Worker service

RabbitMQ consumer that processes recorded audio files with the `news_structurizer` package:
ASR (Whisper) → segmentation → topic/scale classification → attribute extraction.

Outputs are written to `/data/jobs/<job_id>/`:
- `job.json` (status + metadata)
- `transcript.txt`
- `report.json`

