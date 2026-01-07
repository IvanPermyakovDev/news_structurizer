FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core package (with service deps; ML deps should be baked into the image separately if needed)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .[service,ml]

# Install worker-specific deps (pika pin for runtime consistency)
COPY apps/worker/requirements.txt ./requirements-worker.txt
RUN pip install --no-cache-dir -r requirements-worker.txt

COPY apps/worker/app/ ./app/

ENV NS_DATA_DIR=/data

CMD ["python", "-m", "app.main"]

