FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV WARMUP_MODELS=true
ENV QDRANT_TIMEOUT_SECONDS=5
ENV UVICORN_WORKERS=2
# Each worker's ONNX runtime would otherwise take every core.
ENV OMP_NUM_THREADS=2
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
