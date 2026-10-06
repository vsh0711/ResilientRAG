# Single-container demo of the API for a free Hugging Face Space (CPU, 16 GB).
# Vectors live in this process (QDRANT_USE_MEMORY), there is no Redis, and there
# is one worker. Documents are lost when the Space restarts or sleeps.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed
WORKDIR /home/user/app

COPY --chown=user pyproject.toml ./
COPY --chown=user app ./app
RUN pip install --no-cache-dir --user -e .

# Fetch the three embedding models at build time so the first upload is not a
# multi-minute download.
RUN python -c "from app.agent.vectorstore import _dense_model,_sparse_model,_reranker_model; _dense_model(); _sparse_model(); _reranker_model()"

ENV QDRANT_USE_MEMORY=true \
    CACHE_ENABLED=false \
    UPLOAD_DIR=/tmp/resilientrag_uploads \
    MAX_CONCURRENT_QUERIES=4 \
    MAX_CONCURRENT_UPLOADS=2 \
    TRUST_PROXY=true

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
