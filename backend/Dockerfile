# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN addgroup --system backend && adduser --system --ingroup backend worker
USER worker

ENV BACKEND_FAISS_INDEX_PATH=/data/index.faiss
ENV BACKEND_FAISS_META_PATH=/data/index.json
ENV BACKEND_PLAYLIST_INGEST_BATCH_SIZE=75

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
