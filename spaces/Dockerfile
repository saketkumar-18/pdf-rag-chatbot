# Slim image for HF Spaces: no Ollama, LLM via HF serverless Inference API.
FROM python:3.11-slim

WORKDIR /app

# curl for healthchecks; keep image small
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image (avoids cold-start download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app/ ./app/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1 \
    RAG_LLM_BACKEND=hf \
    RAG_HF_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    RAG_CHROMA_DIR=/tmp/chroma \
    RAG_UPLOADS_DIR=/tmp/uploads

# /tmp is guaranteed writable on free Spaces (no persistent storage needed)
RUN mkdir -p /tmp/chroma /tmp/uploads

EXPOSE 7860

# HF Spaces injects APP_PORT (default 7860)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-7860}
