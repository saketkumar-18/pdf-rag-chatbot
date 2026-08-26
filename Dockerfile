# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install build deps + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download embedding model at build time (cached in layer)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime stage
FROM python:3.11-slim

# Install Ollama runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy app code
COPY app/ ./app/
COPY static/ ./static/

# Data directories (will be mounted as volumes in production)
RUN mkdir -p /data/chroma /data/uploads /data/ollama

ENV OLLAMA_MODELS=/data/ollama \
    CHROMA_DIR=/data/chroma \
    UPLOAD_DIR=/data/uploads \
    PYTHONUNBUFFERED=1 \
    RAG_LLM_MODEL=llama3.2:1b \
    RAG_OLLAMA_URL=http://localhost:11434

# Pre-pull the LLM model at build time (optional - comment out to pull at runtime)
# RUN ollama serve & sleep 10 && ollama pull llama3.2:1b

EXPOSE 8000

# Startup script: Ollama daemon + FastAPI
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Pull model if not present
if ! ollama list | grep -q "${RAG_LLM_MODEL:-llama3.2:1b}"; then
    echo "Pulling model ${RAG_LLM_MODEL:-llama3.2:1b}..."
    ollama pull "${RAG_LLM_MODEL:-llama3.2:1b}"
fi

# Start FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]