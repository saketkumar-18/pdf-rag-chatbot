"""Application configuration.

All knobs live here so nothing is hard-coded deep inside the codebase.
Every value can be overridden with an environment variable, e.g.
    RAG_LLM_MODEL=qwen2.5:3b uvicorn app.main:app

Deployment profiles
-------------------
RAG_DEPLOYMENT=local       (default) writable disk, uploads enabled, any backend.
RAG_DEPLOYMENT=serverless  Vercel/read-only FS: prebuilt index bundled with the
                           code, uploads disabled, hosted OpenAI-compatible LLM,
                           fastembed (ONNX) embeddings instead of torch.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env if present

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


# --- deployment profile --------------------------------------------------
DEPLOYMENT = os.getenv("RAG_DEPLOYMENT", "local").strip().lower()
SERVERLESS = DEPLOYMENT == "serverless"

# --- storage ---------------------------------------------------------------
# local: persistent dirs inside the project. serverless: the code bundle's
# data/ dir holds a PREBUILT index (built by scripts/build_index.py);
# nothing is written at runtime.
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
UPLOADS_DIR = Path(os.getenv("RAG_UPLOADS_DIR", str(BASE_DIR / "data" / "uploads")))
PREBUILT_INDEX = Path(
    os.getenv("RAG_PREBUILT_INDEX", str(BASE_DIR / "data" / "index.json"))
)
if not SERVERLESS:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- embeddings ----------------------------------------------------------
# local default: sentence-transformers (torch) all-MiniLM-L6-v2, 384-dim.
# serverless: fastembed ONNX runtime (no torch) with the SAME model so the
# prebuilt index and live queries always share one embedding space.
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "auto")  # auto|st|fastembed
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_SERVERLESS = os.getenv(
    "RAG_EMBEDDING_MODEL_SERVERLESS", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = _env_int("RAG_EMBEDDING_DIM", "384")
CACHE_DIR_RAW = os.getenv("RAG_CACHE_DIR", "")
if not CACHE_DIR_RAW:
    # Serverless: bundle the ONNX model inside the repo for instant cold starts.
    CACHE_DIR = str(BASE_DIR / "assets" / "fastembed-cache") if SERVERLESS else "/tmp/fastembed-cache"
elif not Path(CACHE_DIR_RAW).is_absolute():
    CACHE_DIR = str(BASE_DIR / CACHE_DIR_RAW)
else:
    CACHE_DIR = CACHE_DIR_RAW

# --- LLM -------------------------------------------------------------------
# auto = ollama when RAG_DEPLOYMENT=local, openai-compatible when serverless
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "auto").strip().lower()  # auto|ollama|openai|hf
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2:1b")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))

# OpenAI-compatible settings (works with OpenAI, Groq, OpenRouter, Together, ...)
OPENAI_BASE_URL = os.getenv(
    "RAG_OPENAI_BASE_URL",
    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
).rstrip("/")
_OPENAI_KEY_CANDIDATES = (
    os.getenv("RAG_OPENAI_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("GROQ_API_KEY", "").strip()
    or os.getenv("OPENROUTER_API_KEY", "").strip()
)
OPENAI_API_KEY = _OPENAI_KEY_CANDIDATES or None
OPENAI_LLM_MODEL = os.getenv(
    "RAG_OPENAI_LLM_MODEL", os.getenv("RAG_HF_LLM_MODEL", "llama-3.1-8b-instant")
)

# Legacy HF Inference router (OpenAI-compatible endpoint)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
HF_LLM_MODEL = os.getenv("RAG_HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# Server-side embedding proxy for browser-indexed uploads (keeps the token
# off the client). Same all-MiniLM-L6-v2 space as every other backend.
HF_EMBED_URL = os.getenv(
    "RAG_HF_EMBED_URL",
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
)


def effective_llm_backend() -> str:
    """Resolve the actual backend after 'auto'/'hf' aliases."""
    resolved = _resolve_llm_backend()
    # Graceful degradation: no hosted key configured -> built-in extractive
    # answering (still grounded + cited). Adding a key upgrades automatically.
    if resolved == "openai" and not (OPENAI_API_KEY or HF_TOKEN):
        return "extractive"
    return resolved


def _resolve_llm_backend() -> str:
    if LLM_BACKEND == "ollama":
        return "ollama"
    if LLM_BACKEND in {"openai", "hf"}:
        return "openai"
    # auto: serverless -> openai-compatible; local -> ollama
    return "openai" if SERVERLESS else "ollama"


def effective_embedding_backend() -> str:
    if EMBEDDING_BACKEND != "auto":
        return EMBEDDING_BACKEND
    return "fastembed" if SERVERLESS else "st"


# --- chunking / retrieval ------------------------------------------------
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", "900")        # characters
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", "150")  # characters
TOP_K = _env_int("RAG_TOP_K", "5")                    # chunks per query

# --- agentic RAG (LangGraph) ----------------------------------------------
# AGENT_MODE: auto = use the LangGraph agent when an LLM backend is present,
#             on   = force agent (falls back to classic path if unavailable),
#             off  = classic single-shot RAG only.
AGENT_MODE = os.getenv("RAG_AGENT_MODE", "auto").strip().lower()
AGENT_MAX_REWRITES = _env_int("RAG_AGENT_MAX_REWRITES", "1")
AGENT_CHECK_ANSWER = _env_bool("RAG_AGENT_CHECK_ANSWER", "true")

# Deep-research mode (plan -> parallel subquery retrieval -> synthesize).
# Bounded so a full run stays within serverless timeouts:
# typical = 3 LLM calls (plan, synthesize, check) + N cheap vector searches.
RESEARCH_MAX_SUBQUERIES = _env_int("RAG_RESEARCH_MAX_SUBQUERIES", "3")
RESEARCH_K_PER_SUBQUERY = _env_int("RAG_RESEARCH_K_PER_SUBQUERY", "3")
RESEARCH_MAX_FINDINGS = _env_int("RAG_RESEARCH_MAX_FINDINGS", "8")

# --- upload limits ---------------------------------------------------------
MAX_UPLOAD_MB = _env_int("RAG_MAX_UPLOAD_MB", "50")
ALLOWED_EXTENSIONS = {".pdf"}

# --- CORS ------------------------------------------------------------------
# Comma-separated list of extra allowed origins. Same-origin UI always works;
# "*" only as an explicit opt-in.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("RAG_CORS_ORIGINS", "").split(",") if o.strip()
]

# --- rate limiting (per IP, sliding window) --------------------------------
RATE_LIMIT_ASK = _env_int("RAG_RATE_LIMIT_ASK", "20")      # requests/min
RATE_LIMIT_SEARCH = _env_int("RAG_RATE_LIMIT_SEARCH", "30")
RATE_LIMIT_UPLOAD = _env_int("RAG_RATE_LIMIT_UPLOAD", "10")
RATE_LIMIT_EMBED = _env_int("RAG_RATE_LIMIT_EMBED", "15")  # browser indexing batches
RATE_LIMIT_ENABLED = _env_bool("RAG_RATE_LIMIT_ENABLED", "true")
