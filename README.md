# PDF RAG Chatbot

**Live demo:** https://pdf-rag-chatbot-gamma.vercel.app

Upload PDFs → ask questions → get answers **cited to page numbers**, generated
strictly from your documents. Fully local: no API keys, no cloud.

```
PDF ──pypdf──► page-aware chunks ──MiniLM──► vectors ──► ChromaDB
                                                            │
Question ──MiniLM──► vector search (top-k) ─────────────────┘
                          │
                          ▼
            numbered context blocks + question
                          │
                          ▼
                 Ollama LLM (llama3.2)  ──►  cited answer [1][2]
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | async, pydantic validation, auto docs at `/docs` |
| PDF parsing | pypdf | pure-python text extraction per page |
| Chunking | custom paragraph packer | ~900 chars/chunk, 150-char overlap, page provenance |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, fast on CPU |
| Vector DB | ChromaDB (persistent, cosine HNSW) | zero-setup local persistence |
| LLM | Ollama · `llama3.2:1b` default | open-source weights, swap via env var |

## Deployment modes

| | `local` (default) | `serverless` (Vercel) |
|---|---|---|
| Uploads | ✅ drop PDFs in the UI | ❌ read-only (prebuilt index) |
| Vector store | ChromaDB on disk | bundled flat-file `data/index.json` |
| Embeddings | sentence-transformers (torch) | fastembed (ONNX, bundled in repo) |
| LLM | local Ollama | hosted API key **or built-in retrieval mode** |
| Cost | free | free |

**Zero-auth by design:** there is no signup or login anywhere. Anyone with the
link (or a scan of the QR code — use the 🔗 button in the header) lands
directly in the chat.

## Deploy to Vercel (free)

```bash
npm i -g vercel
vercel login

# one-time: build the read-only index from your PDFs
pip install fastembed pypdf
python scripts/build_index.py data/demo/attention-is-all-you-need-summary.pdf

# optional: add an LLM for full generative answers (else built-in mode is used)
vercel env add RAG_OPENAI_API_KEY          # paste a Groq/OpenAI/OpenRouter key
vercel env add RAG_OPENAI_BASE_URL         # e.g. https://api.groq.com/openai/v1
vercel env add RAG_OPENAI_LLM_MODEL        # e.g. llama-3.1-8b-instant

vercel --prod
```

`RAG_DEPLOYMENT=serverless`, the route table and function limits are already
configured in `vercel.json`. Without an LLM key the app serves cited answers
from the built-in extractive engine (marked as such in each reply); adding a
key upgrades every answer to generative RAG automatically.

To change the knowledge base: add/remove PDFs locally, re-run
`scripts/build_index.py`, redeploy.

## Quick start

```bash
# 1) prerequisites: Python 3.11+, Ollama running with a model pulled
ollama pull llama3.2:1b        # or use the bigger llama3:8b

# 2) install
python -m venv .venv
.venv\Scripts\activate         # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 3) run
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
#    or double-click run.bat
```

Open **http://localhost:8000** — drop a PDF in the sidebar, ask questions.

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | service + Ollama + index status |
| `POST /api/documents` | upload PDF(s) (multipart `files`) |
| `GET /api/documents` | list indexed documents |
| `DELETE /api/documents/{doc_id}` | remove document from index |
| `POST /api/search` | raw vector search, returns chunks + scores |
| `POST /api/ask` | RAG answer with `citations[]` and `sources[]` |

Interactive docs: http://localhost:8000/docs

```bash
curl -X POST localhost:8000/api/documents -F "files=@paper.pdf"
curl -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question":"What datasets were used?"}'
```

`/api/ask` response shape:

```json
{
  "answer": "Trained on eight NVIDIA P100 GPUs ...",
  "citations": [{"ref": 1, "source": "paper.pdf", "pages": "p.3", "score": 0.47, "auto": false}],
  "sources":  [{"text": "...", "source": "paper.pdf", "page_start": 3, "page_end": 3, "score": 0.47}],
  "timing":   {"retrieval_ms": 15, "generation_ms": 9600, "best_similarity": 0.47}
}
```

`auto: true` means the model omitted inline `[n]` markers and the backend
attached provenance automatically.

## Configuration (`.env`, see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_BACKEND` | `ollama` | `ollama` (local) or `hf` (HF serverless Inference API) |
| `RAG_LLM_MODEL` | `llama3.2:1b` | Ollama model tag (ollama backend) |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama daemon |
| `HF_TOKEN` | — | free token from huggingface.co/settings/tokens (hf backend) |
| `RAG_HF_LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | model for hf backend |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 900 / 150 | chunking knobs |
| `RAG_TOP_K` | 5 | chunks retrieved per query |
| `RAG_MAX_UPLOAD_MB` | 50 | upload limit |

## Hugging Face: LLM + embeddings backend (not hosting)

HF **Docker Spaces now require a PRO subscription** (free accounts get static
HTML Spaces only), so HF is used here as the *brain*, not the *host*:

- **LLM** — free serverless Inference API (`router.huggingface.co/v1`,
  OpenAI-compatible). Verified working models on a free token:
  `meta-llama/Llama-3.1-8B-Instruct` (default), plus ~130 more
  (see `GET /v1/models`). Set `HF_TOKEN` env var to enable.
- **Embeddings at build time** — `scripts/build_index.py` can embed via the
  `hf-inference` feature-extraction endpoint instead of local fastembed.

Hosting itself is on Vercel (see above). The `spaces/` folder keeps an
experimental Docker Space image for PRO users.

## Design notes

- **Page-aware chunks** — every chunk stores `page_start/page_end`, so every
  claim traces back to a *page*, not an opaque blob id.
- **Grounding contract** — the system prompt forbids answering outside context;
  if similarity of the best hit is very low, the model is instructed to say it
  doesn't know instead of hallucinating.
- **Idempotent re-upload** — same filename+size maps to one stable `doc_id`;
  re-ingesting replaces previous chunks instead of duplicating them.
- **LLM-agnostic** — Ollama access is one stdlib HTTP call in `app/llm.py`;
  switching models = changing one env var.

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

Covers: page-aware chunking, overlap behavior, store round-trip +
re-upload dedup, delete, and a full API flow including a live LLM answer.

## Known limits

- Scanned/image-only PDFs need OCR (not included).
- No multi-turn conversation memory — each question is answered independently.
- For best citation discipline use a ≥3B model (`RAG_LLM_MODEL=qwen2.5:3b`).
