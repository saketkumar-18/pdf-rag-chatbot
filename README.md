# PDF RAG Chatbot

**Live demo:** https://pdf-rag-chatbot-gamma.vercel.app

Upload PDFs → ask questions → get answers **cited to page numbers**, generated
strictly from your documents. Fully local: no API keys, no cloud.

Now with **two LangGraph agent modes**: a self-correcting **CRAG pipeline**
(grades what it retrieves, rewrites weak queries, checks its own draft for
hallucinations) and a **deep-research mode** (decomposes the question into
sub-queries, retrieves for each, synthesizes one cited answer) — both with a
visible step-by-step trace in the UI.

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

### Agentic mode (LangGraph)

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
START ─► retrieve ─► grade_documents ─┬─ relevant ─► generate ─► check_answer ─► END
                                      │                            │ hallucinated?
                                      ├─ irrelevant & retries left ─► rewrite_query ─┐
                                      │                                              │
                                      └─ irrelevant & exhausted ─► refuse (no LLM)   │
                                                                                     │
        rewrite_query ─► retrieve (loop)                        generate ◄───────────┘
                                                                  (strict mode, 1 retry)
```

- **retrieve** — vector search (original or rewritten query)
- **grade_documents** — one batched LLM call scores every chunk yes/no;
  unparseable replies fall back to a similarity-floor heuristic
- **rewrite_query** — LLM rewrites the question into a keyword-rich query,
  then retrieval runs again (max `RAG_AGENT_MAX_REWRITES`)
- **generate** — cited answer from the surviving chunks only
- **check_answer** — a grader LLM rejects drafts that invent facts; one
  strict-mode regeneration is allowed before giving up
- Every node appends to a **trace** shown in the UI (🤖 Agent panel)

Pick a mode per-question with the header picker (**Classic / 🤖 Agent /
🔬 Research**), or globally with `RAG_AGENT_MODE=auto|on|off`. When no LLM
backend is configured the agents report themselves unavailable and the
classic single-shot path serves answers — nothing breaks.

### Deep-research mode (LangGraph)

```
START ─► plan ─► research ─► synthesize ─► check_answer ─► END
              (N sub-queries,   (1 LLM call,    │ hallucinated?
               dedupe + rank)    cited answer)  ▼
                              synthesize ◄── retry_gate (strict mode, 1 retry)
```

- **plan** — one LLM call decomposes the question into ≤ `RAG_RESEARCH_MAX_SUBQUERIES`
  keyword-rich sub-questions; unparseable replies fall back to the original question
- **research** — one vector search per sub-query (no LLM); results are deduped
  by chunk id, ranked by score, and capped at `RAG_RESEARCH_MAX_FINDINGS`
- **synthesize** — cited answer composed from the union of findings; if nothing
  was found it refuses without spending an LLM call
- **check_answer** — same hallucination grader as the CRAG graph, with one
  strict-mode regeneration allowed

Bounded by design: a full run is typically **3 LLM calls + N cheap vector
searches**, so it fits comfortably inside serverless timeouts.

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | async, pydantic validation, auto docs at `/docs` |
| PDF parsing | pypdf | pure-python text extraction per page |
| Chunking | custom paragraph packer | ~900 chars/chunk, 150-char overlap, page provenance |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, fast on CPU |
| Vector DB | ChromaDB (persistent, cosine HNSW) | zero-setup local persistence |
| LLM | Ollama · `llama3.2:1b` default | open-source weights, swap via env var |
| Agent orchestration | **LangGraph** (2 graphs) | CRAG: retrieve→grade→rewrite→generate→check · Research: plan→multi-query→synthesize→check |

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
| `POST /api/ask` | RAG answer with `citations[]` and `sources[]`; optional `"mode": "classic"\|"agent"\|"research"` (legacy `"agent": true/false` still works) |

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

When an agentic path runs, the response additionally carries an `agent`
block:

```json
{
  "agent": {
    "enabled": true,
    "mode": "agent",
    "trace": [
      {"node": "retrieve", "detail": "5 chunks for \"…\", best score 0.48", "ms": 44},
      {"node": "grade",    "detail": "kept 2/5 chunks", "ms": 1200},
      {"node": "generate", "detail": "118 chars from 2 chunks", "ms": 6800},
      {"node": "check",    "detail": "grounded ✓", "ms": 4900}
    ],
    "rewrites": 0,
    "retries": 0,
    "verdict": "grounded",
    "subqueries": [],
    "total_ms": 13000
  }
}
```

In `"mode": "research"` the trace instead shows `plan → research →
synthesize → check`, and `subqueries` lists the planner's decomposition.

`verdict` is `grounded` (passed the hallucination check), `ungrounded`
(rejected twice — answer still returned, treat with care), or `skipped`
(honest refusal, no generation LLM call was made).

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
| `RAG_AGENT_MODE` | `auto` | `auto` / `on` / `off` — LangGraph agentic path |
| `RAG_AGENT_MAX_REWRITES` | 1 | query rewrites allowed when retrieval is irrelevant |
| `RAG_AGENT_CHECK_ANSWER` | `true` | extra LLM call that screens drafts for hallucinations |
| `RAG_RESEARCH_MAX_SUBQUERIES` | 3 | max sub-questions the research planner may emit |
| `RAG_RESEARCH_K_PER_SUBQUERY` | 3 | vector hits fetched per sub-query |
| `RAG_RESEARCH_MAX_FINDINGS` | 8 | cap on unique chunks passed to synthesis |
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
re-upload dedup, delete, a full API flow including a live LLM answer,
and both LangGraph graphs — CRAG (happy path, rewrite loop, refusal,
self-correction, grading fallbacks) and deep-research (plan fallback,
dedupe/cap, no-findings refusal, self-correction, API integration) —
LLM mocked, no network needed.

## Known limits

- Scanned/image-only PDFs need OCR (not included).
- No multi-turn conversation memory — each question is answered independently.
- For best citation discipline use a ≥3B model (`RAG_LLM_MODEL=qwen2.5:3b`).
