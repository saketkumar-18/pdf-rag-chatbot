---
title: PDF RAG Chatbot
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Upload PDFs, ask questions, get page-cited answers
---

# PDF RAG Chatbot

Upload PDFs → ask questions → get answers cited to page numbers.

- **Embeddings:** all-MiniLM-L6-v2 (runs in the Space)
- **Vector DB:** ChromaDB (in-Space, resets on restart)
- **LLM:** Hugging Face serverless Inference API (Llama-3.1-8B-Instruct, free tier)

Set the `HF_TOKEN` secret in Space settings (Settings → Variables and secrets).
