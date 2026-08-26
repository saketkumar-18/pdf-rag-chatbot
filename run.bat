@echo off
REM Start the PDF RAG chatbot (API + UI on http://localhost:8000)
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
