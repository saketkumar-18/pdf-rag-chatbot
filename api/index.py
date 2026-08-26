"""Vercel serverless entrypoint.

Vercel looks for an ASGI/WSGI app in api/*.py. This file re-exports the
FastAPI app; all logic lives in the `app` package (shared with local mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the lambda cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.main import app  # noqa: E402,F401
