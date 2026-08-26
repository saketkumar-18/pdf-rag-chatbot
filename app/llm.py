"""LLM backends: local Ollama OR any hosted OpenAI-compatible chat API.

Both are stdlib-only (urllib) so swapping backends stays a config change,
not a dependency change.

RAG_LLM_BACKEND=ollama   -> local Ollama daemon (default for local deploys)
RAG_LLM_BACKEND=openai   -> OpenAI-compatible /chat/completions endpoint
                            (OpenAI, Groq, OpenRouter, Together, HF router...)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or errors out."""


# Backwards-compatible alias (older code/tests import OllamaError).
OllamaError = LLMError

_TIMEOUT = 60
_MAX_ATTEMPTS = 2  # one retry on transient failures


def _transient(exc: Exception) -> bool:
    msg = str(exc)
    return "HTTP 429" in msg or "HTTP 5" in msg or "timed out" in msg.lower()


def _post_json(
    url: str, payload: dict, headers: dict | None = None, timeout: int = _TIMEOUT
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach LLM API ({exc.reason}).") from exc
    except TimeoutError as exc:
        raise LLMError("LLM request timed out.") from exc


def ollama_alive() -> bool:
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3
        ) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def openai_backend_ready() -> bool:
    """True when an API key is present for the OpenAI-compatible path."""
    return bool(config.OPENAI_API_KEY or config.HF_TOKEN)


def llm_alive() -> bool:
    backend = config.effective_llm_backend()
    if backend == "extractive":
        return True  # built-in engine is always "up"
    if backend == "ollama":
        return ollama_alive()
    return openai_backend_ready()


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def generate(prompt: str, system: str | None = None, model: str | None = None) -> str:
    """One-shot completion via the configured backend. Raises LLMError on failure."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            if config.effective_llm_backend() == "openai":
                return _openai_generate(prompt, system, model)
            return _ollama_generate(prompt, system, model)
        except LLMError as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1 and _transient(exc):
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    raise last_exc  # pragma: no cover


# --------------------------------------------------------------------------
# Ollama backend
# --------------------------------------------------------------------------
def _ollama_generate(prompt: str, system: str | None, model: str | None) -> str:
    payload = {
        "model": model or config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": 8192,
            "num_predict": 512,
        },
    }
    if system:
        payload["system"] = system
    result = _post_json(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate", payload)
    answer = (result.get("response") or "").strip()
    if not answer:
        raise LLMError("Ollama returned an empty response.")
    return answer


# --------------------------------------------------------------------------
# OpenAI-compatible backend (OpenAI / Groq / OpenRouter / HF router / ...)
# --------------------------------------------------------------------------
def _openai_generate(prompt: str, system: str | None, model: str | None) -> str:
    api_key = config.OPENAI_API_KEY or config.HF_TOKEN
    if not api_key:
        raise LLMError(
            "No LLM API key configured. Set RAG_OPENAI_API_KEY / OPENAI_API_KEY "
            "(or GROQ_API_KEY / OPENROUTER_API_KEY)."
        )

    # Explicit OpenAI-compatible key -> its base URL. HF-token-only ->
    # route through the HF Inference router so RAG_LLM_BACKEND=hf works
    # out of the box instead of hitting api.openai.com with an HF token.
    if config.OPENAI_API_KEY:
        base_url = config.OPENAI_BASE_URL
        default_model = config.OPENAI_LLM_MODEL
    else:
        base_url = config.HF_BASE_URL
        default_model = config.HF_LLM_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or default_model,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": 512,
    }
    result = _post_json(
        f"{base_url}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        answer = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM API response: {result}") from exc
    if not answer:
        raise LLMError("LLM API returned an empty response.")
    return answer
