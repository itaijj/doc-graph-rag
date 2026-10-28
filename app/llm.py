"""Shared OpenAI-compatible LLM client factory and JSON-call helpers.

The backbone LLM does NOT have to be an OpenAI model: any endpoint that speaks
the OpenAI chat-completions API works (OpenAI, Azure OpenAI compatibility mode,
vLLM, Ollama's OpenAI endpoint, LiteLLM proxy, OpenRouter, ...). Configure:

    OPENAI_BASE_URL=https://api.openai.com/v1   # or http://my-vllm:8000/v1, etc.
    OPENAI_MODEL=<any model name the endpoint serves>
    OPENAI_API_KEY=<key, or any dummy value for local servers>
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    """Build an OpenAI client pointing at the configured (possibly non-OpenAI) endpoint."""
    s = get_settings()
    # Local servers (vLLM/Ollama) accept any key; OpenAI requires a real one.
    api_key = s.openai_api_key or "not-set"
    return OpenAI(api_key=api_key, base_url=s.openai_base_url, timeout=s.openai_timeout)


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from free-form model output."""
    text = _strip_fences(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def chat_json(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    """Chat completion that expects a JSON object as the reply.

    Tries JSON mode (`response_format`) first; if the endpoint rejects it
    (not all OpenAI-compatible servers support it), falls back to a plain
    call with regex extraction of the JSON payload.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        data = _extract_json(resp.choices[0].message.content or "")
        if data is not None:
            return data
        logger.warning("JSON mode returned unparseable output; falling back to plain call")
    except Exception as exc:
        logger.warning("JSON mode unsupported by endpoint (%s); falling back to plain call", exc)

    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return data if data is not None else {}
