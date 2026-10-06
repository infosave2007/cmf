"""Pure helpers for the Spark-X2.5 CMF Space.

Kept dependency-free so the request contract can be checked without downloading
a model or starting a Gradio server.
"""
from __future__ import annotations

import re
from typing import Any

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "Spark 1.7B · fastest": {
        "file": "Spark-X2.5-1.7B-q4mix.cmf",
        "size": "1.36 GB",
        "summary": "Smallest download; the default for a shared Space.",
    },
    "Spark 1.7B · detail": {
        "file": "Spark-X2.5-1.7B-q8_2f.cmf",
        "size": "1.73 GB",
        "summary": "Higher-fidelity 1.7B weights for smaller machines.",
    },
    "Spark 4B · balanced": {
        "file": "Spark-X2.5-4B-q4mix.cmf",
        "size": "3.23 GB",
        "summary": "More capacity with a smaller GPU-friendly file.",
    },
    "Spark 4B · quality": {
        "file": "Spark-X2.5-4B-q8_2f.cmf",
        "size": "4.14 GB",
        "summary": "The model-card default; practically lossless quantization.",
    },
}

PRESETS: dict[str, dict[str, float | int]] = {
    "Fast": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256},
    "Balanced": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 512},
    "Deep": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 1024},
}

SYSTEM_MESSAGE = (
    "You are Spark-X2.5 running locally through Cortiq. Be helpful, precise, "
    "and answer in the user's language. Do not claim to have performed an "
    "external action unless a tool result was provided."
)

_THINK = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.IGNORECASE | re.DOTALL)


def model_id(variant: str) -> str:
    """Return the API model id from a visible variant name."""
    try:
        filename = MODEL_VARIANTS[variant]["file"]
    except KeyError as exc:
        raise ValueError("Choose one of the listed Spark variants.") from exc
    return filename.removesuffix(".cmf")


def trim_history(history: list[dict[str, Any]] | None, limit: int = 12) -> list[dict[str, str]]:
    """Turn Gradio message history into a bounded OpenAI-compatible history."""
    valid: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        valid.append({"role": role, "content": content})
    return valid[-limit:]


def build_payload(
    *,
    variant: str,
    preset: str,
    thinking: bool,
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the documented OpenAI-style request without hidden defaults."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write a message first.")
    if len(prompt) > 12_000:
        raise ValueError("For a reliable shared demo, keep one message below 12,000 characters.")
    try:
        sampling = PRESETS[preset]
    except KeyError as exc:
        raise ValueError("Choose Fast, Balanced, or Deep.") from exc

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages.extend(trim_history(history))
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model_id(variant),
        "messages": messages,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "max_tokens": sampling["max_tokens"],
        "enable_thinking": thinking,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def split_thinking(content: str) -> tuple[str, str]:
    """Keep the chat legible while preserving a reasoning trace on request."""
    content = (content or "").strip()
    match = _THINK.fullmatch(content)
    if not match:
        return "", content
    return match.group(1).strip(), match.group(2).strip()


def response_parts(response: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Extract answer, tool calls and usage from a chat-completions response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("The runtime returned no completion.")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        calls = []
    usage = response.get("usage")
    return content, calls, usage if isinstance(usage, dict) else {}
