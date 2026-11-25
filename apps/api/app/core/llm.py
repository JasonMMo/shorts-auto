"""Claude (Anthropic) 텍스트 생성 헬퍼.

원본 저장소의 OpenAI chat.completions / responses 호출을 대체한다.
모든 LLM 호출은 이 모듈을 거치므로 테스트에서는 클라이언트만 가짜로 바꾸면 된다.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000
# 거절(refusal) 시 서버가 카테고리에 맞는 모델로 자동 재시도
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$")


DEFAULT_OPENAI_FALLBACK_MODEL = "gpt-4.1-mini"


class OpenAIFallback:
    """Claude가 한도에 걸렸을 때만 쓰는 OpenAI chat.completions 대체 경로."""

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_FALLBACK_MODEL, client: Any = None):
        self.model = model or DEFAULT_OPENAI_FALLBACK_MODEL
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def complete(self, prompt: str, *, system: str | None, max_tokens: int) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_completion_tokens=max_tokens
        )
        return (response.choices[0].message.content or "").strip()


class ClaudeClientWithFallback:
    """Anthropic 클라이언트에 OpenAI 폴백을 붙인 래퍼. claude_text가 폴백을 찾아 쓴다."""

    def __init__(self, claude: Any, openai_fallback: OpenAIFallback):
        self.beta = claude.beta
        self.openai_fallback = openai_fallback


def is_claude_limit_error(exc: Exception) -> bool:
    """rate limit(429) 또는 크레딧/사용 한도 소진(400)이면 True."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if status == 400:
        message = str(exc).lower()
        return "credit balance" in message or "usage limit" in message
    return False


def create_claude_client(api_key: str, *, openai_api_key: str = "", openai_model: str = ""):
    if not api_key:
        raise ValueError("Claude API key is missing (set CLAUDE_API_KEY)")
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    if openai_api_key:
        return ClaudeClientWithFallback(client, OpenAIFallback(openai_api_key, openai_model))
    return client


def create_llm_client(settings: Any):
    """설정에서 Claude 클라이언트(+OpenAI 키가 있으면 한도 폴백)를 만든다."""
    return create_claude_client(
        settings.claude_api_key,
        openai_api_key=getattr(settings, "openai_api_key", ""),
        openai_model=getattr(settings, "openai_fallback_model", ""),
    )


def claude_text(
    client: Any,
    prompt: str,
    *,
    model: str = DEFAULT_CLAUDE_MODEL,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """단일 user 프롬프트를 보내고 text 블록만 이어붙여 반환한다."""
    kwargs: dict[str, Any] = {
        "model": model or DEFAULT_CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.beta.messages.create(**kwargs)
    except Exception as exc:
        fallback = getattr(client, "openai_fallback", None)
        if fallback is None or not is_claude_limit_error(exc):
            raise
        logger.warning("Claude limit hit (%s); falling back to OpenAI %s", exc, fallback.model)
        return fallback.complete(prompt, system=system, max_tokens=max_tokens)
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise RuntimeError(f"Claude declined the request (category={category})")
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


def strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def parse_json_text(text: str) -> Any:
    """```json 펜스나 앞뒤 설명 문장이 붙어 있어도 JSON 본문을 파싱한다."""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"Claude response is not JSON: {text[:200]!r}")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError(f"Claude response is not JSON: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def claude_json(
    client: Any,
    prompt: str,
    *,
    model: str = DEFAULT_CLAUDE_MODEL,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    json_system = "Respond with valid JSON only. No markdown fences, no commentary."
    if system:
        json_system = f"{system}\n\n{json_system}"
    return parse_json_text(
        claude_text(client, prompt, model=model, system=json_system, max_tokens=max_tokens)
    )
