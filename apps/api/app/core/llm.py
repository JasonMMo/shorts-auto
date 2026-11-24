"""Claude (Anthropic) 텍스트 생성 헬퍼.

원본 저장소의 OpenAI chat.completions / responses 호출을 대체한다.
모든 LLM 호출은 이 모듈을 거치므로 테스트에서는 클라이언트만 가짜로 바꾸면 된다.
"""
from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000
# 거절(refusal) 시 서버가 카테고리에 맞는 모델로 자동 재시도
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$")


def create_claude_client(api_key: str):
    if not api_key:
        raise ValueError("Claude API key is missing (set CLAUDE_API_KEY)")
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


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
    response = client.beta.messages.create(**kwargs)
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
