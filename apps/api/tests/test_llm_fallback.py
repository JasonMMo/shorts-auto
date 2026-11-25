from types import SimpleNamespace

import pytest

from app.core.llm import (
    ClaudeClientWithFallback,
    OpenAIFallback,
    claude_json,
    claude_text,
    is_claude_limit_error,
)


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


class FakeClaude:
    def __init__(self, error: Exception | None = None, text: str = "claude-ok"):
        self.error = error
        self.text = text
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=self.text)],
        )


class FakeOpenAI:
    def __init__(self, text: str = "openai-ok"):
        self.text = text
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(claude: FakeClaude, openai: FakeOpenAI) -> ClaudeClientWithFallback:
    return ClaudeClientWithFallback(claude, OpenAIFallback("sk-test", "gpt-test", client=openai))


def test_uses_claude_when_no_error():
    claude, openai = FakeClaude(), FakeOpenAI()

    assert claude_text(make_client(claude, openai), "hi") == "claude-ok"
    assert openai.requests == []


def test_falls_back_to_openai_on_rate_limit():
    claude, openai = FakeClaude(error=FakeAPIError(429)), FakeOpenAI()

    result = claude_text(make_client(claude, openai), "hi", system="be brief", max_tokens=500)

    assert result == "openai-ok"
    request = openai.requests[0]
    assert request["model"] == "gpt-test"
    assert request["max_completion_tokens"] == 500
    assert request["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_falls_back_on_exhausted_credit_balance():
    error = FakeAPIError(400, "Your credit balance is too low to access the Anthropic API.")
    claude, openai = FakeClaude(error=error), FakeOpenAI(text='{"ok": true}')

    assert claude_json(make_client(claude, openai), "hi") == {"ok": True}


def test_other_errors_are_not_masked_by_fallback():
    claude, openai = FakeClaude(error=FakeAPIError(500, "server error")), FakeOpenAI()

    with pytest.raises(FakeAPIError):
        claude_text(make_client(claude, openai), "hi")
    assert openai.requests == []


def test_rate_limit_without_fallback_raises():
    claude = FakeClaude(error=FakeAPIError(429))

    with pytest.raises(FakeAPIError):
        claude_text(claude, "hi")


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (429, "rate limited", True),
        (400, "Your credit balance is too low", True),
        (400, "invalid request", False),
        (529, "overloaded", False),
        (None, "boom", False),
    ],
)
def test_is_claude_limit_error(status, message, expected):
    assert is_claude_limit_error(FakeAPIError(status, message)) is expected
