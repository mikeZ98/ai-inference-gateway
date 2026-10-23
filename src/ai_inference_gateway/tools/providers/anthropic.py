"""Adapter for the Anthropic Messages API (``POST /v1/messages``)."""

from collections.abc import Iterable, Mapping
from typing import Any, Final, TypedDict

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError

from ai_inference_gateway.core.errors import ProviderError
from ai_inference_gateway.core.ports import ProviderCompletion, StreamCompleted, TextDelta
from ai_inference_gateway.schemas import InferenceRequest, ModelProvider, RouteTarget
from ai_inference_gateway.tools.providers.base import DEFAULT_TIMEOUT_S, BaseProvider, StreamParser
from ai_inference_gateway.tools.providers.messages import to_transcript

ANTHROPIC_BASE_URL: Final = "https://api.anthropic.com"
ANTHROPIC_VERSION: Final = "2023-06-01"
DEFAULT_MAX_TOKENS: Final = 4096
MAX_TEMPERATURE: Final = 1.0
RETRYABLE_STREAM_ERRORS: Final = frozenset({"overloaded_error", "api_error", "rate_limit_error"})


class _ContentBlock(TypedDict, total=False):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _Message(TypedDict, total=False):
    model: str
    content: list[_ContentBlock]
    usage: _Usage


_RESPONSE_ADAPTER: Final = TypeAdapter(_Message)


class _StreamUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int


class _StreamMessage(TypedDict, total=False):
    model: str
    usage: _StreamUsage


class _StreamDelta(TypedDict, total=False):
    type: str
    text: str


class _StreamError(TypedDict, total=False):
    type: str
    message: str


class _StreamEvent(TypedDict, total=False):
    type: str
    message: _StreamMessage
    delta: _StreamDelta
    usage: _StreamUsage
    error: _StreamError


_EVENT_ADAPTER: Final = TypeAdapter(_StreamEvent)


class _AnthropicStreamParser(StreamParser):
    """Parses Messages streaming events (``message_start`` ... ``message_stop``)."""

    def __init__(self, adapter: "AnthropicProvider", target: RouteTarget) -> None:
        self._adapter = adapter
        self._target = target
        self._model: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._stopped = False

    def feed(self, event: str | None, data: str) -> Iterable[TextDelta]:
        try:
            parsed = _EVENT_ADAPTER.validate_json(data, strict=True)
        except ValidationError as exc:
            raise self._adapter._error("anthropic: malformed stream event") from exc
        kind = parsed.get("type", event)
        if kind == "message_start":
            message = parsed.get("message", {})
            self._model = message.get("model")
            usage = message.get("usage", {})
            self._input_tokens = usage.get("input_tokens", 0)
            self._output_tokens = usage.get("output_tokens", 0)
        elif kind == "content_block_delta":
            delta = parsed.get("delta", {})
            text = delta.get("text")
            if delta.get("type") == "text_delta" and text:
                return (TextDelta(text=text),)
        elif kind == "message_delta":
            self._output_tokens = parsed.get("usage", {}).get("output_tokens", self._output_tokens)
        elif kind == "message_stop":
            self._stopped = True
        elif kind == "error":
            error = parsed.get("error", {})
            error_type = error.get("type", "unknown_error")
            raise self._adapter._error(
                f"anthropic: stream error ({error_type}): {error.get('message', '')}",
                retryable=error_type in RETRYABLE_STREAM_ERRORS,
            )
        return ()

    def finish(self) -> StreamCompleted:
        adapter = self._adapter
        if not self._stopped:
            raise ProviderError(
                "anthropic: stream ended before message_stop",
                provider=adapter.provider,
                retryable=True,
            )
        return StreamCompleted(
            model_used=adapter._resolve_model(self._model, self._target),
            provider=adapter.provider,
            tokens_used=max(0, self._input_tokens + self._output_tokens),
        )


class AnthropicProvider(BaseProvider):
    """Claude adapter.

    - Authenticates with ``x-api-key`` and pins ``anthropic-version``.
    - Sends system turns via the top-level ``system`` field (no system role).
    - Sends strictly alternating user/assistant messages starting with ``user``.
    - Always sets ``max_tokens`` (required by the API) and clamps
      ``temperature`` to Anthropic's ``[0, 1]`` range.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: SecretStr,
        base_url: str = ANTHROPIC_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        anthropic_version: str = ANTHROPIC_VERSION,
    ) -> None:
        """Configure credentials, endpoint and request defaults."""
        if default_max_tokens < 1:
            raise ValueError("default_max_tokens must be positive")
        super().__init__(
            client,
            provider=ModelProvider.ANTHROPIC,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        self._key = api_key
        self._default_max_tokens = default_max_tokens
        self._anthropic_version = anthropic_version

    def _path(self) -> str:
        return "/v1/messages"

    def _auth_headers(self) -> Mapping[str, str]:
        return {
            "x-api-key": self._key.get_secret_value(),
            "anthropic-version": self._anthropic_version,
        }

    def _payload(self, request: InferenceRequest, target: RouteTarget) -> dict[str, Any]:
        transcript = to_transcript(request.prompt)
        payload: dict[str, Any] = {
            "model": target.model,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in transcript.messages],
        }
        if transcript.system:
            payload["system"] = transcript.system
        if request.temperature is not None:
            payload["temperature"] = min(request.temperature, MAX_TEMPERATURE)
        return payload

    def _stream_parser(self, target: RouteTarget) -> StreamParser:
        return _AnthropicStreamParser(self, target)

    def _parse(self, body: bytes, target: RouteTarget) -> ProviderCompletion:
        try:
            parsed = _RESPONSE_ADAPTER.validate_json(body, strict=True)
        except ValidationError as exc:
            raise self._error("anthropic: malformed messages response") from exc

        texts = [
            block.get("text", "")
            for block in parsed.get("content", [])
            if block.get("type") == "text"
        ]
        if not texts:
            raise self._error("anthropic: response contained no text content")

        usage = parsed.get("usage")
        tokens = usage["input_tokens"] + usage["output_tokens"] if usage else 0
        return ProviderCompletion(
            content="".join(texts),
            model_used=self._resolve_model(parsed.get("model"), target),
            provider=self.provider,
            tokens_used=max(0, tokens),
        )
