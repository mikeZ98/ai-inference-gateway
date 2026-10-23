"""Adapter for the Anthropic Messages API (``POST /v1/messages``)."""

from collections.abc import Mapping
from typing import Any, Final, TypedDict

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError

from ai_inference_gateway.core.ports import ProviderCompletion
from ai_inference_gateway.schemas import InferenceRequest, ModelProvider, RouteTarget
from ai_inference_gateway.tools.providers.base import DEFAULT_TIMEOUT_S, BaseProvider
from ai_inference_gateway.tools.providers.messages import to_transcript

ANTHROPIC_BASE_URL: Final = "https://api.anthropic.com"
ANTHROPIC_VERSION: Final = "2023-06-01"
DEFAULT_MAX_TOKENS: Final = 4096
MAX_TEMPERATURE: Final = 1.0


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
