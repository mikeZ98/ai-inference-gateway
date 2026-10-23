"""Universal adapter for OpenAI-format ``POST /chat/completions`` APIs."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Final, NotRequired, TypedDict

import httpx
from pydantic import Field, SecretStr, TypeAdapter, ValidationError

from ai_inference_gateway.core.ports import ProviderCompletion
from ai_inference_gateway.schemas import InferenceRequest, ModelProvider, RouteTarget
from ai_inference_gateway.tools.providers.base import DEFAULT_TIMEOUT_S, BaseProvider
from ai_inference_gateway.tools.providers.messages import to_transcript

DEFAULT_BASE_URLS: Final[Mapping[ModelProvider, str]] = MappingProxyType(
    {
        ModelProvider.OPENROUTER: "https://openrouter.ai/api/v1",
        ModelProvider.MISTRAL: "https://api.mistral.ai/v1",
        ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1",
        ModelProvider.GROQ: "https://api.groq.com/openai/v1",
        ModelProvider.TOGETHER: "https://api.together.xyz/v1",
        ModelProvider.OPENAI: "https://api.openai.com/v1",
        ModelProvider.VLLM: "http://localhost:8000/v1",
        ModelProvider.OLLAMA: "http://localhost:11434/v1",
    }
)


class _Message(TypedDict):
    content: str | None


class _Choice(TypedDict):
    message: _Message


class _Usage(TypedDict):
    prompt_tokens: NotRequired[int]
    completion_tokens: NotRequired[int]
    total_tokens: NotRequired[int]


class _ChatCompletion(TypedDict):
    model: NotRequired[str]
    choices: Annotated[list[_Choice], Field(min_length=1)]
    usage: NotRequired[_Usage | None]


_RESPONSE_ADAPTER: Final = TypeAdapter(_ChatCompletion)


class OpenAICompatibleProvider(BaseProvider):
    """Adapter for OpenRouter, Mistral, DeepSeek, Groq, Together, OpenAI, vLLM, Ollama, ...

    ``api_key`` may be ``None`` for unauthenticated local servers (vLLM, Ollama).
    ``extra_headers`` carries optional provider extras such as OpenRouter's
    ``HTTP-Referer`` / ``X-Title``.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        provider: ModelProvider,
        base_url: str | None = None,
        api_key: SecretStr | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the endpoint; ``base_url`` defaults to the provider's public API."""
        resolved = base_url or DEFAULT_BASE_URLS.get(provider)
        if resolved is None:
            raise ValueError(f"base_url is required for provider {provider.value!r}")
        super().__init__(
            client, provider=provider, base_url=resolved, api_key=api_key, timeout_s=timeout_s
        )
        self._extra_headers = dict(extra_headers or {})

    def _path(self) -> str:
        return "/chat/completions"

    def _auth_headers(self) -> Mapping[str, str]:
        headers = dict(self._extra_headers)
        if self._api_key is not None:
            headers["authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        return headers

    def _payload(self, request: InferenceRequest, target: RouteTarget) -> dict[str, Any]:
        transcript = to_transcript(request.prompt)
        messages: list[dict[str, str]] = []
        if transcript.system:
            messages.append({"role": "system", "content": transcript.system})
        messages.extend({"role": m.role, "content": m.content} for m in transcript.messages)
        payload: dict[str, Any] = {"model": target.model, "messages": messages, "stream": False}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _parse(self, body: bytes, target: RouteTarget) -> ProviderCompletion:
        try:
            parsed = _RESPONSE_ADAPTER.validate_json(body, strict=True)
        except ValidationError as exc:
            raise self._error(f"{self.provider.value}: malformed completion response") from exc

        content = parsed["choices"][0]["message"]["content"]
        if content is None:
            raise self._error(f"{self.provider.value}: completion contained no text content")

        usage = parsed.get("usage") or {}
        tokens = usage.get("total_tokens")
        if tokens is None:
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return ProviderCompletion(
            content=content,
            model_used=self._resolve_model(parsed.get("model"), target),
            provider=self.provider,
            tokens_used=max(0, tokens),
        )
