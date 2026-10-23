"""Template-method base for HTTP provider adapters.

Subclasses only describe *what* to send and how to read the answer; transport,
status-code mapping and secret redaction live here so every adapter fails the
same way: with a ``ProviderError`` that never contains credentials.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, Final

import httpx
from pydantic import SecretStr

from ai_inference_gateway.core.errors import ProviderError
from ai_inference_gateway.core.ports import ProviderCompletion
from ai_inference_gateway.schemas import InferenceRequest, ModelProvider, RouteTarget
from ai_inference_gateway.schemas.inference import MODEL_ID_PATTERN

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: Final = 60.0
MAX_ERROR_DETAIL_CHARS: Final = 200

_MODEL_ID_RE: Final = re.compile(MODEL_ID_PATTERN)
# Generic credential shapes (sk-..., sk-ant-..., sk-or-v1-..., gsk_..., Bearer ...).
_KEY_LIKE_RE: Final = re.compile(
    r"(?:\b(?:sk|pk|rk|gsk|xai)[-_][A-Za-z0-9_\-]{8,})|(?:Bearer\s+\S+)", re.IGNORECASE
)
_RETRYABLE_STATUSES: Final = frozenset(
    {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.CONFLICT, HTTPStatus.TOO_MANY_REQUESTS}
)


def normalize_base_url(url: str) -> str:
    """Canonical form used for endpoint matching (no trailing slash)."""
    return url.rstrip("/")


class BaseProvider(ABC):
    """Async provider adapter: ``complete(request, target) -> ProviderCompletion``."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        provider: ModelProvider,
        base_url: str,
        api_key: SecretStr | None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Bind the adapter to a shared HTTP client, provider identity and endpoint."""
        self._client = client
        self._provider = provider
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout_s = timeout_s

    def __repr__(self) -> str:
        """Identify the adapter without leaking credentials."""
        return (
            f"{type(self).__name__}(provider={self._provider.value!r}, base_url={self._base_url!r})"
        )

    @property
    def provider(self) -> ModelProvider:
        """Provider this adapter speaks for."""
        return self._provider

    @property
    def base_url(self) -> str:
        """Normalized endpoint base URL."""
        return self._base_url

    async def complete(self, request: InferenceRequest, target: RouteTarget) -> ProviderCompletion:
        """Run ``request`` against ``target.model`` and normalize the result."""
        if target.provider is not self._provider:
            raise ProviderError(
                f"{self!r} cannot serve provider {target.provider.value!r}",
                provider=target.provider,
            )
        body = await self._post(self._path(), self._payload(request, target))
        return self._parse(body, target)

    @abstractmethod
    def _path(self) -> str:
        """Endpoint path appended to the base URL."""

    @abstractmethod
    def _auth_headers(self) -> Mapping[str, str]:
        """Provider-specific authentication/version headers."""

    @abstractmethod
    def _payload(self, request: InferenceRequest, target: RouteTarget) -> dict[str, Any]:
        """JSON request body."""

    @abstractmethod
    def _parse(self, body: bytes, target: RouteTarget) -> ProviderCompletion:
        """Translate a 2xx response body; raise ``ProviderError`` if unusable."""

    def _resolve_model(self, reported: str | None, target: RouteTarget) -> str:
        """Prefer the model the provider reports (e.g. OpenRouter slugs) when well-formed."""
        if reported and len(reported) <= 128 and _MODEL_ID_RE.fullmatch(reported):  # noqa: PLR2004
            return reported
        return target.model

    def _error(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> ProviderError:
        return ProviderError(
            self._redact(message),
            provider=self._provider,
            status_code=status_code,
            retryable=retryable,
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> bytes:
        name = self._provider.value
        # Transport errors are raised 'from None': httpx exceptions carry the
        # request (incl. auth headers), which must never reach tracebacks/trackers.
        try:
            response = await self._client.post(
                f"{self._base_url}{path}",
                json=payload,
                headers={"content-type": "application/json", **self._auth_headers()},
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException:
            raise self._error(f"{name}: request timed out", retryable=True) from None
        except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise self._error(
                f"{name}: connection failed ({type(exc).__name__})", retryable=True
            ) from None
        except httpx.HTTPError as exc:
            raise self._error(f"{name}: transport error ({type(exc).__name__})") from None

        status = response.status_code
        if response.is_success:
            return response.content

        detail = self._error_detail(response.content)
        if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            message, retryable = f"{name}: authentication rejected (HTTP {status})", False
        elif status == HTTPStatus.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after", "unspecified")
            message = f"{name}: rate limited (HTTP 429, retry-after={retry_after})"
            retryable = True
        elif status >= HTTPStatus.INTERNAL_SERVER_ERROR or status in _RETRYABLE_STATUSES:
            message, retryable = f"{name}: upstream unavailable (HTTP {status})", True
        else:
            message, retryable = f"{name}: request rejected (HTTP {status})", False
        if detail:
            message = f"{message}: {detail}"
        logger.warning(
            "provider call failed",
            extra={"provider": name, "status_code": status, "retryable": retryable},
        )
        raise self._error(message, status_code=status, retryable=retryable)

    def _error_detail(self, body: bytes) -> str:
        """Best-effort short upstream error message (redacted, truncated)."""
        try:
            parsed: object = json.loads(body)
        except ValueError:
            text = body.decode(errors="replace").strip()
        else:
            text = ""
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                text = error["message"]
            elif isinstance(error, str):
                text = error
            elif isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
                text = parsed["message"]
        return self._redact(text)[:MAX_ERROR_DETAIL_CHARS]

    def _redact(self, text: str) -> str:
        if self._api_key is not None:
            secret = self._api_key.get_secret_value()
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return _KEY_LIKE_RE.sub("[REDACTED]", text)
