"""Pure-ASGI ingress guards: body size limit, client IP resolution, rate limiting.

Pure ASGI (not ``BaseHTTPMiddleware``) keeps streaming responses, client
disconnect propagation and request-body streaming intact.
"""

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv4Network, IPv6Network, ip_address, ip_network
from typing import Final

from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_inference_gateway.guardrails.metrics import GuardrailMetrics
from ai_inference_gateway.guardrails.rate_limiter import ClientAddress, RedisTokenBucketLimiter
from ai_inference_gateway.schemas import ErrorDetail, ErrorResponse, TokenBucketStatus

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_BYTES: Final = 1_048_576
# 0.0.0.0: shared bucket for peers without a usable address (a key, never bound).
UNKNOWN_CLIENT: Final = IPv4Address(0)
MAX_FORWARDED_HOPS: Final = 32

type TrustedNetworks = tuple[IPv4Network | IPv6Network, ...]


def parse_networks(values: Iterable[str]) -> TrustedNetworks:
    """Parse CIDRs / bare addresses (``10.0.0.0/8``, ``127.0.0.1``)."""
    return tuple(ip_network(value.strip(), strict=False) for value in values if value.strip())


def _parse_hop(raw: str) -> ClientAddress | None:
    """One ``X-Forwarded-For`` entry; tolerates ``[v6]:port`` and ``v4:port`` forms."""
    value = raw.strip()
    if value.startswith("["):
        value = value[1 : value.find("]")] if "]" in value else ""
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    try:
        return ip_address(value)
    except ValueError:
        return None


def resolve_client_ip(scope: Scope, trusted: TrustedNetworks) -> ClientAddress:
    """Real client address, honoring ``X-Forwarded-For`` only via trusted proxies.

    Walks the chain right-to-left from the socket peer, skipping trusted proxy
    hops; the first untrusted hop is the client. Spoofed left-most entries are
    therefore ignored, and a malformed hop stops the walk at the last trusted
    address. Unknown peers share the ``0.0.0.0`` bucket rather than bypassing
    limits.
    """
    client = scope.get("client")
    current = _parse_hop(client[0]) if client else None
    if current is None:
        return UNKNOWN_CLIENT

    def is_trusted(address: ClientAddress) -> bool:
        return any(address in network for network in trusted)

    if not trusted or not is_trusted(current):
        return current
    forwarded = ",".join(Headers(scope=scope).getlist("x-forwarded-for"))
    hops = [hop for hop in forwarded.split(",") if hop.strip()][-MAX_FORWARDED_HOPS:]
    for raw in reversed(hops):
        hop = _parse_hop(raw)
        if hop is None:
            return current
        if not is_trusted(hop):
            return hop
        current = hop
    return current


def _error(status: int, detail: ErrorDetail, headers: dict[str, str] | None = None) -> JSONResponse:
    body = ErrorResponse(error=detail).model_dump(mode="json")
    return JSONResponse(status_code=status, content=body, headers=headers)


class BodySizeLimitMiddleware:
    """Rejects request bodies over ``max_bytes`` with 413 before deserialization.

    A declared ``Content-Length`` is checked up front; chunked or understated
    bodies are counted as they stream and aborted once the limit is crossed.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = DEFAULT_MAX_BODY_BYTES,
        metrics: GuardrailMetrics | None = None,
    ) -> None:
        """Wrap ``app``."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self._max = max_bytes
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the limit for HTTP requests; pass everything else through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if not declared.isdigit():
                await _error(
                    400, ErrorDetail(code="bad_request", message="Invalid Content-Length.")
                )(scope, receive, send)
                return
            if int(declared) > self._max:
                self._rejected()
                await self._too_large()(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max:
                    self._rejected()
                    # FastAPI re-raises HTTPException from body reading; the
                    # app's handler renders it as a typed 413 ErrorResponse.
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)

    def _rejected(self) -> None:
        if self._metrics is not None:
            self._metrics.record_payload_rejected()

    def _too_large(self) -> JSONResponse:
        return _error(
            413,
            ErrorDetail(
                code="content_too_large",
                message=f"Request body exceeds {self._max} bytes.",
            ),
        )


@dataclass(slots=True)
class RateLimitState:
    """Mutable holder the lifespan fills once Redis is connected."""

    limiter: RedisTokenBucketLimiter | None = None
    metrics: GuardrailMetrics = field(default_factory=GuardrailMetrics)


def rate_limit_headers(
    status: TokenBucketStatus, limiter: RedisTokenBucketLimiter
) -> dict[str, str]:
    """``X-RateLimit-*`` headers (``Reset`` is a Unix timestamp, like GitHub's API)."""
    return {
        "x-ratelimit-limit": str(limiter.policy.capacity),
        "x-ratelimit-remaining": str(math.floor(status.tokens_remaining)),
        "x-ratelimit-reset": str(math.ceil(status.reset_at.timestamp())),
    }


class RateLimitMiddleware:
    """Applies the per-IP token bucket to the configured paths.

    Allowed responses carry ``X-RateLimit-*`` headers; exhausted buckets get a
    429 ``ErrorResponse`` plus ``Retry-After``. If the limiter is missing or
    Redis fails, requests pass through (fail-open) without rate-limit headers.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        state: RateLimitState,
        paths: Iterable[str] = ("/v1/inference",),
        trusted_proxies: TrustedNetworks = (),
    ) -> None:
        """Wrap ``app``."""
        self.app = app
        self._state = state
        self._paths = frozenset(paths)
        self._trusted = trusted_proxies

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Consume a token for protected paths, then forward or reject."""
        if scope["type"] != "http" or scope["path"] not in self._paths:
            await self.app(scope, receive, send)
            return

        address = resolve_client_ip(scope, self._trusted)
        scope.setdefault("state", {})["client_ip"] = str(address)
        limiter = self._state.limiter
        if limiter is None:  # rate limiting disabled
            await self.app(scope, receive, send)
            return
        status = await limiter.check(address)
        self._state.metrics.record_rate_limit(allowed=None if status is None else status.allowed)
        if status is None:  # Redis unavailable: fail open
            await self.app(scope, receive, send)
            return

        headers = rate_limit_headers(status, limiter)
        if not status.allowed:
            retry_after = limiter.retry_after_s(status)
            logger.info(
                "rate limited", extra={"client_ip": str(address), "retry_after_s": retry_after}
            )
            response = _error(
                429,
                ErrorDetail(code="rate_limited", message="Rate limit exceeded.", retryable=True),
                headers={**headers, "retry-after": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                mutable = MutableHeaders(scope=message)
                for name, value in headers.items():
                    mutable[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)
