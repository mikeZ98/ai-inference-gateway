"""Domain exception hierarchy.

Adapters translate every infrastructure failure (HTTP, Qdrant, Redis, ...) into
one of these, so callers never handle third-party exception types.
"""

from ai_inference_gateway.schemas import ModelProvider


class GatewayError(Exception):
    """Base class for all gateway domain errors."""


class EmbeddingError(GatewayError):
    """The prompt could not be embedded."""


class CacheError(GatewayError):
    """The semantic cache could not be read or written."""


class RoutingError(GatewayError):
    """No valid route target could be resolved for a request."""


class ProviderError(GatewayError):
    """An upstream model provider failed or returned an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        provider: ModelProvider,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        """Attach provider context to the error."""
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
