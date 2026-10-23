"""Interfaces the engine depends on; concrete adapters live outside ``core``.

Implementations must raise only :mod:`ai_inference_gateway.core.errors` types.
"""

from typing import Protocol

from pydantic import Field

from ai_inference_gateway.schemas import (
    CacheEntry,
    CacheQuery,
    ComplexityAssessment,
    InferenceRequest,
    ModelProvider,
    RouteTarget,
    StrictModel,
)
from ai_inference_gateway.schemas.inference import MODEL_ID_PATTERN


class ProviderCompletion(StrictModel):
    """Raw result of a single upstream completion, before normalization."""

    content: str
    model_used: str = Field(min_length=1, max_length=128, pattern=MODEL_ID_PATTERN)
    provider: ModelProvider
    tokens_used: int = Field(ge=0)


class SemanticCache(Protocol):
    """Vector store of previous prompt/response pairs. Raises ``CacheError``."""

    async def lookup(self, query: CacheQuery) -> CacheEntry | None:
        """Return the best entry for ``query`` (with ``similarity_score`` set), if any."""
        ...

    async def store(self, entry: CacheEntry) -> None:
        """Persist ``entry`` for future lookups."""
        ...


class ComplexityAssessor(Protocol):
    """Scores prompt complexity. Pure and CPU-bound."""

    def assess(self, prompt: str) -> ComplexityAssessment:
        """Evaluate ``prompt``."""
        ...


class Router(Protocol):
    """Maps a request and its complexity to a route target. Raises ``RoutingError``."""

    def resolve(self, request: InferenceRequest, assessment: ComplexityAssessment) -> RouteTarget:
        """Choose the tier, provider and model for ``request``."""
        ...


class ProviderDispatcher(Protocol):
    """Sends a request to the provider named by a route target.

    Raises ``ProviderError`` on upstream failure, ``RoutingError`` when no
    adapter is configured for the target.
    """

    async def complete(self, target: RouteTarget, request: InferenceRequest) -> ProviderCompletion:
        """Run the completion upstream."""
        ...
