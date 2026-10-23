"""Request lifecycle orchestration: cache -> complexity -> route -> dispatch -> cache."""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from ai_inference_gateway.core.embedding import EmbeddingClient
from ai_inference_gateway.core.errors import CacheError, EmbeddingError, ProviderError
from ai_inference_gateway.core.ports import (
    ComplexityAssessor,
    ProviderDispatcher,
    Router,
    SemanticCache,
)
from ai_inference_gateway.schemas import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheEntry,
    CacheQuery,
    Embedding,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InferenceEngine:
    """Coordinates one inference request end to end.

    The semantic cache is best-effort: embedding or cache failures are logged
    and the request falls through to the provider (fail-open). Routing and
    provider failures propagate as ``RoutingError`` / ``ProviderError``.

    Requests with an explicit ``model`` override bypass the cache entirely, so
    a response produced by a different model is never served for them.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingClient,
        cache: SemanticCache,
        assessor: ComplexityAssessor,
        router: Router,
        dispatcher: ProviderDispatcher,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        clock: Callable[[], datetime] = _utcnow,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Wire the engine to its collaborators."""
        # Validate the threshold now rather than on the first request.
        CacheQuery(embedding=(0.0,), similarity_threshold=similarity_threshold)
        self._embedder = embedder
        self._cache = cache
        self._assessor = assessor
        self._router = router
        self._dispatcher = dispatcher
        self._similarity_threshold = similarity_threshold
        self._clock = clock
        self._timer = timer

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Serve ``request`` from the semantic cache or an upstream provider."""
        started = self._timer()
        embedding = await self._embed(request) if request.model is None else None

        if embedding is not None:
            hit = await self._lookup(embedding)
            if hit is not None:
                return InferenceResponse(
                    content=hit.response,
                    model_used=hit.model_used,
                    provider=hit.provider,
                    tokens_used=0,
                    cached_hit=True,
                    latency_ms=self._elapsed_ms(started),
                )

        assessment = self._assessor.assess(request.prompt)
        target = self._router.resolve(request, assessment)
        completion = await self._dispatcher.complete(target, request)
        if completion.provider is not target.provider:
            raise ProviderError(
                f"dispatcher answered from {completion.provider.value!r}, "
                f"expected {target.provider.value!r}",
                provider=target.provider,
            )
        logger.info(
            "routed request",
            extra={
                "tier": target.tier.value,
                "provider": target.provider.value,
                "complexity_score": assessment.score,
                "heuristics": assessment.matched_heuristics,
            },
        )

        if embedding is not None and completion.content:
            await self._store(
                CacheEntry(
                    prompt=request.prompt,
                    embedding=embedding,
                    response=completion.content,
                    model_used=completion.model_used,
                    provider=completion.provider,
                    timestamp=self._clock(),
                )
            )

        return InferenceResponse(
            content=completion.content,
            model_used=completion.model_used,
            provider=completion.provider,
            tokens_used=completion.tokens_used,
            cached_hit=False,
            latency_ms=self._elapsed_ms(started),
        )

    async def _embed(self, request: InferenceRequest) -> Embedding | None:
        try:
            return await self._embedder.embed(request.prompt)
        except EmbeddingError:
            logger.warning("embedding failed; bypassing semantic cache", exc_info=True)
            return None

    async def _lookup(self, embedding: Embedding) -> CacheEntry | None:
        query = CacheQuery(embedding=embedding, similarity_threshold=self._similarity_threshold)
        try:
            entry = await self._cache.lookup(query)
        except CacheError:
            logger.warning("semantic cache lookup failed; treating as miss", exc_info=True)
            return None
        # Enforce the strict '>' threshold here regardless of adapter semantics.
        if entry is None or entry.similarity_score is None:
            return None
        if not query.is_hit(entry.similarity_score):
            return None
        logger.info("semantic cache hit", extra={"similarity": entry.similarity_score})
        return entry

    async def _store(self, entry: CacheEntry) -> None:
        try:
            await self._cache.store(entry)
        except CacheError:
            logger.warning("semantic cache store failed; response still served", exc_info=True)

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._timer() - started) * 1000.0)
