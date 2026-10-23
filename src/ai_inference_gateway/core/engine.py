"""Request lifecycle orchestration: cache -> complexity -> route -> dispatch -> cache."""

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from datetime import UTC, datetime

from ai_inference_gateway.core.embedding import EmbeddingClient
from ai_inference_gateway.core.errors import CacheError, EmbeddingError, ProviderError
from ai_inference_gateway.core.ports import (
    ComplexityAssessor,
    ProviderDispatcher,
    Router,
    SemanticCache,
    StreamCompleted,
    TextDelta,
)
from ai_inference_gateway.schemas import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheEntry,
    CacheQuery,
    Embedding,
    InferenceRequest,
    InferenceResponse,
    ModelProvider,
    RouteTarget,
    StreamDelta,
    StreamSummary,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InferenceEngine:
    """Coordinates one inference request end to end, buffered or streamed.

    The semantic cache is best-effort: embedding or cache failures are logged
    and the request falls through to the provider (fail-open). Routing and
    provider failures propagate as ``RoutingError`` / ``ProviderError``.

    Requests with an explicit ``model`` override bypass the cache entirely, so
    a response produced by a different model is never served for them. Only
    complete responses are cached; an interrupted stream stores nothing.
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
        embedding = await self._embed(request)
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

        target = self._route(request)
        completion = await self._dispatcher.complete(target, request)
        self._check_provider(target, completion.provider)
        await self._store(request, embedding, completion.content, completion.model_used, target)
        return InferenceResponse(
            content=completion.content,
            model_used=completion.model_used,
            provider=completion.provider,
            tokens_used=completion.tokens_used,
            cached_hit=False,
            latency_ms=self._elapsed_ms(started),
        )

    async def stream(
        self, request: InferenceRequest
    ) -> AsyncGenerator[StreamDelta | StreamSummary]:
        """Stream ``request``: ``StreamDelta`` events then exactly one ``StreamSummary``.

        Closing this generator closes the upstream stream (via ``aclosing``)
        and skips the cache write.
        """
        started = self._timer()
        embedding = await self._embed(request)
        hit = await self._lookup(embedding)
        if hit is not None:
            if hit.response:
                yield StreamDelta(content=hit.response)
            yield StreamSummary(
                model_used=hit.model_used,
                provider=hit.provider,
                tokens_used=0,
                cached_hit=True,
                latency_ms=self._elapsed_ms(started),
            )
            return

        target = self._route(request)
        parts: list[str] = []
        completed: StreamCompleted | None = None
        async with aclosing(self._dispatcher.stream(target, request)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                    yield StreamDelta(content=event.text)
                else:
                    completed = event
        if completed is None:
            raise ProviderError(
                "stream ended without a completion event",
                provider=target.provider,
                retryable=True,
            )
        self._check_provider(target, completed.provider)
        await self._store(request, embedding, "".join(parts), completed.model_used, target)
        yield StreamSummary(
            model_used=completed.model_used,
            provider=completed.provider,
            tokens_used=completed.tokens_used,
            cached_hit=False,
            latency_ms=self._elapsed_ms(started),
        )

    def _route(self, request: InferenceRequest) -> RouteTarget:
        assessment = self._assessor.assess(request.prompt)
        target = self._router.resolve(request, assessment)
        logger.info(
            "routed request",
            extra={
                "tier": target.tier.value,
                "provider": target.provider.value,
                "complexity_score": assessment.score,
                "heuristics": assessment.matched_heuristics,
            },
        )
        return target

    @staticmethod
    def _check_provider(target: RouteTarget, provider: ModelProvider) -> None:
        if provider is not target.provider:
            raise ProviderError(
                f"dispatcher answered from {provider.value!r}, expected {target.provider.value!r}",
                provider=target.provider,
            )

    async def _embed(self, request: InferenceRequest) -> Embedding | None:
        if request.model is not None:
            return None
        try:
            return await self._embedder.embed(request.prompt)
        except EmbeddingError:
            logger.warning("embedding failed; bypassing semantic cache", exc_info=True)
            return None

    async def _lookup(self, embedding: Embedding | None) -> CacheEntry | None:
        if embedding is None:
            return None
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

    async def _store(
        self,
        request: InferenceRequest,
        embedding: Embedding | None,
        content: str,
        model_used: str,
        target: RouteTarget,
    ) -> None:
        if embedding is None or not content:
            return
        entry = CacheEntry(
            prompt=request.prompt,
            embedding=embedding,
            response=content,
            model_used=model_used,
            provider=target.provider,
            timestamp=self._clock(),
        )
        try:
            await self._cache.store(entry)
        except CacheError:
            logger.warning("semantic cache store failed; response still served", exc_info=True)

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._timer() - started) * 1000.0)
