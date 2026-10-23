"""Application factory and lifespan wiring.

Run with ``uvicorn ai_inference_gateway.api.app:create_app --factory``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai_inference_gateway import __version__
from ai_inference_gateway.api.errors import install_error_handlers
from ai_inference_gateway.api.routes import GatewayServices, router
from ai_inference_gateway.api.settings import GatewaySettings
from ai_inference_gateway.core.embedding import (
    EmbeddingClient,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
)
from ai_inference_gateway.core.engine import InferenceEngine
from ai_inference_gateway.core.errors import CacheError
from ai_inference_gateway.core.heuristics import HeuristicComplexityAssessor
from ai_inference_gateway.guardrails import (
    BodySizeLimitMiddleware,
    InstrumentedDispatcher,
    InstrumentedRouter,
    RateLimitMiddleware,
    RateLimitState,
    RedisTokenBucketLimiter,
)
from ai_inference_gateway.schemas import ModelProvider, RouteTarget, RoutingTier
from ai_inference_gateway.tools.providers import (
    AnthropicProvider,
    BaseProvider,
    DynamicProviderDispatcher,
    OpenAICompatibleProvider,
)
from ai_inference_gateway.tools.qdrant_cache import QdrantSemanticCache
from ai_inference_gateway.tools.router import ComplexityRouter, RouterConfig

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S: Final = 5.0
MAX_CONNECTIONS: Final = 200
MAX_KEEPALIVE: Final = 50
REDIS_MAX_CONNECTIONS: Final = 100
REDIS_HEALTH_CHECK_S: Final = 30
RATE_LIMITED_PATHS: Final = ("/v1/inference",)


def build_adapters(settings: GatewaySettings, client: httpx.AsyncClient) -> list[BaseProvider]:
    """Instantiate an adapter for every provider that has credentials / an endpoint."""
    timeout = settings.provider_timeout_s
    keyed = (
        (ModelProvider.OPENROUTER, settings.openrouter_api_key),
        (ModelProvider.MISTRAL, settings.mistral_api_key),
        (ModelProvider.DEEPSEEK, settings.deepseek_api_key),
        (ModelProvider.GROQ, settings.groq_api_key),
        (ModelProvider.TOGETHER, settings.together_api_key),
        (ModelProvider.OPENAI, settings.openai_api_key),
    )
    adapters: list[BaseProvider] = [
        OpenAICompatibleProvider(client, provider=provider, api_key=key, timeout_s=timeout)
        for provider, key in keyed
        if key is not None
    ]
    for provider, url in (
        (ModelProvider.VLLM, settings.vllm_base_url),
        (ModelProvider.OLLAMA, settings.ollama_base_url),
    ):
        if url:
            adapters.append(
                OpenAICompatibleProvider(client, provider=provider, base_url=url, timeout_s=timeout)
            )
    if settings.anthropic_api_key is not None:
        adapters.append(
            AnthropicProvider(client, api_key=settings.anthropic_api_key, timeout_s=timeout)
        )
    return adapters


def build_router(settings: GatewaySettings) -> ComplexityRouter:
    """Router from the tier settings."""
    return ComplexityRouter(
        RouterConfig(
            tier1=RouteTarget(
                tier=RoutingTier.TIER_1,
                provider=settings.tier1_provider,
                model=settings.tier1_model,
            ),
            tier2=RouteTarget(
                tier=RoutingTier.TIER_2,
                provider=ModelProvider.ANTHROPIC,
                model=settings.tier2_model,
            ),
            tier3=RouteTarget(
                tier=RoutingTier.TIER_3,
                provider=ModelProvider.ANTHROPIC,
                model=settings.tier3_model,
            ),
            tier3_min_score=settings.tier3_min_score,
        )
    )


def build_embedder(settings: GatewaySettings, client: httpx.AsyncClient) -> EmbeddingClient:
    """Offline hashing embedder, or an OpenAI-compatible embeddings endpoint."""
    base_url, model = settings.embedding_base_url, settings.embedding_model
    # GatewaySettings guarantees base_url/model for the openai_compatible backend.
    if settings.embedding_backend == "hashing" or base_url is None or model is None:
        return HashingEmbedder(dimension=settings.embedding_dimension)
    return OpenAICompatibleEmbedder(
        client,
        base_url=base_url,
        api_key=settings.embedding_api_key,
        model=model,
        dimension=settings.embedding_dimension,
    )


def build_cache(
    settings: GatewaySettings, qdrant_client: AsyncQdrantClient | None
) -> QdrantSemanticCache:
    """Qdrant cache from an injected client, ``:memory:``, or a remote URL."""
    if qdrant_client is not None:
        return QdrantSemanticCache(
            qdrant_client,
            dimension=settings.embedding_dimension,
            collection=settings.qdrant_collection,
        )
    if settings.qdrant_url == ":memory:":
        return QdrantSemanticCache.in_memory(
            dimension=settings.embedding_dimension, collection=settings.qdrant_collection
        )
    return QdrantSemanticCache.remote(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        dimension=settings.embedding_dimension,
        collection=settings.qdrant_collection,
    )


def create_app(
    settings: GatewaySettings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    qdrant_client: AsyncQdrantClient | None = None,
    redis_client: Redis | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``http_client`` / ``qdrant_client`` / ``redis_client`` may be injected
    (e.g. mock transport, ``:memory:`` Qdrant, fakeredis in tests); injected
    clients are not closed by the app.
    """
    resolved = settings or GatewaySettings()
    rate_limit = RateLimitState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            client = http_client
            if client is None:
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(resolved.provider_timeout_s, connect=CONNECT_TIMEOUT_S),
                    limits=httpx.Limits(
                        max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_KEEPALIVE
                    ),
                    follow_redirects=False,
                )
                stack.push_async_callback(client.aclose)

            cache = build_cache(resolved, qdrant_client)
            if qdrant_client is None:
                stack.push_async_callback(_close_cache, cache)
            try:
                await cache.initialize()
            except CacheError:
                # Fail open: inference works without the cache; /ready reports it.
                logger.warning("semantic cache unavailable at startup", exc_info=True)

            dispatcher = DynamicProviderDispatcher(build_adapters(resolved, client))
            if not dispatcher.providers:
                logger.warning("no upstream providers configured; /ready will report not_ready")

            redis = await _open_redis(resolved, redis_client, stack)
            if redis is not None:
                rate_limit.limiter = RedisTokenBucketLimiter(
                    redis, resolved.token_bucket_policy(), timeout_s=resolved.rate_limit_timeout_s
                )
            else:
                logger.warning("rate limiting disabled (no Redis configured)")
            metrics = rate_limit.metrics

            app.state.services = GatewayServices(
                engine=InferenceEngine(
                    embedder=build_embedder(resolved, client),
                    cache=cache,
                    assessor=HeuristicComplexityAssessor(resolved.complexity_policy()),
                    router=InstrumentedRouter(build_router(resolved), metrics),
                    dispatcher=InstrumentedDispatcher(dispatcher, metrics),
                    similarity_threshold=resolved.similarity_threshold,
                ),
                cache=cache,
                dispatcher=dispatcher,
                metrics=metrics,
            )
            logger.info(
                "gateway started",
                extra={"providers": sorted(p.value for p in dispatcher.providers)},
            )
            try:
                yield
            finally:
                rate_limit.limiter = None
                del app.state.services
                logger.info("gateway stopping")

    app = FastAPI(
        title="AI Inference Gateway",
        version=__version__,
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.include_router(router)
    # Last added runs first: rate limit -> body size limit -> app.
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=resolved.max_body_bytes, metrics=rate_limit.metrics
    )
    app.add_middleware(
        RateLimitMiddleware,
        state=rate_limit,
        paths=RATE_LIMITED_PATHS,
        trusted_proxies=resolved.trusted_networks(),
    )
    return app


async def _open_redis(
    settings: GatewaySettings, injected: Redis | None, stack: AsyncExitStack
) -> Redis | None:
    """Injected client, or a pooled client from ``REDIS_URL`` (closed on shutdown)."""
    if not settings.rate_limit_enabled:
        return None
    redis = injected
    if redis is None:
        if settings.redis_url is None:
            return None
        redis = Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.rate_limit_timeout_s,
            socket_connect_timeout=settings.rate_limit_timeout_s,
            health_check_interval=REDIS_HEALTH_CHECK_S,
            max_connections=REDIS_MAX_CONNECTIONS,
        )
        stack.push_async_callback(redis.aclose)
    try:
        async with asyncio.timeout(settings.rate_limit_timeout_s * 4):
            await redis.ping()
    except (RedisError, OSError, TimeoutError):
        # Keep the limiter: each check fails open until Redis comes back.
        logger.warning("Redis unreachable at startup; rate limiter will fail open")
    return redis


async def _close_cache(cache: QdrantSemanticCache) -> None:
    try:
        await cache.close()
    except CacheError:
        logger.warning("error while closing semantic cache", exc_info=True)
