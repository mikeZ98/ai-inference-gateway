"""Core unit tests: heuristics, embeddings and the inference engine (all ports faked)."""

import json
import math
import time
import traceback
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr
from pytest_httpx import HTTPXMock

from ai_inference_gateway.core import (
    CacheError,
    ComplexityPolicy,
    EmbeddingError,
    HashingEmbedder,
    HeuristicComplexityAssessor,
    InferenceEngine,
    OpenAICompatibleEmbedder,
    ProviderCompletion,
    ProviderError,
    RoutingError,
    estimate_tokens,
)
from ai_inference_gateway.core.ports import ProviderStreamEvent, StreamCompleted, TextDelta
from ai_inference_gateway.schemas import (
    CacheEntry,
    CacheQuery,
    ComplexityAssessment,
    ComplexityTier,
    Embedding,
    InferenceRequest,
    ModelProvider,
    RouteTarget,
    RoutingTier,
    StreamDelta,
    StreamSummary,
)
from tests.support import DUMMY_KEY

T1 = RouteTarget(tier=RoutingTier.TIER_1, provider=ModelProvider.MISTRAL, model="mistral-small")
T2 = RouteTarget(
    tier=RoutingTier.TIER_2, provider=ModelProvider.ANTHROPIC, model="claude-sonnet-5-5"
)
NOW = datetime(2026, 10, 23, tzinfo=UTC)


def cosine(a: Embedding, b: Embedding) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


# --- Heuristics ---------------------------------------------------------------


@pytest.fixture
def assessor() -> HeuristicComplexityAssessor:
    return HeuristicComplexityAssessor()


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("a b c d e") == 5


def test_trivial_prompt_is_low(assessor: HeuristicComplexityAssessor) -> None:
    result = assessor.assess("What is the capital of France?")
    assert result.tier is ComplexityTier.LOW
    assert result.score == 0.0
    assert result.matched_heuristics == ()


@pytest.mark.parametrize(
    ("prompt", "expected_signals", "tier"),
    [
        ("Explain why the sky is blue", {"reasoning_keywords"}, ComplexityTier.LOW),
        (
            "Analyze and compare the trade-offs, then design an algorithm",
            {"dense_reasoning_keywords"},
            ComplexityTier.HIGH,
        ),
        (
            "Debug this:\n```python\nx = 1 / 0\n```",
            {"reasoning_keywords", "code_block"},
            ComplexityTier.HIGH,
        ),
        ("def f():\n    pass\nclass A:\n    pass", {"code_block"}, ComplexityTier.LOW),
        (
            "User: hi\nAssistant: hello\nUser: now prove sqrt(2) is irrational",
            {"multi_turn", "reasoning_keywords"},
            ComplexityTier.HIGH,
        ),
        ("Steps:\n1. a\n2. b\n3. c", {"multi_step_instructions"}, ComplexityTier.LOW),
        ("Solve $x^2 = 4$", {"math_notation"}, ComplexityTier.LOW),
        ("word " * 600, {"long_prompt"}, ComplexityTier.LOW),
        ("word " * 3000, {"very_long_prompt"}, ComplexityTier.LOW),
    ],
)
def test_signals(
    assessor: HeuristicComplexityAssessor,
    prompt: str,
    expected_signals: set[str],
    tier: ComplexityTier,
) -> None:
    result = assessor.assess(prompt)
    assert set(result.matched_heuristics) == expected_signals
    assert result.tier is tier
    assert 0.0 <= result.score <= 1.0


def test_score_is_capped(assessor: HeuristicComplexityAssessor) -> None:
    prompt = (
        "User: analyze, compare, evaluate and design\nAssistant: ok\nUser: prove it\n"
        "```py\ndef f(): pass\n```\n1. a\n2. b\n3. c\n$\\frac{1}{2}$ " + "word " * 3000
    )
    assert assessor.assess(prompt).score == 1.0


def test_policy_threshold_is_tunable() -> None:
    lenient = HeuristicComplexityAssessor(ComplexityPolicy(high_threshold=0.2))
    assert lenient.assess("Explain why the sky is blue").tier is ComplexityTier.HIGH


def test_adversarial_prompt_is_linear_time(assessor: HeuristicComplexityAssessor) -> None:
    started = time.perf_counter()
    assessor.assess(("a" * 50 + "$ \n- x\n```") * 2000)
    assert time.perf_counter() - started < 1.0


# --- Embeddings ---------------------------------------------------------------


async def test_hashing_embedder_properties() -> None:
    embedder = HashingEmbedder()
    a = await embedder.embed("What is the capital of France?")
    assert embedder.dimension == len(a) == 384
    assert math.isclose(math.fsum(x * x for x in a), 1.0)
    assert a == await embedder.embed("what is the  capital of   FRANCE?")
    assert cosine(a, await embedder.embed("What is the capital of France ?")) > 0.8
    assert cosine(a, await embedder.embed("Write a haiku about Rust")) < 0.3


async def test_hashing_embedder_rejects_blank_and_bad_dimension() -> None:
    with pytest.raises(EmbeddingError):
        await HashingEmbedder().embed("   ")
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dimension=0)


async def test_hashing_embedder_cancelled_features_fall_back_to_unit_vector() -> None:
    embedder = HashingEmbedder(dimension=1)
    vector = await embedder.embed("ab")
    assert vector in {(1.0,), (-1.0,)}
    for text in ("a", "b", "abc", "x y"):
        assert len(await embedder.embed(text)) == 1


EMBED_URL = "https://embed.test/v1/embeddings"


@pytest.fixture
async def real_client() -> AsyncGenerator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:  # default transport, intercepted by httpx_mock
        yield client


def embedder(client: httpx.AsyncClient, *, key: str | None = DUMMY_KEY) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        client,
        base_url="https://embed.test/v1/",
        api_key=SecretStr(key) if key else None,
        model="mistral-embed",
        dimension=3,
    )


async def test_openai_embedder_success(
    httpx_mock: HTTPXMock, real_client: httpx.AsyncClient
) -> None:
    httpx_mock.add_response(
        url=EMBED_URL,
        json={"object": "list", "data": [{"embedding": [0.1, 2, 0.3], "index": 0}], "usage": {}},
    )
    client = embedder(real_client)
    assert client.dimension == 3
    assert await client.embed("hello") == (0.1, 2.0, 0.3)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["authorization"] == f"Bearer {DUMMY_KEY}"
    assert json.loads(sent.content) == {"model": "mistral-embed", "input": "hello"}
    assert DUMMY_KEY not in repr(vars(client))


async def test_openai_embedder_without_key(
    httpx_mock: HTTPXMock, real_client: httpx.AsyncClient
) -> None:
    httpx_mock.add_response(url=EMBED_URL, json={"data": [{"embedding": [1, 0, 0]}]})
    await embedder(real_client, key=None).embed("x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "authorization" not in sent.headers


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, json={"data": []}), "unexpected shape"),
        (httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}), "dimension 2"),
        (httpx.Response(200, content=b'{"data":[{"embedding":[NaN,1,1]}]}'), "unexpected shape"),
        (httpx.Response(200, content=b"not json"), "unexpected shape"),
    ],
)
async def test_openai_embedder_failures(
    httpx_mock: HTTPXMock, real_client: httpx.AsyncClient, response: httpx.Response, match: str
) -> None:
    httpx_mock.add_response(
        url=EMBED_URL, status_code=response.status_code, content=response.content
    )
    with pytest.raises(EmbeddingError, match=match):
        await embedder(real_client).embed("x")


async def test_openai_embedder_transport_error_never_chains_secrets(
    httpx_mock: HTTPXMock, real_client: httpx.AsyncClient
) -> None:
    httpx_mock.add_exception(httpx.ConnectError(f"boom {DUMMY_KEY}"))
    with pytest.raises(EmbeddingError, match="ConnectError") as info:
        await embedder(real_client).embed("x")
    assert info.value.__cause__ is None
    assert DUMMY_KEY not in "".join(traceback.format_exception(info.value))


def test_openai_embedder_rejects_bad_dimension(real_client: httpx.AsyncClient) -> None:
    with pytest.raises(ValueError, match="positive"):
        OpenAICompatibleEmbedder(
            real_client, base_url="https://e", api_key=None, model="m", dimension=0
        )


# --- Engine -------------------------------------------------------------------


class MemoryCache:
    def __init__(self) -> None:
        self.entries: list[CacheEntry] = []
        self.fail_lookup = False
        self.fail_store = False
        self.lookups = 0
        self.score_override: float | None = None
        self.drop_score = False

    async def lookup(self, query: CacheQuery) -> CacheEntry | None:
        self.lookups += 1
        if self.fail_lookup:
            raise CacheError("down")
        if not self.entries:
            return None
        best = max(self.entries, key=lambda e: cosine(e.embedding, query.embedding))
        score = self.score_override or cosine(best.embedding, query.embedding)
        return best.model_copy(update={"similarity_score": None if self.drop_score else score})

    async def store(self, entry: CacheEntry) -> None:
        if self.fail_store:
            raise CacheError("down")
        self.entries.append(entry)


class FailingEmbedder:
    dimension = 384

    async def embed(self, text: str) -> Embedding:
        raise EmbeddingError("embedding service down")


class TierRouter:
    def resolve(self, request: InferenceRequest, assessment: ComplexityAssessment) -> RouteTarget:
        if request.model == "unknown":
            raise RoutingError("unknown model")
        return T2 if assessment.tier is ComplexityTier.HIGH else T1


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[RouteTarget] = []
        self.provider_override: ModelProvider | None = None
        self.stream_events: Callable[[RouteTarget], list[ProviderStreamEvent]] | None = None
        self.closed = 0

    async def complete(self, target: RouteTarget, request: InferenceRequest) -> ProviderCompletion:
        self.calls.append(target)
        return ProviderCompletion(
            content=f"answer:{request.prompt[:12]}",
            model_used=target.model,
            provider=self.provider_override or target.provider,
            tokens_used=42,
        )

    async def stream(
        self, target: RouteTarget, request: InferenceRequest
    ) -> AsyncGenerator[ProviderStreamEvent]:
        self.calls.append(target)
        events: list[ProviderStreamEvent] = (
            self.stream_events(target)
            if self.stream_events
            else [
                TextDelta(text="ans"),
                TextDelta(text="wer"),
                StreamCompleted(
                    model_used=target.model,
                    provider=self.provider_override or target.provider,
                    tokens_used=7,
                ),
            ]
        )
        try:
            for event in events:
                yield event
        finally:
            self.closed += 1


class Harness:
    def __init__(self) -> None:
        self.cache = MemoryCache()
        self.dispatcher = FakeDispatcher()
        ticks = iter(range(0, 1_000_000, 5))
        self.engine = InferenceEngine(
            embedder=HashingEmbedder(),
            cache=self.cache,
            assessor=HeuristicComplexityAssessor(),
            router=TierRouter(),
            dispatcher=self.dispatcher,
            clock=lambda: NOW,
            timer=lambda: next(ticks) / 1000,
        )


@pytest.fixture
def h() -> Harness:
    return Harness()


async def collect(
    gen: AsyncGenerator[StreamDelta | StreamSummary],
) -> list[StreamDelta | StreamSummary]:
    return [event async for event in gen]


async def test_miss_then_semantic_hit(h: Harness, simple_request: InferenceRequest) -> None:
    first = await h.engine.infer(simple_request)
    assert not first.cached_hit
    assert first.provider is ModelProvider.MISTRAL
    assert first.tokens_used == 42
    assert first.latency_ms == 5.0
    assert len(h.cache.entries) == 1
    assert h.cache.entries[0].timestamp == NOW
    assert h.cache.entries[0].similarity_score is None

    second = await h.engine.infer(InferenceRequest(prompt="what is the capital of  france?"))
    assert second.cached_hit
    assert second.tokens_used == 0
    assert second.content == first.content
    assert len(h.dispatcher.calls) == 1


async def test_hit_requires_similarity_strictly_above_threshold(
    h: Harness, simple_request: InferenceRequest
) -> None:
    await h.engine.infer(simple_request)
    h.cache.score_override = 0.95
    assert not (await h.engine.infer(simple_request)).cached_hit
    h.cache.score_override = None
    h.cache.drop_score = True
    assert not (await h.engine.infer(simple_request)).cached_hit


async def test_complex_prompt_routes_high(h: Harness, complex_request: InferenceRequest) -> None:
    result = await h.engine.infer(complex_request)
    assert result.provider is ModelProvider.ANTHROPIC
    assert h.dispatcher.calls[-1] == T2


async def test_model_override_bypasses_cache(h: Harness, simple_request: InferenceRequest) -> None:
    await h.engine.infer(simple_request)
    lookups = h.cache.lookups
    override = InferenceRequest(prompt=simple_request.prompt, model="mistral-large")
    result = await h.engine.infer(override)
    assert not result.cached_hit
    assert h.cache.lookups == lookups
    assert len(h.cache.entries) == 1


async def test_cache_failures_fail_open(h: Harness, simple_request: InferenceRequest) -> None:
    h.cache.fail_lookup = True
    assert not (await h.engine.infer(simple_request)).cached_hit
    h.cache.fail_lookup = False
    h.cache.fail_store = True
    result = await h.engine.infer(InferenceRequest(prompt="tea question"))
    assert result.content
    # The lookup failure did not block the write; the store failure lost only its own entry.
    assert [e.prompt for e in h.cache.entries] == [simple_request.prompt]


async def test_embedding_failure_fails_open(simple_request: InferenceRequest) -> None:
    dispatcher = FakeDispatcher()
    cache = MemoryCache()
    engine = InferenceEngine(
        embedder=FailingEmbedder(),
        cache=cache,
        assessor=HeuristicComplexityAssessor(),
        router=TierRouter(),
        dispatcher=dispatcher,
    )
    result = await engine.infer(simple_request)
    assert not result.cached_hit
    assert cache.lookups == 0
    assert cache.entries == []
    assert result.latency_ms >= 0


async def test_routing_and_provider_mismatch_errors(h: Harness) -> None:
    with pytest.raises(RoutingError):
        await h.engine.infer(InferenceRequest(prompt="x", model="unknown"))
    h.dispatcher.provider_override = ModelProvider.OPENROUTER
    with pytest.raises(ProviderError, match="expected 'mistral'") as info:
        await h.engine.infer(InferenceRequest(prompt="new prompt"))
    assert info.value.provider is ModelProvider.MISTRAL


def test_threshold_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        InferenceEngine(
            embedder=HashingEmbedder(),
            cache=MemoryCache(),
            assessor=HeuristicComplexityAssessor(),
            router=TierRouter(),
            dispatcher=FakeDispatcher(),
            similarity_threshold=0.9,
        )


async def test_stream_miss_then_cached_stream(h: Harness, simple_request: InferenceRequest) -> None:
    events = await collect(h.engine.stream(simple_request))
    assert [type(e) for e in events] == [StreamDelta, StreamDelta, StreamSummary]
    summary = events[-1]
    assert isinstance(summary, StreamSummary)
    assert not summary.cached_hit
    assert summary.tokens_used == 7
    assert h.cache.entries[0].response == "answer"
    assert h.dispatcher.closed == 1

    cached = await collect(h.engine.stream(simple_request))
    assert cached == [
        StreamDelta(content="answer"),
        StreamSummary(
            model_used=T1.model,
            provider=ModelProvider.MISTRAL,
            tokens_used=0,
            cached_hit=True,
            latency_ms=cached[-1].latency_ms if isinstance(cached[-1], StreamSummary) else -1,
        ),
    ]


async def test_stream_cached_empty_response_emits_only_summary(h: Harness) -> None:
    h.cache.entries.append(
        CacheEntry(
            prompt="p",
            embedding=await HashingEmbedder().embed("empty answer prompt"),
            response="",
            model_used="m",
            provider=ModelProvider.MISTRAL,
            timestamp=NOW,
        )
    )
    events = await collect(h.engine.stream(InferenceRequest(prompt="empty answer prompt")))
    assert len(events) == 1
    assert isinstance(events[0], StreamSummary)


async def test_stream_without_completion_event_is_retryable_error(h: Harness) -> None:
    h.dispatcher.stream_events = lambda _: [TextDelta(text="partial")]
    with pytest.raises(ProviderError, match="without a completion") as info:
        await collect(h.engine.stream(InferenceRequest(prompt="cut short")))
    assert info.value.retryable
    assert h.cache.entries == []


async def test_stream_provider_mismatch(h: Harness) -> None:
    h.dispatcher.provider_override = ModelProvider.GROQ
    with pytest.raises(ProviderError):
        await collect(h.engine.stream(InferenceRequest(prompt="mismatch")))


async def test_closing_stream_early_closes_upstream_and_skips_cache(h: Harness) -> None:
    gen = h.engine.stream(InferenceRequest(prompt="abandon me"))
    first = await anext(gen)
    assert first == StreamDelta(content="ans")
    await gen.aclose()
    assert h.dispatcher.closed == 1
    assert h.cache.entries == []
