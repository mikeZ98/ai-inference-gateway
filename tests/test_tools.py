"""Adapters: provider HTTP clients (buffered + streaming), dispatcher, router, Qdrant cache."""

import json
import logging
import math
import traceback
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from pytest_httpx import HTTPXMock
from qdrant_client import AsyncQdrantClient, models

from ai_inference_gateway.core import CacheError, HashingEmbedder, ProviderError, RoutingError
from ai_inference_gateway.core.errors import ModelNotAvailableError, ProviderTimeoutError
from ai_inference_gateway.core.ports import ProviderStreamEvent, StreamCompleted, TextDelta
from ai_inference_gateway.schemas import (
    CacheEntry,
    CacheQuery,
    ComplexityAssessment,
    ComplexityTier,
    InferenceRequest,
    ModelProvider,
    RouteTarget,
    RoutingTier,
)
from ai_inference_gateway.tools import (
    AnthropicProvider,
    ComplexityRouter,
    DynamicProviderDispatcher,
    OpenAICompatibleProvider,
    QdrantSemanticCache,
    RouterConfig,
)
from ai_inference_gateway.tools.providers import to_transcript
from ai_inference_gateway.tools.providers.base import iter_sse
from tests.support import (
    DUMMY_KEY,
    FAKE_ANTHROPIC_KEY,
    RecordingQdrant,
    anthropic_sse,
    mock_client,
    openai_sse,
)

P = ModelProvider
T1 = RouteTarget(tier=RoutingTier.TIER_1, provider=P.MISTRAL, model="mistral-small-latest")
T2 = RouteTarget(tier=RoutingTier.TIER_2, provider=P.ANTHROPIC, model="claude-sonnet-5-5")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
REQ = InferenceRequest(prompt="x")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as c:  # default transport: intercepted by httpx_mock
        yield c


@pytest.fixture
def mistral(client: httpx.AsyncClient, api_key: SecretStr) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(client, provider=P.MISTRAL, api_key=api_key)


@pytest.fixture
def anthropic(client: httpx.AsyncClient, api_key: SecretStr) -> AnthropicProvider:
    return AnthropicProvider(client, api_key=api_key)


async def drain(gen: AsyncGenerator[ProviderStreamEvent]) -> list[ProviderStreamEvent]:
    return [event async for event in gen]


def sent_json(httpx_mock: HTTPXMock) -> tuple[httpx.Request, dict[str, Any]]:
    request = httpx_mock.get_request()
    assert request is not None
    return request, json.loads(request.content)


# --- Transcript ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "system", "roles"),
    [
        ("plain question", None, ["user"]),
        (
            "System: be terse\nUser: hi\nUser: again\nAssistant: hello\nHuman: why?",
            "be terse",
            ["user", "assistant", "user"],
        ),
        ("Assistant: hi\nUser: x", None, ["user"]),
        ("Explain:\nUser: foo", None, ["user"]),
        ("System: only system", None, ["user"]),
        ("User:\nUser: real", None, ["user"]),
    ],
)
def test_transcript(prompt: str, system: str | None, roles: list[str]) -> None:
    transcript = to_transcript(prompt)
    assert transcript.system == system
    assert [m.role for m in transcript.messages] == roles


def test_transcript_merges_consecutive_turns() -> None:
    transcript = to_transcript("User: a\nUser: b\nAI: c")
    assert [(m.role, m.content) for m in transcript.messages] == [
        ("user", "a\n\nb"),
        ("assistant", "c"),
    ]


# --- SSE parser ---------------------------------------------------------------


async def test_iter_sse() -> None:
    async def lines() -> AsyncIterator[str]:
        for line in [": comment", "event: a", "data: 1", "data: 2", "", "", "data:x", "id: 7"]:
            yield line

    assert [e async for e in iter_sse(lines())] == [("a", "1\n2"), (None, "x")]


# --- OpenAI-compatible adapter ------------------------------------------------


async def test_openai_complete(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient, api_key: SecretStr
) -> None:
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": "mistralai/mistral-small-3.1:free",
            "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        },
    )
    adapter = OpenAICompatibleProvider(
        client, provider=P.OPENROUTER, api_key=api_key, extra_headers={"X-Title": "gw"}
    )
    target = RouteTarget(tier=RoutingTier.TIER_1, provider=P.OPENROUTER, model="slug")
    result = await adapter.complete(
        target,
        InferenceRequest(prompt="System: terse\nUser: capital?", temperature=0.3, max_tokens=50),
    )
    request, body = sent_json(httpx_mock)
    assert request.headers["authorization"] == f"Bearer {DUMMY_KEY}"
    assert request.headers["x-title"] == "gw"
    assert body == {
        "model": "slug",
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 50,
        "messages": [
            {"role": "system", "content": "terse"},
            {"role": "user", "content": "capital?"},
        ],
    }
    assert (result.content, result.tokens_used, result.model_used) == (
        "Paris",
        9,
        "mistralai/mistral-small-3.1:free",
    )
    assert DUMMY_KEY not in repr(adapter)
    assert adapter.base_url == "https://openrouter.ai/api/v1"


async def test_openai_keyless_local_server(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
) -> None:
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        json={
            "model": "bad model id!",
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )
    adapter = OpenAICompatibleProvider(client, provider=P.OLLAMA)
    target = RouteTarget(tier=RoutingTier.TIER_1, provider=P.OLLAMA, model="llama3.2:3b")
    result = await adapter.complete(target, REQ)
    request, _ = sent_json(httpx_mock)
    assert "authorization" not in request.headers
    assert result.tokens_used == 7
    assert result.model_used == "llama3.2:3b"  # malformed reported model ignored


async def test_openai_without_usage(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider
) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, json={"choices": [{"message": {"content": "hi"}}]})
    assert (await mistral.complete(T1, REQ)).tokens_used == 0


def test_openai_requires_base_url_for_unknown_default(client: httpx.AsyncClient) -> None:
    with pytest.raises(ValueError, match="base_url is required"):
        OpenAICompatibleProvider(client, provider=P.ANTHROPIC)


async def test_adapter_rejects_foreign_target(mistral: OpenAICompatibleProvider) -> None:
    with pytest.raises(ProviderError, match="cannot serve"):
        await mistral.complete(T2, REQ)
    with pytest.raises(ProviderError, match="cannot serve"):
        await drain(mistral.stream(T2, REQ))


@pytest.mark.parametrize(
    ("status", "body", "retryable", "fragment"),
    [
        (
            401,
            {"error": {"message": f"Incorrect API key provided: {DUMMY_KEY}"}},
            False,
            "authentication",
        ),
        (403, None, False, "authentication"),
        (429, {"error": {"message": "slow down"}}, True, "rate limited"),
        (408, None, True, "unavailable"),
        (500, None, True, "unavailable"),
        (
            529,
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
            True,
            "unavailable",
        ),
        (400, {"error": {"message": f"bad {FAKE_ANTHROPIC_KEY}"}}, False, "rejected"),
        (404, {"error": "no such model"}, False, "rejected"),
        (422, {"message": "unprocessable"}, False, "rejected"),
    ],
)
async def test_status_mapping(
    httpx_mock: HTTPXMock,
    mistral: OpenAICompatibleProvider,
    status: int,
    body: dict[str, Any] | None,
    *,
    retryable: bool,
    fragment: str,
) -> None:
    httpx_mock.add_response(
        url=MISTRAL_URL, status_code=status, json=body, headers={"retry-after": "3"}
    )
    with pytest.raises(ProviderError, match=fragment) as info:
        await mistral.complete(T1, REQ)
    error = info.value
    assert (error.status_code, error.retryable, error.provider) == (status, retryable, P.MISTRAL)
    assert DUMMY_KEY not in str(error)
    assert "sk-ant-api03" not in str(error)


async def test_plain_text_error_body_is_included(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider
) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, status_code=502, text="bad gateway upstream")
    with pytest.raises(ProviderError, match="bad gateway upstream"):
        await mistral.complete(T1, REQ)


@pytest.mark.parametrize(
    ("exc", "error_type", "retryable"),
    [
        (httpx.ReadTimeout, ProviderTimeoutError, True),
        (httpx.ConnectTimeout, ProviderTimeoutError, True),
        (httpx.ConnectError, ProviderError, True),
        (httpx.RemoteProtocolError, ProviderError, True),
        (httpx.UnsupportedProtocol, ProviderError, False),
    ],
)
async def test_transport_errors(
    httpx_mock: HTTPXMock,
    mistral: OpenAICompatibleProvider,
    exc: type[httpx.TransportError],
    error_type: type[ProviderError],
    *,
    retryable: bool,
) -> None:
    httpx_mock.add_exception(exc(f"failed with key {DUMMY_KEY}"))
    with pytest.raises(error_type) as info:
        await mistral.complete(T1, REQ)
    assert info.value.retryable is retryable
    assert info.value.__cause__ is None
    assert DUMMY_KEY not in "".join(traceback.format_exception(info.value))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"choices": []}, "malformed"),
        ({"choices": [{"message": {"content": None}}]}, "no text content"),
    ],
)
async def test_openai_unusable_completion(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider, payload: dict[str, Any], match: str
) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, json=payload)
    with pytest.raises(ProviderError, match=match):
        await mistral.complete(T1, REQ)


async def test_openai_stream(httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, content=openai_sse("Par", "", "is"))
    events = await drain(mistral.stream(T1, InferenceRequest(prompt="q", max_tokens=5)))
    _, body = sent_json(httpx_mock)
    assert body["stream"] is True
    assert events == [
        TextDelta(text="Par"),
        TextDelta(text="is"),
        StreamCompleted(model_used="mistral-small-latest", provider=P.MISTRAL, tokens_used=12),
    ]


async def test_openai_stream_finish_reason_without_done(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider
) -> None:
    body = b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"length"}]}\n\n'
    httpx_mock.add_response(url=MISTRAL_URL, content=body)
    events = await drain(mistral.stream(T1, REQ))
    assert events[-1] == StreamCompleted(model_used=T1.model, provider=P.MISTRAL, tokens_used=0)


@pytest.mark.parametrize(
    ("body", "match", "retryable"),
    [
        (openai_sse("cut", done=False), "ended before completion", True),
        (b"data: {not json}\n\n", "malformed stream chunk", False),
        (
            b'data: {"error":{"message":"provider down sk-or-v1-abcdefghijklmnop"}}\n\n',
            "stream error",
            True,
        ),
    ],
)
async def test_openai_stream_failures(
    httpx_mock: HTTPXMock,
    mistral: OpenAICompatibleProvider,
    body: bytes,
    match: str,
    *,
    retryable: bool,
) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, content=body)
    with pytest.raises(ProviderError, match=match) as info:
        await drain(mistral.stream(T1, REQ))
    assert info.value.retryable is retryable
    assert "sk-or-v1" not in str(info.value)


async def test_stream_http_error_before_start(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider
) -> None:
    httpx_mock.add_response(url=MISTRAL_URL, status_code=401, json={"error": {"message": "nope"}})
    with pytest.raises(ProviderError, match="authentication") as info:
        await drain(mistral.stream(T1, REQ))
    assert info.value.status_code == 401


async def test_stream_transport_error(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider
) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeoutError):
        await drain(mistral.stream(T1, REQ))


async def test_closing_stream_closes_upstream_response() -> None:
    closed: list[bool] = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for i in range(1000):
                yield openai_sse(f"t{i}", done=False)

        async def aclose(self) -> None:
            closed.append(True)

    async with mock_client(lambda _: httpx.Response(200, stream=Body())) as c:
        adapter = OpenAICompatibleProvider(c, provider=P.MISTRAL, api_key=SecretStr(DUMMY_KEY))
        gen = adapter.stream(T1, REQ)
        assert await anext(gen) == TextDelta(text="t0")
        await gen.aclose()
    assert closed == [True]


# --- Anthropic adapter --------------------------------------------------------


async def test_anthropic_complete(httpx_mock: HTTPXMock, anthropic: AnthropicProvider) -> None:
    httpx_mock.add_response(
        url=ANTHROPIC_URL,
        json={
            "model": "claude-sonnet-5-5",
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 5},
        },
    )
    prompt = "System: be kind\nUser: a\nUser: b\nAssistant: c\nUser: d"
    result = await anthropic.complete(T2, InferenceRequest(prompt=prompt, temperature=1.7))
    request, body = sent_json(httpx_mock)
    assert request.headers["x-api-key"] == DUMMY_KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    assert body["system"] == "be kind"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 1.0
    assert (result.content, result.tokens_used) == ("Hello world", 16)


async def test_anthropic_minimal_request(
    httpx_mock: HTTPXMock, anthropic: AnthropicProvider
) -> None:
    httpx_mock.add_response(url=ANTHROPIC_URL, json={"content": [{"type": "text", "text": "ok"}]})
    result = await anthropic.complete(T2, InferenceRequest(prompt="hi", max_tokens=10))
    _, body = sent_json(httpx_mock)
    assert body == {
        "model": T2.model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert (result.tokens_used, result.model_used) == (0, T2.model)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"content": [{"type": "tool_use"}]}, "no text content"),
        ({"content": "not a list"}, "malformed"),
    ],
)
async def test_anthropic_unusable(
    httpx_mock: HTTPXMock, anthropic: AnthropicProvider, payload: dict[str, Any], match: str
) -> None:
    httpx_mock.add_response(url=ANTHROPIC_URL, json=payload)
    with pytest.raises(ProviderError, match=match):
        await anthropic.complete(T2, REQ)


def test_anthropic_rejects_bad_max_tokens(client: httpx.AsyncClient, api_key: SecretStr) -> None:
    with pytest.raises(ValueError, match="positive"):
        AnthropicProvider(client, api_key=api_key, default_max_tokens=0)


async def test_anthropic_stream(httpx_mock: HTTPXMock, anthropic: AnthropicProvider) -> None:
    httpx_mock.add_response(url=ANTHROPIC_URL, content=anthropic_sse("Deep ", "answer"))
    events = await drain(anthropic.stream(T2, REQ))
    assert events == [
        TextDelta(text="Deep "),
        TextDelta(text="answer"),
        StreamCompleted(model_used="claude-sonnet-5-5", provider=P.ANTHROPIC, tokens_used=29),
    ]


@pytest.mark.parametrize(
    ("body", "match", "retryable"),
    [
        (anthropic_sse("x", stop=False), "before message_stop", True),
        (b"data: [1, 2]\n\n", "malformed stream event", False),
        (
            b'data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n',
            "overloaded_error",
            True,
        ),
        (
            b'data: {"type":"error","error":{"type":"invalid_request_error"}}\n\n',
            "invalid_request_error",
            False,
        ),
    ],
)
async def test_anthropic_stream_failures(
    httpx_mock: HTTPXMock, anthropic: AnthropicProvider, body: bytes, match: str, *, retryable: bool
) -> None:
    httpx_mock.add_response(url=ANTHROPIC_URL, content=body)
    with pytest.raises(ProviderError, match=match) as info:
        await drain(anthropic.stream(T2, REQ))
    assert info.value.retryable is retryable


async def test_anthropic_stream_ignores_non_text_deltas(
    httpx_mock: HTTPXMock, anthropic: AnthropicProvider
) -> None:
    body = (
        b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta"}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    httpx_mock.add_response(url=ANTHROPIC_URL, content=body)
    events = await drain(anthropic.stream(T2, REQ))
    assert events == [StreamCompleted(model_used=T2.model, provider=P.ANTHROPIC, tokens_used=0)]


async def test_provider_logs_never_contain_keys(
    httpx_mock: HTTPXMock, mistral: OpenAICompatibleProvider, caplog: pytest.LogCaptureFixture
) -> None:
    httpx_mock.add_response(
        url=MISTRAL_URL, status_code=401, json={"error": {"message": DUMMY_KEY}}
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ProviderError):
        await mistral.complete(T1, REQ)
    assert caplog.records
    assert all(DUMMY_KEY not in json.dumps(r.__dict__, default=str) for r in caplog.records)


# --- Dispatcher ---------------------------------------------------------------


@pytest.fixture
def adapters(http_client: httpx.AsyncClient, api_key: SecretStr) -> dict[str, Any]:
    return {
        "mistral": OpenAICompatibleProvider(http_client, provider=P.MISTRAL, api_key=api_key),
        "vllm_a": OpenAICompatibleProvider(
            http_client, provider=P.VLLM, base_url="http://gpu-a:8000/v1/"
        ),
        "vllm_b": OpenAICompatibleProvider(
            http_client, provider=P.VLLM, base_url="http://gpu-b:8000/v1"
        ),
        "anthropic": AnthropicProvider(http_client, api_key=api_key),
    }


async def test_dispatcher_resolution(adapters: dict[str, Any]) -> None:
    dispatcher = DynamicProviderDispatcher(adapters.values())
    vllm = RouteTarget(tier=RoutingTier.TIER_1, provider=P.VLLM, model="qwen")
    assert dispatcher.providers == {P.MISTRAL, P.VLLM, P.ANTHROPIC}
    assert dispatcher.resolve(T1) is adapters["mistral"]
    assert dispatcher.resolve(T2) is adapters["anthropic"]
    assert dispatcher.resolve(vllm) is adapters["vllm_a"]
    pinned = vllm.model_copy(update={"endpoint": "http://gpu-b:8000/v1/"})
    assert dispatcher.resolve(pinned) is adapters["vllm_b"]


@pytest.mark.parametrize(
    "target",
    [
        RouteTarget(tier=RoutingTier.TIER_1, provider=P.GROQ, model="m"),
        RouteTarget(
            tier=RoutingTier.TIER_1, provider=P.VLLM, model="m", endpoint="http://evil:1/v1"
        ),
        RouteTarget(
            tier=RoutingTier.TIER_1, provider=P.MISTRAL, model="m", endpoint="http://gpu-a:8000/v1"
        ),
    ],
    ids=["unregistered-provider", "unknown-endpoint", "endpoint-provider-mismatch"],
)
def test_dispatcher_routing_errors(adapters: dict[str, Any], target: RouteTarget) -> None:
    with pytest.raises(RoutingError):
        DynamicProviderDispatcher(adapters.values()).resolve(target)


def test_dispatcher_rejects_duplicate_endpoint(
    adapters: dict[str, Any], http_client: httpx.AsyncClient
) -> None:
    dispatcher = DynamicProviderDispatcher(adapters.values())
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register(
            OpenAICompatibleProvider(http_client, provider=P.VLLM, base_url="http://gpu-a:8000/v1")
        )


async def test_dispatcher_complete_and_stream(adapters: dict[str, Any]) -> None:
    dispatcher = DynamicProviderDispatcher(adapters.values())
    assert (await dispatcher.complete(T1, REQ)).content == "Paris"
    events = await drain(dispatcher.stream(T2, REQ))
    assert isinstance(events[-1], StreamCompleted)
    assert events[-1].provider is P.ANTHROPIC


# --- Router -------------------------------------------------------------------


def assessment(score: float, tier: ComplexityTier) -> ComplexityAssessment:
    return ComplexityAssessment(score=score, tier=tier)


def test_router_tiers() -> None:
    router = ComplexityRouter()
    assert router.resolve(REQ, assessment(0.1, ComplexityTier.LOW)) == router.config.tier1
    assert router.resolve(REQ, assessment(0.55, ComplexityTier.HIGH)).model == "claude-sonnet-5-5"
    top = router.resolve(REQ, assessment(0.85, ComplexityTier.HIGH))
    assert (top.model, top.tier) == ("claude-opus-5-5", RoutingTier.TIER_3)


def test_router_overrides() -> None:
    deepseek = RouteTarget(tier=RoutingTier.TIER_1, provider=P.DEEPSEEK, model="deepseek-chat")
    extra = RouteTarget(
        tier=RoutingTier.TIER_1, provider=P.OPENROUTER, model="mistralai/mistral-large"
    )
    router = ComplexityRouter(RouterConfig(tier1=deepseek, extra_models=(extra,)))
    low = assessment(0.1, ComplexityTier.LOW)
    assert router.resolve(REQ, low) == deepseek
    assert (
        router.resolve(InferenceRequest(prompt="x", model="claude-opus-5-5"), low).tier
        is RoutingTier.TIER_3
    )
    assert router.resolve(InferenceRequest(prompt="x", model=extra.model), low) == extra
    with pytest.raises(ModelNotAvailableError, match="not available"):
        router.resolve(InferenceRequest(prompt="x", model="gpt-unknown"), low)


def test_router_rejects_unknown_tier() -> None:
    bogus = ComplexityAssessment.model_construct(score=0.5, tier=cast("ComplexityTier", "medium"))
    with pytest.raises(RoutingError, match="unsupported complexity tier"):
        ComplexityRouter().resolve(REQ, bogus)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier1": T2},
        {"tier2": T1.model_copy(update={"tier": RoutingTier.TIER_1})},
        {"tier3": T2},
        {
            "extra_models": (
                RouteTarget(tier=RoutingTier.TIER_1, provider=P.GROQ, model="claude-sonnet-5-5"),
            )
        },
    ],
)
def test_router_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RouterConfig(**kwargs)


# --- Qdrant semantic cache ----------------------------------------------------


NOW = datetime(2026, 10, 23, 12, tzinfo=UTC)


@pytest.fixture
async def cache(qdrant: AsyncQdrantClient) -> QdrantSemanticCache:
    return QdrantSemanticCache(qdrant, dimension=64)


async def entry_for(text: str, response: str = "Paris") -> CacheEntry:
    return CacheEntry(
        prompt=text,
        embedding=await HashingEmbedder(dimension=64).embed(text),
        response=response,
        model_used="mistral-small-latest",
        provider=P.MISTRAL,
        timestamp=NOW,
    )


async def test_cache_lifecycle(cache: QdrantSemanticCache, qdrant: AsyncQdrantClient) -> None:
    entry = await entry_for("capital of France")
    assert await cache.lookup(CacheQuery(embedding=entry.embedding)) is None
    info = await qdrant.get_collection("semantic_cache")
    params = info.config.params.vectors
    assert isinstance(params, models.VectorParams)
    assert (params.size, params.distance) == (64, models.Distance.COSINE)
    assert cache.dimension == 64

    await cache.store(entry)
    await cache.store(entry)
    assert (await qdrant.count("semantic_cache")).count == 1

    hit = await cache.lookup(CacheQuery(embedding=entry.embedding))
    assert hit is not None
    assert (hit.response, hit.provider, hit.timestamp) == ("Paris", P.MISTRAL, NOW)
    assert hit.similarity_score is not None
    assert hit.similarity_score > 0.95
    assert math.isclose(
        math.fsum(a * b for a, b in zip(hit.embedding, entry.embedding, strict=True)),
        1.0,
        rel_tol=1e-5,
    )

    unrelated = await HashingEmbedder(dimension=64).embed("a poem about tea")
    assert await cache.lookup(CacheQuery(embedding=unrelated)) is None
    await cache.ping()


async def test_cache_threshold_is_exclusive(cache: QdrantSemanticCache) -> None:
    entry = await entry_for("threshold edge")
    await cache.store(entry)
    near = tuple(x + (0.05 if i % 2 else -0.05) for i, x in enumerate(entry.embedding))
    norm = math.sqrt(math.fsum(x * x for x in near))
    similarity = math.fsum(a * b for a, b in zip(entry.embedding, near, strict=True)) / norm
    result = await cache.lookup(CacheQuery(embedding=near))
    assert (result is not None) == (similarity > 0.95)


async def test_cache_dimension_checks(cache: QdrantSemanticCache) -> None:
    entry = await entry_for("dims")
    with pytest.raises(CacheError, match="dimension"):
        await cache.lookup(CacheQuery(embedding=(0.1,) * 8))
    with pytest.raises(CacheError, match="dimension"):
        await cache.store(entry.model_copy(update={"embedding": (0.1,) * 8}))
    with pytest.raises(ValueError, match="positive"):
        QdrantSemanticCache.in_memory(dimension=0)


async def test_cache_corrupt_payload(cache: QdrantSemanticCache, qdrant: AsyncQdrantClient) -> None:
    vector = await HashingEmbedder(dimension=64).embed("corrupt")
    await cache.initialize()
    await qdrant.upsert(
        "semantic_cache",
        points=[
            models.PointStruct(
                id=1, vector=list(vector), payload={"prompt": "p", "provider": "nope"}
            )
        ],
    )
    with pytest.raises(CacheError, match="not a valid cache entry"):
        await cache.lookup(CacheQuery(embedding=vector))


async def test_cache_existing_collection_mismatch(qdrant: AsyncQdrantClient) -> None:
    await qdrant.create_collection(
        "semantic_cache", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT)
    )
    with pytest.raises(CacheError, match="expected 64-d/Cosine"):
        await QdrantSemanticCache(qdrant, dimension=64).initialize()


async def test_cache_named_vectors_rejected(qdrant: AsyncQdrantClient) -> None:
    await qdrant.create_collection(
        "semantic_cache",
        vectors_config={"dense": models.VectorParams(size=64, distance=models.Distance.COSINE)},
    )
    with pytest.raises(CacheError, match="single unnamed vector"):
        await QdrantSemanticCache(qdrant, dimension=64).initialize()


async def test_cache_ping_detects_deleted_collection(
    cache: QdrantSemanticCache, qdrant: AsyncQdrantClient
) -> None:
    await cache.initialize()
    await qdrant.delete_collection("semantic_cache")
    with pytest.raises(CacheError, match="missing"):
        await cache.ping()


class BrokenQdrant(AsyncQdrantClient):
    async def query_points(self, *args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("qdrant down")

    async def upsert(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("grpc unavailable")

    async def close(self, *args: Any, **kwargs: Any) -> None:
        raise OSError("already closed")


async def test_cache_translates_client_errors() -> None:
    cache = QdrantSemanticCache(BrokenQdrant(location=":memory:"), dimension=64)
    entry = await entry_for("broken")
    with pytest.raises(CacheError, match="query failed: ConnectError"):
        await cache.lookup(CacheQuery(embedding=entry.embedding))
    with pytest.raises(CacheError, match="upsert failed: RuntimeError"):
        await cache.store(entry)
    with pytest.raises(CacheError, match="close failed"):
        await cache.close()


async def test_cache_vectorless_point_rejected(
    cache: QdrantSemanticCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = await entry_for("no vector")
    await cache.store(entry)
    point = models.ScoredPoint(id=1, version=0, score=0.99, payload={}, vector=None)
    with pytest.raises(CacheError, match="no dense vector"):
        cache._to_entry(point, 0.99)


def test_remote_factory_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_inference_gateway.tools.qdrant_cache.AsyncQdrantClient", RecordingQdrant
    )
    RecordingQdrant.calls.clear()
    cache = QdrantSemanticCache.remote(
        url="https://qdrant.internal:6333", dimension=8, api_key=SecretStr("k")
    )
    assert cache.dimension == 8
    assert RecordingQdrant.calls == [
        {"url": "https://qdrant.internal:6333", "api_key": "k", "timeout": 5}
    ]
