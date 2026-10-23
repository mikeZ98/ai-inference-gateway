"""Strict Pydantic v2 contract tests: closed, immutable, non-coercing, bounded."""

from datetime import UTC, datetime
from ipaddress import IPv4Address
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ai_inference_gateway.core.heuristics import ComplexityPolicy
from ai_inference_gateway.core.ports import ProviderCompletion, StreamCompleted, TextDelta
from ai_inference_gateway.schemas import (
    DEFAULT_SIMILARITY_THRESHOLD,
    OPENAI_COMPATIBLE_PROVIDERS,
    TIER_PROVIDERS,
    CacheEntry,
    CacheQuery,
    ComplexityAssessment,
    ComplexityTier,
    ComponentCheck,
    ErrorDetail,
    ErrorResponse,
    FieldError,
    GuardrailMetricsSnapshot,
    HealthStatus,
    InferenceRequest,
    InferenceResponse,
    ModelProvider,
    RouteTarget,
    RoutingTier,
    StreamDelta,
    StreamError,
    StreamSummary,
    StrictModel,
    TokenBucketStatus,
)
from ai_inference_gateway.tools.providers.messages import ChatMessage, ChatTranscript

NOW = datetime(2026, 10, 23, tzinfo=UTC)

MISTRAL = ModelProvider.MISTRAL
COUNTERS = dict.fromkeys(GuardrailMetricsSnapshot.model_fields, 0)

# model -> (valid payload, (field, value of the wrong type))
CASES: dict[type[BaseModel], tuple[dict[str, Any], tuple[str, Any]]] = {
    InferenceRequest: (
        {"prompt": "hi", "temperature": 0.5, "max_tokens": 10},
        ("max_tokens", "10"),
    ),
    ComplexityAssessment: ({"score": 0.2, "tier": ComplexityTier.LOW}, ("score", "0.2")),
    CacheEntry: (
        {
            "prompt": "hi",
            "embedding": (0.1, 0.2),
            "response": "yo",
            "model_used": "mistral-small-latest",
            "provider": MISTRAL,
            "timestamp": NOW,
        },
        ("embedding", [0.1, 0.2]),
    ),
    CacheQuery: ({"embedding": (0.1, 0.2)}, ("limit", 1.0)),
    RouteTarget: (
        {"tier": RoutingTier.TIER_1, "provider": MISTRAL, "model": "mistral-small-latest"},
        ("provider", "mistral"),
    ),
    InferenceResponse: (
        {
            "content": "x",
            "model_used": "claude-sonnet-5-5",
            "provider": ModelProvider.ANTHROPIC,
            "tokens_used": 3,
            "cached_hit": False,
            "latency_ms": 12.5,
        },
        ("cached_hit", 1),
    ),
    TokenBucketStatus: (
        {
            "client_ip": IPv4Address("10.0.0.1"),
            "tokens_remaining": 4.0,
            "reset_at": NOW,
            "allowed": True,
        },
        ("allowed", "true"),
    ),
    ErrorDetail: ({"code": "bad_thing", "message": "m"}, ("retryable", 0)),
    ErrorResponse: ({"error": ErrorDetail(code="x", message="m")}, ("error", {"code": "x"})),
    FieldError: ({"loc": ("body", "prompt"), "message": "m", "type": "t"}, ("loc", ["body"])),
    HealthStatus: ({"status": "ok"}, ("checks", [])),
    ComponentCheck: ({"name": "qdrant", "ok": True}, ("ok", "yes")),
    StreamDelta: ({"content": "tok"}, ("content", 1)),
    StreamSummary: (
        {
            "model_used": "m",
            "provider": MISTRAL,
            "tokens_used": 1,
            "cached_hit": True,
            "latency_ms": 1.0,
        },
        ("tokens_used", "1"),
    ),
    StreamError: ({"error": ErrorDetail(code="x", message="m")}, ("type", 1)),
    GuardrailMetricsSnapshot: (COUNTERS, ("tokens_consumed", 1.5)),
    ProviderCompletion: (
        {"content": "c", "model_used": "m", "provider": MISTRAL, "tokens_used": 1},
        ("tokens_used", "1"),
    ),
    TextDelta: ({"text": "t"}, ("text", b"t")),
    StreamCompleted: (
        {"model_used": "m", "provider": MISTRAL, "tokens_used": 0},
        ("provider", "m"),
    ),
    ChatMessage: ({"role": "user", "content": "hi"}, ("content", 1)),
    ChatTranscript: (
        {"messages": (ChatMessage(role="user", content="hi"),)},
        ("messages", [ChatMessage(role="user", content="hi")]),
    ),
    ComplexityPolicy: ({}, ("long_prompt_tokens", "512")),
}

MODELS = list(CASES)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_strict_config(model: type[BaseModel]) -> None:
    assert issubclass(model, StrictModel)
    assert model.model_config == {"extra": "forbid", "frozen": True, "strict": True}


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_valid_payload_is_hashable_and_round_trips(model: type[BaseModel]) -> None:
    data, _ = CASES[model]
    instance = model(**data)
    hash(instance)
    assert model.model_validate_json(instance.model_dump_json()) == instance


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_rejects_extra_fields(model: type[BaseModel]) -> None:
    data, _ = CASES[model]
    with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
        model(**data, unexpected=1)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_is_immutable(model: type[BaseModel]) -> None:
    data, (field, _) = CASES[model]
    instance = model(**data)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, field, getattr(instance, field))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_rejects_coercion(model: type[BaseModel]) -> None:
    data, (field, wrong) = CASES[model]
    with pytest.raises(ValidationError):
        model(**{**data, field: wrong})


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (InferenceRequest, {"prompt": "   "}),
        (InferenceRequest, {"prompt": ""}),
        (InferenceRequest, {"prompt": "x" * 100_001}),
        (InferenceRequest, {"prompt": "a", "temperature": 2.5}),
        (InferenceRequest, {"prompt": "a", "temperature": -0.1}),
        (InferenceRequest, {"prompt": "a", "max_tokens": 0}),
        (InferenceRequest, {"prompt": "a", "max_tokens": 40_000}),
        (InferenceRequest, {"prompt": "a", "model": "has space"}),
        (ComplexityAssessment, {"score": 1.5, "tier": ComplexityTier.HIGH}),
        (ComplexityAssessment, {"score": float("nan"), "tier": ComplexityTier.HIGH}),
        (CacheQuery, {"embedding": (0.1,), "similarity_threshold": 0.9}),
        (CacheQuery, {"embedding": (0.1,), "similarity_threshold": 1.0}),
        (CacheQuery, {"embedding": ()}),
        (CacheQuery, {"embedding": (float("inf"),)}),
        (CacheQuery, {"embedding": (0.1,), "limit": 11}),
        (InferenceResponse, {**CASES[InferenceResponse][0], "tokens_used": -1}),
        (InferenceResponse, {**CASES[InferenceResponse][0], "latency_ms": -1.0}),
        (TokenBucketStatus, {**CASES[TokenBucketStatus][0], "client_ip": "999.1.1.1"}),
        (TokenBucketStatus, {**CASES[TokenBucketStatus][0], "tokens_remaining": -1.0}),
        (ErrorDetail, {"code": "Bad-Code", "message": "m"}),
        (HealthStatus, {"status": "maybe"}),
        (StreamDelta, {"content": ""}),
        (ComplexityPolicy, {"long_prompt_tokens": 100, "very_long_prompt_tokens": 100}),
    ],
)
def test_bounds_and_invariants(model: type[BaseModel], data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**data)


def test_naive_timestamp_rejected() -> None:
    data = {**CASES[CacheEntry][0], "timestamp": datetime(2026, 1, 1)}
    with pytest.raises(ValidationError, match="timezone"):
        CacheEntry(**data)


def test_json_input_accepts_json_native_encodings() -> None:
    entry = CacheEntry.model_validate_json(
        '{"prompt":"p","embedding":[1,0.5],"response":"r","model_used":"m",'
        '"provider":"mistral","timestamp":"2026-10-23T00:00:00Z"}'
    )
    assert entry.embedding == (1.0, 0.5)
    assert entry.provider is MISTRAL
    req = InferenceRequest.model_validate_json('{"prompt":"hi","temperature":1}')
    assert req.temperature == 1.0
    assert req.stream is False
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate_json('{"prompt":"a","max_tokens":"10"}')


def test_cache_query_hit_is_strictly_above_threshold() -> None:
    query = CacheQuery(embedding=(0.1,))
    assert query.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD == 0.95
    assert not query.is_hit(0.95)
    assert query.is_hit(0.9500001)


@pytest.mark.parametrize(
    ("tier", "provider", "valid"),
    [
        (RoutingTier.TIER_1, ModelProvider.MISTRAL, True),
        (RoutingTier.TIER_1, ModelProvider.DEEPSEEK, True),
        (RoutingTier.TIER_1, ModelProvider.OLLAMA, True),
        (RoutingTier.TIER_1, ModelProvider.ANTHROPIC, False),
        (RoutingTier.TIER_2, ModelProvider.ANTHROPIC, True),
        (RoutingTier.TIER_2, ModelProvider.MISTRAL, False),
        (RoutingTier.TIER_3, ModelProvider.ANTHROPIC, True),
        (RoutingTier.TIER_3, ModelProvider.OPENROUTER, False),
    ],
)
def test_route_target_tier_provider_pairing(
    tier: RoutingTier, provider: ModelProvider, *, valid: bool
) -> None:
    if valid:
        assert RouteTarget(tier=tier, provider=provider, model="m").provider is provider
    else:
        with pytest.raises(ValidationError, match="not valid for tier"):
            RouteTarget(tier=tier, provider=provider, model="m")


def test_route_target_endpoint_must_be_http() -> None:
    def target(endpoint: str) -> RouteTarget:
        return RouteTarget(
            tier=RoutingTier.TIER_1, provider=ModelProvider.VLLM, model="m", endpoint=endpoint
        )

    assert target("http://gpu:8000/v1").endpoint == "http://gpu:8000/v1"
    with pytest.raises(ValidationError):
        target("ftp://gpu/v1")


def test_provider_sets() -> None:
    assert ModelProvider.ANTHROPIC not in OPENAI_COMPATIBLE_PROVIDERS
    assert TIER_PROVIDERS[RoutingTier.TIER_1] == OPENAI_COMPATIBLE_PROVIDERS
    assert TIER_PROVIDERS[RoutingTier.TIER_3] == frozenset({ModelProvider.ANTHROPIC})
