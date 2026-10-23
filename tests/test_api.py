"""HTTP API: health, readiness, JSON + SSE inference, error mapping, lifecycle, disconnects."""

import asyncio
import gc
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import SecretStr, ValidationError
from qdrant_client import AsyncQdrantClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.types import Message

from ai_inference_gateway.api import GatewaySettings, create_app
from ai_inference_gateway.api.app import build_adapters, build_cache, build_embedder
from ai_inference_gateway.api.errors import (
    _on_gateway_error,
    _on_http_error,
    _on_validation_error,
    gateway_error_detail,
)
from ai_inference_gateway.api.routes import get_services
from ai_inference_gateway.core import (
    CacheError,
    EmbeddingError,
    GatewayError,
    OpenAICompatibleEmbedder,
)
from ai_inference_gateway.core.errors import ProviderError
from ai_inference_gateway.schemas import (
    ErrorResponse,
    HealthStatus,
    InferenceResponse,
    ModelProvider,
    RouteTarget,
    RoutingTier,
    StreamDelta,
    StreamError,
    StreamSummary,
)
from tests.support import (
    DUMMY_KEY,
    AppFactory,
    NetworkBlockedError,
    RecordingQdrant,
    Upstream,
    make_settings,
    mock_client,
    openai_sse,
    serve,
)

type Event = StreamDelta | StreamSummary | StreamError | str
LEAK_MARKERS = (DUMMY_KEY, "api.mistral.ai", "api.anthropic.com", "Traceback", "httpx", "sk-")


FRAME_MODELS: dict[str, type[StreamDelta] | type[StreamSummary] | type[StreamError]] = {
    "delta": StreamDelta,
    "summary": StreamSummary,
    "error": StreamError,
}


def sse_events(text: str) -> list[Event]:
    events: list[Event] = []
    for frame in (f.removeprefix("data: ") for f in text.split("\n\n") if f):
        if frame == "[DONE]":
            events.append(frame)
            continue
        kind = json.loads(frame)["type"]
        model = FRAME_MODELS[kind]
        events.append(model.model_validate_json(frame))
    return events


def error_of(response: httpx.Response) -> ErrorResponse:
    return ErrorResponse.model_validate_json(response.content)


def leak_free(response: httpx.Response) -> bool:
    return all(marker not in response.text for marker in LEAK_MARKERS)


@pytest.fixture
async def gw(app_factory: AppFactory) -> AsyncIterator[httpx.AsyncClient]:
    async with serve(app_factory()) as client:
        yield client


# --- Health / readiness -------------------------------------------------------


async def test_health_ready_openapi(gw: httpx.AsyncClient) -> None:
    health = await gw.get("/health")
    assert health.status_code == 200
    assert HealthStatus.model_validate_json(health.content).status == "ok"

    ready = await gw.get("/ready")
    status = HealthStatus.model_validate_json(ready.content)
    assert ready.status_code == 200
    assert status.status == "ready"
    assert {c.name: c.ok for c in status.checks} == {"qdrant": True, "providers": True}

    schema = (await gw.get("/openapi.json")).json()
    assert {"/v1/inference", "/health", "/ready", "/metrics"} <= set(schema["paths"])


async def test_not_ready_without_providers(app_factory: AppFactory) -> None:
    settings = make_settings(mistral_api_key=None, anthropic_api_key=None)
    async with serve(app_factory(settings=settings)) as gw:
        ready = await gw.get("/ready")
        status = HealthStatus.model_validate_json(ready.content)
        assert ready.status_code == 503
        assert [c.name for c in status.checks if not c.ok] == ["providers"]
        response = await gw.post("/v1/inference", json={"prompt": "hello"})
        assert response.status_code == 503
        assert error_of(response).error.code == "no_route"
        assert "mistral" not in error_of(response).error.message


class UnreachableQdrant(AsyncQdrantClient):
    async def collection_exists(self, *args: Any, **kwargs: Any) -> bool:
        raise httpx.ConnectError("qdrant://admin:pw@10.0.0.5:6333 unreachable")


async def test_qdrant_down_not_ready_but_inference_fails_open(app_factory: AppFactory) -> None:
    app = app_factory(qdrant_client=UnreachableQdrant(location=":memory:"))
    async with serve(app) as gw:
        ready = await gw.get("/ready")
        assert ready.status_code == 503
        assert [
            c.name for c in HealthStatus.model_validate_json(ready.content).checks if not c.ok
        ] == ["qdrant"]
        assert "10.0.0.5" not in ready.text
        response = await gw.post("/v1/inference", json={"prompt": "works without cache?"})
        assert response.status_code == 200
        assert not InferenceResponse.model_validate_json(response.content).cached_hit


# --- JSON inference -----------------------------------------------------------


async def test_json_inference_and_cache(gw: httpx.AsyncClient, upstream: Upstream) -> None:
    first = await gw.post("/v1/inference", json={"prompt": "What is the capital of France?"})
    result = InferenceResponse.model_validate_json(first.content)
    assert first.status_code == 200
    assert (result.provider, result.cached_hit, result.tokens_used) == (
        ModelProvider.MISTRAL,
        False,
        12,
    )

    again = InferenceResponse.model_validate_json(
        (await gw.post("/v1/inference", json={"prompt": "what is the capital of  france?"})).content
    )
    assert again.cached_hit
    assert again.tokens_used == 0
    assert len(upstream.calls) == 1


async def test_complex_prompt_uses_claude(gw: httpx.AsyncClient, upstream: Upstream) -> None:
    prompt = "Debug this and explain why:\n```python\ndef f(): return 1/0\n```"
    response = await gw.post("/v1/inference", json={"prompt": prompt, "temperature": 1.5})
    result = InferenceResponse.model_validate_json(response.content)
    assert (result.provider, result.model_used) == (ModelProvider.ANTHROPIC, "claude-sonnet-5-5")
    assert upstream.calls[-1][1]["temperature"] == 1.0


@pytest.mark.parametrize(
    ("payload", "loc", "kind"),
    [
        ({"prompt": "TOP-SECRET", "evil": 1}, ("body", "evil"), "extra_forbidden"),
        ({"prompt": "TOP-SECRET", "max_tokens": "10"}, ("body", "max_tokens"), "int_type"),
        ({"prompt": "   "}, ("body", "prompt"), "value_error"),
        ({"prompt": "TOP-SECRET", "temperature": 3}, ("body", "temperature"), "less_than_equal"),
        ({}, ("body", "prompt"), "missing"),
    ],
)
async def test_validation_errors_are_typed_and_do_not_echo_input(
    gw: httpx.AsyncClient, payload: dict[str, Any], loc: tuple[str, ...], kind: str
) -> None:
    response = await gw.post("/v1/inference", json=payload)
    error = error_of(response).error
    assert response.status_code == 422
    assert error.code == "invalid_request"
    assert [(f.loc, f.type) for f in error.field_errors] == [(loc, kind)]
    assert "TOP-SECRET" not in response.text


async def test_malformed_json_and_http_errors(gw: httpx.AsyncClient) -> None:
    bad = await gw.post(
        "/v1/inference", content=b'{"prompt": ', headers={"content-type": "application/json"}
    )
    assert (bad.status_code, error_of(bad).error.code) == (422, "invalid_request")
    missing = await gw.get("/nope")
    assert (missing.status_code, error_of(missing).error.code) == (404, "not_found")
    method = await gw.get("/v1/inference")
    assert (method.status_code, error_of(method).error.code) == (405, "method_not_allowed")


async def test_unknown_model_override(gw: httpx.AsyncClient) -> None:
    response = await gw.post("/v1/inference", json={"prompt": "x", "model": "gpt-unknown"})
    assert response.status_code == 400
    assert error_of(response).error.code == "model_not_available"


def _raise(exc: Exception) -> Callable[[httpx.Request, dict[str, Any]], httpx.Response]:
    def handler(request: httpx.Request, _: dict[str, Any]) -> httpx.Response:
        raise exc

    return handler


UPSTREAM_FAILURES: list[
    tuple[str, Callable[[httpx.Request, dict[str, Any]], httpx.Response], int, str, bool]
] = [
    (
        "401",
        lambda q, b: httpx.Response(401, json={"error": {"message": f"bad {DUMMY_KEY}"}}),
        502,
        "upstream_error",
        False,
    ),
    (
        "400",
        lambda q, b: httpx.Response(400, json={"error": {"message": "internal-detail"}}),
        502,
        "upstream_error",
        False,
    ),
    (
        "429",
        lambda q, b: httpx.Response(429, headers={"retry-after": "30"}),
        503,
        "upstream_unavailable",
        True,
    ),
    ("503", lambda q, b: httpx.Response(503), 503, "upstream_unavailable", True),
    ("timeout", _raise(httpx.ReadTimeout("slow")), 504, "upstream_timeout", True),
    (
        "refused",
        _raise(httpx.ConnectError(f"refused {DUMMY_KEY}")),
        503,
        "upstream_unavailable",
        True,
    ),
]


@pytest.mark.parametrize(
    ("label", "handler", "status", "code", "retryable"),
    UPSTREAM_FAILURES,
    ids=[c[0] for c in UPSTREAM_FAILURES],
)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_upstream_failure_mapping(
    gw: httpx.AsyncClient,
    upstream: Upstream,
    *,
    label: str,
    handler: Callable[[httpx.Request, dict[str, Any]], httpx.Response],
    status: int,
    code: str,
    retryable: bool,
    stream: bool,
) -> None:
    upstream.override = handler
    response = await gw.post(
        "/v1/inference", json={"prompt": f"fresh {label} question", "stream": stream}
    )
    error = error_of(response).error
    assert response.status_code == status
    assert response.headers["content-type"].startswith(
        "application/json"
    )  # pre-stream: real status
    assert (error.code, error.retryable) == (code, retryable)
    assert ("retry-after" in response.headers) is retryable
    assert leak_free(response)


# --- SSE streaming ------------------------------------------------------------


async def test_stream_then_cached_stream(gw: httpx.AsyncClient, upstream: Upstream) -> None:
    response = await gw.post(
        "/v1/inference", json={"prompt": "Name a large city in Japan", "stream": True}
    )
    events = sse_events(response.text)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert [type(e).__name__ if not isinstance(e, str) else e for e in events] == [
        "StreamDelta",
        "StreamDelta",
        "StreamSummary",
        "[DONE]",
    ]
    summary = events[2]
    assert isinstance(summary, StreamSummary)
    assert (summary.tokens_used, summary.cached_hit) == (12, False)
    assert upstream.calls[-1][1]["stream"] is True

    cached = sse_events(
        (
            await gw.post(
                "/v1/inference", json={"prompt": "Name a large city in Japan", "stream": True}
            )
        ).text
    )
    assert cached[0] == StreamDelta(content="Paris")
    assert isinstance(cached[1], StreamSummary)
    assert cached[1].cached_hit
    assert len(upstream.calls) == 1


async def test_anthropic_stream(gw: httpx.AsyncClient) -> None:
    prompt = "Debug this and explain why:\n```\nclass A: pass\n```\nthen compare designs"
    events = sse_events(
        (await gw.post("/v1/inference", json={"prompt": prompt, "stream": True})).text
    )
    assert "".join(e.content for e in events if isinstance(e, StreamDelta)) == "Deep answer"
    summary = events[-2]
    assert isinstance(summary, StreamSummary)
    assert (summary.provider, summary.tokens_used) == (ModelProvider.ANTHROPIC, 29)


@pytest.mark.parametrize(
    ("body", "retryable"),
    [
        (
            openai_sse("Hal", "f", done=False)
            + b'data: {"error":{"message":"down sk-or-v1-abcdefghijklmnop"}}\n\n',
            True,
        ),
        (openai_sse("cut", done=False), True),
    ],
    ids=["midstream-error", "truncated"],
)
async def test_failure_after_stream_start(
    gw: httpx.AsyncClient, upstream: Upstream, body: bytes, *, retryable: bool
) -> None:
    upstream.override = lambda q, b: httpx.Response(200, content=body)
    response = await gw.post("/v1/inference", json={"prompt": "will fail midway", "stream": True})
    events = sse_events(response.text)
    assert response.status_code == 200
    assert isinstance(events[-2], StreamError)
    assert events[-2].error.retryable is retryable
    assert events[-1] == "[DONE]"
    assert "sk-or" not in response.text

    upstream.override = None
    again = InferenceResponse.model_validate_json(
        (await gw.post("/v1/inference", json={"prompt": "will fail midway"})).content
    )
    assert not again.cached_hit  # the failed stream was not cached


# --- Lifecycle ----------------------------------------------------------------


async def test_lifespan_owns_only_what_it_creates(http_client: httpx.AsyncClient) -> None:
    app = create_app(make_settings())
    async with app.router.lifespan_context(app):
        tier1 = RouteTarget(tier=RoutingTier.TIER_1, provider=ModelProvider.MISTRAL, model="m")
        owned = get_services(_request(app)).dispatcher.resolve(tier1)._client
        assert not is_closed(owned)
    assert is_closed(owned)
    assert not hasattr(app.state, "services")

    injected = create_app(make_settings(), http_client=http_client)
    async with injected.router.lifespan_context(injected):
        pass
    assert not http_client.is_closed


def is_closed(client: httpx.AsyncClient) -> bool:
    return client.is_closed


def _request(app: FastAPI) -> Request:
    return Request({"type": "http", "app": app, "headers": [], "method": "GET", "path": "/"})


def test_services_require_lifespan() -> None:
    with pytest.raises(RuntimeError, match="lifespan not run"):
        get_services(_request(create_app(make_settings())))


async def test_unhandled_exception_returns_typed_500(app_factory: AppFactory) -> None:
    app = app_factory()

    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    app.add_api_route("/boom", boom)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw"
        ) as gw,
    ):
        response = await gw.get("/boom")
    assert response.status_code == 500
    assert error_of(response).error.code == "internal_error"
    assert "secret internal detail" not in response.text


# --- Error mapping units ------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (CacheError("x"), 503, "dependency_unavailable"),
        (EmbeddingError("x"), 503, "dependency_unavailable"),
        (GatewayError("x"), 500, "internal_error"),
        (ProviderError("x", provider=ModelProvider.MISTRAL), 502, "upstream_error"),
    ],
)
def test_gateway_error_detail(exc: GatewayError, status: int, code: str) -> None:
    mapped_status, detail = gateway_error_detail(exc)
    assert (mapped_status, detail.code) == (status, code)
    assert detail.message != "x"


async def test_handlers_tolerate_unexpected_exception_types() -> None:
    request = _request(create_app(make_settings()))
    for handler in (_on_gateway_error, _on_validation_error, _on_http_error):
        response = await handler(request, ValueError("unexpected"))
        assert response.status_code in {422, 500}
    teapot = await _on_http_error(request, StarletteHTTPException(status_code=599))
    assert teapot.status_code == 599
    assert json.loads(bytes(teapot.body))["error"]["code"] == "error"
    assert (await _on_validation_error(request, RequestValidationError([]))).status_code == 422


# --- Factory builders ---------------------------------------------------------


def test_build_adapters_registers_configured_providers(http_client: httpx.AsyncClient) -> None:
    key = SecretStr(DUMMY_KEY)
    settings = make_settings(
        openrouter_api_key=key,
        deepseek_api_key=key,
        groq_api_key=key,
        together_api_key=key,
        openai_api_key=key,
        vllm_base_url="http://gpu:8000/v1",
        ollama_base_url="http://localhost:11434/v1",
    )
    providers = {a.provider for a in build_adapters(settings, http_client)}
    assert providers == set(ModelProvider)
    assert (
        build_adapters(make_settings(mistral_api_key=None, anthropic_api_key=None), http_client)
        == []
    )


def test_build_embedder_and_cache(
    http_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ai_inference_gateway.tools.qdrant_cache.AsyncQdrantClient", RecordingQdrant
    )
    RecordingQdrant.calls.clear()
    remote_embedder = build_embedder(
        make_settings(
            embedding_backend="openai_compatible",
            embedding_base_url="https://api.mistral.ai/v1",
            embedding_model="mistral-embed",
            embedding_dimension=1024,
        ),
        http_client,
    )
    assert isinstance(remote_embedder, OpenAICompatibleEmbedder)
    assert remote_embedder.dimension == 1024
    remote_cache = build_cache(
        make_settings(qdrant_url="https://qdrant.internal:6333", qdrant_api_key=SecretStr("k")),
        None,
    )
    assert remote_cache.dimension == 384
    assert RecordingQdrant.calls[0]["url"] == "https://qdrant.internal:6333"


def test_settings_validation() -> None:
    with pytest.raises(ValidationError, match="embedding_base_url"):
        GatewaySettings(embedding_backend="openai_compatible")
    with pytest.raises(ValidationError):
        GatewaySettings(similarity_threshold=0.9)


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("GATEWAY_TIER1_PROVIDER", "deepseek")
    monkeypatch.setenv("GATEWAY_SIMILARITY_THRESHOLD", "0.97")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    settings = GatewaySettings()
    assert settings.tier1_provider is ModelProvider.DEEPSEEK
    assert settings.similarity_threshold == 0.97
    assert settings.mistral_api_key is not None
    assert DUMMY_KEY not in repr(settings)


# --- Client disconnect mid-stream ---------------------------------------------


class SlowUpstream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False
        self.chunks = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i in range(10_000):
            self.chunks += 1
            yield openai_sse(f"tok{i} ", done=False)
            await asyncio.sleep(0.005)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_client_disconnect_closes_upstream_deterministically(spec: str) -> None:
    slow = SlowUpstream()
    async with mock_client(lambda _: httpx.Response(200, stream=slow)) as client:
        app = create_app(make_settings(), http_client=client)
        async with app.router.lifespan_context(app):
            body = json.dumps(
                {"prompt": "A very long story about lighthouses", "stream": True}
            ).encode()
            disconnect = asyncio.Event()
            requested = False
            frames = 0

            async def receive() -> Message:
                nonlocal requested
                if not requested:
                    requested = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message: Message) -> None:
                nonlocal frames
                if message["type"] == "http.response.body" and message.get("body"):
                    frames += 1
                    if frames >= 5:
                        if spec == "2.4":
                            raise OSError("client went away")
                        disconnect.set()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": spec},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/inference",
                "raw_path": b"/v1/inference",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json"), (b"host", b"gw")],
                "client": ("127.0.0.1", 5000),
                "server": ("gw", 80),
            }
            before = set(asyncio.all_tasks())
            if spec == "2.4":  # send() raising OSError surfaces as ClientDisconnect
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(app(scope, receive, send), timeout=5)
            else:
                await asyncio.wait_for(app(scope, receive, send), timeout=5)
            closed_at_return = slow.closed  # before any GC or extra loop iterations
            await asyncio.sleep(0.05)
            gc.collect()
            leftover = (
                {t for t in asyncio.all_tasks() if not t.done()} - before - {asyncio.current_task()}
            )

            assert frames >= 5
            assert slow.chunks < 50
            assert closed_at_return
            assert not leftover
            services = get_services(_request(app))
            assert (await services.cache._client.count("semantic_cache")).count == 0


# --- Hermeticity self-check ---------------------------------------------------


def _chain(exc: BaseException | None) -> list[BaseException]:
    """Every exception reachable through causes, contexts and exception groups."""
    seen: list[BaseException] = []
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or current in seen:
            continue
        seen.append(current)
        stack += [current.__cause__, current.__context__]
        if isinstance(current, BaseExceptionGroup):
            stack += list(current.exceptions)
    return seen


@pytest.mark.parametrize("url", ["https://api.anthropic.com/v1/messages", "http://93.184.216.34/"])
async def test_network_guard_blocks_real_calls(url: str) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError) as info:
            await client.get(url)
    assert any(isinstance(e, NetworkBlockedError) for e in _chain(info.value))


def test_environment_is_scrubbed() -> None:
    import os  # noqa: PLC0415 - local to keep the module import surface minimal

    leaked = [
        n
        for n in os.environ
        if n.endswith("_API_KEY") or n.startswith(("GATEWAY_", "REDIS_", "QDRANT_"))
    ]
    assert leaked == []
