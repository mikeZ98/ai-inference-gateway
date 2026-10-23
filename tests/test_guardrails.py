"""Guardrails: token bucket (real Lua via fakeredis + lupa), client IP, body limits, metrics."""

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from ipaddress import ip_address
from typing import Any

import fakeredis
import httpx
import pytest
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

from ai_inference_gateway.api import GatewaySettings
from ai_inference_gateway.guardrails import (
    BodySizeLimitMiddleware,
    GuardrailMetrics,
    RedisTokenBucketLimiter,
    TokenBucketPolicy,
    parse_networks,
    resolve_client_ip,
)
from ai_inference_gateway.guardrails.middleware import UNKNOWN_CLIENT
from ai_inference_gateway.guardrails.rate_limiter import bucket_key
from ai_inference_gateway.schemas import ErrorResponse, GuardrailMetricsSnapshot, InferenceResponse
from tests.support import AppFactory, make_settings, serve

START = 1_800_000_000.0
A = ip_address("203.0.113.7")


class Clock:
    def __init__(self) -> None:
        self.t = START

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


def limiter(
    redis: fakeredis.FakeAsyncRedis, clock: Clock | None, **policy: Any
) -> RedisTokenBucketLimiter:
    shape = {"capacity": 3, "refill_per_second": 1.0, **policy}
    return RedisTokenBucketLimiter(redis, TokenBucketPolicy(**shape), clock=clock)


# --- Token bucket -------------------------------------------------------------


async def test_bucket_dynamics(redis: fakeredis.FakeAsyncRedis, clock: Clock) -> None:
    lim = limiter(redis, clock)
    seq = [await lim.check(A) for _ in range(4)]
    assert all(s is not None for s in seq)
    statuses = [s for s in seq if s is not None]
    assert [s.allowed for s in statuses] == [True, True, True, False]
    assert [s.tokens_remaining for s in statuses] == [2.0, 1.0, 0.0, 0.0]
    assert lim.retry_after_s(statuses[3]) == 1
    assert lim.retry_after_s(statuses[0]) == 0
    assert statuses[2].reset_at == datetime.fromtimestamp(START + 3.0, UTC)
    assert lim.policy.capacity == 3

    clock.t += 1.5
    refilled = await lim.check(A)
    assert refilled is not None
    assert refilled.allowed
    assert refilled.tokens_remaining == pytest.approx(0.5)

    clock.t += 1000
    capped = await lim.check(A)
    assert capped is not None
    assert capped.tokens_remaining == 2.0

    clock.t -= 50
    backwards = await lim.check(A)
    assert backwards is not None
    assert backwards.tokens_remaining == 1.0  # a clock going backwards never adds tokens

    assert 0 < await redis.ttl(bucket_key("gw:rl:", A)) <= 4
    other = await lim.check(ip_address("198.51.100.1"))
    assert other is not None
    assert other.tokens_remaining == 2.0


async def test_cost_per_request(redis: fakeredis.FakeAsyncRedis, clock: Clock) -> None:
    lim = limiter(redis, clock, capacity=5, cost=2)
    results = [await lim.check(A) for _ in range(3)]
    assert [r.allowed for r in results if r] == [True, True, False]
    last = results[2]
    assert last is not None
    assert last.tokens_remaining == 1.0
    assert lim.retry_after_s(last) == 1


def test_policy_validation() -> None:
    with pytest.raises(ValidationError, match="cost cannot exceed capacity"):
        TokenBucketPolicy(capacity=1, cost=2)


async def test_lua_is_atomic_under_concurrency(
    redis: fakeredis.FakeAsyncRedis, clock: Clock
) -> None:
    lim = limiter(redis, clock, capacity=10, refill_per_second=0.0001)
    results = await asyncio.gather(*(lim.check(ip_address("192.0.2.1")) for _ in range(50)))
    assert sum(1 for r in results if r is not None and r.allowed) == 10


async def test_redis_time_is_default_clock(redis: fakeredis.FakeAsyncRedis) -> None:
    status = await limiter(redis, None, capacity=2).check(A)
    assert status is not None
    assert status.allowed
    assert abs(status.reset_at.timestamp() - (time.time() + 1)) < 5


@pytest.mark.parametrize(
    ("address", "key"),
    [
        ("2001:db8::1", "p:2001:db8::/64"),
        ("2001:db8::ffff:1", "p:2001:db8::/64"),
        ("2001:db8:0:1::1", "p:2001:db8:0:1::/64"),
        ("::ffff:203.0.113.7", "p:203.0.113.7"),
        ("203.0.113.7", "p:203.0.113.7"),
    ],
)
def test_bucket_keys(address: str, key: str) -> None:
    assert bucket_key("p:", ip_address(address)) == key


async def test_fail_open_on_connection_error(
    redis: fakeredis.FakeAsyncRedis, redis_server: fakeredis.FakeServer, clock: Clock
) -> None:
    redis_server.connected = False
    assert await limiter(redis, clock).check(A) is None


class SlowRedis(fakeredis.FakeAsyncRedis):
    async def evalsha(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)


class GarbageRedis(fakeredis.FakeAsyncRedis):
    reply: Any = None

    async def evalsha(self, *args: Any, **kwargs: Any) -> Any:
        return self.reply


async def test_fail_open_on_timeout(redis_server: fakeredis.FakeServer) -> None:
    lim = RedisTokenBucketLimiter(SlowRedis(server=redis_server), timeout_s=0.05)
    started = time.perf_counter()
    assert await lim.check(A) is None
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize("reply", [[1, b"nan", b"1"], [1, None, b"1"], [1, b"x", b"1"], "nope"])
async def test_fail_open_on_malformed_reply(redis_server: fakeredis.FakeServer, reply: Any) -> None:
    client = GarbageRedis(server=redis_server)
    client.reply = reply
    assert await RedisTokenBucketLimiter(client).check(A) is None


# --- Client IP resolution -----------------------------------------------------


TRUSTED = parse_networks(["10.0.0.0/8", "127.0.0.1", " "])


def scope(peer: str | None, *xff: str) -> Scope:
    return {
        "type": "http",
        "client": (peer, 1234) if peer else None,
        "headers": [(b"x-forwarded-for", v.encode()) for v in xff],
    }


@pytest.mark.parametrize(
    ("peer", "xff", "trusted", "expected"),
    [
        ("198.51.100.9", ("1.1.1.1",), (), "198.51.100.9"),
        ("198.51.100.9", ("1.1.1.1",), TRUSTED, "198.51.100.9"),
        ("10.0.0.9", ("1.1.1.1, 10.0.0.5",), TRUSTED, "1.1.1.1"),
        ("10.0.0.9", ("6.6.6.6, 1.1.1.1",), TRUSTED, "1.1.1.1"),
        ("10.0.0.9", ("1.1.1.1", "10.0.0.5"), TRUSTED, "1.1.1.1"),
        ("10.0.0.9", ("1.1.1.1, not-an-ip",), TRUSTED, "10.0.0.9"),
        ("10.0.0.9", ("[2001:db8::7]:443",), TRUSTED, "2001:db8::7"),
        ("10.0.0.9", ("[broken",), TRUSTED, "10.0.0.9"),
        ("10.0.0.9", ("1.1.1.1:5000",), TRUSTED, "1.1.1.1"),
        ("10.0.0.9", ("10.0.0.1",), TRUSTED, "10.0.0.1"),
        ("10.0.0.9", (), TRUSTED, "10.0.0.9"),
        (None, (), TRUSTED, str(UNKNOWN_CLIENT)),
        ("not-an-ip", (), TRUSTED, str(UNKNOWN_CLIENT)),
    ],
)
def test_resolve_client_ip(
    peer: str | None, xff: tuple[str, ...], trusted: tuple[Any, ...], expected: str
) -> None:
    assert str(resolve_client_ip(scope(peer, *xff), trusted)) == expected


# --- Body size middleware (raw ASGI) ------------------------------------------


async def echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


async def test_body_limit_passes_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    async def receive() -> Message:
        return {}

    async def send(message: Message) -> None:
        return None

    await BodySizeLimitMiddleware(app, max_bytes=10)({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]
    with pytest.raises(ValueError, match="positive"):
        BodySizeLimitMiddleware(app, max_bytes=0)


async def test_body_limit_allows_small_bodies() -> None:
    metrics = GuardrailMetrics()
    app = BodySizeLimitMiddleware(echo_app, max_bytes=10, metrics=metrics)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/", content=b"small")).text == "small"
    assert metrics.snapshot().payloads_rejected == 0


# --- API integration ----------------------------------------------------------


def guarded_settings(**overrides: Any) -> GatewaySettings:
    base: dict[str, Any] = {
        "rate_limit_capacity": 3,
        "rate_limit_refill_per_second": 0.001,
        "max_body_bytes": 2048,
        "trusted_proxies": "127.0.0.1/32",
    }
    return make_settings(**{**base, **overrides})


@pytest.fixture
async def gw(
    app_factory: AppFactory, redis: fakeredis.FakeAsyncRedis
) -> AsyncIterator[httpx.AsyncClient]:
    async with serve(app_factory(settings=guarded_settings(), redis_client=redis)) as client:
        yield client


def from_ip(ip: str) -> dict[str, str]:
    return {"x-forwarded-for": ip}


async def metrics_of(gw: httpx.AsyncClient) -> GuardrailMetricsSnapshot:
    return GuardrailMetricsSnapshot.model_validate_json((await gw.get("/metrics")).content)


async def test_rate_limit_headers_and_429(gw: httpx.AsyncClient) -> None:
    ok = [
        await gw.post("/v1/inference", json={"prompt": f"q {i}"}, headers=from_ip("203.0.113.50"))
        for i in range(3)
    ]
    assert [r.status_code for r in ok] == [200, 200, 200]
    assert [r.headers["x-ratelimit-remaining"] for r in ok] == ["2", "1", "0"]
    assert all(r.headers["x-ratelimit-limit"] == "3" for r in ok)
    assert int(ok[-1].headers["x-ratelimit-reset"]) > time.time()

    denied = await gw.post(
        "/v1/inference", json={"prompt": "again"}, headers=from_ip("203.0.113.50")
    )
    error = ErrorResponse.model_validate_json(denied.content).error
    assert denied.status_code == 429
    assert (error.code, error.retryable) == ("rate_limited", True)
    assert int(denied.headers["retry-after"]) >= 1
    assert denied.headers["x-ratelimit-remaining"] == "0"

    other = await gw.post(
        "/v1/inference", json={"prompt": "other"}, headers=from_ip("203.0.113.51")
    )
    assert (other.status_code, other.headers["x-ratelimit-remaining"]) == (200, "2")
    spoofed = await gw.post(
        "/v1/inference", json={"prompt": "x"}, headers=from_ip("9.9.9.9, 203.0.113.50")
    )
    assert spoofed.status_code == 429

    health = [await gw.get("/health") for _ in range(5)]
    assert all("x-ratelimit-limit" not in r.headers for r in health)


async def test_sse_carries_rate_limit_headers(gw: httpx.AsyncClient) -> None:
    response = await gw.post(
        "/v1/inference", json={"prompt": "stream", "stream": True}, headers=from_ip("203.0.113.52")
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-ratelimit-remaining"] == "2"
    assert response.text.endswith("data: [DONE]\n\n")


async def test_fail_open_when_redis_dies(
    gw: httpx.AsyncClient, redis_server: fakeredis.FakeServer
) -> None:
    redis_server.connected = False
    response = await gw.post("/v1/inference", json={"prompt": "redis down"})
    assert response.status_code == 200
    assert "x-ratelimit-limit" not in response.headers
    assert (await metrics_of(gw)).rate_limiter_fail_open == 1


async def test_body_size_guard(gw: httpx.AsyncClient) -> None:
    declared = await gw.post(
        "/v1/inference",
        content=b'{"prompt": "' + b"y" * 5000 + b'"}',
        headers={"content-type": "application/json", **from_ip("203.0.113.60")},
    )
    assert declared.status_code == 413
    assert ErrorResponse.model_validate_json(declared.content).error.code == "content_too_large"

    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"prompt": "'
        for _ in range(10):
            yield b"z" * 500
        yield b'"}'

    streamed = await gw.post(
        "/v1/inference",
        content=chunks(),
        headers={"content-type": "application/json", **from_ip("203.0.113.61")},
    )
    assert "content-length" not in streamed.request.headers
    assert streamed.status_code == 413
    assert ErrorResponse.model_validate_json(streamed.content).error.code == "content_too_large"

    invalid = await gw.post(
        "/v1/inference",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": "abc",
            **from_ip("203.0.113.62"),
        },
    )
    assert invalid.status_code == 400
    assert (await metrics_of(gw)).payloads_rejected == 2


async def test_metrics_track_tiers_and_tokens(gw: httpx.AsyncClient) -> None:
    await gw.post(
        "/v1/inference", json={"prompt": "capital of France?"}, headers=from_ip("203.0.113.70")
    )
    await gw.post(
        "/v1/inference",
        json={"prompt": "stream it", "stream": True},
        headers=from_ip("203.0.113.71"),
    )
    await gw.post(
        "/v1/inference",
        json={"prompt": "Debug and explain why:\n```py\nx=1/0\n```"},
        headers=from_ip("203.0.113.72"),
    )
    m = await metrics_of(gw)
    assert (m.complexity_low, m.complexity_high) == (2, 1)
    assert (m.routed_tier1, m.routed_tier2, m.routed_tier3) == (2, 1, 0)
    assert m.upstream_completions == 3
    assert m.tokens_consumed == 12 + 12 + 29
    assert m.requests_allowed == 3


async def test_complexity_tuning_via_settings(app_factory: AppFactory) -> None:
    settings = guarded_settings(complexity_high_threshold=0.2)
    async with serve(app_factory(settings=settings)) as gw:
        response = await gw.post("/v1/inference", json={"prompt": "Explain why the sky is blue"})
    assert InferenceResponse.model_validate_json(response.content).model_used == "claude-sonnet-5-5"


async def test_disabled_rate_limiting(
    app_factory: AppFactory, redis: fakeredis.FakeAsyncRedis
) -> None:
    settings = guarded_settings(rate_limit_enabled=False)
    async with serve(app_factory(settings=settings, redis_client=redis)) as gw:
        responses = [await gw.post("/v1/inference", json={"prompt": "same"}) for _ in range(5)]
        snapshot = await metrics_of(gw)
    assert all(r.status_code == 200 and "x-ratelimit-limit" not in r.headers for r in responses)
    assert (snapshot.rate_limiter_fail_open, snapshot.requests_allowed) == (0, 0)


async def test_no_redis_url_disables_limiter(app_factory: AppFactory) -> None:
    async with serve(app_factory(settings=guarded_settings())) as gw:
        responses = [await gw.post("/v1/inference", json={"prompt": "same"}) for _ in range(5)]
    assert all(r.status_code == 200 for r in responses)


async def test_unreachable_redis_url_fails_open(app_factory: AppFactory) -> None:
    started = time.perf_counter()
    async with serve(
        app_factory(settings=guarded_settings(redis_url="redis://127.0.0.1:1/0"))
    ) as gw:
        responses = [await gw.post("/v1/inference", json={"prompt": "no redis"}) for _ in range(3)]
        snapshot = await metrics_of(gw)
    assert all(r.status_code == 200 for r in responses)
    assert snapshot.rate_limiter_fail_open == 3
    assert time.perf_counter() - started < 10  # owned pool closed, all timeouts bounded


async def test_injected_redis_left_open(
    app_factory: AppFactory, redis: fakeredis.FakeAsyncRedis
) -> None:
    async with serve(app_factory(settings=guarded_settings(), redis_client=redis)):
        pass
    assert await redis.ping()


# --- Settings -----------------------------------------------------------------


def test_guardrail_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_TRUSTED_PROXIES", "10.0.0.0/8, 192.168.1.1")
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/0")
    settings = GatewaySettings()
    assert settings.trusted_proxies == ("10.0.0.0/8", "192.168.1.1")
    assert len(settings.trusted_networks()) == 2
    assert settings.redis_url == "redis://cache:6379/0"
    assert settings.token_bucket_policy().capacity == 60
    assert GatewaySettings(trusted_proxies=("127.0.0.1",)).trusted_proxies == ("127.0.0.1",)


@pytest.mark.parametrize(
    "bad",
    [
        {"trusted_proxies": "not-a-cidr"},
        {"rate_limit_capacity": 1, "rate_limit_cost": 2},
        {"complexity_long_prompt_tokens": 5000},
        {"max_body_bytes": 10},
    ],
)
def test_invalid_guardrail_settings(bad: dict[str, Any]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        GatewaySettings(**bad)
