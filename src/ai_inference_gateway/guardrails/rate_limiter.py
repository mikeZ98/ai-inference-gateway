"""Redis-backed token bucket, evaluated atomically by a Lua script.

Each client key holds ``tokens`` and ``ts`` (last update, epoch seconds). On
every check the bucket is refilled for the elapsed time (capped at capacity)
and ``cost`` tokens are taken if available. The whole read-modify-write runs
inside Redis, so concurrent gateway replicas never double-spend a bucket.

By default the clock is Redis ``TIME``, so replicas with skewed clocks still
share one timeline; tests may inject a clock for determinism.
"""

import asyncio
import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from ipaddress import IPv4Address, IPv6Address, ip_network
from typing import Final, Self

from pydantic import Field, model_validator
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai_inference_gateway.schemas import StrictModel, TokenBucketStatus

logger = logging.getLogger(__name__)

IPV6_BUCKET_PREFIX: Final = 64

BUCKET_SCRIPT_LUA: Final = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
if not now then
  local t = redis.call('TIME')
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000
end

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
return {allowed, tostring(tokens), tostring(now)}
"""

type ClientAddress = IPv4Address | IPv6Address


class TokenBucketPolicy(StrictModel):
    """Bucket shape shared by every client."""

    capacity: int = Field(default=60, ge=1, le=1_000_000)
    refill_per_second: float = Field(default=1.0, gt=0.0, le=1_000_000.0)
    cost: int = Field(default=1, ge=1)
    key_prefix: str = Field(default="gw:rl:", min_length=1, max_length=64)

    @model_validator(mode="after")
    def _cost_fits(self) -> Self:
        if self.cost > self.capacity:
            raise ValueError("cost cannot exceed capacity (requests could never pass)")
        return self


def bucket_key(prefix: str, address: ClientAddress) -> str:
    """Redis key for ``address``; IPv6 clients share a bucket per /64."""
    if isinstance(address, IPv6Address) and address.ipv4_mapped is None:
        return f"{prefix}{ip_network(f'{address}/{IPV6_BUCKET_PREFIX}', strict=False)}"
    mapped = address.ipv4_mapped if isinstance(address, IPv6Address) else None
    return f"{prefix}{mapped or address}"


def _to_float(value: object) -> float:
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str | int | float):
        raise TypeError(f"unexpected token bucket field {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite token bucket field")
    return result


class RedisTokenBucketLimiter:
    """Per-client token bucket. ``check`` never raises: Redis failures fail open."""

    def __init__(
        self,
        redis: Redis,
        policy: TokenBucketPolicy | None = None,
        *,
        timeout_s: float = 0.25,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Bind to a Redis client; ``clock`` overrides Redis ``TIME`` (tests)."""
        self._policy = policy or TokenBucketPolicy()
        self._script = redis.register_script(BUCKET_SCRIPT_LUA)
        self._timeout_s = timeout_s
        self._clock = clock

    @property
    def policy(self) -> TokenBucketPolicy:
        """Active bucket policy."""
        return self._policy

    async def check(self, address: ClientAddress) -> TokenBucketStatus | None:
        """Consume ``cost`` tokens for ``address``.

        Returns the bucket status, or ``None`` when Redis is unavailable (the
        caller must then let the request through: fail-open).
        """
        policy = self._policy
        now = "" if self._clock is None else repr(self._clock())
        try:
            async with asyncio.timeout(self._timeout_s):
                raw = await self._script(
                    keys=[bucket_key(policy.key_prefix, address)],
                    args=[policy.capacity, policy.refill_per_second, policy.cost, now],
                )
            allowed, tokens, server_now = (_to_float(item) for item in raw)
        except (RedisError, OSError, TimeoutError, TypeError, ValueError) as exc:
            logger.warning(
                "rate limiter unavailable; failing open", extra={"error": type(exc).__name__}
            )
            return None

        tokens = max(0.0, min(float(policy.capacity), tokens))
        seconds_to_full = (policy.capacity - tokens) / policy.refill_per_second
        return TokenBucketStatus(
            client_ip=address,
            tokens_remaining=tokens,
            reset_at=datetime.fromtimestamp(server_now + seconds_to_full, UTC),
            allowed=allowed == 1.0,
        )

    def retry_after_s(self, status: TokenBucketStatus) -> int:
        """Whole seconds until ``cost`` tokens are available again (>= 1 when denied)."""
        missing = self._policy.cost - status.tokens_remaining
        if missing <= 0:
            return 0
        return max(1, math.ceil(missing / self._policy.refill_per_second))
