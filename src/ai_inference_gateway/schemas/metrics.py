"""Runtime guardrail metrics payload (per process)."""

from pydantic import Field

from ai_inference_gateway.schemas.base import StrictModel


class GuardrailMetricsSnapshot(StrictModel):
    """Monotonic counters since process start."""

    requests_allowed: int = Field(ge=0)
    requests_rate_limited: int = Field(ge=0)
    rate_limiter_fail_open: int = Field(ge=0)
    payloads_rejected: int = Field(ge=0)
    complexity_low: int = Field(ge=0)
    complexity_high: int = Field(ge=0)
    routed_tier1: int = Field(ge=0)
    routed_tier2: int = Field(ge=0)
    routed_tier3: int = Field(ge=0)
    upstream_completions: int = Field(ge=0)
    tokens_consumed: int = Field(ge=0)
