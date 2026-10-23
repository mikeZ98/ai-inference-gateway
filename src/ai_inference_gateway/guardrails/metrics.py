"""In-process guardrail counters and instrumentation decorators.

The decorators wrap the core ``Router`` / ``ProviderDispatcher`` ports, so the
engine stays unaware of metrics. Counters are per process (one event loop, no
locking needed); aggregate across workers at the scraper.
"""

from collections import Counter
from collections.abc import AsyncGenerator
from contextlib import aclosing

from ai_inference_gateway.core.ports import (
    ProviderCompletion,
    ProviderDispatcher,
    ProviderStreamEvent,
    Router,
    StreamCompleted,
)
from ai_inference_gateway.schemas import (
    ComplexityAssessment,
    ComplexityTier,
    GuardrailMetricsSnapshot,
    InferenceRequest,
    RouteTarget,
    RoutingTier,
)


class GuardrailMetrics:
    """Monotonic counters for rate limiting, routing decisions and token spend."""

    def __init__(self) -> None:
        """Start all counters at zero."""
        self._counts: Counter[str] = Counter()

    def record_rate_limit(self, *, allowed: bool | None) -> None:
        """Count a limiter decision; ``None`` means Redis was unavailable (failed open)."""
        if allowed is None:
            self._counts["rate_limiter_fail_open"] += 1
            self._counts["requests_allowed"] += 1
        elif allowed:
            self._counts["requests_allowed"] += 1
        else:
            self._counts["requests_rate_limited"] += 1

    def record_payload_rejected(self) -> None:
        """Count a refused oversized request body."""
        self._counts["payloads_rejected"] += 1

    def record_route(self, assessment: ComplexityAssessment, target: RouteTarget) -> None:
        """Count a complexity decision and the tier it routed to."""
        tier = "complexity_high" if assessment.tier is ComplexityTier.HIGH else "complexity_low"
        self._counts[tier] += 1
        self._counts[f"routed_tier{target.tier.value}"] += 1

    def record_completion(self, tokens_used: int) -> None:
        """Count an upstream completion and its token usage."""
        self._counts["upstream_completions"] += 1
        self._counts["tokens_consumed"] += tokens_used

    def snapshot(self) -> GuardrailMetricsSnapshot:
        """Immutable view of the current counters."""
        c = self._counts
        return GuardrailMetricsSnapshot(
            requests_allowed=c["requests_allowed"],
            requests_rate_limited=c["requests_rate_limited"],
            rate_limiter_fail_open=c["rate_limiter_fail_open"],
            payloads_rejected=c["payloads_rejected"],
            complexity_low=c["complexity_low"],
            complexity_high=c["complexity_high"],
            routed_tier1=c[f"routed_tier{RoutingTier.TIER_1.value}"],
            routed_tier2=c[f"routed_tier{RoutingTier.TIER_2.value}"],
            routed_tier3=c[f"routed_tier{RoutingTier.TIER_3.value}"],
            upstream_completions=c["upstream_completions"],
            tokens_consumed=c["tokens_consumed"],
        )


class InstrumentedRouter:
    """``Router`` decorator recording complexity tiers and routing decisions."""

    def __init__(self, inner: Router, metrics: GuardrailMetrics) -> None:
        """Wrap ``inner``."""
        self._inner = inner
        self._metrics = metrics

    def resolve(self, request: InferenceRequest, assessment: ComplexityAssessment) -> RouteTarget:
        """Delegate, then record the decision (failures are not counted)."""
        target = self._inner.resolve(request, assessment)
        self._metrics.record_route(assessment, target)
        return target


class InstrumentedDispatcher:
    """``ProviderDispatcher`` decorator recording upstream token consumption."""

    def __init__(self, inner: ProviderDispatcher, metrics: GuardrailMetrics) -> None:
        """Wrap ``inner``."""
        self._inner = inner
        self._metrics = metrics

    async def complete(self, target: RouteTarget, request: InferenceRequest) -> ProviderCompletion:
        """Delegate and count tokens of the completed call."""
        completion = await self._inner.complete(target, request)
        self._metrics.record_completion(completion.tokens_used)
        return completion

    async def stream(
        self, target: RouteTarget, request: InferenceRequest
    ) -> AsyncGenerator[ProviderStreamEvent]:
        """Delegate; count tokens once the stream completes."""
        async with aclosing(self._inner.stream(target, request)) as events:
            async for event in events:
                if isinstance(event, StreamCompleted):
                    self._metrics.record_completion(event.tokens_used)
                yield event
