"""Provider and routing-tier contracts."""

from collections.abc import Mapping
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Final, Self

from pydantic import Field, model_validator

from ai_inference_gateway.schemas.base import StrictModel


class ModelProvider(StrEnum):
    """Upstream inference providers supported by the gateway."""

    OPENROUTER = "openrouter"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"


class RoutingTier(IntEnum):
    """Cascading routing tiers, cheapest first."""

    TIER_1 = 1
    """Low complexity: OpenRouter / Mistral."""
    TIER_2 = 2
    """High complexity: Claude (standard)."""
    TIER_3 = 3
    """Highest complexity: Claude (top model)."""


TIER_PROVIDERS: Final[Mapping[RoutingTier, frozenset[ModelProvider]]] = MappingProxyType(
    {
        RoutingTier.TIER_1: frozenset({ModelProvider.OPENROUTER, ModelProvider.MISTRAL}),
        RoutingTier.TIER_2: frozenset({ModelProvider.ANTHROPIC}),
        RoutingTier.TIER_3: frozenset({ModelProvider.ANTHROPIC}),
    }
)


class RouteTarget(StrictModel):
    """Resolved destination for a request: tier, provider and concrete model."""

    tier: RoutingTier
    provider: ModelProvider
    model: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:/-]+$")

    @model_validator(mode="after")
    def _provider_matches_tier(self) -> Self:
        allowed = TIER_PROVIDERS[self.tier]
        if self.provider not in allowed:
            names = ", ".join(sorted(p.value for p in allowed))
            raise ValueError(
                f"provider {self.provider.value!r} is not valid for tier {self.tier.value} "
                f"(allowed: {names})"
            )
        return self
