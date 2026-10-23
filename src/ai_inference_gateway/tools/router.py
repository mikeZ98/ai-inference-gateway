"""Complexity-based multi-model router (implements the core ``Router`` port)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Self

from pydantic import Field, model_validator

from ai_inference_gateway.core.errors import ModelNotAvailableError, RoutingError
from ai_inference_gateway.schemas import (
    ComplexityAssessment,
    ComplexityTier,
    InferenceRequest,
    ModelProvider,
    RouteTarget,
    RoutingTier,
    StrictModel,
)


class RouterConfig(StrictModel):
    """Which model serves each tier, and which models clients may request by name.

    - LOW complexity -> ``tier1`` (any OpenAI-compatible provider).
    - HIGH complexity -> ``tier2``, or ``tier3`` once the score reaches
      ``tier3_min_score``.
    - An explicit ``InferenceRequest.model`` must name a tier model or an entry
      in ``extra_models``; anything else is rejected (cost/abuse control).
    """

    tier1: RouteTarget = RouteTarget(
        tier=RoutingTier.TIER_1, provider=ModelProvider.MISTRAL, model="mistral-small-latest"
    )
    tier2: RouteTarget = RouteTarget(
        tier=RoutingTier.TIER_2, provider=ModelProvider.ANTHROPIC, model="claude-sonnet-5-5"
    )
    tier3: RouteTarget = RouteTarget(
        tier=RoutingTier.TIER_3, provider=ModelProvider.ANTHROPIC, model="claude-opus-5-5"
    )
    tier3_min_score: float = Field(default=0.8, gt=0.0, le=1.0)
    extra_models: tuple[RouteTarget, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for expected, target in (
            (RoutingTier.TIER_1, self.tier1),
            (RoutingTier.TIER_2, self.tier2),
            (RoutingTier.TIER_3, self.tier3),
        ):
            if target.tier is not expected:
                raise ValueError(f"tier{expected.value} target must have tier={expected.value}")
        seen: dict[str, RouteTarget] = {}
        for target in (self.tier1, self.tier2, self.tier3, *self.extra_models):
            previous = seen.setdefault(target.model, target)
            if previous != target:
                raise ValueError(f"model {target.model!r} is mapped to conflicting targets")
        return self

    def catalog(self) -> Mapping[str, RouteTarget]:
        """Model id -> target for every model that may be requested explicitly."""
        targets = (self.tier1, self.tier2, self.tier3, *self.extra_models)
        return MappingProxyType({target.model: target for target in targets})


class ComplexityRouter:
    """Maps a request plus its complexity assessment to a ``RouteTarget``."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        """Use ``config`` or the default tier models."""
        self._config = config or RouterConfig()
        self._catalog = self._config.catalog()

    @property
    def config(self) -> RouterConfig:
        """Active routing configuration."""
        return self._config

    def resolve(self, request: InferenceRequest, assessment: ComplexityAssessment) -> RouteTarget:
        """Honor a known model override, else route by complexity tier."""
        if request.model is not None:
            target = self._catalog.get(request.model)
            if target is None:
                raise ModelNotAvailableError(
                    f"model {request.model!r} is not available on this gateway"
                )
            return target

        config = self._config
        if assessment.tier is ComplexityTier.LOW:
            return config.tier1
        if assessment.tier is ComplexityTier.HIGH:
            return config.tier3 if assessment.score >= config.tier3_min_score else config.tier2
        raise RoutingError(f"unsupported complexity tier {assessment.tier!r}")
