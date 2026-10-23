"""Strict Pydantic v2 data contracts for the gateway."""

from ai_inference_gateway.schemas.base import StrictModel
from ai_inference_gateway.schemas.cache import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheEntry,
    CacheQuery,
    CosineSimilarity,
    Embedding,
)
from ai_inference_gateway.schemas.inference import (
    ComplexityAssessment,
    ComplexityTier,
    InferenceRequest,
    InferenceResponse,
)
from ai_inference_gateway.schemas.rate_limit import TokenBucketStatus
from ai_inference_gateway.schemas.routing import (
    OPENAI_COMPATIBLE_PROVIDERS,
    TIER_PROVIDERS,
    ModelProvider,
    RouteTarget,
    RoutingTier,
)

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "OPENAI_COMPATIBLE_PROVIDERS",
    "TIER_PROVIDERS",
    "CacheEntry",
    "CacheQuery",
    "ComplexityAssessment",
    "ComplexityTier",
    "CosineSimilarity",
    "Embedding",
    "InferenceRequest",
    "InferenceResponse",
    "ModelProvider",
    "RouteTarget",
    "RoutingTier",
    "StrictModel",
    "TokenBucketStatus",
]
