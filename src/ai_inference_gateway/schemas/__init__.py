"""Strict Pydantic v2 data contracts for the gateway."""

from ai_inference_gateway.schemas.base import StrictModel
from ai_inference_gateway.schemas.cache import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheEntry,
    CacheQuery,
    CosineSimilarity,
    Embedding,
)
from ai_inference_gateway.schemas.errors import ErrorDetail, ErrorResponse, FieldError
from ai_inference_gateway.schemas.health import ComponentCheck, HealthStatus
from ai_inference_gateway.schemas.inference import (
    ComplexityAssessment,
    ComplexityTier,
    InferenceRequest,
    InferenceResponse,
)
from ai_inference_gateway.schemas.metrics import GuardrailMetricsSnapshot
from ai_inference_gateway.schemas.rate_limit import TokenBucketStatus
from ai_inference_gateway.schemas.routing import (
    OPENAI_COMPATIBLE_PROVIDERS,
    TIER_PROVIDERS,
    ModelProvider,
    RouteTarget,
    RoutingTier,
)
from ai_inference_gateway.schemas.streaming import (
    StreamDelta,
    StreamError,
    StreamEvent,
    StreamSummary,
)

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "OPENAI_COMPATIBLE_PROVIDERS",
    "TIER_PROVIDERS",
    "CacheEntry",
    "CacheQuery",
    "ComplexityAssessment",
    "ComplexityTier",
    "ComponentCheck",
    "CosineSimilarity",
    "Embedding",
    "ErrorDetail",
    "ErrorResponse",
    "FieldError",
    "GuardrailMetricsSnapshot",
    "HealthStatus",
    "InferenceRequest",
    "InferenceResponse",
    "ModelProvider",
    "RouteTarget",
    "RoutingTier",
    "StreamDelta",
    "StreamError",
    "StreamEvent",
    "StreamSummary",
    "StrictModel",
    "TokenBucketStatus",
]
