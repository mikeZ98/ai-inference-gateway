"""Server-Sent Events frame payloads for streaming inference.

A stream is ``delta*`` then exactly one ``summary`` (success) or ``error``
(failure after the stream started), followed by the literal ``[DONE]`` frame.
"""

from typing import Literal

from pydantic import Field, FiniteFloat

from ai_inference_gateway.schemas.base import StrictModel
from ai_inference_gateway.schemas.errors import ErrorDetail
from ai_inference_gateway.schemas.inference import MODEL_ID_PATTERN
from ai_inference_gateway.schemas.routing import ModelProvider


class StreamDelta(StrictModel):
    """Incremental piece of generated content."""

    type: Literal["delta"] = "delta"
    content: str = Field(min_length=1)


class StreamSummary(StrictModel):
    """Final metadata; mirrors ``InferenceResponse`` without the content."""

    type: Literal["summary"] = "summary"
    model_used: str = Field(min_length=1, max_length=128, pattern=MODEL_ID_PATTERN)
    provider: ModelProvider
    tokens_used: int = Field(ge=0)
    cached_hit: bool
    latency_ms: FiniteFloat = Field(ge=0.0)


class StreamError(StrictModel):
    """Failure after the response headers were already sent."""

    type: Literal["error"] = "error"
    error: ErrorDetail


type StreamEvent = StreamDelta | StreamSummary | StreamError
