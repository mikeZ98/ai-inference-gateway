"""Inference request/response and complexity contracts."""

from enum import StrEnum
from typing import Final

from pydantic import Field, FiniteFloat, field_validator

from ai_inference_gateway.schemas.base import StrictModel
from ai_inference_gateway.schemas.routing import ModelProvider

MAX_PROMPT_CHARS: Final = 100_000
MAX_OUTPUT_TOKENS: Final = 32_768
MODEL_ID_PATTERN: Final = r"^[A-Za-z0-9._:/-]+$"


class InferenceRequest(StrictModel):
    """Incoming client payload.

    Only JSON-native scalar types are used so strict validation behaves the
    same for parsed-dict and raw-JSON input.
    """

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=MODEL_ID_PATTERN,
        description="Explicit model override; bypasses complexity routing.",
    )
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class ComplexityTier(StrEnum):
    """Coarse complexity bucket produced by the heuristics."""

    LOW = "low"
    HIGH = "high"


class ComplexityAssessment(StrictModel):
    """Result of scoring a prompt's complexity."""

    score: FiniteFloat = Field(ge=0.0, le=1.0)
    tier: ComplexityTier
    matched_heuristics: tuple[str, ...] = ()


class InferenceResponse(StrictModel):
    """Normalized response returned to clients, whatever the upstream provider."""

    content: str
    model_used: str = Field(min_length=1, max_length=128, pattern=MODEL_ID_PATTERN)
    provider: ModelProvider
    tokens_used: int = Field(ge=0)
    cached_hit: bool
    latency_ms: FiniteFloat = Field(ge=0.0)
