"""Semantic cache (Qdrant) contracts."""

from typing import Annotated, Final

from pydantic import AwareDatetime, Field, FiniteFloat

from ai_inference_gateway.schemas.base import StrictModel
from ai_inference_gateway.schemas.inference import MODEL_ID_PATTERN
from ai_inference_gateway.schemas.routing import ModelProvider

DEFAULT_SIMILARITY_THRESHOLD: Final = 0.95

Embedding = Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
CosineSimilarity = Annotated[FiniteFloat, Field(ge=-1.0, le=1.0)]


class CacheEntry(StrictModel):
    """A cached prompt/response pair as stored in (and returned from) Qdrant."""

    prompt: str = Field(min_length=1)
    embedding: Embedding
    response: str
    model_used: str = Field(min_length=1, max_length=128, pattern=MODEL_ID_PATTERN)
    provider: ModelProvider
    similarity_score: CosineSimilarity | None = Field(
        default=None, description="Set on lookup hits; None for freshly stored entries."
    )
    timestamp: AwareDatetime


class CacheQuery(StrictModel):
    """Nearest-neighbour lookup; a hit requires similarity strictly above the threshold."""

    embedding: Embedding
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.95, lt=1.0)
    limit: int = Field(default=1, ge=1, le=10)

    def is_hit(self, score: float) -> bool:
        """Return whether ``score`` clears the (exclusive) similarity threshold."""
        return score > self.similarity_threshold
