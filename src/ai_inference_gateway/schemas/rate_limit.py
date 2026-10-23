"""Rate-limiting contracts."""

from pydantic import AwareDatetime, Field, FiniteFloat, IPvAnyAddress

from ai_inference_gateway.schemas.base import StrictModel


class TokenBucketStatus(StrictModel):
    """Outcome of a per-IP token bucket check."""

    client_ip: IPvAnyAddress
    tokens_remaining: FiniteFloat = Field(ge=0.0)
    reset_at: AwareDatetime = Field(description="When the bucket is next full again.")
    allowed: bool
