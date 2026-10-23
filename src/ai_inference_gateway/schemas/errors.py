"""Public error contract returned by every failing endpoint."""

from pydantic import Field

from ai_inference_gateway.schemas.base import StrictModel


class FieldError(StrictModel):
    """One request validation failure (the offending input is never echoed)."""

    loc: tuple[str | int, ...]
    message: str
    type: str


class ErrorDetail(StrictModel):
    """Machine-readable code plus a safe, human-readable message."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str
    retryable: bool = False
    field_errors: tuple[FieldError, ...] = ()


class ErrorResponse(StrictModel):
    """Envelope: ``{"error": {...}}``."""

    error: ErrorDetail
