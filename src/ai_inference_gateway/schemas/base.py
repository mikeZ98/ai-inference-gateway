"""Base contract model shared by every schema in the gateway."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Immutable, closed, non-coercing contract.

    - ``extra="forbid"``: unknown fields are rejected.
    - ``frozen=True``: instances are immutable (and hashable).
    - ``strict=True``: no implicit type coercion (``"1"`` is not an ``int``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
