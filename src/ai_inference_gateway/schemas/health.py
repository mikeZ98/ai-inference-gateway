"""Liveness / readiness payloads."""

from typing import Literal

from ai_inference_gateway.schemas.base import StrictModel


class ComponentCheck(StrictModel):
    """Result of probing one dependency."""

    name: str
    ok: bool


class HealthStatus(StrictModel):
    """``ok`` for liveness; ``ready`` / ``not_ready`` for readiness."""

    status: Literal["ok", "ready", "not_ready"]
    checks: tuple[ComponentCheck, ...] = ()
