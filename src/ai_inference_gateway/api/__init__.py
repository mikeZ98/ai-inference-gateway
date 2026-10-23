"""HTTP API: FastAPI app factory, routes, error mapping and SSE streaming."""

from ai_inference_gateway.api.app import create_app
from ai_inference_gateway.api.settings import GatewaySettings

__all__ = ["GatewaySettings", "create_app"]
