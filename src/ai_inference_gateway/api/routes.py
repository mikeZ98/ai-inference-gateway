"""HTTP endpoints."""

from dataclasses import dataclass
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from ai_inference_gateway.api.sse import SSEResponse, sse_frames
from ai_inference_gateway.core.engine import InferenceEngine
from ai_inference_gateway.core.errors import CacheError
from ai_inference_gateway.schemas import (
    ComponentCheck,
    ErrorResponse,
    HealthStatus,
    InferenceRequest,
    InferenceResponse,
)
from ai_inference_gateway.tools.providers import DynamicProviderDispatcher
from ai_inference_gateway.tools.qdrant_cache import QdrantSemanticCache

_ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status: {"model": ErrorResponse} for status in (400, 422, 500, 502, 503, 504)
}


@dataclass(frozen=True, slots=True)
class GatewayServices:
    """Long-lived collaborators created in the app lifespan."""

    engine: InferenceEngine
    cache: QdrantSemanticCache
    dispatcher: DynamicProviderDispatcher


def get_services(request: Request) -> GatewayServices:
    """Dependency: services attached to ``app.state`` by the lifespan."""
    services = getattr(request.app.state, "services", None)
    if isinstance(services, GatewayServices):
        return services
    raise RuntimeError("gateway services are not initialized (lifespan not run)")


ServicesDep = Annotated[GatewayServices, Depends(get_services)]

router = APIRouter()


@router.get("/health", response_model=HealthStatus, tags=["ops"])
async def health() -> HealthStatus:
    """Liveness: the process is up and serving requests."""
    return HealthStatus(status="ok")


@router.get(
    "/ready",
    response_model=HealthStatus,
    responses={503: {"model": HealthStatus}},
    tags=["ops"],
)
async def ready(services: ServicesDep) -> JSONResponse:
    """Readiness: Qdrant reachable with a valid collection, and >= 1 provider configured."""
    try:
        await services.cache.ping()
        qdrant_ok = True
    except CacheError:
        qdrant_ok = False
    checks = (
        ComponentCheck(name="qdrant", ok=qdrant_ok),
        ComponentCheck(name="providers", ok=bool(services.dispatcher.providers)),
    )
    is_ready = all(check.ok for check in checks)
    body = HealthStatus(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump(mode="json"))


@router.post(
    "/v1/inference",
    response_model=InferenceResponse,
    responses={
        **_ERROR_RESPONSES,
        200: {
            "content": {"text/event-stream": {}},
            "description": "JSON InferenceResponse, or an SSE stream when `stream` is true.",
        },
    },
    tags=["inference"],
)
async def inference(body: InferenceRequest, services: ServicesDep) -> Response:
    """Run inference; ``stream=true`` returns ``text/event-stream`` frames ending in ``[DONE]``."""
    if not body.stream:
        result = await services.engine.infer(body)
        return JSONResponse(content=result.model_dump(mode="json"))

    events = services.engine.stream(body)
    try:
        # Pull the first event before committing to a 200 so that routing,
        # auth and upstream connection failures still map to HTTP statuses.
        first = await anext(events)
    except BaseException:
        await events.aclose()
        raise
    return SSEResponse(sse_frames(first, events))
