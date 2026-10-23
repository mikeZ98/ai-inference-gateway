"""Domain error -> HTTP mapping with fixed, non-leaking public messages.

Exception messages can contain upstream details, URLs or model ids from
configuration, so they are logged server-side only; clients receive a stable
``code`` and a canned ``message`` (except ``ModelNotAvailableError``, whose
message only echoes the client's own validated model id).
"""

import logging
from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_inference_gateway.core.errors import (
    CacheError,
    EmbeddingError,
    GatewayError,
    ModelNotAvailableError,
    ProviderError,
    ProviderTimeoutError,
    RoutingError,
)
from ai_inference_gateway.schemas import ErrorDetail, ErrorResponse, FieldError

logger = logging.getLogger(__name__)

RETRY_AFTER_S = "1"


def gateway_error_detail(exc: GatewayError) -> tuple[int, ErrorDetail]:
    """Map a domain error to ``(status, public detail)``."""
    status: int
    if isinstance(exc, ModelNotAvailableError):
        status, detail = (
            HTTPStatus.BAD_REQUEST,
            ErrorDetail(code="model_not_available", message=str(exc)),
        )
    elif isinstance(exc, RoutingError):
        status, detail = (
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorDetail(code="no_route", message="No upstream is configured for this request."),
        )
    elif isinstance(exc, ProviderTimeoutError):
        status, detail = (
            HTTPStatus.GATEWAY_TIMEOUT,
            ErrorDetail(
                code="upstream_timeout", message="The upstream model timed out.", retryable=True
            ),
        )
    elif isinstance(exc, ProviderError) and exc.retryable:
        status, detail = (
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorDetail(
                code="upstream_unavailable",
                message="The upstream model is temporarily unavailable.",
                retryable=True,
            ),
        )
    elif isinstance(exc, ProviderError):
        status, detail = (
            HTTPStatus.BAD_GATEWAY,
            ErrorDetail(code="upstream_error", message="The upstream model rejected the request."),
        )
    elif isinstance(exc, CacheError | EmbeddingError):
        status, detail = (
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorDetail(
                code="dependency_unavailable",
                message="A gateway dependency is unavailable.",
                retryable=True,
            ),
        )
    else:
        status, detail = HTTPStatus.INTERNAL_SERVER_ERROR, _INTERNAL
    return status, detail


_INTERNAL: Final = ErrorDetail(code="internal_error", message="Internal gateway error.")


def _response(status: int, detail: ErrorDetail) -> JSONResponse:
    headers = {"retry-after": RETRY_AFTER_S} if detail.retryable else None
    body = ErrorResponse(error=detail).model_dump(mode="json")
    return JSONResponse(status_code=status, content=body, headers=headers)


async def _on_gateway_error(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, GatewayError):
        return await _on_unhandled(_, exc)
    status, detail = gateway_error_detail(exc)
    logger.warning(
        "request failed",
        extra={"error_type": type(exc).__name__, "status_code": status, "detail": str(exc)},
    )
    return _response(status, detail)


async def _on_validation_error(_: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else ()
    field_errors = tuple(
        FieldError(
            loc=tuple(part for part in error.get("loc", ()) if isinstance(part, str | int)),
            message=str(error.get("msg", "invalid value")),
            type=str(error.get("type", "value_error")),
        )
        for error in errors
    )
    detail = ErrorDetail(
        code="invalid_request", message="Request validation failed.", field_errors=field_errors
    )
    return _response(HTTPStatus.UNPROCESSABLE_ENTITY, detail)


async def _on_http_error(_: Request, exc: Exception) -> JSONResponse:
    status = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
    phrase = HTTPStatus(status).phrase if status in HTTPStatus else "Error"
    code = phrase.lower().replace(" ", "_").replace("-", "_")
    return _response(status, ErrorDetail(code=code, message=phrase))


async def _on_unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled error", exc_info=exc)
    return _response(HTTPStatus.INTERNAL_SERVER_ERROR, _INTERNAL)


def install_error_handlers(app: FastAPI) -> None:
    """Register all handlers so every error body is an ``ErrorResponse``."""
    app.add_exception_handler(GatewayError, _on_gateway_error)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(StarletteHTTPException, _on_http_error)
    app.add_exception_handler(Exception, _on_unhandled)
