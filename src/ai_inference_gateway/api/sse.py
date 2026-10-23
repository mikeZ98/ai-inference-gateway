"""Server-Sent Events framing and a leak-free streaming response."""

import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Final

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Send

from ai_inference_gateway.api.errors import gateway_error_detail
from ai_inference_gateway.core.errors import GatewayError
from ai_inference_gateway.schemas import StreamDelta, StreamError, StreamEvent, StreamSummary

logger = logging.getLogger(__name__)

DONE_FRAME: Final = b"data: [DONE]\n\n"
CLOSE_TIMEOUT_S: Final = 5.0
SSE_HEADERS: Final = {"cache-control": "no-cache", "x-accel-buffering": "no"}


def frame(event: StreamEvent) -> bytes:
    """Encode one event as an SSE ``data:`` frame (compact JSON, single line)."""
    return b"data: " + event.model_dump_json().encode() + b"\n\n"


async def sse_frames(
    first: StreamDelta | StreamSummary,
    events: AsyncGenerator[StreamDelta | StreamSummary],
) -> AsyncGenerator[bytes]:
    """Frame an already-started engine stream, always ending with ``[DONE]``.

    Errors after the headers were sent become an ``error`` frame instead of
    an HTTP status. Closing this generator closes ``events`` (and thereby the
    upstream connection).
    """
    async with aclosing(events):
        try:
            yield frame(first)
            async for event in events:
                yield frame(event)
        except GatewayError as exc:
            _, detail = gateway_error_detail(exc)
            logger.warning(
                "stream failed after start",
                extra={"error_type": type(exc).__name__, "detail": str(exc)},
            )
            yield frame(StreamError(error=detail))
        yield DONE_FRAME


class SSEResponse(StreamingResponse):
    """``text/event-stream`` response that always closes its generator.

    Starlette stops iterating on client disconnect (cancellation or
    ``ClientDisconnect``) but never calls ``aclose()`` on the body iterator,
    leaving the generator chain (and the upstream HTTP stream) suspended until
    garbage collection. This closes it deterministically, shielded from the
    surrounding cancellation and bounded by ``CLOSE_TIMEOUT_S``.
    """

    def __init__(self, body: AsyncGenerator[bytes]) -> None:
        """Stream ``body`` as SSE."""
        super().__init__(body, media_type="text/event-stream", headers=SSE_HEADERS)
        self._body = body

    async def stream_response(self, send: Send) -> None:
        """Send the stream, then close the body generator no matter how it ended."""
        try:
            await super().stream_response(send)
        finally:
            with anyio.move_on_after(CLOSE_TIMEOUT_S, shield=True):
                await self._body.aclose()
